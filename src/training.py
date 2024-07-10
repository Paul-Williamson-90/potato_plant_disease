from typing import Any, Union
import logging
import os
import time

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.logger import TBXLoggerCallback

from src.dataset import dataset_factory
from src.trainer import Trainer
from src.transform import ImageTransform
from src.models.resnet import ResNet
from src.models.model import ModelConfig
from src.loss_functions import CategoricalFocalLoss

from utils.decorators import inject_config_params, inject_train_config

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

SEARCH_METHODS = {
    "choice": tune.choice,
    "uniform": tune.uniform,
    "loguniform": tune.loguniform,
    "randint": tune.randint,
    "grid_search": tune.grid_search,
    "lograndint": tune.lograndint,
    "quniform": tune.quniform,
    "qloguniform": tune.qloguniform,
    "qrandint": tune.qrandint,
    "qlograndint": tune.qlograndint,
    "randn": tune.randn,
    "qrandn": tune.qrandn,
    "register_env": tune.register_env,
    "register_trainable": tune.register_trainable,
}
SEARCH_ARG_UNPACK = [
    "uniform",
    "loguniform",
    "randint",
    "grid_search",
    "lograndint",
    "quniform",
    "qloguniform",
    "qrandint",
    "qlograndint",
    "randn",
    "qrandn",
]

def check_dir_and_subdir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_paths():
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    new_save_path = os.path.abspath(f"model_saves/resnet/{now}")
    check_dir_and_subdir_exists(new_save_path)

    logging_path = os.path.abspath(f"logs/model_training.csv")
    check_dir_and_subdir_exists("logs")
    return new_save_path, logging_path


def get_transform(output_size, rescale, random_crop):
    return ImageTransform(
        output_size=output_size,
        rescale=rescale,
        random_crop=random_crop,
        to_tensor=True
    )

@inject_config_params("image_meta_path", "image_data_path")
def prepare_dataset(image_meta_path, image_data_path, num_workers, batch_size, transform):
    train_dataset, train_loader = dataset_factory(
        image_meta_path=image_meta_path,
        image_data_path=image_data_path,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=transform
    )
    _, val_loader = dataset_factory(
        image_meta_path=image_meta_path,
        image_data_path=image_data_path,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        transform=transform.no_augment()
    )

    class_weights = train_dataset.return_class_weights()
    return train_loader, val_loader, class_weights

def construct_model(config):
    config = ModelConfig(
        **config
    )
    model = ResNet(
        config=config
    )
    return model

def create_training_artefacts(model, learning_rate, weight_decay, gamma, reduction, class_weights):
    loss_fn = CategoricalFocalLoss(
        gamma=gamma,
        alpha=class_weights,
        reduction=reduction
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LinearLR(optimizer)
    return loss_fn, optimizer, scheduler

def setup_train(config):
    transform = get_transform(config["output_size"], config["rescale"], config["random_crop"])
    train_loader, val_loader, class_weights = prepare_dataset(
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        transform=transform
    )
    model = construct_model(config)
    loss_fn, optimizer, scheduler = create_training_artefacts(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        gamma=config["gamma"],
        reduction=config["reduction"],
        class_weights=class_weights
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config["n_epochs"],
        save_location=config["save_location"],
        training_log=config["training_log"],
        early_stopping=config["early_stopping"],
        early_stopping_metric=config["early_stopping_metric"],
        min_or_max=config["min_or_max"],
        patience=config["patience"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        verbose=config["verbose"],
        additional_reporting=config,
    )
    return trainer

@inject_train_config(config_path='model_train_config.yaml')
def train_one_model(config:dict):
    new_save_path, logging_path = create_paths()
    config["save_location"] = new_save_path
    config["training_log"] = logging_path
    trainer = setup_train(config)
    trainer.train()

def fetch_search_method(search_method:str):
    return SEARCH_METHODS[search_method]

def unpack_search_args(search_method:str, values:Any):
    try:
        if search_method in SEARCH_ARG_UNPACK:
            return fetch_search_method(search_method)(*values)
        return fetch_search_method(search_method)(values)
    except Exception as e:
        logger.error(f"Error in unpacking search args: {search_method} with values: {values}")
        raise e

def objective(config:dict):
    trainer = setup_train(config)

    for n_epoch in range(trainer.n_epochs):
        stop_training, metrics = trainer.epoch(n_epoch)
        
        train.report(metrics)

        if stop_training:
            break

@inject_train_config(config_path='model_hpo_train_config.yaml')
def hpo_train(config:dict[str, Union[dict[str, Any], Any]]):

    search_space = {
        key: unpack_search_args(value["search_method"], value["values"]) for key, value in config.items()
        if isinstance(value, dict)
    }

    new_save_path, logging_path = create_paths()
    search_space["save_location"] = new_save_path
    search_space["training_log"] = logging_path
    search_space["verbose"] = logging_path

    tb_callback = TBXLoggerCallback()
    storage_path = os.path.abspath("raytune_experiments")
    check_dir_and_subdir_exists(storage_path)

    algo = OptunaSearch()

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric=config["metric"],
            mode=config["mode"],
            search_alg=algo,
            num_samples=config["num_samples"],
            max_concurrent_trials=config["max_concurrent_trials"],
        ),
        param_space=search_space,
        run_config=train.RunConfig(callbacks=[tb_callback],storage_path=storage_path),
    )
    results = tuner.fit()
    logging.info("Best config is: %s", results.get_best_result().config)