from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from src.dataset import dataset_factory
from src.trainer import Trainer
from src.transform import ImageTransform
from src.models.resnet import ResNet
from src.models.model import ModelConfig
from src.loss_functions import CategoricalFocalLoss

from utils.decorators import inject_config_params


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