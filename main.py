import os
import time

from src.training import setup_train

def check_dir_and_subdir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    new_save_path = f"model_saves/resnet/{now}"
    check_dir_and_subdir_exists(new_save_path)

    logging_path = f"logs/model_training.csv"
    check_dir_and_subdir_exists(logging_path)

    config = {
        "output_size": [255, 255],
        "rescale": True,
        "random_crop": True,
        "num_workers": 0,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gamma": 2.0,
        "reduction": "mean",
        "n_epochs": 10,
        "save_location": new_save_path,
        "training_log": logging_path,
        "early_stopping": True,
        "early_stopping_metric": "val_loss",
        "min_or_max": "min",
        "patience": 5,
        "gradient_accumulation_steps": 8,
        "verbose": True,
    }

    trainer = setup_train(config)
    trainer.train()

if __name__ == "__main__":
    main()