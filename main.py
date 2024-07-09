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
        # DATA PREPROCESSING ARGS
        "output_size": [255, 255],
        "input_channels": 3,
        "rescale": True,
        "random_crop": True,
        # DATA LOADING ARGS
        "num_workers": 0,
        "batch_size": 64,
        # TRAINING ARGS
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
        # MODEL ARGS
        "resnet_blocks": 3,
        "resnet_channels": [128, 64, 32],
        "resnet_kernel_sizes": [5, 4, 3],
        "resnet_strides": [1, 1, 1],
        "resnet_padding_sizes": [0, 0, 0],
        "resnet_layers": [2, 2, 1],
        "pool_kernel_size": 2,
        "pool_stride": 2,
        "dropout": 0.0,
        "fc1_output_dims": 256,
        "fc2_output_dims": 128,
        "n_classes": 3
    }

    trainer = setup_train(config)
    trainer.train()

if __name__ == "__main__":
    main()