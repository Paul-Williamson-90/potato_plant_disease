import os
import time

import argparse

from src.training import setup_train
from utils.decorators import inject_config_single_model_train

def check_dir_and_subdir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_paths():
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    new_save_path = f"model_saves/resnet/{now}"
    check_dir_and_subdir_exists(new_save_path)

    logging_path = f"logs/model_training.csv"
    check_dir_and_subdir_exists("logs")
    return new_save_path, logging_path

@inject_config_single_model_train()
def train_one_model(config:dict):
    new_save_path, logging_path = create_paths()
    config["save_location"] = new_save_path
    config["training_log"] = logging_path
    trainer = setup_train(config)
    trainer.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hpo', 
        action='store_true', 
        help='Whether to run hyperparameter optimization or single model train'
    )
    args = parser.parse_args()

    if args.hpo:
        raise NotImplementedError("Hyperparameter optimization not yet implemented")
    else:
        train_one_model()

if __name__ == "__main__":
    main()