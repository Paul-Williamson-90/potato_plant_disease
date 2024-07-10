import argparse

from src.training import hpo_train, train_one_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hpo', 
        action='store_true', 
        help='Whether to run hyperparameter optimization (True) or single model train (False, default)'
    )
    args = parser.parse_args()

    if args.hpo:
        hpo_train()
    else:
        train_one_model()

if __name__ == "__main__":
    main()