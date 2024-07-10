# Potato Plant Disease Classification

## Dataset Setup
1. Download the dataset
```bash
pip install -r requirements.txt
kaggle datasets download -d hafiznouman786/potato-plant-diseases-data
```
2. Unzip into your root dir of the project
3. Run the data preparation script
```bash
python utils/dataset_setup.py
```

## Training a Model
There are two options:
1. Run a single model training experiment
    - Use model_train_config.yaml to set desired hyperparameters
    - Run the following in the CLI:
```bash
python main.py
```
2. Run RayTune with Optuna for HPO
    - Use model_hpo_train_config.yaml to set desired search space
    - The search space has named functions from Tune (e.g. choice, uniform, etc)
    - Run the following in CLI:
```bash
python main.py --hpo
```