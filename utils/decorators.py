import yaml

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

def inject_config_params(*param_names, config_path='config.yaml'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = load_config(config_path)
            injected_params = {param: config[param] for param in param_names if param in config}
            return func(*args, **kwargs, **injected_params)
        return wrapper
    return decorator