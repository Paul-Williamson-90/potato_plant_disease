import json
import os

import torch
from torch import nn

class ModelConfig:

    def __init__(
            self,
            **kwargs
    ):
        self.config: dict = kwargs

    def save_config(
            self,
            save_path: str,
    ):
        with open(save_path, "w") as f:
            json.dump(self.config, f)

    @classmethod
    def load_config(
            cls,
            config_path: str,
    ):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config)
    
class Model(nn.Module):

    def __init__(
            self,
            config: ModelConfig,
    ):
        super(Model, self).__init__()
        self.config = config

    def save_pretrained(
            self,
            save_location: str,
    ):
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        self.config.save_config(os.path.join(save_location, "config.json"))
        torch.save(self.state_dict(), os.path.join(save_location, "model.pth"))

    @classmethod
    def load_pretrained(
            cls,
            model_dir: str,
    ):
        config = ModelConfig.load_config(os.path.join(model_dir, "config.json"))
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
        return model