import torch
from torch import nn
import torch.nn.functional as F

from src.models.model import Model, ModelConfig

class ResNetBlock(nn.Module):

    def __init__(
            self, 
            input_size:int,
            input_channels:int,
            n_channels:int, 
            kernel_size:int, 
            stride:int, 
            padding:int,
            layers:int = 2,
            pool_kernel_size:int = 2,
            pool_stride:int = 2,
            dropout: float = 0.2,
        ):
        super(ResNetBlock, self).__init__()
        self.layers = layers
        self.input_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm_initial = nn.BatchNorm2d(n_channels)
        self.pool_initial = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        output_size = self._calculate_output_size_conv(
            input_size=input_size,
            kernel_size=kernel_size,
            padding_size=padding,
            stride=stride
        )
        output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.conv = nn.ModuleList([nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ) for i in range(layers)])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm2d(n_channels) 
            for _ in range(layers)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        input_size = output_size
        output_size = self._calculate_output_size_flow(
            input_size=output_size,
            kernel_size=kernel_size,
            padding_size=padding,
            stride=stride,
            resnet_layers=layers,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            with_pool=False,
        )
        if input_size != output_size:
            self.downsample = True
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=int(((input_size - 1) - (output_size - 1) * 1 + 2 * 0) / 1 + 1),
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(n_channels)
            )
        else:
            self.downsample = False
            self.downsample_layer = None
        self.pool_out = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )

    def _calculate_output_size_flow(
            self,
            input_size,
            kernel_size,
            stride,
            padding_size,
            resnet_layers,
            pool_kernel_size,
            pool_stride,
            with_pool=True
        ):
        output_size = input_size
        for _ in range(resnet_layers):
            output_size = self._calculate_output_size_conv(
                input_size=output_size,
                kernel_size=kernel_size,
                padding_size=padding_size,
                stride=stride
            )
        if with_pool:
            output_size = self._calculate_output_size_pool(
                input_size=output_size,
                kernel_size=pool_kernel_size,
                stride=pool_stride
            )
        return output_size
    
    def _calculate_output_size_conv(self, input_size, kernel_size, padding_size, stride):
        return ((input_size - kernel_size + 2 * padding_size) // stride) + 1
    
    def _calculate_output_size_pool(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1

    def forward(self, x):
        x = self.input_conv(x)
        x = self.batch_norm_initial(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool_initial(x)
        identity = x
        for i in range(self.layers):
            if i == self.layers - 1:
                x = self.conv[i](x)
                x = self.batch_norm[i](x)
                if self.downsample:
                    identity = self.downsample_layer(identity)
                x += identity
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.conv[i](x)
                x = self.batch_norm[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        x = self.pool_out(x)
        return x
    
class ResNet(Model):

    def __init__(
            self,
            config: ModelConfig,
        ):
        super(ResNet, self).__init__(config)
        self.resnet_blocks, output_size = self._create_resnet_blocks(
            resnet_blocks=self.config.config["resnet_blocks"],
            resnet_channels=self.config.config["resnet_channels"],
            resnet_kernel_sizes=self.config.config["resnet_kernel_sizes"],
            resnet_strides=self.config.config["resnet_strides"],
            resnet_padding_sizes=self.config.config["resnet_padding_sizes"],
            resnet_layers=self.config.config["resnet_layers"],
            pool_kernel_size=self.config.config["pool_kernel_size"],
            pool_stride=self.config.config["pool_stride"],
            input_size=self.config.config["output_size"][0],
            input_channels=self.config.config["input_channels"],
            dropout=self.config.config["dropout"]
        )
        fc1_input_dims = self.config.config["resnet_channels"][-1] * output_size * output_size
        
        self.fc1 = nn.Linear(fc1_input_dims, self.config.config["fc1_output_dims"])
        self.fc2 = nn.Linear(self.config.config["fc1_output_dims"], self.config.config["fc2_output_dims"])
        self.fc3 = nn.Linear(self.config.config["fc2_output_dims"], self.config.config["n_classes"])
        self.output_activation = nn.Softmax(dim=1)

    def _create_resnet_blocks(
            self,
            resnet_blocks,
            resnet_channels,
            resnet_kernel_sizes,
            resnet_strides,
            resnet_padding_sizes,
            resnet_layers,
            pool_kernel_size,
            pool_stride,
            input_size,
            input_channels,
            dropout
        ):
        block_list = []
        for i in range(resnet_blocks):
            new_block = ResNetBlock(
                input_size=input_size,
                input_channels=input_channels,
                n_channels=resnet_channels[i],
                kernel_size=resnet_kernel_sizes[i],
                padding=resnet_padding_sizes[i],
                stride=resnet_strides[i],
                layers=resnet_layers[i],
                pool_kernel_size=pool_kernel_size,
                pool_stride=pool_stride,
                dropout=dropout
            )
            input_size = new_block.output_size
            input_channels = resnet_channels[i]
            block_list.append(new_block)
        return nn.ModuleList(block_list), input_size

    def _calculate_output_size_conv(self, input_size, kernel_size, padding_size, stride):
        return ((input_size - kernel_size + 2 * padding_size) // stride) + 1
    
    def _calculate_output_size_pool(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x / 255.0
        for block in self.resnet_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x: torch.Tensor)->torch.Tensor:
        self.eval()
        with torch.no_grad():
            x = self.output_activation(self.forward(x))
        return x
    
 