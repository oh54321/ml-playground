from typing import Iterator, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkAccessorMixin:
    network: nn.Sequential

    def __getitem__(self, key: Union[int, slice, str]) -> nn.Module:
        if isinstance(key, str):
            return getattr(self.network, key)
        return self.network[key]

    def __len__(self) -> int:
        return len(self.network)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.network)

    def named_layers(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.network.named_children()


class ResBlock(NetworkAccessorMixin, nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 3,
        layers: int = 2,
        include_batchnorm: bool = True
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.layers = layers
        self.include_batchnorm = include_batchnorm
        self.network = self.create_network()

    def create_layer(
        self,
        include_relu: bool
    ) -> nn.Module:
        module_dict = OrderedDict()
        module_dict['conv'] = nn.Conv2d(
            self.n_channels,
            self.n_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size//2,
            bias=not self.include_batchnorm
        )
        if self.include_batchnorm:
            module_dict['batchnorm'] = nn.BatchNorm2d(self.n_channels)
        if include_relu:
            module_dict['relu'] = nn.ReLU()
        return nn.Sequential(module_dict)

    def create_network(self) -> nn.Sequential:
        module_dict = OrderedDict()
        for idx in range(1, self.layers):
            module_dict[f'layer{idx}'] = self.create_layer(include_relu=True)
        module_dict[f'layer{self.layers}'] = self.create_layer(include_relu=False)
        return nn.Sequential(module_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.network(x))


class FlattenHead(NetworkAccessorMixin, nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(out_channels * height * width, output_dim))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class ResNetConfig:
    in_channels: int
    n_channels: int
    blocks: int
    height: int
    width: int
    output_dim: int
    head_channels: int = 32
    kernel_size: int = 3
    block_layers: int = 2
    include_batchnorm: bool = True


class ResNet(NetworkAccessorMixin, nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config
        self.network = self.create_network()

    def create_stem(self) -> nn.Module:
        module_dict = OrderedDict()
        module_dict['conv'] = nn.Conv2d(
            self.config.in_channels,
            self.config.n_channels,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size//2,
            bias=not self.config.include_batchnorm
        )
        if self.config.include_batchnorm:
            module_dict['batchnorm'] = nn.BatchNorm2d(self.config.n_channels)
        module_dict['relu'] = nn.ReLU()
        return nn.Sequential(module_dict)

    def create_body(self) -> nn.Module:
        module_dict = OrderedDict()
        for idx in range(1, self.config.blocks + 1):
            module_dict[f'layer{idx}'] = ResBlock(
                self.config.n_channels,
                kernel_size=self.config.kernel_size,
                layers=self.config.block_layers,
                include_batchnorm=self.config.include_batchnorm
            )
        return nn.Sequential(module_dict)

    def create_head(self) -> nn.Module:
        return FlattenHead(
            self.config.n_channels,
            self.config.head_channels,
            self.config.height,
            self.config.width,
            self.config.output_dim
        )

    def create_network(self) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            ('stem', self.create_stem()),
            ('body', self.create_body()),
            ('head', self.create_head())
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
