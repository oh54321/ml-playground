from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, Tuple, Union

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
        include_groupnorm: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.layers = layers
        self.include_groupnorm = include_groupnorm
        self.network = self.create_network()

    def create_layer(self, include_relu: bool) -> nn.Module:
        module_dict = OrderedDict()
        module_dict["conv"] = nn.Conv2d(
            self.n_channels,
            self.n_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=not self.include_groupnorm,
        )
        if self.include_groupnorm:
            module_dict["groupnorm"] = nn.GroupNorm(4, self.n_channels)
        if include_relu:
            module_dict["relu"] = nn.ReLU()
        return nn.Sequential(module_dict)

    def create_network(self) -> nn.Sequential:
        module_dict = OrderedDict()
        for idx in range(1, self.layers):
            module_dict[f"layer{idx}"] = self.create_layer(include_relu=True)
        module_dict[f"layer{self.layers}"] = self.create_layer(include_relu=False)
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
        output_dim: int,
        include_groupnorm: bool = True,
    ) -> None:
        super().__init__()
        module_dict = OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=not include_groupnorm,
                    ),
                ),
            ]
        )
        if include_groupnorm:
            module_dict["groupnorm"] = nn.GroupNorm(4, out_channels)
        module_dict["relu"] = nn.ReLU()
        module_dict["flatten"] = nn.Flatten()
        module_dict["linear"] = nn.Linear(out_channels * height * width, output_dim)
        self.network = nn.Sequential(module_dict)

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
    include_groupnorm: bool = True

@dataclass(frozen=True)
class ActorCriticOutput:
    action_logits: torch.Tensor
    value: torch.Tensor


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = self.create_trunk()
        self.action_head = self.create_head(self.config.output_dim)
        self.value_head = self.create_head(1)

    def create_stem(self) -> nn.Module:
        module_dict = OrderedDict()
        module_dict["conv"] = nn.Conv2d(
            self.config.in_channels,
            self.config.n_channels,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2,
            bias=not self.config.include_groupnorm,
        )
        if self.config.include_groupnorm:
            module_dict["groupnorm"] = nn.GroupNorm(4, self.config.n_channels)
        module_dict["relu"] = nn.ReLU()
        return nn.Sequential(module_dict)

    def create_body(self) -> nn.Module:
        module_dict = OrderedDict()
        for idx in range(1, self.config.blocks + 1):
            module_dict[f"layer{idx}"] = ResBlock(
                self.config.n_channels,
                kernel_size=self.config.kernel_size,
                layers=self.config.block_layers,
                include_groupnorm=self.config.include_groupnorm,
            )
        return nn.Sequential(module_dict)

    def create_head(self, output_dim: int) -> nn.Module:
        return FlattenHead(
            self.config.n_channels,
            self.config.head_channels,
            self.config.height,
            self.config.width,
            output_dim,
            include_groupnorm=self.config.include_groupnorm,
        )

    def create_trunk(self) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    ("stem", self.create_stem()),
                    ("body", self.create_body()),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> ActorCriticOutput:
        trunk_output = self.trunk(x)
        action_logits = self.action_head(trunk_output)
        value = self.value_head(trunk_output).squeeze(dim=-1)
        return ActorCriticOutput(action_logits=action_logits, value=value)
