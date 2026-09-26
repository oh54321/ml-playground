"""Convolutional actor-critic models for spatial observations."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from core.models.actor_critic import ActorCriticModel, ActorCriticOutput


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


class ConvResidualBlock(NetworkAccessorMixin, nn.Module):
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
class ConvActorCriticConfig:
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
    # With separate trunks, n_channels and blocks apply to EACH network.
    shared_trunk: bool = True


class ConvActorCriticModel(ActorCriticModel):
    """A standard residual CNN with policy and value heads."""

    def __init__(self, config: ConvActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        if config.shared_trunk:
            self.trunk = self.create_trunk()
        else:
            self.actor_trunk = self.create_trunk()
            self.critic_trunk = self.create_trunk()
        self.action_head = self.create_action_head()
        self.value_head = self.create_value_head()

    @property
    def policy_trunk(self) -> nn.Sequential:
        """The policy encoder, exposed for model-specific visualization."""
        return self.trunk if self.config.shared_trunk else self.actor_trunk

    @property
    def has_shared_representation(self) -> bool:
        return self.config.shared_trunk

    def representation_parameters(self) -> Iterable[nn.Parameter]:
        if self.config.shared_trunk:
            yield from self.trunk.parameters()
        else:
            yield from self.actor_trunk.parameters()
            yield from self.critic_trunk.parameters()

    def create_stem(self) -> nn.Module:
        modules = OrderedDict()
        modules["conv"] = nn.Conv2d(
            self.config.in_channels,
            self.config.n_channels,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2,
            bias=not self.config.include_groupnorm,
        )
        if self.config.include_groupnorm:
            modules["groupnorm"] = nn.GroupNorm(4, self.config.n_channels)
        modules["relu"] = nn.ReLU()
        return nn.Sequential(modules)

    def create_body(self) -> nn.Sequential:
        modules = OrderedDict()
        for idx in range(1, self.config.blocks + 1):
            modules[f"layer{idx}"] = ConvResidualBlock(
                self.config.n_channels,
                kernel_size=self.config.kernel_size,
                layers=self.config.block_layers,
                include_groupnorm=self.config.include_groupnorm,
            )
        return nn.Sequential(modules)

    def create_head(self, output_dim: int) -> FlattenHead:
        return FlattenHead(
            self.config.n_channels,
            self.config.head_channels,
            self.config.height,
            self.config.width,
            output_dim,
            include_groupnorm=self.config.include_groupnorm,
        )

    def create_action_head(self) -> nn.Module:
        return self.create_head(self.config.output_dim)

    def create_value_head(self) -> nn.Module:
        return self.create_head(1)

    def create_trunk(self) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    ("stem", self.create_stem()),
                    ("body", self.create_body()),
                ]
            )
        )

    def forward(self, observations: torch.Tensor) -> ActorCriticOutput:
        if self.config.shared_trunk:
            policy_features = value_features = self.trunk(observations)
        else:
            policy_features = self.actor_trunk(observations)
            value_features = self.critic_trunk(observations)
        return ActorCriticOutput(
            action_logits=self.action_head(policy_features),
            value=self.value_head(value_features).squeeze(dim=-1),
        )
