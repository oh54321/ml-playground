"""Residual convolutional actor-critic for spatial observations."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from core.models.actor_critic import ActorCriticModel, ActorCriticOutput


@dataclass(frozen=True)
class ConvActorCriticConfig:
    """Observation shape, action count, and sizes for each independent trunk."""

    in_channels: int
    height: int
    width: int
    output_dim: int
    n_channels: int = 64
    blocks: int = 4
    head_channels: int = 32
    kernel_size: int = 3
    block_layers: int = 2

    def __post_init__(self) -> None:
        if min(self.in_channels, self.height, self.width, self.output_dim,
               self.n_channels, self.head_channels) <= 0:
            raise ValueError("Input dimensions, action count, and channel counts must be positive")
        if self.blocks < 0:
            raise ValueError("The number of residual blocks cannot be negative")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd to preserve spatial dimensions")
        if self.block_layers <= 0:
            raise ValueError("Each residual block must have at least one layer")


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, block_layers: int = 2):
        super().__init__()
        layers = OrderedDict()
        for index in range(1, block_layers + 1):
            modules = OrderedDict([
                ("conv", nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)),
                ("groupnorm", nn.GroupNorm(4, channels)),
            ])
            if index < block_layers:
                modules["relu"] = nn.ReLU()
            layers[f"layer{index}"] = nn.Sequential(modules)
        self.network = nn.Sequential(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(inputs + self.network(inputs))


class FlattenHead(nn.Module):
    def __init__(self, channels: int, head_channels: int, height: int, width: int, outputs: int):
        super().__init__()
        self.network = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, head_channels, 1, bias=False)),
            ("groupnorm", nn.GroupNorm(4, head_channels)),
            ("relu", nn.ReLU()),
            ("flatten", nn.Flatten()),
            ("linear", nn.Linear(head_channels * height * width, outputs)),
        ]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ConvActorCriticModel(ActorCriticModel):
    """Separate policy and value CNNs over a fixed spatial observation shape."""

    architecture = "cnn"

    def __init__(self, config: ConvActorCriticConfig):
        super().__init__()
        if self.architecture == "cnn" and (config.n_channels % 4 or config.head_channels % 4):
            raise ValueError("CNN trunk and head channels must be divisible by 4")
        self.config = config
        self.actor_trunk = self.create_trunk()
        self.critic_trunk = self.create_trunk()
        self.action_head = self.create_action_head()
        self.value_head = self.create_value_head()

    def create_stem(self) -> nn.Module:
        config = self.config
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(config.in_channels, config.n_channels, config.kernel_size,
                               padding=config.kernel_size // 2, bias=False)),
            ("groupnorm", nn.GroupNorm(4, config.n_channels)),
            ("relu", nn.ReLU()),
        ]))

    def create_body(self) -> nn.Module:
        config = self.config
        blocks = OrderedDict()
        for index in range(1, config.blocks + 1):
            blocks[f"layer{index}"] = ConvResidualBlock(
                config.n_channels, config.kernel_size, config.block_layers,
            )
        return nn.Sequential(blocks)

    def create_trunk(self) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            ("stem", self.create_stem()), ("body", self.create_body()),
        ]))

    def create_action_head(self) -> nn.Module:
        return FlattenHead(self.config.n_channels, self.config.head_channels,
                           self.config.height, self.config.width, self.config.output_dim)

    def create_value_head(self) -> nn.Module:
        return FlattenHead(self.config.n_channels, self.config.head_channels,
                           self.config.height, self.config.width, 1)

    def forward(self, observations: torch.Tensor) -> ActorCriticOutput:
        expected = (self.config.in_channels, self.config.height, self.config.width)
        if observations.ndim != 4 or observations.shape[1:] != expected:
            raise ValueError(f"Expected observations shaped (batch, {expected[0]}, {expected[1]}, {expected[2]})")
        return ActorCriticOutput(
            action_logits=self._forward_actor(observations),
            value=self._forward_critic(observations),
        )

    def _forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.actor_trunk(observations))

    def _forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(observations)).squeeze(-1)

    def compile_trunks(self) -> None:
        """Compile calls while keeping model parameters and checkpoint keys intact."""
        self._forward_actor = torch.compile(self._forward_actor, fullgraph=True)
        self._forward_critic = torch.compile(self._forward_critic, fullgraph=True)
