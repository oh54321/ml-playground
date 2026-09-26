"""D4-aware convolutional actor-critic models for square-grid observations."""

from collections import OrderedDict
from typing import Mapping
from torch import nn

from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
from core.models.d4_layers import (
    D4ActionHead,
    D4GroupNorm,
    D4LiftConv,
    D4ResBlock,
    D4ValueHead,
    NUM_SYMMETRIES,
)
from core.symmetries.d4_transforms import BoardTransform, D4_TRANSFORMS


class D4ConvActorCriticModel(ConvActorCriticModel):
    """A group-equivariant D4 CNN with invariant value and equivariant policy."""

    def __init__(
        self,
        config: ConvActorCriticConfig,
        action_permutations: Mapping[BoardTransform, tuple[int, ...]],
    ) -> None:
        if set(action_permutations) != set(BoardTransform):
            raise ValueError("D4 action mappings must define all eight transforms")
        if config.height != config.width:
            raise ValueError("D4 convolution requires a square board")
        if config.n_channels % NUM_SYMMETRIES:
            raise ValueError("D4 channels must be divisible by 8")
        if config.output_dim != len(action_permutations[D4_TRANSFORMS[0]]):
            raise ValueError("D4 action mappings must match the configured output size")
        self.__dict__["action_permutations"] = action_permutations
        super().__init__(config)

    def create_stem(self) -> nn.Module:
        families = self.config.n_channels // NUM_SYMMETRIES
        modules = OrderedDict(
            [
                (
                    "conv",
                    D4LiftConv(
                        self.config.in_channels,
                        families,
                        self.config.kernel_size,
                        bias=not self.config.include_groupnorm,
                    ),
                ),
            ]
        )
        if self.config.include_groupnorm:
            modules["groupnorm"] = D4GroupNorm(families)
        modules["relu"] = nn.ReLU()
        return nn.Sequential(modules)

    def create_body(self) -> nn.Sequential:
        modules = OrderedDict()
        families = self.config.n_channels // NUM_SYMMETRIES
        for idx in range(1, self.config.blocks + 1):
            modules[f"layer{idx}"] = D4ResBlock(
                families,
                kernel_size=self.config.kernel_size,
                layers=self.config.block_layers,
                include_groupnorm=self.config.include_groupnorm,
            )
        return nn.Sequential(modules)

    def create_action_head(self) -> nn.Module:
        return D4ActionHead(
            self.config.n_channels,
            self.config.height,
            self.config.width,
            self.config.head_channels,
            self.action_permutations,
        )

    def create_value_head(self) -> nn.Module:
        return D4ValueHead(
            self.config.n_channels,
            self.config.height,
            self.config.width,
            self.config.head_channels,
        )
