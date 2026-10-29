"""D4-equivariant convolutional actor-critic for square spatial observations."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
from torch import nn

from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from core.models.d4_layers import (
    D4ActionHead,
    D4GroupConv,
    D4GroupNorm,
    D4LiftConv,
    D4ResBlock,
    D4ValueHead,
    NUM_SYMMETRIES,
    _D4ScalarProbe,
)
from core.symmetries.d4_transforms import BoardTransform, D4_TRANSFORMS


class D4ConvActorCriticModel(ConvActorCriticModel):
    """A group-equivariant CNN with invariant value and equivariant policy."""

    architecture = "d4"

    def __init__(
        self,
        config: ConvActorCriticConfig,
        action_permutations: Mapping[BoardTransform, tuple[int, ...]],
    ) -> None:
        if config.height != config.width:
            raise ValueError("D4 observations must have equal height and width")
        if config.n_channels % NUM_SYMMETRIES:
            raise ValueError("D4 trunk channels must be divisible by 8")
        if set(action_permutations) != set(BoardTransform):
            raise ValueError("D4 action mappings must define all eight transforms")
        if len(action_permutations[D4_TRANSFORMS[0]]) != config.output_dim:
            raise ValueError("D4 action mappings must match the configured output size")
        self.action_permutations = action_permutations
        super().__init__(config)

    def create_action_head(self) -> nn.Module:
        return D4ActionHead(self.config.n_channels, self.config.height, self.config.width,
                            self.config.head_channels, self.action_permutations)

    def create_value_head(self) -> nn.Module:
        return D4ValueHead(self.config.n_channels, self.config.height, self.config.width,
                           self.config.head_channels)

    @contextmanager
    def prepared_inference(self) -> Iterator[None]:
        """Reuse expanded D4 parameters while fixed weights serve a rollout."""
        layers = [layer for layer in self.modules()
                  if isinstance(layer, (D4LiftConv, D4GroupConv, _D4ScalarProbe))]
        if any(getattr(layer, "_prepared_parameters", None) is not None for layer in layers):
            raise RuntimeError("D4 inference preparation scopes cannot be nested")
        try:
            with torch.no_grad():
                for layer in layers:
                    layer._prepared_parameters = layer._expand_parameters()
            yield
        finally:
            for layer in layers:
                layer._prepared_parameters = None

    def create_stem(self) -> nn.Module:
        families = self.config.n_channels // NUM_SYMMETRIES
        return nn.Sequential(OrderedDict([
            ("conv", D4LiftConv(self.config.in_channels, families, self.config.kernel_size)),
            ("groupnorm", D4GroupNorm(families)),
            ("relu", nn.ReLU()),
        ]))

    def create_body(self) -> nn.Sequential:
        modules = OrderedDict()
        families = self.config.n_channels // NUM_SYMMETRIES
        for idx in range(1, self.config.blocks + 1):
            modules[f"layer{idx}"] = D4ResBlock(families, self.config.kernel_size, self.config.block_layers)
        return nn.Sequential(modules)
