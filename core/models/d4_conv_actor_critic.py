"""D4-aware convolutional actor-critic models for square-grid observations."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
from torch import nn

from core.models.actor_critic import ActorCriticOutput
from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
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


__all__ = ["D4ConvActorCriticModel"]


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

    def _forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.actor_trunk(observations))

    def _forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(observations)).squeeze(-1)

    def forward(self, observations: torch.Tensor) -> ActorCriticOutput:
        if self.config.shared_trunk:
            return super().forward(observations)
        observations = self.encode_observations(observations)
        return ActorCriticOutput(
            self._forward_actor(observations), self._forward_critic(observations)
        )

    def compile_trunks(self) -> None:
        """Compile independent actor/critic calls without changing checkpoint keys."""
        if self.config.shared_trunk:
            raise ValueError("Compilation currently requires separate actor/critic trunks")
        import torch._functorch.config

        # PPO measures several component gradients from the same retained graph.
        # Buffer donation would prevent those repeated backward calls in PyTorch 2.8.
        torch._functorch.config.donated_buffer = False
        self._forward_actor = torch.compile(self._forward_actor, fullgraph=True)
        self._forward_critic = torch.compile(self._forward_critic, fullgraph=True)

    @contextmanager
    def prepared_inference(self) -> Iterator[None]:
        """Prepare D4 parameters for one rollout; clear before updates.

        Model weights and device must remain fixed inside this scope.
        """
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
