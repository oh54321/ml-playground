from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from games.rl_2048.d4 import (
    D4ActionHead,
    D4GroupNorm,
    D4LiftConv,
    D4ResBlock,
    D4ValueHead,
    NUM_SYMMETRIES as D4_NUM_SYMMETRIES,
    inverse_transform as _d4_inverse_transform,
    transform as _d4_transform,
)


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


class D4ResidualStem(nn.Module):
    """A batched, residual feature mixer over 2048's eight board symmetries.

    The shared stem is applied to every transformed view.  Each result is
    aligned back to the input frame before the views are averaged, so the
    mixed feature map remains spatially equivariant.  The zero-initialized
    projection makes this module start exactly as the identity-view stem and
    learn to use the symmetry path during training.
    """

    NUM_SYMMETRIES = D4_NUM_SYMMETRIES

    def __init__(self, shared_stem: nn.Module, n_channels: int) -> None:
        super().__init__()
        self.shared_stem = shared_stem
        self.residual_projection = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        views = torch.stack(
            [_d4_transform(x, index) for index in range(self.NUM_SYMMETRIES)],
            dim=1,
        )
        view_features = self.shared_stem(views.flatten(0, 1))
        view_features = view_features.reshape(
            batch_size,
            self.NUM_SYMMETRIES,
            view_features.shape[1],
            view_features.shape[2],
            view_features.shape[3],
        )
        aligned_features = torch.stack(
            [
                _d4_inverse_transform(view_features[:, index], index)
                for index in range(self.NUM_SYMMETRIES)
            ],
            dim=1,
        )
        mixed_features = aligned_features.mean(dim=1)
        identity_features = aligned_features[:, 0]
        return identity_features + self.residual_projection(mixed_features)


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
    # With separate trunks, n_channels and blocks apply to EACH network.
    # True preserves the architecture and state-dict keys of old checkpoints.
    shared_trunk: bool = True
    symmetry_adapter: bool = False
    group_equivariant: bool = False

@dataclass(frozen=True)
class ActorCriticOutput:
    action_logits: torch.Tensor
    value: torch.Tensor


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        if config.group_equivariant:
            if config.symmetry_adapter:
                raise ValueError("The group-equivariant model replaces the symmetry adapter")
            if config.height != config.width or config.n_channels % 8:
                raise ValueError("D4 layers require a square board and channels divisible by 8")
            if config.output_dim != 4:
                raise ValueError("The D4 action head requires four 2048 directions")
        self.config = config
        if config.shared_trunk:
            self.trunk = self.create_trunk()
        else:
            self.actor_trunk = self.create_trunk()
            self.critic_trunk = self.create_trunk()
        self.action_head = self.create_head(self.config.output_dim)
        self.value_head = self.create_head(1)

    @property
    def policy_trunk(self) -> nn.Sequential:
        """The trunk producing action features, for activation inspection."""
        return self.trunk if self.config.shared_trunk else self.actor_trunk

    def representation_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the shared trunk or the union of independent trunks.

        Heads are excluded to preserve the scope of existing PPO diagnostics.
        In separate mode, policy/value gradient cosine is structurally zero.
        """
        if self.config.shared_trunk:
            yield from self.trunk.parameters()
        else:
            yield from self.actor_trunk.parameters()
            yield from self.critic_trunk.parameters()

    def create_stem(self) -> nn.Module:
        if self.config.group_equivariant:
            families = self.config.n_channels // 8
            modules = OrderedDict([
                ("conv", D4LiftConv(
                    self.config.in_channels,
                    families,
                    self.config.kernel_size,
                    bias=not self.config.include_groupnorm,
                )),
            ])
            if self.config.include_groupnorm:
                modules["groupnorm"] = D4GroupNorm(families)
            modules["relu"] = nn.ReLU()
            return nn.Sequential(modules)
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
        stem = nn.Sequential(module_dict)
        if self.config.symmetry_adapter:
            return D4ResidualStem(stem, self.config.n_channels)
        return stem

    def create_body(self) -> nn.Module:
        module_dict = OrderedDict()
        for idx in range(1, self.config.blocks + 1):
            if self.config.group_equivariant:
                module_dict[f"layer{idx}"] = D4ResBlock(
                    self.config.n_channels // 8,
                    kernel_size=self.config.kernel_size,
                    layers=self.config.block_layers,
                    include_groupnorm=self.config.include_groupnorm,
                )
            else:
                module_dict[f"layer{idx}"] = ResBlock(
                    self.config.n_channels,
                    kernel_size=self.config.kernel_size,
                    layers=self.config.block_layers,
                    include_groupnorm=self.config.include_groupnorm,
                )
        return nn.Sequential(module_dict)

    def create_head(self, output_dim: int) -> nn.Module:
        if self.config.group_equivariant:
            head = D4ActionHead if output_dim == self.config.output_dim else D4ValueHead
            return head(
                self.config.n_channels,
                self.config.height,
                self.config.width,
                self.config.head_channels,
            )
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
        if self.config.shared_trunk:
            actor_features = critic_features = self.trunk(x)
        else:
            actor_features = self.actor_trunk(x)
            critic_features = self.critic_trunk(x)
        action_logits = self.action_head(actor_features)
        value = self.value_head(critic_features).squeeze(dim=-1)
        return ActorCriticOutput(action_logits=action_logits, value=value)
