"""Convolutional actor-critic models for spatial observations."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from core.models.actor_critic import ActorCriticModel, ActorCriticOutput


__all__ = ["ConvActorCriticConfig", "ConvActorCriticModel"]


NUMERIC_OBSERVATION = "numeric"
NUMERIC_ONEHOT_OBSERVATION = "numeric_onehot"
OBSERVATION_ENCODINGS = (NUMERIC_OBSERVATION, NUMERIC_ONEHOT_OBSERVATION)


def observation_input_channels(height: int, width: int, encoding: str) -> int:
    """Return the convolution input channels for a board encoding."""
    if height <= 0 or width <= 0:
        raise ValueError("Board dimensions must be positive")
    if encoding == NUMERIC_OBSERVATION:
        return 1
    if encoding == NUMERIC_ONEHOT_OBSERVATION:
        cells = height * width
        # One numeric magnitude plane plus empty and exponents 1..cells+1.
        return cells + 3
    raise ValueError(
        f"Unknown observation_encoding {encoding!r}; expected one of {OBSERVATION_ENCODINGS}"
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
    # Input tensors stay in the original one-plane log2 form; the model expands
    # that plane when numeric and categorical channels are enabled.
    observation_encoding: str = NUMERIC_OBSERVATION

    @property
    def in_channels(self) -> int:
        """Number of channels produced by the configured observation encoding."""
        return observation_input_channels(
            self.height, self.width, self.observation_encoding
        )


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
        observations = self.encode_observations(observations)
        if self.config.shared_trunk:
            features = self.trunk(observations)
            return ActorCriticOutput(
                action_logits=self.action_head(features),
                value=self.value_head(features).squeeze(dim=-1),
            )
        return ActorCriticOutput(
            action_logits=self._forward_actor(observations),
            value=self._forward_critic(observations),
        )

    def _forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.actor_trunk(observations))

    def _forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(observations)).squeeze(dim=-1)

    def compile_trunks(self) -> None:
        """Compile actor and critic calls without changing checkpoint keys."""
        if self.config.shared_trunk:
            raise ValueError(
                "Compilation currently requires separate actor/critic trunks"
            )
        import torch._functorch.config

        # PPO measures several component gradients from the same retained graph.
        # Buffer donation would prevent those repeated backward calls in PyTorch 2.8.
        torch._functorch.config.donated_buffer = False
        self._forward_actor = torch.compile(self._forward_actor, fullgraph=True)
        self._forward_critic = torch.compile(self._forward_critic, fullgraph=True)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Expand the stable one-plane log2 input for categorical experiments."""
        if self.config.observation_encoding == NUMERIC_OBSERVATION:
            return observations

        expected_shape = (self.config.height, self.config.width)
        if observations.ndim != 4 or observations.shape[1] != 1 or tuple(
            observations.shape[-2:]
        ) != expected_shape:
            raise ValueError(
                "numeric_onehot encoding expects observations shaped "
                f"(batch, 1, {expected_shape[0]}, {expected_shape[1]}); "
                f"got {tuple(observations.shape)}"
            )
        values = observations[:, 0]
        rounded = values.round()
        max_exponent = self.config.height * self.config.width + 1
        invalid = (
            (~torch.isfinite(values))
            | (values < 0)
            | (values != rounded)
            | (values > max_exponent)
        )
        if torch.any(invalid):
            finite_values = values[torch.isfinite(values)]
            maximum = (
                float(finite_values.max().item())
                if finite_values.numel()
                else float("nan")
            )
            raise ValueError(
                "numeric_onehot encoding requires empty cells or tile exponents "
                f"from 1 through {max_exponent}; found an invalid value "
                f"(finite maximum {maximum})"
            )
        categories = max_exponent + 1  # empty=0 and exponents 1..max_exponent
        categorical = F.one_hot(rounded.to(torch.long), num_classes=categories)
        categorical = categorical.permute(0, 3, 1, 2).to(dtype=observations.dtype)
        return torch.cat((observations, categorical), dim=1)
