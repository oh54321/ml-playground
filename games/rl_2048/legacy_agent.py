"""Load models from checkpoint configs written by the former ResNet API."""

from dataclasses import asdict, dataclass
from typing import Mapping

from core.models.actor_critic import ActorCriticModel
from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from core.models.legacy.d4_residual_adapter import (
    D4ResidualAdapterActorCriticModel,
    D4ResidualStem,
)
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


@dataclass(frozen=True)
class ResNetConfig:
    """Legacy config shape used in existing checkpoints."""

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
    shared_trunk: bool = True
    symmetry_adapter: bool = False
    group_equivariant: bool = False


def model_from_legacy_config(config: Mapping | ResNetConfig) -> ActorCriticModel:
    """Build the separated implementation described by an old config."""
    config_dict = (
        asdict(config)
        if isinstance(config, ResNetConfig)
        else asdict(ResNetConfig(**dict(config)))
    )
    symmetry_adapter = bool(config_dict.pop("symmetry_adapter", False))
    group_equivariant = bool(config_dict.pop("group_equivariant", False))
    if symmetry_adapter and group_equivariant:
        raise ValueError("A checkpoint cannot enable both D4 model variants")
    model_config = ConvActorCriticConfig(**config_dict)
    if group_equivariant:
        action_permutations = {
            spec.transform: spec.action_permutation
            for spec in SQUARE_SYMMETRY_SPECS
        }
        return D4ConvActorCriticModel(model_config, action_permutations)
    if symmetry_adapter:
        return D4ResidualAdapterActorCriticModel(model_config)
    return ConvActorCriticModel(model_config)


def ResNet(config: ResNetConfig) -> ActorCriticModel:
    """Deprecated factory retained for old scripts and checkpoints."""
    return model_from_legacy_config(config)


__all__ = [
    "D4ResidualStem",
    "ResNet",
    "ResNetConfig",
    "model_from_legacy_config",
]
