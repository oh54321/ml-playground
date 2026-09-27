"""Checkpoint persistence and warm-start compatibility for PPO 2048."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from algorithms.ppo import PPOConfig
from core.models.actor_critic import ActorCriticModel
from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from core.models.legacy.d4_residual_adapter import D4ResidualAdapterActorCriticModel


__all__ = ["save_checkpoint", "warm_start_from_checkpoint"]


def _legacy_model_config(
    model: ActorCriticModel,
    model_config: ConvActorCriticConfig,
) -> dict:
    """Keep the checkpoint model-config shape understood by older tools."""
    config = asdict(model_config)
    config["symmetry_adapter"] = isinstance(model, D4ResidualAdapterActorCriticModel)
    config["group_equivariant"] = isinstance(model, D4ConvActorCriticModel)
    return config


def save_checkpoint(
    output_path: Path,
    model: ActorCriticModel,
    optimizer: Optimizer,
    ppo_config: PPOConfig,
    model_config: ConvActorCriticConfig,
    outer_iteration: int,
    mean_score: float,
    median_score: float,
    mean_moves: float,
    symmetry_augmentation: str | None = None,
    exploration_schedule: dict | None = None,
) -> None:
    """Atomically save the state needed to inspect or warm-start a run."""
    checkpoint = {
        "outer_iteration": outer_iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ppo_config": asdict(ppo_config),
        "resnet_config": _legacy_model_config(model, model_config),
        "evaluation": {
            "mean_score": mean_score,
            "median_score": median_score,
            "mean_moves": mean_moves,
        },
        "symmetry_augmentation": symmetry_augmentation,
        "exploration_schedule": exploration_schedule,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(checkpoint, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def warm_start_from_checkpoint(
    checkpoint_path: Path,
    model: ActorCriticModel,
    optimizer: Optimizer,
    scheduler: ExponentialLR,
    ppo_config: PPOConfig,
    model_config: ConvActorCriticConfig,
    symmetry_augmentation: str | None,
) -> int:
    """Restore compatible model/optimizer state and return the next iteration."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["resnet_config"] != _legacy_model_config(model, model_config):
        raise ValueError("Checkpoint model configuration does not match this run")
    if checkpoint["ppo_config"] != asdict(ppo_config):
        raise ValueError("Checkpoint PPO configuration does not match this run")
    if checkpoint.get("symmetry_augmentation") != symmetry_augmentation:
        raise ValueError("Checkpoint symmetry augmentation does not match this run")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_iteration = int(checkpoint["outer_iteration"])
    scheduler.last_epoch = completed_iteration
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return completed_iteration + 1
