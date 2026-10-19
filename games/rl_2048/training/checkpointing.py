"""Save and restore the current model, optimizer, and learning-rate schedule."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch

from games.rl_2048.environment.game_2048 import ObservationEncoding


def save_checkpoint(
    path, model, optimizer, scheduler, ppo_config, iteration, *, observation_encoding,
) -> None:
    checkpoint = {
        "iteration": iteration,
        "architecture": model.architecture,
        "observation_encoding": ObservationEncoding(observation_encoding).value,
        "model_config": asdict(model.config),
        "ppo_config": asdict(ppo_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary_path = Path(file.name)
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def warm_start_from_checkpoint(
    path, model, optimizer, scheduler, ppo_config, *, observation_encoding,
) -> int:
    """Restore training weights and schedule; rollout games start from new boards."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != model.architecture:
        raise ValueError("Checkpoint architecture does not match this run")
    if checkpoint["observation_encoding"] != ObservationEncoding(observation_encoding).value:
        raise ValueError("Checkpoint observation encoding does not match this run")
    if checkpoint["model_config"] != asdict(model.config):
        raise ValueError("Checkpoint model configuration does not match this run")
    if checkpoint["ppo_config"] != asdict(ppo_config):
        raise ValueError("Checkpoint PPO configuration does not match this run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Retain the new run's decay (e.g. 1.0 for a constant-LR continuation),
    # while restoring the checkpoint's learning rate and scheduler counters.
    decay = scheduler.gamma
    scheduler.load_state_dict(checkpoint["scheduler"])
    scheduler.gamma = decay
    return int(checkpoint["iteration"]) + 1
