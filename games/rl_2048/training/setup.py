"""Prepare PPO state and run artifacts for the visible training loop."""

from dataclasses import dataclass, replace
from datetime import datetime
import math
from pathlib import Path

import torch
from torch.optim import Optimizer

from algorithms.ppo import PPO, PPOConfig
from core.models.actor_critic import ActorCriticModel
from core.models.conv_actor_critic import ConvActorCriticConfig
from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.training.checkpointing import warm_start_from_checkpoint
from games.rl_2048.training.config import (
    ExplorationSchedule,
    LinearExplorationSchedule,
    NoUniformExploration,
    TrainingConfig,
)
from games.rl_2048.training.evaluation import EvaluationRun
from games.rl_2048.training.schedulers import (
    FloorExponentialLR,
    clamp_restored_learning_rate,
)


@dataclass
class TrainingRun:
    device: torch.device
    model: ActorCriticModel
    optimizer: Optimizer
    scheduler: FloorExponentialLR
    ppo: PPO
    ppo_config: PPOConfig
    model_config: ConvActorCriticConfig
    first_iteration: int
    last_iteration: int
    exploration: ExplorationSchedule
    exploration_data: dict[str, object]
    output_dir: Path
    raw_summary_path: Path
    run_summary_path: Path
    evaluation: EvaluationRun
    parameter_count: int


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_config(config: TrainingConfig) -> None:
    if not config.run_name or Path(config.run_name).name != config.run_name:
        raise ValueError("run_name must be a non-empty directory name")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config.min_learning_rate is not None and (
        not math.isfinite(config.min_learning_rate)
        or config.min_learning_rate < 0
        or config.min_learning_rate > config.learning_rate
    ):
        raise ValueError(
            "min_learning_rate must be finite and between zero and learning_rate"
        )
    if config.outer_iterations <= 0:
        raise ValueError("outer_iterations must be positive")
    if not isinstance(config.saved_checkpoints, (tuple, list)):
        raise ValueError("saved_checkpoints must be a list or tuple of iterations")
    if any(
        isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0
        for iteration in config.saved_checkpoints
    ):
        raise ValueError("saved_checkpoints must contain positive integer iterations")
    if config.num_parallel_games <= 0:
        raise ValueError("num_parallel_games must be positive")
    if config.steps_per_game_per_rollout <= 0:
        raise ValueError("steps_per_game_per_rollout must be positive")
    if config.evaluation_interval <= 0 or config.evaluation_games <= 0:
        raise ValueError("evaluation interval and game count must be positive")
    if not 0 < config.scheduler_decay <= 1:
        raise ValueError("scheduler_decay must be in (0, 1]")


def _resolve_training_end(
    config: TrainingConfig,
    first_iteration: int,
) -> tuple[int, ExplorationSchedule, dict[str, object]]:
    last_iteration = config.outer_iterations
    schedule = config.exploration_schedule
    if last_iteration < first_iteration:
        raise ValueError("Checkpoint has already completed the requested iterations")
    if isinstance(schedule, LinearExplorationSchedule):
        if schedule.end_iteration < first_iteration:
            raise ValueError("Exploration schedule end iteration precedes its start")
        for probability in (schedule.start_probability, schedule.end_probability):
            if not math.isfinite(probability) or not 0 <= probability < 1:
                raise ValueError("Exploration probabilities must be finite and in [0, 1)")
        if schedule.end_probability > schedule.start_probability:
            raise ValueError("Exploration end probability must not exceed its start probability")
        if schedule.start_iteration != first_iteration:
            schedule = replace(schedule, start_iteration=first_iteration)
    elif not isinstance(schedule, NoUniformExploration):
        raise TypeError("Unsupported exploration schedule")
    schedule_data = schedule.to_checkpoint()
    return last_iteration, schedule, schedule_data


def _create_output_dir(run_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("artifacts") / "ppo_2048" / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def prepare_training_run(
    config: TrainingConfig,
    ppo_config: PPOConfig,
    device: torch.device,
    model: ActorCriticModel,
    model_config: ConvActorCriticConfig,
    optimizer: Optimizer,
    scheduler: FloorExponentialLR,
) -> TrainingRun:
    """Prepare warm start, PPO state, and output files around caller-built objects."""
    _validate_config(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    first_iteration = 1
    if config.warm_start_from is not None:
        first_iteration = warm_start_from_checkpoint(
            config.warm_start_from,
            model,
            optimizer,
            scheduler,
            ppo_config,
            model_config,
            config.symmetry_augmentation,
        )
        clamp_restored_learning_rate(
            optimizer,
            scheduler,
            config.min_learning_rate,
        )
    last_iteration, exploration, exploration_data = _resolve_training_end(
        config,
        first_iteration,
    )

    ppo = PPO(
        model,
        optimizer,
        ppo_config,
        game_factory=lambda seed: Game2048(seed=seed),
        symmetry_augmentation=config.symmetry_augmentation,
    )
    output_dir = _create_output_dir(config.run_name)
    evaluation = EvaluationRun(
        output_dir=output_dir,
        seed_base=config.evaluation_seed,
        num_games=config.evaluation_games,
        save_traces=config.save_traces,
        symmetry_augmentation=config.symmetry_augmentation,
        exploration_schedule=exploration_data,
    )
    run_summary_path = output_dir / "summary.txt"
    run_summary_path.write_text(
        "PPO 2048 evaluation summary\n"
        f"Training seed: {ppo_config.seed}\n"
        f"Evaluation seed base: {config.evaluation_seed}\n"
        f"Model: {type(model).__name__}\n"
        f"Symmetry augmentation: {config.symmetry_augmentation}\n"
        f"Exploration schedule: {exploration_data}\n"
        f"Warm start: {config.warm_start_from or 'none'}\n\n",
        encoding="utf-8",
    )
    return TrainingRun(
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ppo=ppo,
        ppo_config=ppo_config,
        model_config=model_config,
        first_iteration=first_iteration,
        last_iteration=last_iteration,
        exploration=exploration,
        exploration_data=exploration_data,
        output_dir=output_dir,
        raw_summary_path=output_dir / "raw_summary.txt",
        run_summary_path=run_summary_path,
        evaluation=evaluation,
        parameter_count=parameter_count,
    )
