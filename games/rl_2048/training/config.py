"""Training configuration fields and validation; experiment settings live in train_2048.py."""

from dataclasses import dataclass
from pathlib import Path

from games.rl_2048.environment.game_2048 import ObservationEncoding


@dataclass(frozen=True)
class TrainingConfig:
    height: int
    width: int
    architecture: str
    observation_encoding: ObservationEncoding
    run_name: str
    learning_rate: float
    scheduler_decay: float
    scheduler_decay_until: int | None
    seed: int
    outer_iterations: int
    num_parallel_games: int
    steps_per_game_per_rollout: int
    evaluation_interval: int
    evaluation_games: int
    evaluation_seed: int
    tensorboard: bool
    compile_model: bool

    def __post_init__(self) -> None:
        if min(self.height, self.width) <= 0:
            raise ValueError("Board dimensions must be positive")
        if self.height * self.width < 3:
            raise ValueError("Training requires at least three board cells")
        if self.architecture == "d4" and self.height != self.width:
            raise ValueError("D4 requires a square board; use CNN for rectangular boards")
        if self.architecture not in ("cnn", "d4"):
            raise ValueError("Unknown architecture")
        object.__setattr__(self, "observation_encoding", ObservationEncoding(self.observation_encoding))
        if not self.run_name or self.run_name in (".", "..") or Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must be a directory name")
        if not 0 < self.learning_rate < float("inf") or not 0 < self.scheduler_decay <= 1:
            raise ValueError("Invalid learning rate or scheduler decay")
        if self.scheduler_decay_until is not None and self.scheduler_decay_until < 0:
            raise ValueError("scheduler_decay_until must be nonnegative or None")
        if min(self.outer_iterations, self.num_parallel_games, self.steps_per_game_per_rollout,
               self.evaluation_interval, self.evaluation_games) <= 0:
            raise ValueError("Iteration counts, intervals, and game counts must be positive")
