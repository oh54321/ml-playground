"""Experiment settings and exploration schedule for PPO 2048 runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "NO_UNIFORM_EXPLORATION",
    "NoUniformExploration",
    "LinearExplorationSchedule",
    "ExplorationSchedule",
    "exploration_schedule_from_checkpoint",
    "SymmetryAugmentation",
    "TrainingConfig",
    "linear_exploration_probability",
]


SymmetryAugmentation = Literal["step", "episode"] | None


@dataclass(frozen=True)
class NoUniformExploration:
    """An explicit schedule that disables uniform legal-action exploration."""

    def probability(self, iteration: int) -> float:
        return 0.0

    def to_checkpoint(self) -> dict[str, object]:
        return {"kind": "disabled"}


NO_UNIFORM_EXPLORATION = NoUniformExploration()


@dataclass(frozen=True)
class LinearExplorationSchedule:
    """Uniform legal-action exploration that decreases linearly over training."""

    start_probability: float
    end_probability: float
    end_iteration: int
    start_iteration: int = 1

    def probability(self, iteration: int) -> float:
        if self.end_iteration <= self.start_iteration:
            return self.start_probability
        progress = min(
            1.0,
            max(
                0.0,
                (iteration - self.start_iteration)
                / (self.end_iteration - self.start_iteration),
            ),
        )
        return self.start_probability + progress * (
            self.end_probability - self.start_probability
        )

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "kind": "linear",
            "start_iteration": self.start_iteration,
            "end_iteration": self.end_iteration,
            "start_probability": self.start_probability,
            "end_probability": self.end_probability,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, object]) -> "LinearExplorationSchedule":
        if checkpoint.get("kind") != "linear":
            raise ValueError("Only linear exploration schedules are supported")
        return cls(
            start_probability=float(checkpoint["start_probability"]),
            end_probability=float(checkpoint["end_probability"]),
            end_iteration=int(checkpoint["end_iteration"]),
            start_iteration=int(checkpoint.get("start_iteration", 1)),
        )


ExplorationSchedule = LinearExplorationSchedule | NoUniformExploration


def exploration_schedule_from_checkpoint(
    checkpoint: dict[str, object],
) -> ExplorationSchedule:
    """Load the supported schedule kinds, including historical linear records."""
    kind = checkpoint.get("kind", "linear")
    if kind == "disabled":
        return NO_UNIFORM_EXPLORATION
    if kind == "linear":
        return LinearExplorationSchedule.from_checkpoint({**checkpoint, "kind": "linear"})
    raise ValueError(f"Unsupported exploration schedule kind: {kind!r}")


@dataclass(frozen=True)
class TrainingConfig:
    """Settings that define one reproducible PPO 2048 experiment."""

    run_name: str
    learning_rate: float
    seed: int
    # The last outer iteration to run, inclusive.
    outer_iterations: int
    warm_start_from: Path | None
    num_parallel_games: int
    steps_per_game_per_rollout: int
    symmetry_augmentation: SymmetryAugmentation
    exploration_schedule: ExplorationSchedule
    evaluation_interval: int
    evaluation_games: int
    evaluation_seed: int
    save_traces: bool
    scheduler_decay: float
    tensorboard: bool = False
    # Iterations at which to save a model/optimizer snapshot, whether or not
    # that iteration also runs an evaluation.
    saved_checkpoints: tuple[int, ...] | list[int] = ()


def linear_exploration_probability(
    iteration: int,
    start_iteration: int,
    end_iteration: int,
    start_probability: float,
    end_probability: float,
) -> float:
    """Return the linearly annealed exploration probability for one iteration."""
    return LinearExplorationSchedule(
        start_probability=start_probability,
        end_probability=end_probability,
        end_iteration=end_iteration,
        start_iteration=start_iteration,
    ).probability(iteration)
