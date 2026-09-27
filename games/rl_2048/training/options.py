"""Command-line input options for launching a PPO 2048 experiment."""

import argparse
from dataclasses import replace
from pathlib import Path

from games.rl_2048.training.config import LinearExplorationSchedule, TrainingConfig


def read_training_options(defaults: TrainingConfig) -> TrainingConfig:
    """Return experiment settings with any command-line overrides applied."""
    parser = argparse.ArgumentParser(description="Train the 2048 PPO agent")
    parser.add_argument("--warm-start-from", type=Path)
    parser.add_argument("--outer-iterations", type=int)
    parser.add_argument("--num-parallel-games", type=int)
    parser.add_argument("--steps-per-game-per-rollout", type=int)
    parser.add_argument("--exploration-start", type=float)
    parser.add_argument("--exploration-end", type=float)
    parser.add_argument("--exploration-end-iteration", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--evaluation-games", type=int)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="write live scalar metrics for TensorBoard",
    )
    args = parser.parse_args()

    schedule = defaults.exploration_schedule
    if any(
        value is not None
        for value in (
            args.exploration_start,
            args.exploration_end,
            args.exploration_end_iteration,
        )
    ):
        schedule = LinearExplorationSchedule(
            start_probability=(
                schedule.start_probability
                if args.exploration_start is None
                else args.exploration_start
            ),
            end_probability=(
                schedule.end_probability
                if args.exploration_end is None
                else args.exploration_end
            ),
            end_iteration=(
                schedule.end_iteration
                if args.exploration_end_iteration is None
                else args.exploration_end_iteration
            ),
        )

    return replace(
        defaults,
        warm_start_from=(
            defaults.warm_start_from
            if args.warm_start_from is None
            else args.warm_start_from
        ),
        outer_iterations=(
            defaults.outer_iterations
            if args.outer_iterations is None
            else args.outer_iterations
        ),
        num_parallel_games=(
            defaults.num_parallel_games
            if args.num_parallel_games is None
            else args.num_parallel_games
        ),
        steps_per_game_per_rollout=(
            defaults.steps_per_game_per_rollout
            if args.steps_per_game_per_rollout is None
            else args.steps_per_game_per_rollout
        ),
        evaluation_interval=(
            defaults.evaluation_interval
            if args.evaluation_interval is None
            else args.evaluation_interval
        ),
        evaluation_games=(
            defaults.evaluation_games
            if args.evaluation_games is None
            else args.evaluation_games
        ),
        tensorboard=defaults.tensorboard or args.tensorboard,
        exploration_schedule=schedule,
    )
