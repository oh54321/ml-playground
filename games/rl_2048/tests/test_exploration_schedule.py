"""Focused tests for disabled and linear exploration schedules."""

import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

from games.rl_2048.training.config import (
    NO_UNIFORM_EXPLORATION,
    LinearExplorationSchedule,
    TrainingConfig,
    exploration_schedule_from_checkpoint,
)
from games.rl_2048.training.options import read_training_options
from games.rl_2048.training.setup import _resolve_training_end, _validate_config


def _config(schedule, outer_iterations=1200):
    return TrainingConfig(
        run_name="test",
        learning_rate=1e-3,
        seed=1,
        outer_iterations=outer_iterations,
        warm_start_from=None,
        num_parallel_games=1,
        steps_per_game_per_rollout=1,
        symmetry_augmentation=None,
        exploration_schedule=schedule,
        evaluation_interval=1,
        evaluation_games=1,
        evaluation_seed=1,
        save_traces=False,
        scheduler_decay=1.0,
    )


class ExplorationScheduleTests(unittest.TestCase):
    def test_disabled_schedule_has_no_iteration_bound(self):
        last, schedule, data = _resolve_training_end(
            _config(NO_UNIFORM_EXPLORATION), first_iteration=1001
        )
        self.assertEqual(last, 1200)
        self.assertEqual(schedule.probability(1001), 0.0)
        self.assertEqual(schedule.probability(1200), 0.0)
        self.assertEqual(data["kind"], "disabled")
        self.assertEqual(data, {"kind": "disabled"})

    def test_checkpoint_loader_reads_new_disabled_and_old_linear_records(self):
        disabled_data = NO_UNIFORM_EXPLORATION.to_checkpoint()
        self.assertIs(
            exploration_schedule_from_checkpoint(disabled_data),
            NO_UNIFORM_EXPLORATION,
        )
        old_linear = {
            "start_iteration": 5,
            "end_iteration": 1000,
            "start_probability": 0.1,
            "end_probability": 0.0,
        }
        loaded = exploration_schedule_from_checkpoint(old_linear)
        self.assertEqual(loaded, LinearExplorationSchedule(0.1, 0.0, 1000, 5))
        old_disabled = {"kind": "disabled", "start_iteration": 1001, "end_iteration": 1200}
        self.assertIs(exploration_schedule_from_checkpoint(old_disabled), NO_UNIFORM_EXPLORATION)

    def test_linear_schedule_owns_bounds_clamps_and_serializes_them(self):
        schedule = LinearExplorationSchedule(0.2, 0.0, 11, start_iteration=5)
        self.assertEqual(schedule.probability(4), 0.2)
        self.assertEqual(schedule.probability(8), 0.1)
        self.assertEqual(schedule.probability(15), 0.0)
        self.assertEqual(exploration_schedule_from_checkpoint(schedule.to_checkpoint()), schedule)

    def test_cli_numeric_override_enables_linear_with_requested_last_iteration(self):
        argv = [
            "train",
            "--outer-iterations",
            "1500",
            "--exploration-start",
            "0.08",
        ]
        with patch.object(sys, "argv", argv):
            result = read_training_options(_config(NO_UNIFORM_EXPLORATION))
        self.assertEqual(result.exploration_schedule, LinearExplorationSchedule(0.08, 0.0, 1500))

    def test_cli_end_iteration_only_keeps_disabled_probabilities(self):
        with patch.object(
            sys,
            "argv",
            ["train", "--exploration-end-iteration", "1500"],
        ):
            result = read_training_options(_config(NO_UNIFORM_EXPLORATION))
        self.assertEqual(result.exploration_schedule, LinearExplorationSchedule(0.0, 0.0, 1500))

    def test_existing_linear_cli_override_retains_omitted_values(self):
        initial = LinearExplorationSchedule(0.2, 0.01, 900)
        with patch.object(sys, "argv", ["train", "--exploration-end", "0.0"]):
            result = read_training_options(_config(initial))
        self.assertEqual(result.exploration_schedule, LinearExplorationSchedule(0.2, 0.0, 900))

    def test_cli_override_preserves_configured_linear_start_bound(self):
        initial = LinearExplorationSchedule(0.2, 0.01, 900, start_iteration=40)
        with patch.object(sys, "argv", ["train", "--exploration-end", "0.0"]):
            result = read_training_options(_config(initial))
        self.assertEqual(result.exploration_schedule.start_iteration, 40)

    def test_resume_binds_linear_start_to_first_iteration(self):
        configured = LinearExplorationSchedule(0.2, 0.0, 1200)
        _, resumed, data = _resolve_training_end(_config(configured), first_iteration=1001)
        self.assertEqual(resumed.start_iteration, 1001)
        self.assertEqual(resumed.probability(1001), 0.2)
        self.assertEqual(data["start_iteration"], 1001)

    def test_minimum_learning_rate_must_be_finite_and_at_most_initial_rate(self):
        config = _config(NO_UNIFORM_EXPLORATION)
        _validate_config(replace(config, min_learning_rate=1e-4))
        for invalid in (float("nan"), float("inf"), -1e-5, 2e-3):
            with self.subTest(min_learning_rate=invalid):
                with self.assertRaises(ValueError):
                    _validate_config(replace(config, min_learning_rate=invalid))


if __name__ == "__main__":
    unittest.main()
