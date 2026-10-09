"""Evaluation win rates use the terminal board's actual tile values."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from games.rl_2048.training.evaluation import (
    CompletedGame,
    EvaluationResult,
    EvaluationRun,
    _evaluate_games,
)
from games.rl_2048.training.evaluation_reporting import write_evaluation_scalars
from games.rl_2048.utils.plot_training_diagnostics import (
    save_evaluation_plots,
    save_training_diagnostic_plots,
)


class EvaluationWinRateTests(unittest.TestCase):
    def test_win_rate_counts_2048_and_higher_tiles(self) -> None:
        result = EvaluationResult(
            scores=[0, 100, 200],
            moves=[10, 20, 30],
            worst_game=CompletedGame(1, 0, 10, 1),
            best_game=CompletedGame(3, 200, 30, 3),
            max_tiles=[1024, 2048, 4096],
        )
        self.assertEqual(result.win_rate, 2 / 3)

    def test_evaluator_records_actual_terminal_tile_values(self) -> None:
        class FakeGame:
            def __init__(self, seed: int):
                self.seed = seed
                self.total_score = 0
                self.state = SimpleNamespace(
                    grid=np.array([[2048 if seed == 1 else 4096]])
                )

            def current_state(self):
                return self.state

            def step(self, action):
                return SimpleNamespace(action_applied=True, terminated=True, reward=0)

        fake_ppo = SimpleNamespace(
            generate_next_actions=lambda games, deterministic: [
                SimpleNamespace(
                    action=0,
                    illegal_actions_mask=torch.zeros(4, dtype=torch.bool),
                )
                for _ in games
            ]
        )
        with patch("games.rl_2048.training.evaluation.Game2048", FakeGame):
            result = _evaluate_games(fake_ppo, [1, 2], False, None)

        self.assertEqual(result.max_tiles, [2048, 4096])
        self.assertEqual(result.win_rate, 1.0)

    def test_evaluation_plot_saves_rate_as_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                "games.rl_2048.utils.plot_training_diagnostics.save_line_plot"
            ) as save_plot:
                save_evaluation_plots(
                    {
                        "mean": [1], "median": [1], "best": [1], "worst": [1],
                        "mean_moves": [1], "win_rate": [0.25],
                    },
                    [10],
                    output,
                )
            call = next(
                call for call in save_plot.call_args_list
                if call.args[0].name == "evaluation_win_rate.png"
            )
            self.assertEqual(call.args[2][0][1], [25.0])
            self.assertEqual(call.args[1], [10])
            self.assertEqual(call.kwargs["y_max"], 100)

    def test_legacy_summary_without_win_rate_omits_win_rate_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw_summary.txt"
            summary = root / "summary.txt"
            raw.write_text("Outer iteration 1\n", encoding="utf-8")
            summary.write_text(
                "Outer iteration 1\nRaw scores: [100, 200]\n",
                encoding="utf-8",
            )
            with patch(
                "games.rl_2048.utils.plot_training_diagnostics.save_line_plot"
            ) as save_plot:
                save_training_diagnostic_plots(raw, root, summary)
            self.assertNotIn(
                "evaluation_win_rate.png",
                [call.args[0].name for call in save_plot.call_args_list],
            )

    def test_summary_plot_rebuilds_win_rate_from_saved_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw_summary.txt"
            summary = root / "summary.txt"
            raw.write_text("Outer iteration 1\n", encoding="utf-8")
            summary.write_text(
                "Outer iteration 1\nRaw scores: [100, 200]\n"
                "Win rate (2048+): 50.0% (1/2 games)\n",
                encoding="utf-8",
            )
            with patch(
                "games.rl_2048.utils.plot_training_diagnostics.save_line_plot"
            ) as save_plot:
                save_training_diagnostic_plots(raw, root, summary)
            call = next(
                call for call in save_plot.call_args_list
                if call.args[0].name == "evaluation_win_rate.png"
            )
            self.assertEqual(call.args[1], [1])
            self.assertEqual(call.args[2][0][1], [50.0])

    def test_tensorboard_logs_evaluation_win_rate_tag(self) -> None:
        class FakeWriter:
            def __init__(self):
                self.scalars = []

            def add_scalar(self, *args):
                self.scalars.append(args)

        run = EvaluationRun(Path("."), 1, 2, False)
        run.history["win_rate"].append(0.5)
        run.history["mean"].append(900.0)
        writer = FakeWriter()
        write_evaluation_scalars(writer, run, 20)
        self.assertIn(("evaluation/win_rate", 0.5, 20), writer.scalars)


if __name__ == "__main__":
    unittest.main()
