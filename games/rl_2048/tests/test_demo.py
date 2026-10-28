"""The demo uses checkpoint metadata and the same legal moves as training."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.models.actor_critic import ActorCriticOutput
from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.demo import DemoGame
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


class DemoTests(unittest.TestCase):
    def test_history_snapshots_leave_live_game_unchanged(self):
        with patch("games.rl_2048.demo.secrets.randbits", return_value=123):
            demo = DemoGame()
            control = DemoGame()
        states = [demo.state()]
        for _ in range(12):
            action = demo.game.legal_actions()[0]
            states.append(demo.move(action))
            control.move(action)
        live = demo.state()
        for number, state in enumerate(states):
            snapshot = demo.snapshot(123, number)
            for field in ("board", "score", "moves", "last_move"):
                self.assertEqual(snapshot[field], state[field])
            self.assertEqual(demo.state(), live)
        action = demo.game.legal_actions()[0]
        self.assertEqual(demo.move(action), control.move(action))
        with self.assertRaises(ValueError):
            demo.snapshot(123, -1)
        with self.assertRaises(ValueError):
            demo.snapshot(123, 14)
        with patch("games.rl_2048.demo.secrets.randbits", return_value=456):
            reset = demo.new_game()
        self.assertEqual(demo.boards, [reset["board"]])
        with self.assertRaises(ValueError):
            demo.snapshot(123, 0)

    def test_manual_moves_and_reset(self):
        demo = DemoGame()
        demo.game.replace_board(np.array([[2, 2, 0, 0], [0, 0, 0, 0],
                                          [0, 0, 0, 0], [0, 0, 0, 0]]))
        before = demo.state()
        self.assertEqual(demo.move(1), before)  # Up cannot move this board.
        after = demo.move(0)
        self.assertEqual((after["score"], after["moves"], after["board"][0][0]), (4, 1, 4))
        record = after["last_move"]
        merged = np.array(after["before_spawn"])
        self.assertEqual(np.count_nonzero(merged), 1)
        self.assertEqual(merged[0, 0], 4)
        spawn = record["spawn"]
        self.assertEqual(merged[spawn["row"], spawn["col"]], 0)
        merged[spawn["row"], spawn["col"]] = spawn["value"]
        np.testing.assert_array_equal(merged, after["board"])
        with self.assertRaises(ValueError):
            demo.move(True)
        with self.assertRaises(ValueError):
            demo.model_step()
        with patch("games.rl_2048.demo.secrets.randbits", return_value=123):
            reset = demo.new_game()
        self.assertEqual((reset["score"], reset["moves"], reset["last_move"]), (0, 0, None))
        self.assertIsNone(reset["before_spawn"])
        np.testing.assert_array_equal(reset["board"], Game2048(seed=123).state.grid)

    def test_checkpoint_variants_and_greedy_legal_action(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            cases = (("cnn", ObservationEncoding.NUMERIC), ("d4", ObservationEncoding.NUMERIC_ONEHOT))
            for architecture, encoding in cases:
                with self.subTest(architecture=architecture, encoding=encoding):
                    height, width = (3, 5) if architecture == "cnn" else (4, 4)
                    config = ConvActorCriticConfig(encoding.in_channels(height, width), height, width, 4, 8, 0, 8)
                    model = (ConvActorCriticModel(config) if architecture == "cnn" else D4ConvActorCriticModel(
                        config, {spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS}))
                    with TemporaryDirectory() as directory:
                        path = Path(directory) / "best.pt"
                        torch.save({"model_config": asdict(config), "architecture": architecture,
                                    "observation_encoding": encoding.value, "model": model.state_dict(),
                                    "iteration": 12}, path)
                        demo = DemoGame(path)
                    self.assertEqual(demo.game.state.grid.shape, (height, width))
                    torch.testing.assert_close(demo.model.state_dict(), model.state_dict(), rtol=0, atol=0)
                    board = np.zeros((height, width), dtype=np.int64)
                    board[0, 0] = 2
                    demo.game.replace_board(board)
                    # Left/up score highest but are illegal; right beats down.
                    output = ActorCriticOutput(torch.tensor([[100., 99., 2., 1.]]), torch.zeros(1))
                    with patch.object(demo.model, "forward", return_value=output):
                        state = demo.model_step()
                    self.assertEqual((state["last_move"]["action"], state["moves"]), (2, 1))
                    terminal = np.where(np.indices((height, width)).sum(axis=0) % 2, 2, 4)
                    demo.game.replace_board(terminal)
                    before = demo.state()
                    self.assertTrue(before["done"])
                    self.assertEqual(demo.model_step(), before)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
