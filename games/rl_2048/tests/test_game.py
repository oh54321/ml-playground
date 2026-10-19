"""Game rules, reproducible moves, immutable states, and observation encodings."""

import unittest

import numpy as np
import torch

from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding, State2048
from games.rl_2048.environment._game_2048_kernels import move_int64
from core.symmetries.d4_transforms import transform_array
from games.rl_2048.environment.symmetry_2048 import RECTANGULAR_SYMMETRY_SPECS, SQUARE_SYMMETRY_SPECS


class GameTests(unittest.TestCase):
    def test_moves_and_merge_scores(self):
        cases = (
            ([[0, 2, 0, 4]], 0, [[2, 4, 0, 0]], 0),
            ([[2, 2, 2, 2]], 0, [[4, 4, 0, 0]], 8),
            ([[2, 2, 2, 2]], 2, [[0, 0, 4, 4]], 8),
            ([[2, 2, 4, 0]], 0, [[4, 4, 0, 0]], 4),
            ([[2], [2], [2], [2]], 1, [[4], [4], [0], [0]], 8),
            ([[2], [2], [2], [2]], 3, [[0], [0], [4], [4]], 8),
        )
        for board, action, expected, reward in cases:
            with self.subTest(board=board, action=action):
                board = np.array(board, dtype=np.int64)
                result = np.zeros_like(board)
                self.assertEqual(move_int64(board, action, result), (reward, True))
                np.testing.assert_array_equal(result, expected)

    def test_legal_actions_and_game_over(self):
        game = Game2048()
        for board, legal in (([[2, 4, 0, 0]], [2]), ([[2, 2]], [0, 2]),
                             ([[2, 4], [4, 2]], [])):
            game.replace_board(np.array(board))
            self.assertEqual(game.legal_actions(), legal)
            self.assertEqual(game.done, not legal)
        with self.assertRaises(RuntimeError):
            game.step(0)

    def test_step_scores_and_spawns_only_after_a_move(self):
        game, control = Game2048(seed=17), Game2048(seed=17)
        board = np.array([[2, 4, 0, 0]])
        for instance in (game, control):
            instance.replace_board(board)
        result = game.step(0)
        self.assertEqual((result.reward, result.action_applied), (0, False))
        np.testing.assert_array_equal(game.state.grid, board)
        # The no-op must not consume randomness or affect the next spawn.
        game.step(2)
        control.step(2)
        np.testing.assert_array_equal(game.state.grid, control.state.grid)
        game.replace_board(np.array([[2, 2, 0, 0]]), total_score=10)
        result = game.step(0)
        self.assertEqual((result.reward, result.action_applied, game.total_score), (4, True, 14))
        self.assertEqual(game.state.grid[0, 0], 4)
        self.assertEqual(np.count_nonzero(game.state.grid), 2)
        self.assertIn(game.state.grid.sum(), (6, 8))
        with self.assertRaises(ValueError):
            game.step(4)

    def test_same_seed_reproduces_games(self):
        first, second = Game2048(seed=123), Game2048(seed=123)
        for seed in (123, 456):
            first.reset(seed)
            second.reset(seed)
            np.testing.assert_array_equal(first.state.grid, second.state.grid)
            self.assertEqual(np.count_nonzero(first.state.grid), 2)
            for _ in range(8):
                action = first.legal_actions()[0]
                first.step(action)
                second.step(action)
                np.testing.assert_array_equal(first.state.grid, second.state.grid)
                self.assertEqual(first.total_score, second.total_score)

    def test_dimensions_encoding_and_snapshots_survive_game_changes(self):
        for height, width in ((4, 4), (6, 6), (3, 5)):
            for encoding in ObservationEncoding:
                with self.subTest(shape=(height, width), encoding=encoding):
                    game = Game2048(seed=17, height=height, width=width, observation_encoding=encoding)
                    snapshot = game.state
                    original = snapshot.grid.copy()
                    copied = snapshot.copy()
                    stepped = game.step(game.legal_actions()[0]).next_state
                    game.reset(seed=18)
                    reset = game.state
                    # Replacing the board retains the game's current state encoding.
                    game.replace_board(State2048(original))
                    for state in (game.state, snapshot, copied, stepped, reset):
                        self.assertEqual((state.height, state.width), (height, width))
                        self.assertEqual(state.observation_encoding, encoding)
                        with self.assertRaises(ValueError):
                            state.grid[0, 0] = 8
                    for state in (snapshot, copied, game.state):
                        np.testing.assert_array_equal(state.grid, original)

    def test_numeric_and_onehot_tensor_values(self):
        for height, width in ((4, 4), (6, 6), (3, 5)):
            exponents = np.zeros((height, width), dtype=np.int64)
            exponents[0, :4] = [0, 1, 2, 3]
            exponents[-1, -1] = height * width + 1
            grid = np.where(exponents == 0, 0, 2 ** exponents)
            for encoding in ObservationEncoding:
                state = State2048(grid, encoding)
                tensor = State2048.batch_to_tensor([state, state.copy()])
                channels = 1 if encoding is ObservationEncoding.NUMERIC else height * width + 3
                self.assertEqual(tensor.shape, (2, channels, height, width))
                self.assertEqual(encoding.in_channels(height, width), channels)
                torch.testing.assert_close(tensor[0], state.to_tensor())
                torch.testing.assert_close(tensor[:, 0], torch.from_numpy(exponents).float().expand(2, -1, -1))
                if encoding is ObservationEncoding.NUMERIC_ONEHOT:
                    torch.testing.assert_close(tensor[:, 1:].sum(1), torch.ones(2, height, width))
                    torch.testing.assert_close(tensor[:, 1:].argmax(1), tensor[:, 0].long())
        with self.assertRaises(ValueError):
            State2048.batch_to_tensor([State2048(), State2048(observation_encoding="numeric_onehot")])
        with self.assertRaises(ValueError):
            State2048(np.full((4, 4), 2 ** 18), "numeric_onehot").to_tensor()

    def test_symmetry_action_mappings_preserve_moves(self):
        board = np.array([[2, 2, 0, 4], [0, 8, 8, 0], [2, 0, 4, 4], [0, 2, 0, 8]])
        for board, specs in ((board, SQUARE_SYMMETRY_SPECS),
                             (board[:3], RECTANGULAR_SYMMETRY_SPECS)):
            for spec in specs:
                transformed = transform_array(board, spec.transform)
                self.assertEqual(transformed.shape, board.shape)
                for action in range(4):
                    expected, actual = np.zeros_like(board), np.zeros_like(board)
                    reward, moved = move_int64(board, action, expected)
                    self.assertEqual(move_int64(transformed, spec.action_permutation[action], actual), (reward, moved))
                    np.testing.assert_array_equal(actual, transform_array(expected, spec.transform))
