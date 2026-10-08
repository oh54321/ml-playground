from itertools import product
import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from games.rl_2048.environment.game_2048 import Game2048, State2048


def _simulation_oracle(game: Game2048) -> list[int]:
    return [
        action
        for action in range(4)
        if game._simulate_move(game.state.grid, action)[2]
    ]


class _SimulationLegalityGame(Game2048):
    """Reference behavior: derive legal actions by simulating all directions."""

    def legal_actions(self) -> list[int]:
        return _simulation_oracle(self)

    def _is_game_over(self) -> bool:
        return not _simulation_oracle(self)


class TestGameLegality(unittest.TestCase):
    def test_every_four_cell_line_matches_simulation_oracle(self):
        values = (0, 2, 4, 8, 16)
        for shape in ((1, 4), (4, 1)):
            game = Game2048(*shape, seed=1)
            for cells in product(values, repeat=4):
                with self.subTest(shape=shape, cells=cells):
                    grid = np.asarray(cells, dtype=np.int64).reshape(shape)
                    game.replace_board(grid)
                    oracle = _simulation_oracle(game)
                    self.assertEqual(game.legal_actions(), oracle)
                    self.assertEqual(game._is_game_over(), not oracle)
                    self.assertEqual(game.done, not oracle)

    def test_representative_rectangular_boards_match_simulation_oracle(self):
        boards = (
            np.zeros((3, 5), dtype=np.int64),
            np.full((3, 5), 2, dtype=np.int64),
            np.array([[0, 2, 0], [4, 4, 8]], dtype=np.int64),
            np.array([[2, 4, 8, 16], [4, 8, 16, 2]], dtype=np.int64),
            np.array([[2], [0], [4], [4], [0]], dtype=np.int64),
        )
        for grid in boards:
            with self.subTest(grid=grid.tolist()):
                game = Game2048(*grid.shape, seed=2)
                game.replace_board(grid)
                oracle = _simulation_oracle(game)
                self.assertEqual(game.legal_actions(), oracle)
                self.assertEqual(game._is_game_over(), not oracle)
                self.assertEqual(game.done, not oracle)

    def test_one_cell_board_has_no_legal_move(self):
        game = Game2048(1, 1, seed=12)
        for value in (0, 2):
            with self.subTest(value=value):
                game.replace_board(np.array([[value]], dtype=np.int64))
                self.assertEqual(game.legal_actions(), [])
                self.assertEqual(game.legal_actions(), _simulation_oracle(game))
                self.assertTrue(game._is_game_over())

    def test_action_ids_keep_their_direction(self):
        cases = (
            ([[0, 2]], [0]),
            ([[2, 0]], [2]),
            ([[0], [2]], [1]),
            ([[2], [0]], [3]),
            ([[2, 2]], [0, 2]),
            ([[2], [2]], [1, 3]),
        )
        for board, expected in cases:
            with self.subTest(board=board):
                grid = np.asarray(board, dtype=np.int64)
                game = Game2048(*grid.shape, seed=13)
                game.replace_board(grid)
                self.assertEqual(game.legal_actions(), expected)

    def test_random_boards_match_simulation_oracle(self):
        rng = np.random.default_rng(17)
        tile_values = np.array([0, 2, 4, 8, 16, 32, 64])
        for shape in ((2, 7), (3, 4), (4, 4), (6, 2)):
            game = Game2048(*shape, seed=3)
            for index in range(100):
                with self.subTest(shape=shape, index=index):
                    game.replace_board(rng.choice(tile_values, size=shape))
                    oracle = _simulation_oracle(game)
                    self.assertEqual(game.legal_actions(), oracle)
                    self.assertEqual(game._is_game_over(), not oracle)
                    self.assertEqual(game.done, not oracle)

    def test_game_board_is_read_only_and_legal_actions_returns_a_fresh_list(self):
        game = Game2048(2, 3, seed=4)
        board = np.array([[2, 0, 0], [4, 8, 16]], dtype=np.int64)
        game.replace_board(board)
        expected = _simulation_oracle(game)
        actions = game.legal_actions()
        actions.clear()
        self.assertEqual(game.legal_actions(), expected)
        with self.assertRaises(AttributeError):
            game.state = State2048(grid=np.zeros((2, 3), dtype=np.int64))
        with self.assertRaises(AttributeError):
            game.state.grid = np.zeros((2, 3), dtype=np.int64)
        with self.assertRaises(ValueError):
            game.state.grid[0, 0] = 0
        with self.assertRaises(ValueError):
            game.state.grid.setflags(write=True)
        with self.assertRaises(AttributeError):
            game.state.height = 1
        original_shape = game.state.grid.shape
        original_dtype = game.state.grid.dtype
        self.assertFalse(game.state.grid.base.flags.writeable)
        game.state.grid.shape = (1, 6)
        game.state.grid.dtype = np.int32
        self.assertEqual(game.state.grid.shape, original_shape)
        self.assertEqual(game.state.grid.dtype, original_dtype)
        self.assertEqual(game.legal_actions(), expected)

        # Replacement copies aliases and refreshes terminal state and score.
        board[:, :] = 0
        self.assertEqual(game.legal_actions(), expected)
        game.replace_board(np.zeros((2, 3), dtype=np.int64))
        self.assertTrue(game.done)
        self.assertEqual(game.legal_actions(), [])
        game.replace_board(np.array([[0, 2, 0], [0, 0, 0]], dtype=np.int64))
        self.assertFalse(game.done)
        self.assertEqual(game.legal_actions(), _simulation_oracle(game))

    def test_spawn_and_reset_preserve_readonly_board_and_rng_behavior(self):
        game = Game2048(2, 3, seed=91)
        game.replace_board(np.array([[2, 0, 0], [4, 8, 16]], dtype=np.int64))
        before = int(np.count_nonzero(game.state.grid))
        working = game.state.grid.copy()
        self.assertTrue(game._add_random_tile(working))
        self.assertEqual(int(np.count_nonzero(working)), before + 1)
        game.replace_board(working)
        self.assertFalse(game.state.grid.flags.writeable)
        self.assertEqual(game.legal_actions(), _simulation_oracle(game))

        game.reset(seed=92)
        expected = Game2048(2, 3, seed=92)
        np.testing.assert_array_equal(game.state.grid, expected.state.grid)
        self.assertFalse(game.done)
        self.assertEqual(game.rng.bit_generator.state, expected.rng.bit_generator.state)
        with self.assertRaises(ValueError):
            game.state.grid.setflags(write=True)

    def test_replace_board_copies_input_resets_score_and_recomputes_done(self):
        game = Game2048(seed=40)
        game.total_score = 123
        rng_state = copy.deepcopy(game.rng.bit_generator.state)
        board = np.array([[2, 4], [8, 16]], dtype=np.int64)

        game.replace_board(board, total_score=7)
        self.assertEqual(game.total_score, 7)
        self.assertTrue(game.done)
        self.assertEqual(game.legal_actions(), [])
        self.assertEqual(game.rng.bit_generator.state, rng_state)
        installed_state = game.state
        for invalid, error in (
            (np.empty((0, 2), dtype=np.int64), ValueError),
            (np.array([[2.5]]), ValueError),
            (np.array([[3]], dtype=np.int64), AssertionError),
        ):
            with self.subTest(invalid=invalid.tolist()):
                with self.assertRaises(error):
                    game.replace_board(invalid)
                self.assertIs(game.state, installed_state)
                self.assertEqual(game.total_score, 7)
                self.assertTrue(game.done)
                self.assertEqual(game.rng.bit_generator.state, rng_state)
        board[:, :] = 0
        self.assertEqual(game.legal_actions(), [])

        game.replace_board(np.array([[2, 0], [4, 8]], dtype=np.int64))
        self.assertFalse(game.done)
        self.assertGreater(len(game.legal_actions()), 0)
        self.assertEqual(game.total_score, 0)

    def test_standalone_states_copy_validate_and_freeze_external_grids(self):
        external = np.array([[2, 0]], dtype=np.int64)
        state = State2048(grid=external)
        self.assertFalse(state.grid.flags.writeable)
        external[0, 0] = 3
        self.assertEqual(state.grid.tolist(), [[2, 0]])
        with self.assertRaises(ValueError):
            state.grid[0, 1] = 4
        with self.assertRaises(AttributeError):
            state.grid = np.array([[8, 16, 32]], dtype=np.int64)
        with self.assertRaises(ValueError):
            state.grid.setflags(write=True)
        with self.assertRaises(ValueError):
            State2048(height=0, width=2)
        with self.assertRaises(ValueError):
            State2048(height=2, width=0)

    def test_immutable_snapshots_skip_revalidation_safely(self):
        game = Game2048(2, 3, seed=41)
        game.replace_board(np.array([[2, 0, 0], [4, 8, 16]], dtype=np.int64))
        original = game.state
        original_grid = original.grid.copy()

        with patch.object(
            State2048, "_validate", side_effect=AssertionError("unexpected validation")
        ):
            copied = original.copy()
            self.assertFalse(copied.grid.flags.writeable)
            self.assertFalse(np.shares_memory(copied.grid, original.grid))
            with self.assertRaises(ValueError):
                copied.grid[0, 0] = 3
            np.testing.assert_array_equal(original.grid, original_grid)

            transformed = game.symmetries()[0].transform_state(original)
            self.assertFalse(transformed.grid.flags.writeable)
            transformed_copy = transformed.copy()
            self.assertFalse(transformed_copy.grid.flags.writeable)
            self.assertFalse(np.shares_memory(transformed_copy.grid, transformed.grid))

            result = game.step(0)
            self.assertIs(result.next_state, game.state)
            next_copy = result.next_state.copy()
            self.assertFalse(next_copy.grid.flags.writeable)
            self.assertFalse(np.shares_memory(next_copy.grid, result.next_state.grid))
            held_snapshot = original.grid.copy()
            game.step(1)
            np.testing.assert_array_equal(original.grid, held_snapshot)

    def test_external_aliases_are_isolated_and_invalid_inputs_remain_rejected(self):
        backing = np.array([[2, 4], [8, 16]], dtype=np.int64)
        readonly_view = backing.view()
        readonly_view.flags.writeable = False
        state = State2048(grid=readonly_view)
        backing[0, 0] = 3
        np.testing.assert_array_equal(state.grid, [[2, 4], [8, 16]])
        self.assertFalse(state.grid.flags.writeable)
        np.testing.assert_array_equal(state.copy().grid, state.grid)
        transformed = Game2048().symmetries()[0].transform_state(state)
        np.testing.assert_array_equal(transformed.grid, state.grid)
        self.assertFalse(transformed.grid.flags.writeable)

        mutable_input = np.array([[2, 4]], dtype=np.int64)
        immutable = State2048(grid=mutable_input)
        mutable_input[0, 0] = 3
        np.testing.assert_array_equal(immutable.grid, [[2, 4]])
        with self.assertRaises(AssertionError):
            State2048(grid=np.array([[3]], dtype=np.int64))

        restored = State2048.__new__(State2048)
        with self.assertRaises(AssertionError):
            restored.__setstate__(
                {"_grid": np.array([[3]], dtype=np.int64), "_grid_readonly": True}
            )

    def test_legacy_mutable_and_frozen_state_pickle_compatibility(self):
        expected = np.array([[2, 4], [8, 16]], dtype=np.int64)
        expected_tensor = State2048(grid=expected).to_tensor().numpy()
        for payload in (
            {"grid": expected.copy(), "height": 2, "width": 2},
            {"_grid": expected.copy(), "_grid_frozen": True},
            {"_grid": expected.copy(), "_grid_readonly": False},
        ):
            restored = State2048.__new__(State2048)
            restored.__setstate__(payload)
            np.testing.assert_array_equal(restored.grid, expected)
            source = payload.get("_grid", payload.get("grid"))
            self.assertFalse(np.shares_memory(restored.grid, source))
            with self.assertRaises(ValueError):
                restored.grid[0, 0] = 4
            with self.assertRaises(ValueError):
                restored.grid.setflags(write=True)

        standalone = State2048(grid=expected)
        game = Game2048(seed=29)
        game.replace_board(expected)
        for original in (standalone, game.state):
            for restored in (pickle.loads(pickle.dumps(original)), copy.deepcopy(original)):
                np.testing.assert_array_equal(restored.grid, expected)
                np.testing.assert_array_equal(restored.to_tensor().numpy(), expected_tensor)
                self.assertEqual(restored.height, 2)
                self.assertEqual(restored.width, 2)
                with self.assertRaises(ValueError):
                    restored.grid[0, 0] = 4
                with self.assertRaises(ValueError):
                    restored.grid.setflags(write=True)

    def test_published_board_snapshots_survive_step_reset_and_replacement(self):
        game = Game2048(4, 4, seed=101)
        game.replace_board(
            np.array(
                [[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]],
                dtype=np.int64,
            )
        )
        initial_state = game.state
        initial_grid = initial_state.grid.copy()
        step_result = game.step(0)
        with self.assertRaises(ValueError):
            step_result.next_state.grid.setflags(write=True)
        np.testing.assert_array_equal(initial_state.grid, initial_grid)
        step_state = step_result.next_state
        step_grid = step_state.grid.copy()

        game.replace_board(np.array([[2, 4], [8, 16]], dtype=np.int64))
        np.testing.assert_array_equal(step_state.grid, step_grid)
        replacement_state = game.state
        replacement_grid = replacement_state.grid.copy()
        game.reset(seed=102)
        np.testing.assert_array_equal(replacement_state.grid, replacement_grid)

    def test_legality_and_game_over_do_not_simulate_moves(self):
        game = Game2048(3, 3, seed=5)
        game.replace_board(
            np.array([[2, 4, 8], [16, 32, 64], [128, 256, 512]], dtype=np.int64)
        )
        with patch.object(
            game,
            "_simulate_move",
            side_effect=AssertionError("legality should not simulate a complete move"),
        ):
            self.assertEqual(game.legal_actions(), [])
            self.assertTrue(game._is_game_over())

    def test_terminal_detection_and_step_do_not_call_legal_actions(self):
        game = Game2048(2, 2, seed=14)
        terminal = np.array([[2, 4], [8, 16]], dtype=np.int64)
        game.replace_board(terminal)
        with patch.object(
            game, "legal_actions", side_effect=AssertionError("should not be called")
        ):
            self.assertTrue(game._is_game_over())

        game.replace_board(np.array([[2, 0], [4, 8]], dtype=np.int64))
        with patch.object(
            game, "legal_actions", side_effect=AssertionError("should not be called")
        ):
            game.step(0)

    def test_seeded_gameplay_matches_simulation_legality(self):
        for shape in ((4, 4), (3, 5), (1, 4)):
            for seed in (0, 23, 109):
                with self.subTest(shape=shape, seed=seed):
                    optimized = Game2048(*shape, seed=seed)
                    reference = _SimulationLegalityGame(*shape, seed=seed)
                    self.assertEqual(
                        optimized.legal_actions(), reference.legal_actions()
                    )

                    for turn in range(2000):
                        self.assertEqual(
                            optimized.legal_actions(), reference.legal_actions()
                        )
                        if optimized.done:
                            break
                        action = (turn * 7 + seed) % 4
                        fast_result = optimized.step(action)
                        reference_result = reference.step(action)
                        np.testing.assert_array_equal(
                            optimized.state.grid, reference.state.grid
                        )
                        self.assertEqual(fast_result.reward, reference_result.reward)
                        self.assertEqual(
                            fast_result.terminated, reference_result.terminated
                        )
                        self.assertEqual(
                            fast_result.action_applied, reference_result.action_applied
                        )
                        self.assertEqual(optimized.total_score, reference.total_score)
                        self.assertEqual(
                            optimized.rng.bit_generator.state,
                            reference.rng.bit_generator.state,
                        )
                    else:
                        self.fail("reference game did not terminate within turn limit")


if __name__ == "__main__":
    unittest.main()
