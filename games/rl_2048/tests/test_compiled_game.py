from copy import deepcopy
import pickle
import unittest

import numpy as np

from games.rl_2048.environment.game_2048 import Game2048, State2048


def numpy_move(grid: np.ndarray, action: int):
    """Independent copy of the pre-kernel NumPy movement implementation."""
    with np.errstate(over="ignore"):
        rotated = np.rot90(grid, k=action)
        rows, reward = [], 0
        for row in rotated:
            tiles = row[row != 0]
            merged = []
            index = 0
            while index < len(tiles):
                if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
                    value = tiles[index] * 2
                    merged.append(value)
                    reward += value
                    index += 2
                else:
                    merged.append(tiles[index])
                    index += 1
            rows.append(np.array(merged + [0] * (len(row) - len(merged)), dtype=row.dtype))
        result = np.rot90(np.array(rows), k=-action)
    return result, reward, not np.array_equal(result, grid)


def numpy_terminal(grid: np.ndarray) -> bool:
    if not np.any(grid):
        return True
    if np.any(grid == 0):
        return False
    return not (
        np.any(grid[:, :-1] == grid[:, 1:])
        or np.any(grid[:-1, :] == grid[1:, :])
    )


def numpy_step(grid: np.ndarray, action: int, rng: np.random.Generator):
    moved_grid, reward, moved = numpy_move(grid, action)
    moved_grid = np.array(moved_grid, copy=True, order="C")
    State2048(grid=moved_grid)  # Legacy validation happened before spawning.
    if moved:
        empty = np.argwhere(moved_grid == 0)
        if len(empty):
            row, col = empty[rng.integers(len(empty))]
            moved_grid[row, col] = 2 if rng.random() < 0.9 else 4
    return moved_grid, reward, moved, numpy_terminal(moved_grid)


def numpy_spawn(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.int64)
    for _ in range(2):
        empty = np.argwhere(grid == 0)
        if not len(empty):
            break
        row, col = empty[rng.integers(len(empty))]
        grid[row, col] = 2 if rng.random() < 0.9 else 4
    return grid


def clone_rng(rng: np.random.Generator) -> np.random.Generator:
    clone = np.random.default_rng()
    clone.bit_generator.state = deepcopy(rng.bit_generator.state)
    return clone


class TestCompiledGame(unittest.TestCase):
    def assert_move_matches(self, game, board, action):
        expected, reward, moved = numpy_move(board, action)
        actual, actual_reward, actual_moved = game._simulate_move(board, action)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_reward, reward)
        self.assertEqual(actual_moved, moved)
        return expected, reward, moved

    def assert_step_matches(self, game, board, action, rng, score=None):
        expected, reward, moved, done = numpy_step(board, action, rng)
        result = game.step(action)
        np.testing.assert_array_equal(result.next_state.grid, expected)
        self.assertEqual(result.reward, reward)
        self.assertEqual(result.action_applied, moved)
        self.assertEqual(result.terminated, done)
        self.assertEqual(game.done, done)
        self.assertFalse(game.state.grid.flags.writeable)
        self.assertEqual(game.rng.bit_generator.state, rng.bit_generator.state)
        if score is not None:
            self.assertEqual(game.total_score, score + reward)
        return expected, reward, moved, done

    def test_compiled_move_legality_and_terminal_match_numpy(self):
        rng = np.random.default_rng(802)
        values = np.array([0, 2, 4, 8, 16, 32, 64, 2**61, 2**62], dtype=np.int64)
        for shape in ((4, 4), (3, 5), (1, 4), (4, 1), (1, 1)):
            game = Game2048(*shape, seed=3)
            for _ in range(120):
                board = rng.choice(values, size=shape)
                game.replace_board(board)
                legal = [a for a in range(4) if numpy_move(board, a)[2]]
                for action in range(4):
                    self.assert_move_matches(game, board, action)
                self.assertEqual(game.legal_actions(), legal)
                self.assertEqual(game._is_game_over(), numpy_terminal(board))
                self.assertEqual(game._is_game_over(), not legal)

    def test_canonical_merges_and_tile_mass(self):
        cases = (
            ([[2, 2, 2, 2]], 0, [[4, 4, 0, 0]], 8),
            ([[2, 2, 2, 2]], 2, [[0, 0, 4, 4]], 8),
            ([[2, 2, 4, 0]], 0, [[4, 4, 0, 0]], 4),
            ([[2], [2], [2], [2]], 1, [[4], [4], [0], [0]], 8),
            ([[2], [2], [2], [2]], 3, [[0], [0], [4], [4]], 8),
        )
        for board, action, expected, reward in cases:
            board, expected = np.array(board, dtype=np.int64), np.array(expected, dtype=np.int64)
            with self.subTest(board=board.tolist(), action=action):
                actual, actual_reward, moved = self.assert_move_matches(
                    Game2048(*board.shape, seed=1), board, action
                )
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_reward, reward)
                self.assertTrue(moved)
                self.assertEqual(int(actual.sum()), int(board.sum()))

        chooser = np.random.default_rng(9021)
        values = np.array([0, 2, 4, 8, 16, 32, 64], dtype=np.int64)
        game, tested = Game2048(seed=1), 0
        for shape in ((2, 7), (3, 5), (4, 4), (1, 6), (6, 1)):
            for _ in range(100):
                board = chooser.choice(values, size=shape)
                for action in range(4):
                    moved, _, _ = game._simulate_move(board, action)
                    self.assertEqual(int(board.sum()), int(moved.sum()))
                    tested += 1
        self.assertEqual(tested, 2000)

    def test_d4_moves_match_numpy_reference(self):
        chooser = np.random.default_rng(159)
        values = np.array([0, 2, 4, 8, 16, 32], dtype=np.int64)
        for shape in ((4, 4), (3, 5)):
            game = Game2048(*shape, seed=2)
            for _ in range(12):
                board = chooser.choice(values, size=shape)
                game.replace_board(board)
                base = State2048(grid=board)
                for symmetry in game.symmetries():
                    transformed = symmetry.transform_state(base).grid
                    for action in range(4):
                        actual, reward, moved = self.assert_move_matches(
                            game, transformed, symmetry.to_transformed_action(action)
                        )
                        expected_base, expected_reward, expected_moved = numpy_move(board, action)
                        expected = symmetry.transform_state(State2048(grid=expected_base)).grid
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(reward, expected_reward)
                        self.assertEqual(moved, expected_moved)

    def test_readonly_strided_move_input_is_unchanged(self):
        owner = np.array(
            [[2, 0, 4, 0, 8, 0], [16, 0, 32, 0, 64, 0],
             [0, 0, 0, 0, 0, 0], [2, 0, 2, 0, 2, 0]], dtype=np.int64
        )
        board = owner[:, ::2]
        board.flags.writeable = False
        before = board.copy()
        self.assertFalse(board.flags.c_contiguous)
        for action in range(4):
            self.assert_move_matches(Game2048(seed=5), board, action)
            np.testing.assert_array_equal(board, before)
            self.assertFalse(board.flags.writeable)

    def test_seeded_trajectories_match_numpy_through_resets(self):
        chooser, steps, resets = np.random.default_rng(9911), 0, 0
        for shape in ((4, 4), (3, 5), (1, 4), (4, 1), (1, 1)):
            for seed in (2, 31, 407):
                game = Game2048(*shape, seed=seed)
                board, rng, score = game.state.grid.copy(), clone_rng(game.rng), game.total_score
                for _ in range(400):
                    if game.done:
                        seed = int(chooser.integers(0, 2**32))
                        game.reset(seed=seed)
                        rng = np.random.default_rng(seed)
                        board = numpy_spawn(*shape, rng)
                        score = 0
                        self.assertEqual(game.rng.bit_generator.state, rng.bit_generator.state)
                        resets += 1
                    action = int(chooser.integers(0, 4))
                    board, reward, _, _ = self.assert_step_matches(game, board, action, rng, score)
                    score += reward
                    steps += 1
        self.assertEqual(steps, 6000)
        self.assertGreater(resets, 0)

    def test_reset_seed_and_set_seed_match_numpy_rng(self):
        game = Game2048(3, 5, seed=71)
        rng = np.random.default_rng(71)
        np.testing.assert_array_equal(game.state.grid, numpy_spawn(3, 5, rng))
        self.assertEqual(game.rng.bit_generator.state, rng.bit_generator.state)
        game.step(0)
        rng = clone_rng(game.rng)
        game.reset()
        expected = numpy_spawn(3, 5, rng)
        np.testing.assert_array_equal(game.state.grid, expected)
        self.assertEqual(game.rng.bit_generator.state, rng.bit_generator.state)

        for seed in (714, 0, 123, 2**31):
            game.reset(seed=seed)
            rng = np.random.default_rng(seed)
            np.testing.assert_array_equal(game.state.grid, numpy_spawn(3, 5, rng))
            self.assertEqual(game.rng.bit_generator.state, rng.bit_generator.state)

        game.replace_board(np.array([[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        before = game.state.grid.copy()
        game.set_seed(83)
        np.testing.assert_array_equal(game.state.grid, before)
        rng = np.random.default_rng(83)
        self.assert_step_matches(game, before, 0, rng)

    def test_noop_invalid_and_terminal_steps_preserve_rng(self):
        game = Game2048(2, 2, seed=17)
        game.replace_board(np.array([[2, 0], [4, 8]], dtype=np.int64))
        rng_state, board = deepcopy(game.rng.bit_generator.state), game.state
        result = game.step(0)
        self.assertFalse(result.action_applied)
        self.assertEqual(result.reward, 0)
        self.assertEqual(game.rng.bit_generator.state, rng_state)
        board = game.state

        for action in (-1, 4):
            with self.subTest(action=action), self.assertRaises(AssertionError):
                game.step(action)
            self.assertIs(game.state, board)
            self.assertEqual(game.rng.bit_generator.state, rng_state)

        game.replace_board(np.array([[2, 4], [8, 16]], dtype=np.int64))
        rng_state, board = deepcopy(game.rng.bit_generator.state), game.state
        with self.assertRaises(RuntimeError):
            game.step(0)
        self.assertIs(game.state, board)
        self.assertEqual(game.rng.bit_generator.state, rng_state)

    def test_integer_dtypes_and_non_native_endian_boards_match_numpy(self):
        cases = (
            (np.int8, [[2, 2, 0]]),
            (np.int16, [[2, 2, 0]]),
            (np.int32, [[2, 2, 0]]),
            (np.uint64, [[2**63, 2**63, 0]]),
            (">i8", [[0, 2, 0], [4, 0, 8]]),
            (">u8", [[0, 2, 0], [4, 0, 8]]),
            (">i4", [[0, 2, 0], [4, 0, 8]]),
        )
        for dtype, data in cases:
            with self.subTest(dtype=dtype):
                board = np.asarray(data, dtype=dtype)
                game = Game2048(*board.shape, seed=347)
                rng_state = deepcopy(game.rng.bit_generator.state)
                game.replace_board(board)
                self.assertEqual(game.state.grid.dtype, board.dtype)
                self.assertEqual(game.rng.bit_generator.state, rng_state)
                rng = clone_rng(game.rng)
                expected, reward, moved, done = numpy_step(board, 0, rng)
                with np.errstate(over="ignore"):
                    result = game.step(0)
                self.assertEqual(result.next_state.grid.dtype, expected.dtype)
                np.testing.assert_array_equal(result.next_state.grid, expected)
                self.assertEqual(result.reward, reward)
                self.assertEqual(result.action_applied, moved)
                self.assertEqual(result.terminated, done)
                self.assertEqual(game.rng.bit_generator.state, rng.bit_generator.state)

        for dtype in (">i8", ">u8", ">i4"):
            game = Game2048(2, 2, seed=1)
            game.replace_board(np.array([[2, 4], [8, 16]], dtype=dtype))
            self.assertEqual(game.legal_actions(), [])
            self.assertTrue(game._is_game_over())

    def test_high_tiles_and_signed_overflow_match_legacy(self):
        game = Game2048(1, 3, seed=27)
        game.replace_board(np.array([[2**61, 2**61, 0]], dtype=np.int64))
        self.assertEqual(game.step(0).reward, np.int64(2**62))

        game.replace_board(np.array([[2**61] * 4], dtype=np.int64))
        with np.errstate(over="ignore"):
            result = game.step(0)
        self.assertEqual(result.reward, np.int64(-(2**63)))
        self.assertEqual(game.total_score, np.int64(-(2**63)))
        self.assertEqual(result.next_state.grid[0, 0], 2**62)
        self.assertEqual(result.next_state.grid[0, 1], 2**62)
        self.assertIn(result.next_state.grid[0, 2], (2, 4))
        self.assertEqual(result.next_state.grid[0, 3], 0)

        game.replace_board(np.array([[2**62, 2**62, 0]], dtype=np.int64))
        before = game.state.grid.copy(), deepcopy(game.rng.bit_generator.state), game.total_score
        with self.assertRaises(AssertionError):
            game.step(0)
        np.testing.assert_array_equal(game.state.grid, before[0])
        self.assertEqual(game.rng.bit_generator.state, before[1])
        self.assertEqual(game.total_score, before[2])
        self.assert_move_matches(game, game.state.grid, 0)  # Private move still wraps as NumPy did.

        game.replace_board(np.array([[8, 8, 0]], dtype=np.int64))
        game.total_score = np.int64(np.iinfo(np.int64).max - 10)
        with np.errstate(over="ignore"):
            result = game.step(0)
        self.assertEqual(result.reward, np.int64(16))
        self.assertEqual(game.total_score, np.int64(np.iinfo(np.int64).min + 5))

    def test_pickle_and_deepcopy_preserve_readonly_state_and_rng_trajectory(self):
        game = Game2048(seed=183)
        rng = clone_rng(game.rng)
        board, _, _, _ = numpy_step(game.state.grid.copy(), 0, rng)
        game.step(0)
        np.testing.assert_array_equal(game.state.grid, board)

        for clone in (pickle.loads(pickle.dumps(game)), deepcopy(game)):
            self.assertFalse(clone.state.grid.flags.writeable)
            with self.assertRaises(ValueError):
                clone.state.grid.setflags(write=True)
            reference, rng = clone.state.grid.copy(), clone_rng(clone.rng)
            for action in (1, 2, 3, 0) * 20:
                if clone.done:
                    break
                reference, reward, moved, done = numpy_step(reference, action, rng)
                result = clone.step(action)
                np.testing.assert_array_equal(clone.state.grid, reference)
                self.assertEqual(result.reward, reward)
                self.assertEqual(result.action_applied, moved)
                self.assertEqual(result.terminated, done)
                self.assertEqual(clone.rng.bit_generator.state, rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
