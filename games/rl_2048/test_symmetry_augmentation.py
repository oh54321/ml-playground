"""Regression tests for per-move 2048 symmetry augmentation."""

from dataclasses import asdict
import io
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.ppo import PPO, PPOConfig
from core.game import SymmetryAugmentedGame, batch_current_states
from games.rl_2048.agent import ResNet, ResNetConfig
from games.rl_2048.game import Game2048, State2048, SYMMETRIES_2048


TEST_BOARD = np.array(
    [
        [2, 2, 4, 8],
        [16, 32, 0, 64],
        [128, 256, 512, 1024],
        [2, 4, 8, 16],
    ],
    dtype=np.int64,
)


class SymmetryDefinitionTests(unittest.TestCase):
    def test_all_square_symmetries_commute_with_moves(self):
        game = Game2048(seed=0)
        state = State2048(grid=TEST_BOARD)
        self.assertEqual(len(game.symmetries()), 8)

        for symmetry in SYMMETRIES_2048:
            transformed_board = symmetry.transform_state(state).grid
            for canonical_action in range(4):
                with self.subTest(
                    symmetry=symmetry.name,
                    canonical_action=canonical_action,
                ):
                    transformed_action = symmetry.to_transformed_action(
                        canonical_action
                    )
                    canonical_next, canonical_reward, canonical_moved = (
                        game._simulate_move(TEST_BOARD, canonical_action)
                    )
                    transformed_next, transformed_reward, transformed_moved = (
                        game._simulate_move(transformed_board, transformed_action)
                    )
                    expected_next = symmetry.transform_state(
                        State2048(grid=canonical_next)
                    ).grid
                    np.testing.assert_array_equal(transformed_next, expected_next)
                    self.assertEqual(transformed_reward, canonical_reward)
                    self.assertEqual(transformed_moved, canonical_moved)
                    self.assertEqual(
                        symmetry.to_canonical_action(transformed_action),
                        canonical_action,
                    )

    def test_rectangular_boards_expose_only_shape_preserving_symmetries(self):
        game = Game2048(height=3, width=4, seed=0)
        self.assertEqual(
            [symmetry.name for symmetry in game.symmetries()],
            ["identity", "rotate_180", "flip_left_right", "flip_up_down"],
        )

    def test_batched_encoding_matches_individual_encoding(self):
        states = [
            State2048(grid=TEST_BOARD),
            State2048(grid=np.rot90(TEST_BOARD, 2).copy()),
        ]
        expected = torch.stack([state.to_tensor() for state in states])
        actual = State2048.batch_to_tensor(states)
        self.assertTrue(torch.equal(actual, expected))


class SymmetryAugmentedGameTests(unittest.TestCase):
    def make_wrapped_game(
        self,
        seed: int,
        *,
        resample: str = "step",
    ) -> SymmetryAugmentedGame:
        base = Game2048(seed=seed)
        base.state = State2048(grid=TEST_BOARD.copy())
        base.done = False
        base.total_score = 0
        return SymmetryAugmentedGame(base, seed=seed, resample=resample)

    def test_batched_current_states_match_individual_views(self):
        games = [self.make_wrapped_game(100 + index) for index in range(8)]
        for index, game in enumerate(games):
            game.symmetry_index = index
        expected = [game.current_state() for game in games]

        with patch.object(
            games[0].game,
            "transform_states",
            wraps=games[0].game.transform_states,
        ) as transform_states:
            actual = batch_current_states(games)

        transform_states.assert_called_once()
        for actual_state, expected_state in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_state.grid, expected_state.grid)

    def test_view_is_stable_until_step_and_action_maps_back(self):
        seed = 777
        wrapped = self.make_wrapped_game(seed)
        reference = Game2048(seed=seed)
        reference.state = State2048(grid=TEST_BOARD.copy())
        reference.done = False
        reference.total_score = 0

        symmetry = wrapped.current_symmetry
        first_view = wrapped.current_state()
        second_view = wrapped.current_state()
        np.testing.assert_array_equal(first_view.grid, second_view.grid)
        first_view.grid[0, 0] = 0
        np.testing.assert_array_equal(wrapped.game.state.grid, TEST_BOARD)

        expected_legal = sorted(
            symmetry.to_transformed_action(action)
            for action in reference.legal_actions()
        )
        self.assertEqual(wrapped.legal_actions(), expected_legal)
        canonical_action = reference.legal_actions()[0]
        transformed_action = symmetry.to_transformed_action(canonical_action)
        next_symmetry = (wrapped.symmetry_index + 1) % len(wrapped.symmetries())

        with patch.object(
            wrapped,
            "_sample_symmetry_index",
            return_value=next_symmetry,
        ) as sample_symmetry:
            wrapped_result = wrapped.step(transformed_action)
        reference_result = reference.step(canonical_action)

        sample_symmetry.assert_called_once()
        self.assertEqual(wrapped.symmetry_index, next_symmetry)
        self.assertEqual(wrapped_result.reward, reference_result.reward)
        self.assertEqual(wrapped_result.terminated, reference_result.terminated)
        self.assertEqual(wrapped_result.action_applied, reference_result.action_applied)
        np.testing.assert_array_equal(wrapped.game.state.grid, reference.state.grid)
        np.testing.assert_array_equal(
            wrapped_result.next_state.grid,
            wrapped.current_state().grid,
        )

    def test_episode_mode_keeps_one_symmetry_between_steps(self):
        wrapped = self.make_wrapped_game(888, resample="episode")
        symmetry_index = wrapped.symmetry_index
        transformed_action = wrapped.legal_actions()[0]
        with patch.object(wrapped, "_sample_symmetry_index") as sample_symmetry:
            result = wrapped.step(transformed_action)
        self.assertTrue(result.action_applied)
        sample_symmetry.assert_not_called()
        self.assertEqual(wrapped.symmetry_index, symmetry_index)


class SymmetryPPOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_rollout_and_update(self):
        torch.manual_seed(123)
        model = ResNet(
            ResNetConfig(
                in_channels=1,
                n_channels=8,
                blocks=1,
                height=4,
                width=4,
                output_dim=4,
                head_channels=8,
            )
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        config = PPOConfig(
            gamma=1.0,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coefficient=0.003,
            entropy_coefficient=0.01,
            min_update_epochs=1,
            max_update_epochs=1,
            target_kl=0.02,
            num_parallel_games=4,
            steps_per_game_per_rollout=8,
            max_episode_steps=100,
            mini_batch_size=16,
            reward_multiplier=0.01,
            value_transform="identity",
            denominator_epsilon=1e-8,
            gradient_clip_norm=None,
            seed=123,
            print_moves=False,
            print_timestamps=False,
        )
        ppo = PPO(
            model,
            optimizer,
            config,
            lambda seed: SymmetryAugmentedGame(
                Game2048(seed=seed),
                seed=seed,
                resample="step",
            ),
        )

        with patch("algorithms.ppo.batch_current_states", wraps=batch_current_states) as batch_states:
            with unittest.mock.patch("sys.stdout", new=io.StringIO()):
                statistics = ppo.run_outer_iteration()

        self.assertGreaterEqual(batch_states.call_count, config.steps_per_game_per_rollout)
        self.assertEqual(len(statistics), 1)
        self.assertTrue(
            all(math.isfinite(value) for value in asdict(statistics[0]).values())
        )
        self.assertTrue(math.isfinite(ppo.last_outer_iteration_policy_kl))


if __name__ == "__main__":
    unittest.main()
