"""Regression tests for per-move 2048 symmetry augmentation."""

from dataclasses import asdict
import io
import math
import unittest

import numpy as np
import torch

from algorithms.ppo import PPO, PPOConfig
from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
from core.symmetries.d4_transforms import transform_tensor
from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.environment.symmetry_2048 import (
    RECTANGULAR_SYMMETRY_SPECS,
    SQUARE_SYMMETRY_SPECS,
)


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
    def test_symmetry_objects_are_shared_by_board_shape(self):
        square = Game2048(4, 4, seed=1).symmetries()
        rectangular = Game2048(3, 4, seed=2).symmetries()
        self.assertIs(square, Game2048(2, 2, seed=3).symmetries())
        self.assertIs(rectangular, Game2048(5, 7, seed=4).symmetries())
        self.assertIsNot(square, rectangular)

    def test_all_square_symmetries_commute_with_moves(self):
        game = Game2048(seed=0)
        state = State2048(grid=TEST_BOARD)
        game_symmetries = game.symmetries()
        self.assertEqual(len(game_symmetries), len(SQUARE_SYMMETRY_SPECS))

        for index, symmetry in enumerate(game_symmetries):
            transformed_board = symmetry.transform_state(state).grid
            for canonical_action in range(4):
                with self.subTest(
                    symmetry=index,
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
        expected_action_permutations = tuple(
            spec.action_permutation
            for spec in RECTANGULAR_SYMMETRY_SPECS
        )
        self.assertEqual(
            tuple(symmetry.action_permutation for symmetry in game.symmetries()),
            expected_action_permutations,
        )

    def test_batched_encoding_matches_individual_encoding(self):
        states = [
            State2048(grid=TEST_BOARD),
            State2048(grid=np.rot90(TEST_BOARD, 2).copy()),
        ]
        expected = torch.stack([state.to_tensor() for state in states])
        actual = State2048.batch_to_tensor(states)
        self.assertTrue(torch.equal(actual, expected))

    def test_tensor_transform_matches_the_game_state_transform(self):
        state = State2048(grid=TEST_BOARD)
        tensor = state.to_tensor().unsqueeze(0)
        for index, (symmetry, spec) in enumerate(
            zip(Game2048().symmetries(), SQUARE_SYMMETRY_SPECS, strict=True)
        ):
            with self.subTest(symmetry=index):
                expected = symmetry.transform_state(state).to_tensor().unsqueeze(0)
                actual = transform_tensor(tensor, spec.transform)
                torch.testing.assert_close(actual, expected)


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
        model = ConvActorCriticModel(
            ConvActorCriticConfig(
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
            lambda seed: Game2048(seed=seed),
            symmetry_augmentation="step",
        )
        self.assertTrue(all(isinstance(game, Game2048) for game in ppo.games))
        with unittest.mock.patch("sys.stdout", new=io.StringIO()):
            statistics = ppo.run_outer_iteration()
        self.assertEqual(len(statistics), 1)
        self.assertTrue(
            all(math.isfinite(value) for value in asdict(statistics[0]).values())
        )
        self.assertTrue(math.isfinite(ppo.last_outer_iteration_policy_kl))


if __name__ == "__main__":
    unittest.main()
