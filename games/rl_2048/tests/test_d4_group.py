"""Regression tests for the 2048 D4-equivariant actor-critic."""

from dataclasses import replace
import io
import contextlib
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.ppo import PPO, PPOConfig
from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from core.models.d4_layers import (
    D4GroupConv,
    D4LiftConv,
    RELATIVE_ORIENTATIONS,
    _D4ScalarProbe,
    transform_features,
)
from core.symmetries.d4_transforms import D4_TRANSFORMS, transform_tensor
from core.symmetries.d4_transforms import (
    compose_transforms,
    inverse_transform,
    transform_array,
)
from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


CONFIG = ConvActorCriticConfig(
    in_channels=1, n_channels=64, blocks=4, height=4, width=4,
    output_dim=4, head_channels=32, shared_trunk=False,
)
ACTION_PERMUTATIONS = {
    spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS
}


class D4GroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_group_composition_and_inverses_match_board_transforms(self) -> None:
        board = np.arange(16).reshape(4, 4)
        for outer in D4_TRANSFORMS:
            with self.subTest(outer=outer, operation="inverse"):
                recovered = transform_array(
                    transform_array(board, outer),
                    inverse_transform(outer),
                )
                np.testing.assert_array_equal(recovered, board)
            for inner in D4_TRANSFORMS:
                with self.subTest(outer=outer, inner=inner):
                    composed = compose_transforms(outer, inner)
                    sequential_result = transform_array(
                        transform_array(board, inner), outer
                    )
                    composed_result = transform_array(board, composed)
                    np.testing.assert_array_equal(
                        composed_result, sequential_result
                    )

    def test_fast_expansions_match_previous_outputs_and_gradients(self) -> None:
        torch.manual_seed(7)
        lift = D4LiftConv(1, 2)
        group = D4GroupConv(2, 2)
        probe = _D4ScalarProbe(16, 4, 4, 5)
        cases = [
            (
                lift, torch.randn(3, 1, 4, 4, requires_grad=True),
                lambda x: F.conv2d(
                    x,
                    torch.stack([
                        transform_tensor(lift.weight, symmetry)
                        for symmetry in D4_TRANSFORMS
                    ], 1)
                    .flatten(0, 1),
                    padding=1,
                ),
            ),
            (
                group, torch.randn(3, 16, 4, 4, requires_grad=True),
                lambda x: F.conv2d(
                    x,
                    torch.stack([
                        transform_tensor(
                            group.weight.index_select(
                                2, RELATIVE_ORIENTATIONS[index]
                            ),
                            symmetry,
                        )
                        for index, symmetry in enumerate(D4_TRANSFORMS)
                    ], 1).reshape(16, 16, 3, 3),
                    padding=1,
                ),
            ),
            (
                probe, torch.randn(3, 16, 4, 4, requires_grad=True),
                lambda x: probe.network(torch.stack([
                    transform_features(x, symmetry)
                    for symmetry in D4_TRANSFORMS
                ], 1).flatten(0, 1).flatten(1)).reshape(3, 8),
            ),
        ]
        for module, x, previous_forward in cases:
            with self.subTest(module=type(module).__name__):
                previous = previous_forward(x)
                current = module(x)
                torch.testing.assert_close(current, previous, rtol=1e-5, atol=1e-6)
                parameters = (x, *module.parameters())
                previous_grads = torch.autograd.grad(previous.sum(), parameters)
                current_grads = torch.autograd.grad(current.sum(), parameters)
                for current_grad, previous_grad in zip(current_grads, previous_grads):
                    torch.testing.assert_close(
                        current_grad, previous_grad, rtol=1e-5, atol=1e-5
                    )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_fast_model_forward_and_backward_on_mps(self) -> None:
        torch.manual_seed(9)
        model = D4ConvActorCriticModel(CONFIG, ACTION_PERMUTATIONS).to("mps")
        boards = torch.randn(2, 1, 4, 4, device="mps")
        original = model(boards)
        for symmetry in SQUARE_SYMMETRY_SPECS:
            transformed = model(transform_tensor(boards, symmetry.transform))
            expected = torch.empty_like(original.action_logits)
            for action, transformed_action in enumerate(symmetry.action_permutation):
                expected[:, transformed_action] = original.action_logits[:, action]
            torch.testing.assert_close(
                transformed.action_logits, expected, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                transformed.value, original.value, rtol=1e-4, atol=1e-4
            )
        (original.action_logits.square().mean() + original.value.square().mean()).backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ))

    def test_entire_model_is_equivariant_and_within_budget(self) -> None:
        torch.manual_seed(12)
        model = D4ConvActorCriticModel(CONFIG, ACTION_PERMUTATIONS).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 139826)
        boards = torch.randn(5, 1, 4, 4)
        with torch.no_grad():
            original = model(boards)
            for symmetry in SQUARE_SYMMETRY_SPECS:
                transformed = model(transform_tensor(boards, symmetry.transform))
                expected_logits = torch.empty_like(original.action_logits)
                for action, transformed_action in enumerate(symmetry.action_permutation):
                    expected_logits[:, transformed_action] = original.action_logits[:, action]
                torch.testing.assert_close(
                    transformed.action_logits, expected_logits, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    transformed.value, original.value, rtol=1e-5, atol=1e-5
                )

    def test_ppo_updates_group_weights_with_step_augmentation(self) -> None:
        torch.manual_seed(22)
        config = replace(CONFIG, n_channels=8, blocks=1, head_channels=8)
        model = D4ConvActorCriticModel(config, ACTION_PERMUTATIONS)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        ppo_config = PPOConfig(
            gamma=1.0, gae_lambda=0.95, clip_epsilon=0.2,
            value_coefficient=0.003, entropy_coefficient=0.01,
            min_update_epochs=1, max_update_epochs=1, target_kl=0.02,
            num_parallel_games=2, steps_per_game_per_rollout=8,
            max_episode_steps=100, mini_batch_size=8,
            reward_multiplier=0.01, value_transform="identity",
            denominator_epsilon=1e-8, gradient_clip_norm=None,
            seed=123, print_moves=False, print_timestamps=False,
        )
        ppo = PPO(
            model, optimizer, ppo_config,
            lambda seed: Game2048(seed=seed),
            symmetry_augmentation="step",
        )
        initial = model.actor_trunk.body.layer1.network[0].weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            statistics = ppo.run_outer_iteration()
        self.assertEqual(len(statistics), 1)
        self.assertFalse(torch.equal(
            initial, model.actor_trunk.body.layer1.network[0].weight
        ))

    def test_masked_policy_reflects_with_a_real_board(self) -> None:
        model = D4ConvActorCriticModel(
            replace(CONFIG, n_channels=8, blocks=1, head_channels=8),
            ACTION_PERMUTATIONS,
        ).eval()
        board = State2048(grid=np.array([
            [2, 2, 4, 8],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
        ]))
        game = Game2048(seed=0)
        states = [symmetry.transform_state(board) for symmetry in Game2048().symmetries()]
        with torch.no_grad():
            logits = model(State2048.batch_to_tensor(states)).action_logits
        probabilities = []
        for row, state in enumerate(states):
            game.state = state
            legal = game.legal_actions()
            mask = torch.ones(4, dtype=torch.bool)
            mask[legal] = False
            probabilities.append(torch.softmax(logits[row].masked_fill(mask, -torch.inf), 0))
        for index, symmetry in enumerate(SQUARE_SYMMETRY_SPECS):
            expected = torch.empty_like(probabilities[0])
            for action, transformed_action in enumerate(symmetry.action_permutation):
                expected[transformed_action] = probabilities[0][action]
            torch.testing.assert_close(probabilities[index], expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
