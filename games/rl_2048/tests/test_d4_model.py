"""D4 symmetry, gradients, and inference-cache correctness."""

import unittest

import numpy as np
import torch

from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from core.symmetries.d4_transforms import (
    D4_TRANSFORMS, compose_transforms, transform_index, transform_tensor,
)
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding, State2048
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


ACTION_PERMUTATIONS = {
    spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS
}


class D4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_square_non2048_shape_and_action_mapping(self):
        # Actions represent all eight orientations, rather than four directions.
        permutations = {
            outer: tuple(transform_index(compose_transforms(outer, inner)) for inner in D4_TRANSFORMS)
            for outer in D4_TRANSFORMS
        }
        model = D4ConvActorCriticModel(
            ConvActorCriticConfig(2, 3, 3, 8, 8, 1, 7), permutations,
        )
        observations = torch.randn(2, 2, 3, 3)
        with torch.no_grad():
            original = model(observations)
            self.assertEqual(original.action_logits.shape, (2, 8))
            for transform, permutation in permutations.items():
                actual = model(transform_tensor(observations, transform))
                expected = torch.empty_like(original.action_logits)
                expected[:, list(permutation)] = original.action_logits
                torch.testing.assert_close(actual.action_logits, expected, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(actual.value, original.value, rtol=1e-5, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "equal height and width"):
            D4ConvActorCriticModel(ConvActorCriticConfig(2, 3, 4, 8, 8, 1, 7), permutations)
        with self.assertRaisesRegex(ValueError, "output size"):
            D4ConvActorCriticModel(ConvActorCriticConfig(2, 3, 3, 5, 8, 1, 7), permutations)

    def test_2048_policy_equivariance_value_invariance_and_gradients(self):
        exponents = np.arange(16).reshape(4, 4)
        grid = np.where(exponents == 0, 0, 2 ** exponents)
        for kernel_size, block_layers in ((3, 2), (1, 1), (5, 3)):
            for encoding in ObservationEncoding:
                with self.subTest(encoding=encoding, kernel_size=kernel_size, block_layers=block_layers):
                    torch.manual_seed(21)
                    config = ConvActorCriticConfig(
                        encoding.in_channels(4, 4), 4, 4, 4, 8, 1, 8,
                        kernel_size=kernel_size, block_layers=block_layers,
                    )
                    model = D4ConvActorCriticModel(config, ACTION_PERMUTATIONS)
                    observations = State2048.batch_to_tensor([State2048(grid, encoding)])
                    with torch.no_grad():
                        original = model(observations)
                        for spec in SQUARE_SYMMETRY_SPECS:
                            actual = model(transform_tensor(observations, spec.transform))
                            expected = torch.empty_like(original.action_logits)
                            expected[:, list(spec.action_permutation)] = original.action_logits
                            torch.testing.assert_close(actual.action_logits, expected, rtol=1e-5, atol=1e-5)
                            torch.testing.assert_close(actual.value, original.value, rtol=1e-5, atol=1e-5)
                    output = model(observations)
                    (output.action_logits.square().mean() + output.value.square().mean()).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_prepared_inference_refreshes_after_update(self):
        torch.manual_seed(21)
        model = D4ConvActorCriticModel(ConvActorCriticConfig(1, 4, 4, 4, 8, 1, 8), ACTION_PERMUTATIONS)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        game = Game2048(seed=23)
        observations = State2048.batch_to_tensor([game.state, game.state.copy()])
        for _ in range(2):
            with torch.no_grad():
                expected = model(observations)
                with model.prepared_inference():
                    actual = model(observations)
                torch.testing.assert_close(actual.action_logits, expected.action_logits, rtol=0, atol=0)
                torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
            optimizer.zero_grad()
            output = model(observations)
            (output.action_logits.square().mean() + output.value.square().mean()).backward()
            optimizer.step()
