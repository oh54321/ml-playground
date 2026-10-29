"""Generic CNN behavior and integration with encoded 2048 states."""

import unittest

import torch

from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding, State2048


class CNNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_generic_rectangular_cnn_has_requested_output_and_gradients(self):
        for kernel_size, block_layers in ((3, 2), (1, 1), (5, 3)):
            with self.subTest(kernel_size=kernel_size, block_layers=block_layers):
                config = ConvActorCriticConfig(
                    in_channels=3, height=3, width=5, output_dim=7,
                    n_channels=8, blocks=1, head_channels=8,
                    kernel_size=kernel_size, block_layers=block_layers,
                )
                model = ConvActorCriticModel(config)
                observations = torch.randn(2, 3, 3, 5)
                output = model(observations)
                self.assertEqual(tuple(output.action_logits.shape), (2, 7))
                self.assertEqual(tuple(output.value.shape), (2,))
                (output.action_logits.square().mean() + output.value.square().mean()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cnn_accepts_encoded_2048_observations(self):
        for height, width in ((4, 4), (6, 6), (3, 5)):
            for encoding in ObservationEncoding:
                with self.subTest(height=height, width=width, encoding=encoding):
                    game = Game2048(seed=23, height=height, width=width, observation_encoding=encoding)
                    config = ConvActorCriticConfig(
                        in_channels=encoding.in_channels(height, width),
                        height=height, width=width, output_dim=4,
                        n_channels=8, blocks=1, head_channels=8,
                    )
                    model = ConvActorCriticModel(config)
                    observations = State2048.batch_to_tensor([game.state, game.state.copy()])
                    output = model(observations)
                    self.assertEqual(tuple(output.action_logits.shape), (2, 4))
                    self.assertEqual(tuple(output.value.shape), (2,))


if __name__ == "__main__":
    unittest.main()
