import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from algorithms.ppo import PPOConfig
from core.models.conv_actor_critic import (
    NUMERIC_OBSERVATION,
    NUMERIC_ONEHOT_OBSERVATION,
    ConvActorCriticConfig,
    ConvActorCriticModel,
    observation_input_channels,
)
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from core.symmetries.d4_transforms import transform_tensor
from games.rl_2048.legacy_agent import model_from_legacy_config
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS
from games.rl_2048.environment.game_2048 import State2048
from games.rl_2048.training.checkpointing import save_checkpoint


def _model_config(encoding: str) -> ConvActorCriticConfig:
    return ConvActorCriticConfig(
        n_channels=4,
        blocks=1,
        height=2,
        width=3,
        output_dim=4,
        head_channels=4,
        include_groupnorm=False,
        observation_encoding=encoding,
    )


def _ppo_config() -> PPOConfig:
    return PPOConfig(
        gamma=1.0,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.003,
        entropy_coefficient=0.01,
        min_update_epochs=1,
        max_update_epochs=1,
        target_kl=0.02,
        num_parallel_games=1,
        steps_per_game_per_rollout=1,
        max_episode_steps=10,
        mini_batch_size=1,
        reward_multiplier=0.01,
        value_transform="identity",
        denominator_epsilon=1e-8,
        gradient_clip_norm=None,
        seed=123,
        print_moves=False,
        print_timestamps=False,
    )


class ObservationEncodingTests(unittest.TestCase):
    def test_channel_count_scales_with_rectangular_board(self) -> None:
        self.assertEqual(observation_input_channels(2, 3, NUMERIC_OBSERVATION), 1)
        self.assertEqual(observation_input_channels(2, 3, NUMERIC_ONEHOT_OBSERVATION), 9)
        self.assertEqual(observation_input_channels(4, 4, NUMERIC_ONEHOT_OBSERVATION), 19)
        numeric_config = _model_config(NUMERIC_OBSERVATION)
        onehot_config = _model_config(NUMERIC_ONEHOT_OBSERVATION)
        self.assertEqual(numeric_config.in_channels, 1)
        self.assertEqual(onehot_config.in_channels, 9)
        self.assertNotIn("in_channels", asdict(onehot_config))

    def test_numeric_onehot_keeps_magnitude_and_encodes_empty_and_tile_exponents(self) -> None:
        model = ConvActorCriticModel(_model_config(NUMERIC_ONEHOT_OBSERVATION))
        # Empty, 2, 4, 64, 128 (the maximum 2x3 exponent), and another empty.
        log_board = torch.tensor([[[[0.0, 1.0, 2.0], [6.0, 7.0, 0.0]]]])
        expanded = model.encode_observations(log_board)

        self.assertEqual(tuple(expanded.shape), (1, 9, 2, 3))
        torch.testing.assert_close(expanded[:, :1], log_board)
        for row, col, category in (
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 0, 6),
            (1, 1, 7),
        ):
            self.assertEqual(expanded[0, 1 + category, row, col].item(), 1.0)
            self.assertEqual(expanded[0, 1:, row, col].sum().item(), 1.0)

    def test_numeric_mode_preserves_original_tensor_and_forward_shapes(self) -> None:
        model = ConvActorCriticModel(_model_config(NUMERIC_OBSERVATION))
        observations = torch.tensor([[[[0.0, 1.0, 2.0], [6.0, 7.0, 0.0]]]])
        self.assertIs(model.encode_observations(observations), observations)
        output = model(observations)
        self.assertEqual(tuple(output.action_logits.shape), (1, 4))
        self.assertEqual(tuple(output.value.shape), (1,))

    def test_numeric_onehot_rejects_non_exponents_and_values_above_board_bound(self) -> None:
        model = ConvActorCriticModel(_model_config(NUMERIC_ONEHOT_OBSERVATION))
        with self.assertRaisesRegex(ValueError, "exponents from 1 through 7"):
            model.encode_observations(torch.tensor([[[[8.0, 0, 0], [0, 0, 0]]]]))
        with self.assertRaisesRegex(ValueError, "exponents from 1 through 7"):
            model.encode_observations(torch.tensor([[[[1.5, 0, 0], [0, 0, 0]]]]))

    def test_batched_and_single_state_encoding_match(self) -> None:
        model = ConvActorCriticModel(_model_config(NUMERIC_ONEHOT_OBSERVATION))
        states = [
            State2048(grid=torch.tensor([[0, 2, 4], [8, 16, 32]]).numpy()),
            State2048(grid=torch.tensor([[64, 128, 0], [0, 2, 4]]).numpy()),
        ]
        raw_batch = State2048.batch_to_tensor(states)
        batch_encoded = model.encode_observations(raw_batch)
        single_encoded = torch.cat(
            [model.encode_observations(state.to_tensor().unsqueeze(0)) for state in states]
        )
        torch.testing.assert_close(batch_encoded, single_encoded)

    def test_forward_and_checkpoint_reload_preserve_both_encodings(self) -> None:
        ppo_config = _ppo_config()
        for encoding in (NUMERIC_OBSERVATION, NUMERIC_ONEHOT_OBSERVATION):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as directory:
                config = _model_config(encoding)
                model = ConvActorCriticModel(config)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                checkpoint_path = Path(directory) / "model.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    ppo_config,
                    config,
                    outer_iteration=1,
                    mean_score=None,
                    median_score=None,
                    mean_moves=None,
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                restored = model_from_legacy_config(checkpoint["resnet_config"])
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                raw = torch.zeros((1, 1, 2, 3))
                torch.testing.assert_close(model(raw).action_logits, restored(raw).action_logits)
                self.assertEqual(
                    restored.config.observation_encoding,
                    asdict(config)["observation_encoding"],
                )

    def test_d4_model_accepts_the_expanded_board_encoding(self) -> None:
        config = ConvActorCriticConfig(
            n_channels=8,
            blocks=1,
            height=4,
            width=4,
            output_dim=4,
            head_channels=8,
            observation_encoding=NUMERIC_ONEHOT_OBSERVATION,
        )
        action_permutations = {
            spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS
        }
        model = D4ConvActorCriticModel(config, action_permutations).eval()
        board = torch.arange(16, dtype=torch.float32).remainder(18).reshape(1, 1, 4, 4)
        with torch.no_grad():
            original = model(board)
            for spec in SQUARE_SYMMETRY_SPECS:
                transformed = model(transform_tensor(board, spec.transform))
                expected_logits = torch.empty_like(original.action_logits)
                for action, mapped_action in enumerate(spec.action_permutation):
                    expected_logits[:, mapped_action] = original.action_logits[:, action]
                torch.testing.assert_close(
                    transformed.action_logits, expected_logits, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    transformed.value, original.value, rtol=1e-5, atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
