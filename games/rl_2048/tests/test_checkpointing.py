"""Scheduled snapshots can be saved without inventing evaluation results."""

import tempfile
import unittest
from pathlib import Path

import torch

from algorithms.ppo import PPOConfig
from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from games.rl_2048.training.checkpointing import save_checkpoint


class CheckpointingTests(unittest.TestCase):
    def test_checkpoint_without_evaluation_keeps_iteration_and_empty_metrics(self) -> None:
        model_config = ConvActorCriticConfig(
            in_channels=1,
            n_channels=4,
            blocks=1,
            height=4,
            width=4,
            output_dim=4,
            head_channels=4,
            include_groupnorm=False,
        )
        model = ConvActorCriticModel(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        ppo_config = PPOConfig(
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint_0003.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                ppo_config,
                model_config,
                outer_iteration=3,
                mean_score=None,
                median_score=None,
                mean_moves=None,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["outer_iteration"], 3)
        self.assertEqual(
            checkpoint["evaluation"],
            {"mean_score": None, "median_score": None, "mean_moves": None},
        )
        self.assertIsNone(checkpoint["evaluation_iteration"])
        self.assertIn("state", checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    unittest.main()
