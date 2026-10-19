"""Model, Adam, and schedule recovery across 2048 model variants."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from algorithms.ppo import PPO, PPOConfig
from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS
from games.rl_2048.training.checkpointing import save_checkpoint, warm_start_from_checkpoint


ACTION_PERMUTATIONS = {
    spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS
}

PPO_CONFIG = PPOConfig(
    gamma=1.0, gae_lambda=0.95, clip_epsilon=0.2,
    value_coefficient=0.003, entropy_coefficient=0.03,
    min_update_epochs=2, max_update_epochs=2, target_kl=0.02,
    num_parallel_games=2, steps_per_game_per_rollout=4,
    max_episode_steps=100, mini_batch_size=4,
    reward_multiplier=0.01, denominator_epsilon=1e-8, seed=123,
)


def make_trainer(config: ConvActorCriticConfig, architecture: str,
                 encoding: ObservationEncoding, decay: float):
    torch.manual_seed(123)
    model = (ConvActorCriticModel(config) if architecture == "cnn"
             else D4ConvActorCriticModel(config, ACTION_PERMUTATIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    ppo = PPO(model, optimizer, PPO_CONFIG,
              lambda seed: Game2048(seed=seed, height=config.height, width=config.width,
                                    observation_encoding=encoding))
    return ppo, scheduler


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_model_variants_restore_and_continue_same_update(self):
        cases = ((architecture, encoding, height, width)
                 for architecture in ("cnn", "d4") for encoding in ObservationEncoding
                 for height, width in (((4, 4), (6, 6), (3, 5)) if architecture == "cnn" else ((4, 4), (6, 6))))
        for architecture, encoding, height, width in cases:
            with self.subTest(architecture=architecture, encoding=encoding, height=height, width=width):
                config = ConvActorCriticConfig(encoding.in_channels(height, width), height, width, 4, 8, 1, 8,
                                               kernel_size=5, block_layers=3)
                ppo, scheduler = make_trainer(config, architecture, encoding, 0.999)
                ppo.run_outer_iteration()
                scheduler.step()
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    save_checkpoint(path, ppo.model, ppo.optimizer, scheduler, ppo.config, 7,
                                    observation_encoding=encoding)
                    restored, restored_scheduler = make_trainer(config, architecture, encoding, 0.999)
                    next_iteration = warm_start_from_checkpoint(
                        path, restored.model, restored.optimizer, restored_scheduler, restored.config,
                        observation_encoding=encoding,
                    )
                    self.assertEqual(next_iteration, 8)
                    torch.testing.assert_close(restored.model.state_dict(), ppo.model.state_dict(), rtol=0, atol=0)
                    torch.testing.assert_close(restored.optimizer.state_dict(), ppo.optimizer.state_dict(), rtol=0, atol=0)
                    self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
                    other_config = replace(config, n_channels=16)
                    other_model = (ConvActorCriticModel(other_config) if architecture == "cnn"
                                   else D4ConvActorCriticModel(other_config, ACTION_PERMUTATIONS))
                    with self.assertRaisesRegex(ValueError, "model configuration"):
                        warm_start_from_checkpoint(
                            path, other_model, restored.optimizer,
                            restored_scheduler, restored.config, observation_encoding=encoding,
                        )
                    if height == width:
                        other_model = (D4ConvActorCriticModel(config, ACTION_PERMUTATIONS)
                                       if architecture == "cnn" else ConvActorCriticModel(config))
                        with self.assertRaisesRegex(ValueError, "architecture"):
                            warm_start_from_checkpoint(
                                path, other_model, restored.optimizer,
                                restored_scheduler, restored.config, observation_encoding=encoding,
                            )
                    other_encoding = (ObservationEncoding.NUMERIC if encoding is ObservationEncoding.NUMERIC_ONEHOT
                                      else ObservationEncoding.NUMERIC_ONEHOT)
                    with self.assertRaisesRegex(ValueError, "encoding"):
                        warm_start_from_checkpoint(
                            path, restored.model, restored.optimizer, restored_scheduler, restored.config,
                            observation_encoding=other_encoding,
                        )
                    with self.assertRaisesRegex(ValueError, "PPO configuration"):
                        warm_start_from_checkpoint(
                            path, restored.model, restored.optimizer, restored_scheduler,
                            replace(restored.config, min_update_epochs=1), observation_encoding=encoding,
                        )
                records, mean, std = ppo.rollout_collector.collect_batched_information()
                for trainer in (ppo, restored):
                    trainer.updater.rng = np.random.default_rng(456)
                    trainer.updater.run_epoch(trainer.updater.prepare_batch(records), mean, std)
                scheduler.step()
                restored_scheduler.step()
                torch.testing.assert_close(restored.model.state_dict(), ppo.model.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(restored.optimizer.state_dict(), ppo.optimizer.state_dict(), rtol=0, atol=0)
                self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_new_constant_schedule_keeps_restored_learning_rate(self):
        encoding = ObservationEncoding.NUMERIC
        config = ConvActorCriticConfig(encoding.in_channels(4, 4), 4, 4, 4, 8, 1, 8)
        ppo, scheduler = make_trainer(config, "cnn", encoding, 0.5)
        ppo.run_outer_iteration()
        scheduler.step()
        rate = ppo.optimizer.param_groups[0]["lr"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, ppo.model, ppo.optimizer, scheduler, ppo.config, 1,
                            observation_encoding=encoding)
            restored, constant = make_trainer(config, "cnn", encoding, 1.0)
            self.assertEqual(
                warm_start_from_checkpoint(path, restored.model, restored.optimizer, constant, restored.config,
                                           observation_encoding=encoding), 2,
            )
        self.assertEqual(constant.last_epoch, 1)
        self.assertEqual(constant.gamma, 1.0)
        for _ in range(2):
            restored.run_outer_iteration()
            constant.step()
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], rate)


if __name__ == "__main__":
    unittest.main()
