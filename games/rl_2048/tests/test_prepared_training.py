"""Regression checks for reusing fixed PPO inputs across update epochs."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.ppo import PPO, PPOConfig
from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.train_2048 import PPO_SETTINGS


class PreparedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_ppo(self, value_transform="identity"):
        torch.manual_seed(123)
        model = ConvActorCriticModel(ConvActorCriticConfig(
            n_channels=8, blocks=1, height=4, width=4, output_dim=4,
            head_channels=8, shared_trunk=False,
        ))
        config = PPOConfig(
            **dict(PPO_SETTINGS, mini_batch_size=9, min_update_epochs=3,
                   max_update_epochs=3, max_episode_steps=5,
                   value_transform=value_transform, print_timestamps=False),
            num_parallel_games=4, steps_per_game_per_rollout=7, seed=123,
        )
        return PPO(model, torch.optim.Adam(model.parameters(), lr=3e-4),
                   config, lambda seed: Game2048(seed=seed), "step")

    def collect(self, ppo):
        with patch("sys.stdout", new=io.StringIO()):
            return ppo.collect_batched_information()

    def test_reuse_matches_repreparing_each_epoch_including_partial_batch(self):
        for transform in ("identity", "signed_log1p", "signed_sqrt1p"):
            with self.subTest(value_transform=transform):
                reference = self.make_ppo(transform)
                prepared = self.make_ppo(transform)
                records, mean, std = self.collect(reference)
                self.assertTrue(any(step.truncated for step in records))
                self.assertEqual(len(records) % reference.config.mini_batch_size, 1)
                prepared.rng.bit_generator.state = deepcopy(reference.rng.bit_generator.state)
                batch = prepared.updater.prepare_batch(records)
                # Object-list and index shuffles must consume the same RNG stream.
                a, b = np.random.default_rng(456), np.random.default_rng(456)
                self.assertEqual(list(a.permutation(records)), [records[i] for i in b.permutation(len(records))])
                for epoch in range(3):
                    left = reference.run_epoch(records, mean, std, measure_gradient_diagnostics=epoch == 0)
                    right = prepared.run_epoch(batch, mean, std, measure_gradient_diagnostics=epoch == 0)
                    self.assertEqual(asdict(left), asdict(right))
                self.assertEqual(reference.last_gradient_statistics, prepared.last_gradient_statistics)
                for name, tensor in reference.model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, prepared.model.state_dict()[name]), name)
                for key, state in reference.optimizer.state_dict()["state"].items():
                    for name, tensor in state.items():
                        self.assertTrue(torch.equal(tensor, prepared.optimizer.state_dict()["state"][key][name]), name)
                self.assertEqual(reference.rng.bit_generator.state, prepared.rng.bit_generator.state)

    def test_outer_iteration_prepares_each_field_once_and_retains_early_stop(self):
        ppo = self.make_ppo()
        ppo.config = replace(ppo.config, min_update_epochs=1, target_kl=-1e30)
        records, mean, std = self.collect(ppo)
        with patch.object(ppo, "collect_batched_information", return_value=(records, mean, std)), \
             patch.object(ppo.rollout_collector, "steps_to_tensor", wraps=ppo.rollout_collector.steps_to_tensor) as convert, \
             patch.object(ppo.optimizer, "step", wraps=ppo.optimizer.step) as update:
            stats = ppo.run_outer_iteration()
        self.assertEqual(len(stats), 2)
        self.assertEqual(update.call_count, 8)  # Four minibatches per epoch, including the partial batch.
        self.assertCountEqual([call.args[1] for call in convert.call_args_list],
                              ["advantage", "log_prob", "legal_actions_mask", "action", "value"])
        self.assertTrue(np.isfinite(ppo.last_outer_iteration_policy_kl))


if __name__ == "__main__":
    unittest.main()
