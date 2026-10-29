"""Core PPO math, rollout boundaries, minibatches, and update stopping."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithms.ppo import PPO, PPOConfig
from algorithms.ppo_types import PolicyRolloutInformation
from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make_ppo(self):
        torch.manual_seed(123)
        model = D4ConvActorCriticModel(ConvActorCriticConfig(1, 4, 4, 4, 8, 1, 8), {
            spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS
        })
        config = PPOConfig(
            gamma=1.0, gae_lambda=0.95, clip_epsilon=0.2,
            value_coefficient=0.003, entropy_coefficient=0.03,
            min_update_epochs=2, max_update_epochs=4, target_kl=0.02,
            mini_batch_size=9, max_episode_steps=5,
            reward_multiplier=0.01, denominator_epsilon=1e-8,
            num_parallel_games=4, steps_per_game_per_rollout=7, seed=123,
        )
        return PPO(model, torch.optim.Adam(model.parameters(), lr=3e-4),
                   config, lambda seed: Game2048(seed=seed))

    def test_gae_stops_at_termination_and_bootstraps_truncation_and_rollout_end(self):
        ppo = self.make_ppo()
        ppo.rollout_collector.config = replace(ppo.config, gamma=0.9, gae_lambda=0.8)
        state = State2048()
        def step(reward, value, done=False, truncated=False):
            return PolicyRolloutInformation(state, 0, state, 0, value, reward,
                                            done, truncated, torch.zeros(4, dtype=torch.bool))
        # Episode A terminates; B truncates; C ends only at the rollout boundary.
        records = [step(1, 2), step(3, 4, done=True),
                   step(5, 6, truncated=True), step(7, 8)]
        with patch.object(ppo.rollout_collector, "predict_value", return_value=10) as predict:
            result = ppo.rollout_collector._add_advantages([records])
        torch.testing.assert_close(torch.tensor([r.advantage for r in result]),
                                   torch.tensor([1.88, -1.0, 8.0, 8.0]))
        self.assertEqual(predict.call_count, 2)

    def test_clipping_value_targets_and_illegal_action_probabilities(self):
        ppo = self.make_ppo()
        records, _, _ = ppo.rollout_collector.collect_batched_information()
        data = ppo.updater.prepare_batch(records[:2])
        data["advantages"] = torch.tensor([2.0, -2.0])
        data["values"] = torch.tensor([3.0, 4.0])
        data["actions"] = torch.tensor([0, 0])
        data["illegal_actions_mask"] = torch.tensor([[False, True, False, False]] * 2)
        output = ppo.model(data["observations"])
        log_probs = output.action_logits.masked_fill(data["illegal_actions_mask"], -torch.inf).log_softmax(-1)
        data["old_log_probs"] = log_probs[:, 0].detach() - torch.tensor([1.5, 0.5]).log()
        objectives = ppo.updater.compute_objective(data, 0.0, 2.0)
        torch.testing.assert_close(objectives.policy, torch.tensor([-1.2, 0.8]))
        torch.testing.assert_close(objectives.value_target, torch.tensor([5.0, 2.0]))
        torch.testing.assert_close(objectives.value, (torch.tensor([5.0, 2.0]) - output.value).square())
        self.assertTrue(objectives.outside_clip_range.all())
        self.assertTrue(torch.isfinite(objectives.negative_entropy).all())
        self.assertTrue((log_probs[:, 1].exp() == 0).all())

    def test_partial_minibatches_and_minimum_update_epochs(self):
        for minimum in (1, 2):
            with self.subTest(min_update_epochs=minimum):
                ppo = self.make_ppo()
                records, mean, std = ppo.rollout_collector.collect_batched_information()
                self.assertEqual(len(records), 28)
                self.assertEqual(sum(step.truncated for step in records), 4)
                ppo.config = replace(ppo.config, min_update_epochs=minimum, target_kl=1e-12)
                # Force positive KL without changing the underlying optimizer updates.
                original = ppo.updater.run_epoch
                def update(*args):
                    return replace(original(*args), approximate_kl=1.0)
                with patch.object(ppo.rollout_collector, "collect_batched_information", return_value=(records, mean, std)), \
                     patch.object(ppo.updater, "run_epoch", side_effect=update), \
                     patch.object(ppo.optimizer, "step", wraps=ppo.optimizer.step) as optimizer_step, \
                     patch.object(ppo.updater, "compute_objective", wraps=ppo.updater.compute_objective) as objective:
                    epochs = ppo.run_outer_iteration()
                self.assertEqual(len(epochs), minimum)
                self.assertEqual(optimizer_step.call_count, 4 * minimum)
                self.assertEqual([len(call.args[0]["actions"]) for call in objective.call_args_list],
                                 [9, 9, 9, 1] * minimum)

    def test_update_runs_to_maximum_when_kl_is_within_target(self):
        ppo = self.make_ppo()
        original = ppo.updater.run_epoch
        def update(*args):
            return replace(original(*args), approximate_kl=ppo.config.target_kl)
        with patch.object(ppo.updater, "run_epoch", side_effect=update):
            epochs = ppo.run_outer_iteration()
        self.assertEqual(len(epochs), ppo.config.max_update_epochs)

    def test_rollouts_use_untransformed_game_states(self):
        ppo = self.make_ppo()
        collector = ppo.rollout_collector
        initial = [game.current_state().grid.copy() for game in collector.games]
        records, _, _ = collector.collect_batched_information()
        length = ppo.config.steps_per_game_per_rollout
        for index, game in enumerate(collector.games):
            episode = records[index * length:(index + 1) * length]
            np.testing.assert_array_equal(episode[0].state.grid, initial[index])
            for previous, following in zip(episode, episode[1:]):
                if not previous.done and not previous.truncated:
                    np.testing.assert_array_equal(previous.next_state.grid, following.state.grid)
            np.testing.assert_array_equal(episode[-1].next_state.grid, game.state.grid)
