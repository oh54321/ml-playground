import unittest

import numpy as np
import torch

from games.rl_2048.scripts.balanced_sampling_ppo import (
    balanced_action_counts,
    balanced_sample_weights,
    clipped_policy_loss,
    select_first_kl_passing_divisor,
)
from games.rl_2048.scripts.audit_ppo_update import counterfactual_gae_horizon, generalized_advantage_prefix


class BalancedSamplingPpoTests(unittest.TestCase):
    def test_balanced_strata_and_natural_sampling_target_same_old_behavior_objective(self):
        probabilities = {0: 0.97, 1: 0.02, 3: 0.01}
        action_mean_losses = {0: -2.0, 1: 7.0, 3: 21.0}
        counts = balanced_action_counts([0, 1, 3], 32, offset=2)
        self.assertEqual(sum(counts.values()), 32)
        self.assertEqual(set(counts), set(probabilities))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        weights = balanced_sample_weights(probabilities, counts, n_states=1)
        balanced_estimate = sum(counts[a] * weights[a] * action_mean_losses[a] for a in counts)
        natural_target = sum(probabilities[a] * action_mean_losses[a] for a in probabilities)
        self.assertAlmostEqual(balanced_estimate, natural_target)
        self.assertAlmostEqual(sum(counts[a] * weights[a] for a in counts), 1.0)

    def test_state_weighting_and_all_legal_actions_are_preserved(self):
        counts = balanced_action_counts([0, 1, 2, 3], 32, offset=1)
        weights = balanced_sample_weights({0: .1, 1: .2, 2: .3, 3: .4}, counts, n_states=24)
        self.assertEqual(counts, {0: 8, 1: 8, 2: 8, 3: 8})
        self.assertAlmostEqual(sum(counts[a] * weights[a] for a in counts), 1 / 24)
        with self.assertRaises(ValueError):
            balanced_action_counts([0, 2, 3], 2)

    def test_clipped_objective_respects_advantage_sign(self):
        ratios = torch.tensor([1.3, 0.7, 1.3, 0.7])
        advantages = torch.tensor([2.0, 2.0, -2.0, -2.0])
        losses = clipped_policy_loss(ratios, advantages, .2)
        np.testing.assert_allclose(losses.tolist(), [-2.4, -1.4, 2.6, 1.6], atol=1e-6)

    def test_posthoc_lr_selector_uses_first_divisor_passing_both_arms(self):
        probes = [
            {"divisor": 4, "first_replicate_training_kl": {"natural": .04, "balanced": .03}, "heldout_return": 10000},
            {"divisor": 8, "first_replicate_training_kl": {"natural": .018, "balanced": .021}, "heldout_return": -10000},
            {"divisor": 16, "first_replicate_training_kl": {"natural": .019, "balanced": .017}, "heldout_return": 0},
            {"divisor": 32, "first_replicate_training_kl": {"natural": .001, "balanced": .001}, "heldout_return": 99999},
        ]
        self.assertEqual(select_first_kl_passing_divisor(probes, .02), 16)

    def test_counterfactual_gae_cap_uses_branch_terminal_and_not_source_termination(self):
        cap = counterfactual_gae_horizon(segment_remaining=40, episode_age=100, max_episode_steps=150)
        self.assertEqual(cap, 40)
        short_branch = generalized_advantage_prefix([1.0], [10.0, 0.0], 1.0, .95, cap)
        longer_branch = generalized_advantage_prefix([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0, 0.0], 1.0, .95, 3)
        self.assertAlmostEqual(short_branch, -9.0)
        self.assertAlmostEqual(longer_branch, 11.0 + .95 * 12.0 + .95**2 * 13.0)


if __name__ == "__main__":
    unittest.main()
