import unittest

from games.rl_2048.scripts.audit_ppo_action_probability import (
    count_samples_by_board,
    direction,
    match_boards,
    summarize_advantages,
)


class ActionProbabilityAuditTests(unittest.TestCase):
    def test_matches_by_exact_board_and_critic_action(self):
        reference = [{"id": 71, "board": [[1]], "critic_action": 2}]
        bank = [
            {"id": 0, "board": [[9]], "critic_action": 2},
            {"id": 4, "board": [[1]], "critic_action": 2},
            {"id": 5, "board": [[1]], "critic_action": 3},
        ]
        self.assertEqual(match_boards(reference, bank), {71: 4})

    def test_rejects_ambiguous_board_action_match(self):
        reference = [{"board": [[1]], "critic_action": 2}]
        bank = [
            {"id": 4, "board": [[1]], "critic_action": 2},
            {"id": 5, "board": [[1]], "critic_action": 2},
        ]
        with self.assertRaisesRegex(ValueError, "Expected one exact"):
            match_boards(reference, bank)

    def test_direction_uses_explicit_probability_tolerance(self):
        self.assertEqual(direction(0.2, 0.2 + 2e-10), "increased")
        self.assertEqual(direction(0.2, 0.2 - 2e-10), "decreased")
        self.assertEqual(direction(0.2, 0.2 + 0.5e-10), "unchanged")

    def test_advantage_summary_uses_fixed_full_rollout_normalization(self):
        summary = summarize_advantages([-2.0, 2.0], mean=0.0, std=2.0)
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["positive_fraction"], 0.5)
        self.assertAlmostEqual(summary["normalized_mean_fixed_full_rollout"], 0.0)
        self.assertAlmostEqual(summary["normalized_positive_fraction_fixed_full_rollout"], 0.5)

    def test_reconstructs_per_board_action_counts_from_manifest(self):
        boards = [
            {"id": 0, "legal_actions": [0, 2]},
            {"id": 1, "legal_actions": [1, 2, 3]},
        ]
        samples = [[0, 0, 3], [0, 2, 2], [0, 2, 7], [1, 3, 1], [1, 1, 0]]
        self.assertEqual(count_samples_by_board(samples, boards), [
            {"0": 1, "2": 2},
            {"1": 1, "2": 0, "3": 1},
        ])


if __name__ == "__main__":
    unittest.main()
