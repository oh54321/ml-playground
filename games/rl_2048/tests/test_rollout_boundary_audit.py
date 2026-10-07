import unittest

from games.rl_2048.scripts.audit_rollout_boundaries import gae, gae_vector


class RolloutBoundaryGaeTests(unittest.TestCase):
    def test_bootstrap_and_lambda_trace(self):
        # Two unit rewards, zero values, and a boundary value of ten.
        self.assertAlmostEqual(gae([1, 1], [0, 0], [False, False], 1, 1, 10), 12)
        self.assertAlmostEqual(gae([1, 1], [0, 0], [False, False], 1, .5, 10), 6.5)

    def test_terminal_zeros_bootstrap_and_cuts_trace(self):
        self.assertAlmostEqual(gae([1], [0], [True], 1, .99, 100), 1)
        # A terminal transition after a nonterminal one stops propagation.
        self.assertAlmostEqual(gae([1, 2], [0, 0], [False, True], 1, .5, 100), 2)

    def test_boundary_bootstrap_is_not_terminal(self):
        self.assertAlmostEqual(gae([2], [3], [False], 1, .95, 7), 6)

    def test_truncation_bootstraps_but_stops_lambda_trace(self):
        # The final step bootstraps to 5 and its GAE propagates backward, but
        # the trace stops at the truncated step itself.
        values = gae_vector([1, 2], [0, 0], [False, False], 1, .5, 9,
                            truncated=[False, True], truncation_bootstrap=[0, 5])
        self.assertEqual(values, [4.5, 7])


if __name__ == "__main__":
    unittest.main()
