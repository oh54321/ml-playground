"""Focused mathematical checks for the one-iteration PPO audit."""
import unittest

import numpy as np

from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.scripts.audit_ppo_update import (
    clipped_surrogate_gradient_active,
    choose_large_delta,
    counterfactual_gae_horizon,
    counterfactual_gae_prefix_length,
    generalized_advantage_prefix,
    exact_action_q,
    mixture_probabilities,
)


class PPOUpdateAuditTests(unittest.TestCase):
    def test_counterfactual_gae_uses_branch_terminal_before_common_rollout_cap(self):
        # The recorded actor action may terminate after one transition, while
        # the forced alternative survives. Its GAE follows its own terminal to
        # the same scheduled segment/time-limit cap, not the actor's death time.
        cap=counterfactual_gae_horizon(segment_remaining=81,episode_age=943,max_episode_steps=1_048_576)
        self.assertEqual(cap,81)
        self.assertEqual(counterfactual_gae_prefix_length(cap,1),1)
        self.assertEqual(counterfactual_gae_prefix_length(cap,7),7)
        # One-step captured actor terminates at zero; the alternative survives
        # the short fixture to the cap and bootstraps from V after step three.
        captured_actor=generalized_advantage_prefix([1.],[10.,0.],1.,.95,cap)
        alt_to_terminal=generalized_advantage_prefix([1.,2.],[10.,20.,0.],1.,.95,cap)
        alt_survives_cap=generalized_advantage_prefix([1.,2.,3.,4.],[10.,20.,30.,40.,0.],1.,.95,3)
        self.assertAlmostEqual(captured_actor,-9.)
        self.assertAlmostEqual(alt_to_terminal,-6.1)
        self.assertAlmostEqual(alt_survives_cap,11.+.95*12.+(.95**2)*13.)
        self.assertEqual(counterfactual_gae_horizon(500,1_047_900,1_048_576),500)
        self.assertEqual(counterfactual_gae_horizon(500,1_048_100,1_048_576),476)
        with self.assertRaises(ValueError): counterfactual_gae_horizon(1,1_048_576,1_048_576)

    def test_large_delta_selection_enforces_threshold_and_diversity(self):
        def row(gap, index, game, board):
            return dict(delta_q=gap,rollout_index=index,game_index=game,board=[[board]])
        qualifying,selected=choose_large_delta([
            row(500,0,1,1), row(400,1,1,2), row(300,2,2,1),
            row(250,3,3,3), row(249.99,4,4,4)],3,250)
        self.assertEqual([r['delta_q'] for r in qualifying],[500,400,300,250])
        self.assertEqual([r['rollout_index'] for r in selected],[0,3])
        self.assertEqual([r['game_index'] for r in selected],[1,3])
        self.assertEqual(len({tuple(v for line in r['board'] for v in line) for r in selected}),2)

    def test_large_delta_selection_reports_shortfall_and_rejects_bad_settings(self):
        qualifying,selected=choose_large_delta([dict(delta_q=251,rollout_index=0,game_index=0,board=[[1]])],6,250)
        self.assertEqual((len(qualifying),len(selected)),(1,1))
        with self.assertRaises(ValueError): choose_large_delta([],0,250)
        with self.assertRaises(ValueError): choose_large_delta([],1,0)

    def test_exploration_mixture_matches_training_distribution(self):
        policy=np.array([.8,0,.2,0])
        probs=mixture_probabilities(policy,[0,2],.1)
        np.testing.assert_allclose(probs,[.77,0,.23,0])
        self.assertAlmostEqual(float(probs.sum()),1.)
        with self.assertRaises(ValueError): mixture_probabilities(policy,[0,2],1.)

    def test_clipped_gradient_mask_uses_advantage_sign(self):
        eps=.2
        self.assertTrue(clipped_surrogate_gradient_active(1.1,2.,eps))
        self.assertFalse(clipped_surrogate_gradient_active(1.21,2.,eps))
        self.assertTrue(clipped_surrogate_gradient_active(.9,-2.,eps))
        self.assertFalse(clipped_surrogate_gradient_active(.79,-2.,eps))
        # At the boundary PyTorch's min has a nonzero split derivative.
        self.assertTrue(clipped_surrogate_gradient_active(1.2,2.,eps))
        self.assertTrue(clipped_surrogate_gradient_active(.8,-2.,eps))
        self.assertFalse(clipped_surrogate_gradient_active(.5,0.,eps))

    def test_exact_q_uses_reward_multiplier_and_exact_uniform_spawns(self):
        grid=np.array([[2,2,4,8],[16,32,64,128],[256,512,1024,2048],[4096,8192,16384,32768]])
        game=Game2048(seed=1); game.state=State2048(grid=grid); game.done=False
        legal=game.legal_actions()
        self.assertIn(0,legal)
        # gamma=0 isolates immediate game-point reward; the successor
        # value callback must not influence Q even though spawns are enumerated.
        q=exact_action_q(None,game,legal,0.,.25,lambda states:[1e9]*len(states))
        moved,reward,applied=game._simulate_move(grid,0)
        self.assertTrue(applied)
        self.assertAlmostEqual(q[0],reward)

    def test_terminal_board_has_no_eligible_actions(self):
        grid=np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]])
        game=Game2048(seed=1); game.state=State2048(grid=grid); game.done=False
        self.assertEqual(game.legal_actions(),[])


if __name__=='__main__': unittest.main()
