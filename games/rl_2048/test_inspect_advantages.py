"""CPU tests for the frozen-policy advantage audit."""
import contextlib
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
from games.rl_2048.inspect_advantages import (
    FrozenAgent, branch_trials, choose_boards, compare_pair, confirmation_plan,
    evaluate, game_from_board, read_json, select, sha256, stream_seed,
    summarize_trials, trajectory_signals, validate_manifest,
)


DANGER_BOARD = [[16,32,64,256],[8,16,32,64],[4,8,16,8],[2,2,8,2]]


def checkpoint(path, shared=False, transform='identity', value=2.):
    cfg = ConvActorCriticConfig(1,4,1,4,4,4,head_channels=4,shared_trunk=shared)
    model = ConvActorCriticModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.value_head.network.linear.bias.fill_(value)
    model_cfg = asdict(cfg)
    if shared:
        del model_cfg['shared_trunk']  # Old checkpoint compatibility.
    torch.save(dict(resnet_config=model_cfg,model_state_dict=model.state_dict(),ppo_config=dict(
        gamma=1.,gae_lambda=.95,reward_multiplier=.01,value_transform=transform,
        steps_per_game_per_rollout=16,max_episode_steps=10000)),path)


class AdvantageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_gae_matches_weighted_td_errors(self):
        rewards=[4.,0.,16.]
        values=[13.,17.,6.,0.]
        for gamma in (0.,.9,1.):
            for lam in (0.,.95,1.):
                for horizon in (1,2,3,10):
                    with self.subTest(gamma=gamma,lam=lam,horizon=horizon):
                        result=trajectory_signals(rewards,values,gamma,lam,horizon)
                        td=np.array(rewards)+gamma*np.array(values[1:])-np.array(values[:-1])
                        expected=sum((gamma*lam)**t*td[t] for t in range(min(3,horizon)))
                        self.assertAlmostEqual(result['gae'],expected)
                        self.assertAlmostEqual(result['game_score'],20.)
                        self.assertEqual(result['boundary_bootstrapped'],horizon<3)

    def test_lambda_one_is_mc_minus_baseline(self):
        rewards=[4.,0.,16.]
        values=[13.,1700.,600.,0.]
        for gamma in (.9,1.):
            result=trajectory_signals(rewards,values,gamma,1.,100)
            self.assertAlmostEqual(result['full_episode_gae'],result['discounted_return']-values[0])
            self.assertAlmostEqual(result['gae'],result['full_episode_gae'])
        # A one-step boundary must include successor value, not fake termination.
        result=trajectory_signals(rewards,values,1.,.95,1)
        self.assertEqual(result['gae'],4+1700-13)
        terminal=trajectory_signals([4.],[200.,0.],1.,.95,1)
        self.assertEqual(terminal['gae'],-196.)

    def test_reject_incomplete_trajectories(self):
        for rewards,values in [([], [0.]), ([4.],[2.]), ([4.],[2.,3.]), ([float('nan')],[2.,0.])]:
            with self.assertRaises(ValueError):
                trajectory_signals(rewards,values,1.,.95,1)

    def test_bad_successor_values_can_reverse_action_ranking(self):
        better=trajectory_signals([0.,10.],[0.,-200.,0.],1.,.95,2)
        worse=trajectory_signals([0.,0.],[0.,200.,0.],1.,.95,2)
        self.assertGreater(better['discounted_return'],worse['discounted_return'])
        self.assertLess(better['gae'],worse['gae'])
        # Removing bootstrap bias with a complete lambda=1 return recovers the
        # ranking. This is an analytic fixture, not a recommendation to set lambda.
        better_mc=trajectory_signals([0.,10.],[0.,-200.,0.],1.,1.,2)
        worse_mc=trajectory_signals([0.,0.],[0.,200.,0.],1.,1.,2)
        self.assertGreater(better_mc['gae'],worse_mc['gae'])

    def test_selection_unique_reproducible_and_grouped(self):
        pools=[]
        for i in range(12):
            pools.append([dict(board=[[2**(i+1)]*4]*4,source_step=1,empty_cells=0,max_tile=1024),
                          dict(board=[[0,2**(i+1),0,0]]*4,source_step=2,empty_cells=12,max_tile=1024)])
        counts=dict(representative=3,crowded=2,high_tile=2)
        first=choose_boards(pools,counts,987,2,1024)
        self.assertEqual(first,choose_boards(pools,counts,987,2,1024))
        self.assertEqual(len({b['source_game'] for b in first}),7)
        self.assertEqual(len({str(b['board']) for b in first}),7)
        self.assertTrue(all(b['empty_cells']<=2 for b in first if b['group']=='crowded'))
        with self.assertRaisesRegex(ValueError,'only 0 games qualify'):
            choose_boards(pools,dict(representative=0,crowded=0,high_tile=1),987,2,2048)

    def test_rng_namespaces_and_subset_stability(self):
        seeds={stream_seed(123,phase,board,trial,stream) for phase in (0,1,2)
               for board in range(3) for trial in range(8) for stream in (0,1)}
        self.assertEqual(len(seeds),3*3*8*2)
        self.assertEqual(stream_seed(123,2,5,7,0),stream_seed(123,2,5,7,0))

    def test_pairwise_classification_and_centering(self):
        def trial(ret,gae):
            return dict(game_score=ret,discounted_return=ret,gae=gae,full_episode_gae=gae)
        raw={0:[trial(100+x,-10+x) for x in range(8)],2:[trial(10+x,20+x) for x in range(8)]}
        result=summarize_trials(raw,[.75,0,.25,0],[(0,2)],1)
        pair=result['comparisons'][0]
        self.assertEqual(pair['status'],'opposite')
        self.assertEqual(pair['discounted_return']['mean'],90.)
        self.assertEqual(pair['gae']['mean'],-30.)
        self.assertEqual(result['best_mc_action'],0)
        self.assertEqual(result['best_gae_action'],2)
        self.assertEqual(result['actions']['0']['mc_advantage']['mean'],22.5)
        noisy={str(a):dict(trials=[trial(100,sign*10) for sign in (-1,1)]) for a in (0,2)}
        self.assertEqual(compare_pair(noisy,0,2)['status'],'inconclusive')

    def test_load_legacy_and_separate_transforms(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt'
            for shared in (True,False):
                for transform,expected in [('identity',2.),('signed_sqrt1p',8.),('signed_log1p',np.expm1(2.))]:
                    checkpoint(path,shared,transform)
                    agent=FrozenAgent.load(path)
                    probs,values,legal=agent.predict([game_from_board(DANGER_BOARD,8)])
                    self.assertAlmostEqual(float(values[0]),expected/.01,places=3)
                    self.assertEqual(legal,[[0,2]])
                    np.testing.assert_array_equal(probs,[[.5,0,.5,0]])
                    self.assertFalse(agent.model.training)
                    self.assertTrue(all(not p.requires_grad for p in agent.model.parameters()))

    def test_branches_complete_repeatable_and_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt'
            checkpoint(path)
            before=sha256(path)
            agent=replace(FrozenAgent.load(path),gae_lambda=1.)
            board=dict(id=7,board=DANGER_BOARD,gae_horizon=10000)
            first=branch_trials(agent,board,2,123,1,10000,16)
            self.assertEqual(first,branch_trials(agent,board,2,123,1,10000,16))
            for action,trials in first.items():
                self.assertIn(action,(0,2))
                self.assertEqual(len(trials),2)
                for row in trials:
                    self.assertAlmostEqual(row['gae'],row['game_score']-200.)
                    self.assertFalse(row['boundary_bootstrapped'])
            self.assertEqual(sha256(path),before)
            with self.assertRaisesRegex(RuntimeError,'no truncated results'):
                branch_trials(agent,board,2,123,1,1,16)

    def test_end_to_end_select_discover_confirm_resume(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            source=Path(temp)/'source.pt'; checkpoint(source)
            before=sha256(source); output=Path(temp)/'audit'
            select(SimpleNamespace(checkpoint=source,output=output,games=4,representative=1,
                crowded=1,high_tile=0,crowded_cells=2,high_tile_threshold=1024,
                seed=321,max_moves=10000,batch_size=16))
            manifest=validate_manifest(output)
            self.assertEqual(len(manifest['boards']),2)
            args=SimpleNamespace(command='evaluate',folder=output,trials=2,max_moves=10000,
                                 batch_size=16,threads=1,resume=False)
            evaluate(args)
            result_path=output/'discovery'/'results.json'
            first=result_path.read_bytes()
            args.resume=True; evaluate(args)
            self.assertEqual(first,result_path.read_bytes())
            args.trials=3
            with self.assertRaisesRegex(ValueError,'Resume settings'):
                evaluate(args)
            args.trials=2; args.command='confirm'; args.resume=False; args.boards=[0]; args.top=3
            plan=confirmation_plan(output,manifest,[0],3)
            evaluate(args)
            confirmed=read_json(output/'confirmation'/'results.json')
            self.assertEqual(confirmed['settings']['plan'],plan)
            self.assertEqual(len(confirmed['boards']),1)
            self.assertEqual(len(confirmed['boards'][0]['comparisons']),1)
            self.assertTrue((output/'confirmation'/'report.md').exists())
            self.assertEqual(sha256(source),before)
            self.assertEqual(sha256(output/'checkpoint.pt'),before)
            # A changed checkpoint must be refused before any more trials.
            checkpoint(output/'checkpoint.pt',value=3.)
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                validate_manifest(output)


if __name__=='__main__':
    unittest.main()
