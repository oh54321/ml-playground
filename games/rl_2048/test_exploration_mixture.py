"""The rollout sampler and PPO loss must use the same exploration policy."""

import contextlib
import io
import math
import unittest

import torch

from algorithms.ppo import PPO, PPOConfig
from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.train_2048 import linear_exploration_probability


def small_ppo() -> PPO:
    model = ConvActorCriticModel(ConvActorCriticConfig(
        in_channels=1, n_channels=8, blocks=1, height=4, width=4,
        output_dim=4, head_channels=8,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig(
        gamma=1.0, gae_lambda=0.95, clip_epsilon=0.2,
        value_coefficient=0.003, entropy_coefficient=0.01,
        min_update_epochs=1, max_update_epochs=1, target_kl=0.02,
        num_parallel_games=2, steps_per_game_per_rollout=8,
        max_episode_steps=100, mini_batch_size=8,
        reward_multiplier=0.01, value_transform="identity",
        denominator_epsilon=1e-8, gradient_clip_norm=None,
        seed=123, print_moves=False, print_timestamps=False,
    )
    return PPO(model, optimizer, config, lambda seed: Game2048(seed=seed))


class ExplorationMixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_linear_schedule_and_validation(self) -> None:
        schedule = lambda iteration: linear_exploration_probability(
            iteration, 1001, 1201, 0.05, 0.0,
        )
        self.assertAlmostEqual(schedule(1001), 0.05)
        self.assertAlmostEqual(schedule(1101), 0.025)
        self.assertAlmostEqual(schedule(1201), 0.0)
        self.assertAlmostEqual(schedule(1250), 0.0)
        ppo = small_ppo()
        for invalid in (-0.1, 1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ppo.set_exploration_probability(invalid)

    def test_mixture_is_normalized_masked_and_differentiable(self) -> None:
        ppo = small_ppo()
        ppo.set_exploration_probability(0.2)
        logits = torch.tensor([[2.0, -1.0, 0.0, 4.0]], requires_grad=True)
        illegal = torch.tensor([[False, True, False, True]])
        mixed = ppo._action_log_probabilities(logits.masked_fill(illegal, -torch.inf), illegal)
        expected = 0.8 * torch.softmax(torch.tensor([2.0, 0.0]), dim=0) + 0.1
        self.assertTrue(torch.allclose(mixed[0, [0, 2]].exp(), expected))
        self.assertTrue(torch.isneginf(mixed[0, [1, 3]]).all())
        self.assertAlmostEqual(mixed.exp().sum().item(), 1.0)
        mixed[0, 2].backward()
        self.assertGreater(logits.grad[0, 2].abs().item(), 0)

    def test_rollout_log_probability_matches_mixture_and_eval_is_raw(self) -> None:
        ppo = small_ppo()
        ppo.set_exploration_probability(0.2)
        games = ppo.games
        decisions = ppo.generate_next_actions(games)
        states = [game.current_state() for game in games]
        with torch.no_grad():
            logits = ppo.model(ppo._state_batch_tensor(states)).action_logits
            for row, (game, decision) in enumerate(zip(games, decisions, strict=True)):
                legal = game.legal_actions()
                base = torch.softmax(logits[row, legal], dim=-1)
                sampled_position = legal.index(decision.action)
                expected = 0.8 * base[sampled_position] + 0.2 / len(legal)
                self.assertAlmostEqual(math.exp(decision.log_prob), expected.item(), places=6)

            deterministic = ppo.generate_next_actions(games, deterministic=True)
            for row, (game, decision) in enumerate(zip(games, deterministic, strict=True)):
                legal = game.legal_actions()
                self.assertEqual(decision.action, legal[logits[row, legal].argmax().item()])
                raw = torch.log_softmax(logits[row, legal], dim=-1)
                self.assertAlmostEqual(decision.log_prob, raw[legal.index(decision.action)].item(), places=6)

    def test_full_ppo_update_with_mixture(self) -> None:
        ppo = small_ppo()
        ppo.set_exploration_probability(0.2)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = ppo.run_outer_iteration()
        self.assertEqual(len(stats), 1)
        self.assertTrue(math.isfinite(stats[0].approximate_kl))
        self.assertTrue(math.isfinite(ppo.last_outer_iteration_policy_kl))


if __name__ == "__main__":
    unittest.main()
