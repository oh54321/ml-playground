"""CPU regression tests: python -m unittest games.rl_2048.tests.test_agent."""

import contextlib
from dataclasses import asdict, replace
import io
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from algorithms.ppo import PPO, PPOConfig
from core.models.actor_critic import ActorCriticOutput
from games.rl_2048.legacy_agent import model_from_legacy_config
from core.models.conv_actor_critic import ConvActorCriticModel, ConvActorCriticConfig
from core.models.legacy.d4_residual_adapter import (
    D4ResidualAdapterActorCriticModel,
    D4ResidualStem,
)
from core.symmetries.d4_transforms import (
    D4_TRANSFORMS,
    inverse_transform,
    transform_tensor,
)
from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.scripts.inspect_game import ActivationRecorder
from games.rl_2048.utils.plot_training_diagnostics import _parse_raw_summary
from games.rl_2048.utils.reporting import print_gradient_diagnostics


BASE_CONFIG = ConvActorCriticConfig(
    in_channels=1, n_channels=8, blocks=1, height=4, width=4,
    output_dim=4, head_channels=8,
)


class ActorCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_output_and_checkpoint_roundtrip(self):
        inputs = torch.randn(3, 1, 4, 4)
        for shared in (True, False):
            with self.subTest(shared=shared):
                config = replace(BASE_CONFIG, shared_trunk=shared)
                model = ConvActorCriticModel(config)
                output = model(inputs)
                self.assertIsInstance(output, ActorCriticOutput)
                self.assertEqual(output.action_logits.shape, (3, 4))
                self.assertEqual(output.value.shape, (3,))
                self.assertEqual(model(inputs[:1]).value.shape, (1,))
                checkpoint = io.BytesIO()
                torch.save(dict(config=asdict(config), state=model.state_dict()), checkpoint)
                checkpoint.seek(0)
                loaded = torch.load(checkpoint, weights_only=True)
                restored = ConvActorCriticModel(ConvActorCriticConfig(**loaded['config']))
                restored.load_state_dict(loaded['state'], strict=True)
                self.assertTrue(torch.equal(restored(inputs).action_logits, output.action_logits))
                self.assertTrue(torch.equal(restored(inputs).value, output.value))

    def test_legacy_config_and_shared_state_keys(self):
        legacy_config = asdict(BASE_CONFIG)
        del legacy_config['shared_trunk']
        model = model_from_legacy_config(legacy_config)
        self.assertTrue(model.config.shared_trunk)
        self.assertEqual(
            {key.split('.')[0] for key in model.state_dict()},
            {'trunk', 'action_head', 'value_head'},
        )
        inputs = torch.randn(2, 1, 4, 4)
        features = model.trunk(inputs)
        self.assertTrue(torch.equal(model(inputs).action_logits, model.action_head(features)))
        self.assertTrue(torch.equal(model(inputs).value, model.value_head(features).squeeze(-1)))

    def test_independent_gradients_and_adam_updates(self):
        model = ConvActorCriticModel(replace(BASE_CONFIG, shared_trunk=False))
        actor = tuple(model.actor_trunk.parameters()) + tuple(model.action_head.parameters())
        critic = tuple(model.critic_trunk.parameters()) + tuple(model.value_head.parameters())
        self.assertFalse({id(p) for p in actor} & {id(p) for p in critic})
        self.assertEqual({id(p) for p in actor + critic}, {id(p) for p in model.parameters()})
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.randn(4, 1, 4, 4)
        # Populate Adam's momentum in BOTH networks before checking isolation.
        output = model(inputs)
        (output.action_logits.square().mean() + output.value.square().mean()).backward()
        optimizer.step()
        for update_actor in (False, True):
            with self.subTest(update_actor=update_actor):
                before = model(inputs)
                untouched = critic if update_actor else actor
                saved = [p.detach().clone() for p in untouched]
                optimizer.zero_grad(set_to_none=True)
                output = model(inputs)
                loss = output.action_logits.square().mean() if update_actor else output.value.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is None for p in untouched))
                optimizer.step()
                self.assertTrue(all(torch.equal(p, old) for p, old in zip(untouched, saved)))
                after = model(inputs)
                if update_actor:
                    self.assertTrue(torch.equal(before.value, after.value))
                    self.assertFalse(torch.equal(before.action_logits, after.action_logits))
                else:
                    self.assertTrue(torch.equal(before.action_logits, after.action_logits))
                    self.assertFalse(torch.equal(before.value, after.value))

    def test_parameter_budgets(self):
        baseline = replace(BASE_CONFIG, n_channels=32, blocks=8, head_channels=32)
        near = replace(baseline, shared_trunk=False, blocks=4)
        count = lambda config: sum(p.numel() for p in ConvActorCriticModel(config).parameters())
        self.assertEqual(count(baseline), 153573)
        self.assertEqual(count(near), 153925)
        budgeted_adapter = replace(near, head_channels=16)
        adapter = D4ResidualAdapterActorCriticModel(budgeted_adapter)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 153669)

    def test_d4_residual_stem_is_batched_and_starts_as_identity_stem(self):
        model = D4ResidualAdapterActorCriticModel(BASE_CONFIG)
        stem = model.trunk._modules["stem"]
        self.assertIsInstance(stem, D4ResidualStem)

        inputs = torch.randn(3, 1, 4, 4)
        for symmetry in D4_TRANSFORMS:
            transformed = transform_tensor(inputs, symmetry)
            recovered = transform_tensor(
                transformed, inverse_transform(symmetry)
            )
            self.assertTrue(torch.equal(recovered, inputs))

        with torch.no_grad():
            expected = stem.shared_stem(inputs)
            actual = stem(inputs)
        self.assertTrue(torch.equal(actual, expected))

        output = model(inputs)
        (output.action_logits.square().mean() + output.value.square().mean()).backward()
        self.assertIsNotNone(stem.residual_projection.weight.grad)
        self.assertGreater(stem.residual_projection.weight.grad.abs().sum().item(), 0)

    def test_activation_recorder_uses_actor_features(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                model = ConvActorCriticModel(replace(BASE_CONFIG, shared_trunk=shared))
                inputs = torch.randn(2, 1, 4, 4)
                recorder = ActivationRecorder(model)
                model(inputs)
                self.assertEqual(set(recorder.activations), {'trunk', 'stem', 'body'})
                captured = recorder.activations['trunk'].clone()
                recorder.close()
                self.assertTrue(torch.equal(captured, model.policy_trunk(inputs)))
                self.assertEqual(len(model.policy_trunk._forward_hooks), 0)

    def test_ppo_rollout_update_and_diagnostics(self):
        config = PPOConfig(
            gamma=1.0, gae_lambda=.95, clip_epsilon=.2, value_coefficient=.003,
            entropy_coefficient=.01, min_update_epochs=1, max_update_epochs=1,
            target_kl=.02, num_parallel_games=2, steps_per_game_per_rollout=8,
            max_episode_steps=100, mini_batch_size=8, reward_multiplier=.01,
            value_transform='identity', denominator_epsilon=1e-8,
            gradient_clip_norm=None, seed=123, print_moves=False, print_timestamps=False,
        )
        presets = [
            replace(BASE_CONFIG, shared_trunk=True),
            replace(BASE_CONFIG, shared_trunk=False),
            replace(BASE_CONFIG, n_channels=32, blocks=8, head_channels=32),
            replace(BASE_CONFIG, shared_trunk=False, n_channels=32, blocks=4, head_channels=32),
        ]
        for model_config in presets:
            shared = model_config.shared_trunk
            with self.subTest(model_config=model_config):
                model = ConvActorCriticModel(model_config)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                ppo = PPO(model, optimizer, config, lambda seed: Game2048(seed=seed))
                initial = {name: p.detach().clone() for name, p in model.named_parameters()}
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = ppo.run_outer_iteration()
                self.assertEqual(len(stats), 1)
                self.assertTrue(all(math.isfinite(v) for v in asdict(stats[0]).values()))
                gradients = ppo.last_gradient_statistics
                self.assertEqual(gradients.shared_trunk, shared)
                self.assertGreater(gradients.policy_norm, 0)
                self.assertGreater(gradients.value_norm, 0)
                self.assertGreater(gradients.relative_parameter_update_size, 0)
                self.assertTrue(math.isfinite(ppo.last_outer_iteration_policy_kl))
                trunks = ('trunk',) if shared else ('actor_trunk', 'critic_trunk')
                for trunk in trunks:
                    self.assertTrue(any(
                        not torch.equal(initial[name], parameter)
                        for name, parameter in model.named_parameters() if name.startswith(trunk + '.')
                    ))
                if not shared:
                    self.assertEqual(gradients.policy_value_cosine_similarity, 0.0)
                # Existing plot parsing must retain statistics for either mode.
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    print('Outer iteration 1')
                    print_gradient_diagnostics(gradients)
                with patch.object(Path, 'read_text', return_value=printed.getvalue()):
                    record = _parse_raw_summary(Path('unused'))[0]
                self.assertAlmostEqual(record['policy_gradient'], gradients.policy_norm, places=3)
                self.assertAlmostEqual(record['value_gradient'], gradients.value_norm, places=3)
                self.assertGreater(record['parameter_update'], 0)


if __name__ == '__main__':
    unittest.main()
