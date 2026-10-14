"""Compiling callables preserves checkpoint keys and independent gradients."""
import unittest
from unittest.mock import patch

import torch

from core.models.conv_actor_critic import ConvActorCriticConfig
from core.models.conv_actor_critic import ConvActorCriticModel
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


class CompiledTrunkTests(unittest.TestCase):
    def test_shared_trunk_runs_once_for_plain_and_d4_models(self):
        config = ConvActorCriticConfig(
            n_channels=8, blocks=1, height=4, width=4, output_dim=4,
            head_channels=8, shared_trunk=True,
        )
        mappings = {s.transform: s.action_permutation for s in SQUARE_SYMMETRY_SPECS}
        observations = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) % 12
        for model in (ConvActorCriticModel(config), D4ConvActorCriticModel(config, mappings)):
            with torch.no_grad():
                encoded = model.encode_observations(observations)
                features = model.trunk(encoded)
                expected_logits = model.action_head(features)
                expected_value = model.value_head(features).squeeze(dim=-1)
            calls = []
            handle = model.trunk.register_forward_hook(lambda *_: calls.append(1))
            try:
                with torch.no_grad():
                    actual = model(observations)
            finally:
                handle.remove()
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(actual.action_logits, expected_logits)
            torch.testing.assert_close(actual.value, expected_value)

    def test_plain_cnn_compilation_preserves_outputs_gradients_and_keys(self):
        config = ConvActorCriticConfig(
            n_channels=8, blocks=1, height=4, width=4, output_dim=4,
            head_channels=8, shared_trunk=False,
        )
        model = ConvActorCriticModel(config)
        observations = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) % 12
        parameters = tuple(model.parameters())
        keys = tuple(model.state_dict())

        def probe():
            output = model(observations)
            objective = output.action_logits.square().mean() + output.value.square().mean()
            gradients = torch.autograd.grad(objective, parameters)
            return output, gradients

        reference = probe()
        compiler = torch.compile
        with patch("torch.compile", side_effect=lambda fn, **kw: compiler(fn, backend="eager", **kw)):
            model.compile_trunks()
        actual = probe()
        for field in ("action_logits", "value"):
            torch.testing.assert_close(getattr(actual[0], field), getattr(reference[0], field), rtol=0, atol=0)
        for expected, result in zip(reference[1], actual[1]):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(tuple(model.state_dict()), keys)

    def test_compilation_preserves_checkpoints_and_retained_gradients(self):
        config = ConvActorCriticConfig(
            n_channels=8, blocks=1, height=4, width=4, output_dim=4,
            head_channels=8, shared_trunk=False,
        )
        mappings = {s.transform: s.action_permutation for s in SQUARE_SYMMETRY_SPECS}
        model = D4ConvActorCriticModel(config, mappings)
        observations = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) % 12
        parameters = tuple(model.parameters())
        keys = tuple(model.state_dict())
        optimizer = torch.optim.Adam(parameters)

        def probe():
            output = model(observations)
            policy = output.action_logits.square().mean()
            value = output.value.square().mean()
            # Match PPO's component probes followed by the optimizer backward.
            for objective in (policy, value, policy + value):
                torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            gradients = torch.autograd.grad(policy + value, parameters)
            with torch.no_grad(), model.prepared_inference():
                prepared = model(observations)
            return (
                (output.action_logits, output.value), gradients,
                (prepared.action_logits, prepared.value),
            )

        reference = probe()
        compiler = torch.compile
        # Capture real graphs without requiring a platform C++/CUDA toolchain.
        with patch('torch.compile', side_effect=lambda fn, **kw: compiler(fn, backend='eager', **kw)):
            model.compile_trunks()
        actual = probe()
        for expected_group, actual_group in zip(reference, actual):
            for expected, result in zip(expected_group, actual_group):
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertTrue(all(a is b for a, b in zip(parameters, model.parameters())))
        self.assertTrue(all(a is b for a, b in zip(parameters, optimizer.param_groups[0]['params'])))
        restored = D4ConvActorCriticModel(config, mappings)
        restored.load_state_dict(model.state_dict(), strict=True)
        expected, result = model(observations), restored(observations)
        for field in ('action_logits', 'value'):
            torch.testing.assert_close(getattr(result, field), getattr(expected, field), rtol=0, atol=0)
