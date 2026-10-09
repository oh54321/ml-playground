"""PPO objective calculation and optimizer updates."""

import dataclasses
from typing import List

import numpy as np

import torch
from torch.distributions import categorical

from core.models.actor_critic import ActorCriticModel, ActorCriticOutput
from .ppo_types import (
    BatchObjectives,
    EpochStatistics,
    GradientStatistics,
    PolicyRolloutInformation,
    PPOConfig,
)
from .ppo_utils import transform_value_target
from .ppo_rollout import RolloutCollector


class PPOUpdater:
    """Own PPO loss evaluation, optimization, and gradient diagnostics."""

    def __init__(
        self,
        model: ActorCriticModel,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        rng: np.random.Generator,
        rollout_collector: RolloutCollector,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.rng = rng
        self.rollout_collector = rollout_collector
        self.last_gradient_statistics: GradientStatistics | None = None

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def prepare_batch(
        self,
        steps: List[PolicyRolloutInformation] | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Assemble rollout tensors once; already prepared batches pass through."""
        if isinstance(steps, dict):
            return steps
        collector = self.rollout_collector
        return {
            "observations": collector.state_batch_tensor([step.state for step in steps]),
            "advantages": collector.steps_to_tensor(steps, "advantage"),
            "old_log_probs": collector.steps_to_tensor(steps, "log_prob"),
            "illegal_actions_mask": collector.steps_to_tensor(steps, "legal_actions_mask", dtype=torch.bool),
            "actions": collector.steps_to_tensor(steps, "action", dtype=torch.long),
            "values": collector.steps_to_tensor(steps, "value"),
        }

    def run_epoch(
        self,
        policy_rollout_information_list: List[PolicyRolloutInformation] | dict[str, torch.Tensor],
        advantage_mean: float,
        advantage_std: float,
        measure_gradient_diagnostics: bool = False,
    ) -> EpochStatistics:
        data = self.prepare_batch(policy_rollout_information_list)
        shuffled_indices = self.rng.permutation(len(data["actions"]))
        policy_objectives = []
        value_objectives = []
        entropy_objectives = []
        kl_divergences = []
        outside_ranges = []
        value_errors = []
        value_targets = []
        for i in range(0, len(shuffled_indices), self.config.mini_batch_size):
            indices = torch.as_tensor(
                shuffled_indices[i : i + self.config.mini_batch_size],
                device=self.device(), dtype=torch.long,
            )
            minibatch = {name: tensor.index_select(0, indices) for name, tensor in data.items()}
            self.optimizer.zero_grad()
            objectives = self.compute_objective(minibatch, advantage_mean, advantage_std)
            policy_objectives.append(objectives.policy.detach())
            value_objectives.append(objectives.value.detach())
            entropy_objectives.append(objectives.negative_entropy.detach())
            kl_divergences.append(objectives.approximate_kl.detach())
            outside_ranges.append(objectives.outside_clip_range.detach())
            value_errors.append(objectives.value_error.detach())
            value_targets.append(objectives.value_target.detach())
            objective = (
                objectives.policy
                + self.config.value_coefficient * objectives.value
                + self.config.entropy_coefficient * objectives.negative_entropy
            ).mean()
            if measure_gradient_diagnostics and i == 0:
                self.last_gradient_statistics = self._measure_gradient_statistics(
                    objectives,
                )
                parameters_before_update = tuple(
                    parameter.detach().clone()
                    for parameter in self.model.representation_parameters()
                )
            else:
                parameters_before_update = None
            objective.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.optimizer.step()
            if parameters_before_update is not None:
                assert self.last_gradient_statistics is not None
                self.last_gradient_statistics = dataclasses.replace(
                    self.last_gradient_statistics,
                    relative_parameter_update_size=self._relative_parameter_update_size(
                        parameters_before_update,
                        tuple(self.model.representation_parameters()),
                    ),
                )
        all_value_targets = torch.cat(value_targets)
        all_value_errors = torch.cat(value_errors)
        all_predicted_values = all_value_targets - all_value_errors
        return EpochStatistics(
            policy_loss=torch.cat(policy_objectives).mean().item(),
            value_loss=torch.cat(value_objectives).mean().item(),
            negative_entropy=torch.cat(entropy_objectives).mean().item(),
            approximate_kl=torch.cat(kl_divergences).mean().item(),
            clip_fraction=torch.cat(outside_ranges).float().mean().item(),
            explained_variance=1 - all_value_errors.var().item() /(all_value_targets.var().item() + self.config.denominator_epsilon),
            value_target_mean=all_value_targets.mean().item(),
            value_target_std=all_value_targets.std(unbiased=False).item(),
            predicted_value_mean=all_predicted_values.mean().item(),
            predicted_value_std=all_predicted_values.std(unbiased=False).item(),
            value_error_rms=torch.sqrt(all_value_errors.pow(2).mean()).item(),
            entropy=-torch.cat(entropy_objectives).mean().item(),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )

    @staticmethod
    def _gradient_vector(
        gradients: tuple[torch.Tensor | None, ...],
        parameters: tuple[torch.nn.Parameter, ...],
    ) -> torch.Tensor:
        """Flatten gradients while preserving one entry per parameter."""
        return torch.cat(
            [
                (gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                for gradient, parameter in zip(gradients, parameters)
            ]
        )

    def _measure_gradient_statistics(
        self,
        objectives: BatchObjectives,
    ) -> GradientStatistics:
        """Measure weighted gradients in the same trunk-parameter space.

        Independent trunks have disjoint policy/value gradients: their cosine
        is zero by construction, not evidence that their objectives agree.
        """
        trunk_parameters = tuple(self.model.representation_parameters())
        policy_loss = objectives.policy.mean()
        value_loss = self.config.value_coefficient * objectives.value.mean()
        entropy_loss = self.config.entropy_coefficient * objectives.negative_entropy.mean()
        policy_gradients = torch.autograd.grad(
            policy_loss,
            trunk_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        value_gradients = torch.autograd.grad(
            value_loss,
            trunk_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        entropy_gradients = torch.autograd.grad(
            entropy_loss,
            trunk_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        policy_vector = self._gradient_vector(policy_gradients, trunk_parameters)
        value_vector = self._gradient_vector(value_gradients, trunk_parameters)
        entropy_vector = self._gradient_vector(entropy_gradients, trunk_parameters)
        policy_norm = torch.linalg.vector_norm(policy_vector).item()
        value_norm = torch.linalg.vector_norm(value_vector).item()
        entropy_norm = torch.linalg.vector_norm(entropy_vector).item()
        value_to_policy_ratio = value_norm / (policy_norm + self.config.denominator_epsilon)
        policy_value_cosine_similarity = torch.nn.functional.cosine_similarity(
            policy_vector,
            value_vector,
            dim=0,
            eps=self.config.denominator_epsilon,
        ).item()
        return GradientStatistics(
            policy_norm=policy_norm,
            value_norm=value_norm,
            entropy_norm=entropy_norm,
            value_to_policy_ratio=value_to_policy_ratio,
            policy_value_cosine_similarity=policy_value_cosine_similarity,
            shared_trunk=getattr(
                self.model,
                "has_shared_representation",
                self.model.config.shared_trunk,
            ),
        )

    def _relative_parameter_update_size(
        self,
        parameters_before_update: tuple[torch.Tensor, ...],
        parameters_after_update: tuple[torch.nn.Parameter, ...],
    ) -> float:
        """Return the applied trunk update norm relative to its starting norm."""
        update_squared_norm = sum(
            (parameter.detach() - parameter_before).pow(2).sum()
            for parameter_before, parameter in zip(
                parameters_before_update,
                parameters_after_update,
            )
        )
        parameter_squared_norm = sum(
            parameter_before.pow(2).sum()
            for parameter_before in parameters_before_update
        )
        return (
            torch.sqrt(update_squared_norm).item()
            / (torch.sqrt(parameter_squared_norm).item() + self.config.denominator_epsilon)
        )

    def compute_objective(
        self,
        steps: List[PolicyRolloutInformation] | dict[str, torch.Tensor],
        advantage_mean: float,
        advantage_std: float,
    ) -> BatchObjectives:
        data = self.prepare_batch(steps)
        advantages = data["advantages"]
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + self.config.denominator_epsilon)
        old_policy_log_probs = data["old_log_probs"]
        # dimension 0 is the batch dimension
        actor_critic_output: ActorCriticOutput = self.model(data["observations"])
        action_logits = actor_critic_output.action_logits
        transformed_predicted_values = actor_critic_output.value
        predicted_values = self.rollout_collector.untransform_predicted_value(transformed_predicted_values)
        legal_actions_mask = data["illegal_actions_mask"]
        action_logits = action_logits.masked_fill(legal_actions_mask, -torch.inf)
        log_probs = self.rollout_collector.action_log_probabilities(action_logits, legal_actions_mask)
        sampled_actions = data["actions"]
        current_log_prob = log_probs[torch.arange(log_probs.shape[0], device=self.device()), sampled_actions]
        clip_multiplier = torch.where(normalized_advantages > 0, 1 + self.config.clip_epsilon, 1 - self.config.clip_epsilon)
        advantage_clip = clip_multiplier * normalized_advantages
        approximate_kl = old_policy_log_probs - current_log_prob
        probability_ratio = torch.exp(-approximate_kl)
        policy_objective = -torch.minimum(probability_ratio * normalized_advantages, advantage_clip)
        outside_clip_range = torch.abs(probability_ratio - 1) > self.config.clip_epsilon
        values = data["values"]
        value_target = advantages + values
        transformed_value_target = transform_value_target(
            value_target,
            self.config.value_transform,
        )
        value_error = value_target - predicted_values
        value_objective = (transformed_value_target - transformed_predicted_values) ** 2
        dist = categorical.Categorical(logits=log_probs)
        entropy_objective = -dist.entropy()
        return BatchObjectives(
            policy=policy_objective,
            value=value_objective,
            negative_entropy=entropy_objective,
            approximate_kl=approximate_kl,
            outside_clip_range=outside_clip_range,
            value_error=value_error,
            value_target=value_target
        )

    @torch.no_grad()
    def policy_log_probabilities(
        self,
        steps: List[PolicyRolloutInformation],
    ) -> torch.Tensor:
        """Return masked action log-probabilities for a fixed state batch."""
        if not steps:
            raise ValueError("Cannot evaluate policy probabilities for an empty batch")
        actor_critic_output: ActorCriticOutput = self.model(
            self.rollout_collector.state_batch_tensor([step.state for step in steps])
        )
        illegal_actions_mask = torch.stack(
            [step.legal_actions_mask for step in steps],
            dim=0,
        ).to(device=self.device(), dtype=torch.bool)
        masked_logits = actor_critic_output.action_logits.masked_fill(
            illegal_actions_mask,
            -torch.inf,
        )
        return self.rollout_collector.action_log_probabilities(masked_logits, illegal_actions_mask)
