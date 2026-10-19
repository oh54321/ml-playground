"""PPO objective calculation and optimizer updates."""


import numpy as np
import torch
from torch.distributions import categorical

from core.models.actor_critic import ActorCriticModel
from .ppo_types import (
    BatchObjectives,
    EpochStatistics,
    PolicyRolloutInformation,
    PPOConfig,
)
from .ppo_rollout import RolloutCollector


class PPOUpdater:
    """Compute the clipped PPO loss and apply minibatch updates."""

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

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def prepare_batch(
        self,
        steps: list[PolicyRolloutInformation],
    ) -> dict[str, torch.Tensor]:
        """Assemble rollout tensors once for all update epochs."""
        collector = self.rollout_collector
        return {
            "observations": collector.state_batch_tensor([step.state for step in steps]),
            "advantages": collector.steps_to_tensor(steps, "advantage"),
            "old_log_probs": collector.steps_to_tensor(steps, "log_prob"),
            "illegal_actions_mask": collector.steps_to_tensor(steps, "illegal_actions_mask", dtype=torch.bool),
            "actions": collector.steps_to_tensor(steps, "action", dtype=torch.long),
            "values": collector.steps_to_tensor(steps, "value"),
        }

    def run_epoch(
        self,
        data: dict[str, torch.Tensor],
        advantage_mean: float,
        advantage_std: float,
    ) -> EpochStatistics:
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
            objective.backward()
            self.optimizer.step()
        all_value_targets = torch.cat(value_targets)
        all_value_errors = torch.cat(value_errors)
        return EpochStatistics(
            policy_loss=torch.cat(policy_objectives).mean().item(),
            value_loss=torch.cat(value_objectives).mean().item(),
            approximate_kl=torch.cat(kl_divergences).mean().item(),
            clip_fraction=torch.cat(outside_ranges).float().mean().item(),
            explained_variance=1 - all_value_errors.var(unbiased=False).item() / (
                all_value_targets.var(unbiased=False).item() + self.config.denominator_epsilon
            ),
            entropy=-torch.cat(entropy_objectives).mean().item(),
        )

    def compute_objective(
        self,
        data: dict[str, torch.Tensor],
        advantage_mean: float,
        advantage_std: float,
    ) -> BatchObjectives:
        advantages = data["advantages"]
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + self.config.denominator_epsilon)
        old_policy_log_probs = data["old_log_probs"]
        output = self.model(data["observations"])
        action_logits = output.action_logits
        predicted_values = output.value
        illegal_actions_mask = data["illegal_actions_mask"]
        action_logits = action_logits.masked_fill(illegal_actions_mask, -torch.inf)
        log_probs = torch.log_softmax(action_logits, dim=-1)
        sampled_actions = data["actions"]
        current_log_prob = log_probs[torch.arange(log_probs.shape[0], device=self.device()), sampled_actions]
        clip_multiplier = torch.where(normalized_advantages > 0, 1 + self.config.clip_epsilon, 1 - self.config.clip_epsilon)
        advantage_clip = clip_multiplier * normalized_advantages
        approximate_kl = old_policy_log_probs - current_log_prob
        probability_ratio = torch.exp(-approximate_kl)
        policy_objective = -torch.minimum(probability_ratio * normalized_advantages, advantage_clip)
        outside_clip_range = torch.abs(probability_ratio - 1) > self.config.clip_epsilon
        value_target = advantages + data["values"]
        value_error = value_target - predicted_values
        value_objective = value_error ** 2
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
