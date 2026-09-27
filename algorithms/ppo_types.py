from dataclasses import dataclass
from typing import Literal

import torch

from core.environments.state import State


@dataclass(frozen=True)
class PolicyRolloutInformation:
    state: State
    action: int
    next_state: State
    log_prob: float
    value: float
    reward: float
    done: bool # environment ended naturally
    truncated: bool # rollout was truncated due to max_episode_steps
    legal_actions_mask: torch.Tensor
    advantage: float = None


@dataclass(frozen=True)
class ActionDecision:
    action: int
    log_prob: float
    value: float
    illegal_actions_mask: torch.Tensor


@dataclass(frozen=True)
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    min_update_epochs: int
    max_update_epochs: int
    target_kl: float
    num_parallel_games: int
    steps_per_game_per_rollout: int
    max_episode_steps: int
    mini_batch_size: int
    reward_multiplier: float
    value_transform: Literal["identity", "signed_log1p", "signed_sqrt1p"]
    denominator_epsilon: float
    gradient_clip_norm: float | None
    seed: int
    print_moves: bool
    print_timestamps: bool = True


@dataclass(frozen=True)
class BatchObjectives:
    policy: torch.Tensor
    value: torch.Tensor
    negative_entropy: torch.Tensor
    approximate_kl: torch.Tensor
    outside_clip_range: torch.Tensor
    value_error: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class EpochStatistics:
    policy_loss: float
    value_loss: float
    negative_entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    value_target_mean: float = 0.0
    value_target_std: float = 0.0
    predicted_value_mean: float = 0.0
    predicted_value_std: float = 0.0
    value_error_rms: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 0.0


@dataclass(frozen=True)
class GradientStatistics:
    """Gradient diagnostics over the shared trunk or both independent trunks."""

    policy_norm: float
    value_norm: float
    entropy_norm: float
    value_to_policy_ratio: float
    policy_value_cosine_similarity: float
    relative_parameter_update_size: float = 0.0
    shared_trunk: bool = True

