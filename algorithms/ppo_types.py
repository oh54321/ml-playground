from dataclasses import dataclass

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
    illegal_actions_mask: torch.Tensor
    advantage: float | None = None


@dataclass(frozen=True)
class ActionDecision:
    state: State
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
    denominator_epsilon: float
    seed: int


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
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    entropy: float
