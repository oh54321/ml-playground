from datetime import datetime
from typing import Callable, List, Literal

import numpy as np
import torch

from core.environments.game import Game
from core.environments.state import State
from core.models.actor_critic import ActorCriticModel
from .ppo_rollout import RolloutCollector
from .ppo_types import (
    ActionDecision,
    BatchObjectives,
    EpochStatistics,
    GradientStatistics,
    PolicyRolloutInformation,
    PPOConfig,
)
from .ppo_update import PPOUpdater
from .ppo_utils import forward_policy_kl


__all__ = [
    "PPO",
    "PPOConfig",
    "PolicyRolloutInformation",
    "ActionDecision",
    "BatchObjectives",
    "EpochStatistics",
    "GradientStatistics",
]


class PPO:
    """Coordinate rollout collection and PPO updates for one training run."""

    def __init__(
        self,
        model: ActorCriticModel,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        game_factory: Callable[[int], Game],
        symmetry_augmentation: Literal["step", "episode"] | None = None,
    ) -> None:
        self._validate_config(config, symmetry_augmentation)
        torch.manual_seed(config.seed)
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.rng = np.random.default_rng(config.seed)
        self.rollout_collector = RolloutCollector(
            model,
            config,
            game_factory,
            self.rng,
            symmetry_augmentation,
        )
        self.updater = PPOUpdater(
            model,
            optimizer,
            config,
            self.rng,
            self.rollout_collector,
        )
        self.last_outer_iteration_policy_kl: float | None = None

    @staticmethod
    def _validate_config(
        config: PPOConfig,
        symmetry_augmentation: Literal["step", "episode"] | None,
    ) -> None:
        if config.value_transform not in ("identity", "signed_log1p", "signed_sqrt1p"):
            raise ValueError(f"Unsupported value transform: {config.value_transform}")
        if config.num_parallel_games <= 0:
            raise ValueError("num_parallel_games must be positive")
        if config.steps_per_game_per_rollout <= 0:
            raise ValueError("steps_per_game_per_rollout must be positive")
        if config.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if config.min_update_epochs <= 0:
            raise ValueError("min_update_epochs must be positive")
        if config.max_update_epochs < config.min_update_epochs:
            raise ValueError("max_update_epochs must be at least min_update_epochs")
        if config.target_kl <= 0:
            raise ValueError("target_kl must be positive")
        if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if symmetry_augmentation not in (None, "step", "episode"):
            raise ValueError("symmetry_augmentation must be None, 'step', or 'episode'")

    @property
    def games(self) -> list[Game]:
        return self.rollout_collector.games

    @property
    def exploration_probability(self) -> float:
        return self.rollout_collector.exploration_probability

    @exploration_probability.setter
    def exploration_probability(self, probability: float) -> None:
        self.rollout_collector.set_exploration_probability(probability)

    @property
    def last_gradient_statistics(self) -> GradientStatistics | None:
        return self.updater.last_gradient_statistics

    def set_exploration_probability(self, probability: float) -> None:
        self.rollout_collector.set_exploration_probability(probability)

    def generate_next_actions(
        self,
        games: List[Game],
        deterministic: bool = False,
        states: List[State] | None = None,
        legal_actions: List[list[int]] | None = None,
    ) -> List[ActionDecision]:
        return self.rollout_collector.generate_next_actions(
            games,
            deterministic=deterministic,
            states=states,
            legal_actions=legal_actions,
        )

    def collect_batched_information(
        self,
    ) -> tuple[list[PolicyRolloutInformation], float, float]:
        return self.rollout_collector.collect_batched_information()

    def run_epoch(
        self,
        policy_rollout_information_list: List[PolicyRolloutInformation] | dict[str, torch.Tensor],
        advantage_mean: float,
        advantage_std: float,
        measure_gradient_diagnostics: bool = False,
    ) -> EpochStatistics:
        return self.updater.run_epoch(
            policy_rollout_information_list,
            advantage_mean,
            advantage_std,
            measure_gradient_diagnostics,
        )

    def _compute_objective(
        self,
        steps: List[PolicyRolloutInformation],
        advantage_mean: float,
        advantage_std: float,
    ) -> BatchObjectives:
        return self.updater.compute_objective(steps, advantage_mean, advantage_std)

    def _state_batch_tensor(self, states: List[State]) -> torch.Tensor:
        return self.rollout_collector.state_batch_tensor(states)

    def _predict_value(self, state: State) -> float:
        return self.rollout_collector.predict_value(state)

    def _untransform_predicted_value(self, value: torch.Tensor) -> torch.Tensor:
        return self.rollout_collector.untransform_predicted_value(value)

    def _action_log_probabilities(
        self,
        masked_logits: torch.Tensor,
        illegal_actions_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.rollout_collector.action_log_probabilities(
            masked_logits,
            illegal_actions_mask,
        )

    def _print_timestamp(self, message: str) -> None:
        if self.config.print_timestamps:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{timestamp}] {message}")

    def run_outer_iteration(self) -> List[EpochStatistics]:
        self._print_timestamp("Outer iteration starting; collecting rollouts.")
        rollouts, advantage_mean, advantage_std = self.collect_batched_information()
        old_policy_log_probabilities = self.updater.policy_log_probabilities(rollouts)
        training_batch = self.updater.prepare_batch(rollouts)
        self._print_timestamp("Rollouts collected; starting PPO updates.")
        epoch_stats = []
        self.updater.last_gradient_statistics = None
        self.last_outer_iteration_policy_kl = None
        self._print_timestamp(f"PPO update epoch 0/{self.config.max_update_epochs}.")

        for epoch in range(1, self.config.max_update_epochs + 1):
            current_epoch_stats = self.run_epoch(
                training_batch,
                advantage_mean,
                advantage_std,
                measure_gradient_diagnostics=epoch == 1,
            )
            epoch_stats.append(current_epoch_stats)
            if epoch % 10 == 0:
                self._print_timestamp(
                    f"PPO update epoch {epoch}/{self.config.max_update_epochs} complete."
                )
            if (
                epoch > self.config.min_update_epochs
                and current_epoch_stats.approximate_kl > self.config.target_kl
            ):
                self._print_timestamp(
                    "Stopping PPO updates early: "
                    f"approximate KL {current_epoch_stats.approximate_kl:.6f} "
                    f"exceeded target {self.config.target_kl:.6f} "
                    f"after {epoch} epochs."
                )
                break
        new_policy_log_probabilities = self.updater.policy_log_probabilities(rollouts)
        self.last_outer_iteration_policy_kl = forward_policy_kl(
            old_policy_log_probabilities,
            new_policy_log_probabilities,
        ).item()
        self._print_timestamp("Outer iteration complete.")
        return epoch_stats
