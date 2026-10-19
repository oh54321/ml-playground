"""Proximal policy optimization: collect rollouts, then update on fixed data."""

from typing import Callable

import numpy as np
import torch

from core.environments.game import Game
from core.models.actor_critic import ActorCriticModel
from .ppo_rollout import RolloutCollector
from .ppo_types import PPOConfig, EpochStatistics
from .ppo_update import PPOUpdater


class PPO:
    def __init__(
        self,
        model: ActorCriticModel,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        game_factory: Callable[[int], Game],
    ) -> None:
        if min(config.num_parallel_games, config.steps_per_game_per_rollout,
               config.max_episode_steps, config.mini_batch_size, config.min_update_epochs) <= 0:
            raise ValueError("Rollout sizes, minibatch size, and update epochs must be positive")
        if config.max_update_epochs < config.min_update_epochs or config.target_kl <= 0:
            raise ValueError("Invalid update epoch bounds or target KL")
        torch.manual_seed(config.seed)
        self.config = config
        self.model = model
        self.optimizer = optimizer
        rng = np.random.default_rng(config.seed)
        self.rollout_collector = RolloutCollector(model, config, game_factory, rng)
        self.updater = PPOUpdater(model, optimizer, config, rng, self.rollout_collector)

    def run_outer_iteration(self) -> list[EpochStatistics]:
        rollouts, mean, std = self.rollout_collector.collect_batched_information()
        batch = self.updater.prepare_batch(rollouts)
        statistics = []
        for epoch in range(1, self.config.max_update_epochs + 1):
            stats = self.updater.run_epoch(batch, mean, std)
            statistics.append(stats)
            if epoch >= self.config.min_update_epochs and stats.approximate_kl > self.config.target_kl:
                break
        return statistics
