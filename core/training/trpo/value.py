from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer
from typing import List
from core.training.trajectory import Trajectory
from core.training.trainer import Trainer, TrainingIteration, TrainerConfig


class ValueTrainingIteration(TrainingIteration):
    def __init__(
        self,
        value_network: nn.Module,
        trajectories: List[Trajectory],
        gamma: float,
        epsilon: float,
        optimizer: Optimizer,
        max_steps: int,
    ) -> None:
        self.value_network = value_network
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.n_steps = 0
        self.states = torch.concat(
            [trajectory.states() for trajectory in trajectories]
        )
        self.returns = self._get_returns(trajectories)
        self.initial_values = self.values().detach()
        self.current_values = self.initial_values
        self.sigma_2 = self.loss(self.initial_values, self.returns).item()

    @torch.no_grad()
    def is_done(self) -> bool:
        upper_bound = 2 * self.epsilon * self.sigma_2
        distance = self.loss(self.current_values, self.initial_values)
        return bool(distance >= upper_bound) or self.n_steps >= self.max_steps

    def values(self) -> torch.Tensor:
        return self.value_network(self.states).squeeze(-1)

    @torch.no_grad()
    def _bootstrap_value(self, trajectory: Trajectory) -> torch.Tensor:
        next_state = trajectory[-1].next_state
        if next_state is None:
            return torch.zeros(())
        return self.value_network(next_state.unsqueeze(0)).squeeze()

    @torch.no_grad()
    def _returns(self, trajectory: Trajectory) -> torch.Tensor:
        rewards = trajectory.rewards()
        returns = [self._bootstrap_value(trajectory)]
        for reward in rewards.flip(0):
            returns.append(reward + self.gamma * returns[-1])
        return torch.stack(returns[1:]).flip(0)

    def _get_returns(self, trajectories: List[Trajectory]) -> torch.Tensor:
        return torch.concat([self._returns(trajectory) for trajectory in trajectories])

    def loss(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (u - v).pow(2).mean()

    def step(self) -> None:
        self.optimizer.zero_grad()
        values = self.values()
        loss = self.loss(values, self.returns)
        loss.backward()
        self.optimizer.step()
        self.current_values = values.detach()
        self.n_steps += 1


@dataclass(frozen=True)
class ValueTrainerConfig(TrainerConfig):
    value_network: nn.Module
    trajectories: List[Trajectory]
    optimizer: Optimizer


class ValueTrainer(Trainer[ValueTrainerConfig]):
    def __init__(self, gamma: float, epsilon: float, max_steps: int = 100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps

    def create_iteration(
        self, config: ValueTrainerConfig
    ) -> ValueTrainingIteration:
        return ValueTrainingIteration(
            gamma=self.gamma,
            epsilon=self.epsilon,
            max_steps=self.max_steps,
            optimizer=config.optimizer,
            value_network=config.value_network,
            trajectories=config.trajectories,
        )
