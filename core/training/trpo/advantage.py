import torch.nn as nn
import torch
from core.training.trajectory import Trajectory


class AdvantageEstimator:
    def __init__(self, gamma: float, lambda_: float, value_network: nn.Module):
        self.gamma = gamma
        self.value_network = value_network
        self.lambda_ = lambda_

    @torch.no_grad()
    def _next_values(self, trajectory: Trajectory) -> torch.Tensor:
        has_next_state = trajectory.has_next_state()
        next_values = torch.zeros(len(trajectory))
        if has_next_state.any():
            next_values[has_next_state] = self.value_network(
                trajectory.next_states()
            ).squeeze(-1)
        return next_values

    @torch.no_grad()
    def get_advantages(self, trajectory: Trajectory) -> torch.Tensor:
        rewards = trajectory.rewards()
        values = self.value_network(trajectory.states()).squeeze(-1)
        next_values = self._next_values(trajectory)
        deltas = rewards + self.gamma * next_values - values

        advantages = [0]
        for delta in deltas.flip(0):
            advantage = advantages[-1]
            next_advantage = delta + (self.gamma * self.lambda_) * advantage
            advantages.append(next_advantage)
        return torch.stack(advantages[1:]).flip(0)
