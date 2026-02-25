from collections import deque
import torch

from core.environments.gym.trajectory import Trajectory
from core.environments.gym.buffer import ReplayBuffer


class ReinforceReplayBuffer(ReplayBuffer):
    def __init__(self, n: int, n_input: int, gamma: float):
        super().__init__(n, n_input)
        self.gamma = gamma
        self._future_rewards = deque(maxlen=n)

    def future_rewards(self) -> torch.Tensor:
        return torch.tensor(self._future_rewards)

    def add_future_rewards(self, trajectory: Trajectory):
        future_rewards = []
        total_reward = 0
        for node in trajectory[::-1]:
            total_reward = self.gamma * total_reward + node.reward
            future_rewards.append(total_reward)
        future_rewards = future_rewards[::-1]
        for reward in future_rewards:
            self._future_rewards.append(reward)

    def add(self, trajectory: Trajectory):
        self.add_future_rewards(trajectory)
        super().add(trajectory)
