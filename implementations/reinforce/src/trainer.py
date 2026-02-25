from typing import Optional
from gymnasium.core import Env
from torch.optim import Optimizer, AdamW
import torch.nn as nn
from matplotlib.animation import FuncAnimation
import torch

from core.environments.gym.trajectory import Trajectory
from core.environments.gym.env import PolicyEnvironment

from implementations.reinforce.src.buffer import REINFORCEReplayBuffer


class REINFORCETrainer:
    def __init__(
        self,
        model: nn.Module,
        env: Env,
        input_buffer_size: int = 3,
        replay_buffer_size: int = 100,
        gamma: float = 0.99,
        optimizer: Optional[Optimizer] = None,
    ):
        self.env = PolicyEnvironment(env, input_buffer_size, model)
        self.model = model
        self.gamma = gamma
        self.replay = REINFORCEReplayBuffer(
            replay_buffer_size, input_buffer_size, gamma
        )
        if optimizer is None:
            self.optimizer = AdamW(model.parameters())
        else:
            self.optimizer = optimizer
        self._rewards = []

    def sample(self) -> Trajectory:
        return self.env.sample_trajectory()

    def display(self) -> FuncAnimation:
        return self.env.display()

    def fetch_output(self):
        inputs = self.replay.inputs()
        actions = self.replay.actions()
        output = torch.log_softmax(self.model.forward(inputs), dim=1)
        indices = torch.arange(output.size(0))
        future_rewards = self.replay.future_rewards()
        values = output[indices, actions] * future_rewards
        return values.mean()

    def add_reward(self, trajectory: Trajectory):
        reward = 0
        for node in trajectory.nodes[::-1]:
            reward = self.gamma * reward + node.reward
        self._rewards.append(reward)
        if len(self._rewards) > 10:
            print(sum(self._rewards[-10:]) / len(self._rewards[-10:]))

    def iterate(self) -> None:
        self.replay.reset()
        self.optimizer.zero_grad()
        while not self.replay.is_full():
            trajectory = self.sample()
            self.replay.add(trajectory)
            self.add_reward(trajectory)
        output = self.fetch_output()
        output.backward()
        self.optimizer.step()
