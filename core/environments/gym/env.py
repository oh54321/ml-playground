from dataclasses import dataclass
from gymnasium import Env
import numpy as np
from typing import List, Union, Tuple
import torch.nn as nn
import torch
from matplotlib.animation import FuncAnimation

from core.environments.gym.display import GameDisplay
from core.environments.gym.buffer import FrameBuffer


@dataclass
class TrajectoryNode:
    observation: np.ndarray
    action: int
    reward: float


class Trajectory:
    def __init__(self):
        self.nodes: List[TrajectoryNode] = []

    def add(self, observation: np.ndarray, action: int, reward: float):
        node = TrajectoryNode(observation, action, reward)
        self.nodes.append(node)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TrajectoryNode, List[TrajectoryNode]]:
        return self.nodes[index]


class PolicyEnvironment:
    def __init__(self, env: Env, buffer_size: int, model: nn.Module):
        self.env = env
        self.buffer_size = buffer_size
        self.buffer = FrameBuffer(buffer_size)
        self.model = model
        self.displayer = GameDisplay()

    def reset(self) -> np.ndarray:
        observation, _ = self.env.reset()
        self.displayer.reset()
        self.buffer = FrameBuffer(self.buffer_size)
        self.buffer.add(observation)
        self.displayer.add(observation)
        return observation

    def sample(self) -> int:
        input_tensor = self.buffer.to_tensor()
        input_tensor = input_tensor.detach().unsqueeze(0)
        logits = self.model(input_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        return int(distribution.sample())

    def step(self) -> Tuple[int, np.ndarray, float, bool]:
        action = self.sample()
        observation, reward, done, _, _ = self.env.step(action)
        self.displayer.add(observation)
        return (action, observation, reward, done)

    def sample_trajectory(self) -> Trajectory:
        observation = self.reset()
        done = False
        trajectory = Trajectory()
        while not done:
            action, next_observation, reward, done = self.step()
            trajectory.add(observation, action, reward)
            observation = next_observation
        return trajectory

    def sample_trajectories(self, n: int) -> List[Trajectory]:
        return [self.sample_trajectory() for _ in range(n)]

    def display(self) -> FuncAnimation:
        return self.displayer.display()
