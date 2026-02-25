from dataclasses import dataclass
from typing import List, Union
import numpy as np


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

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TrajectoryNode, List[TrajectoryNode]]:
        return self.nodes[index]
