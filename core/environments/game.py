"""Environment contract for discrete-action reinforcement learning."""

import abc
from dataclasses import dataclass

import numpy as np

from core.environments.state import State


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    terminated: bool
    action_applied: bool


class Game(abc.ABC):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def current_state(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def legal_actions(self) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: int) -> StepResult:
        raise NotImplementedError
