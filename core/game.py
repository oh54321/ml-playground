import abc
from dataclasses import dataclass

import numpy as np

from core.game_symmetry import GameSymmetry
from core.state import State


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    terminated: bool
    action_applied: bool


class Game(abc.ABC):
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def reset(self, seed: int = None):
        raise NotImplementedError

    @abc.abstractmethod
    def current_state(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def legal_actions(self) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def action_name(self, action: int) -> str:
        raise NotImplementedError

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def symmetries(self) -> tuple[GameSymmetry, ...]:
        """Return usable game symmetries; games opt in by overriding this."""
        return ()

    @abc.abstractmethod
    def step(self, action: int) -> StepResult:
        raise NotImplementedError
