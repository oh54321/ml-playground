import abc
from typing import Tuple

import numpy as np
import torch


class State(abc.ABC):
    @abc.abstractmethod
    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self) -> "State":
        raise NotImplementedError


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

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[State, float, bool, bool]:
        raise NotImplementedError
