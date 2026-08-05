from typing import Tuple
import abc
import torch


class State(abc.ABC):
    @abc.abstractmethod
    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self) -> "State":
        raise NotImplementedError


class Game(abc.ABC):
    @abc.abstractmethod
    def reset(self, seed: int = None):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[State, float, bool, bool]:
        raise NotImplementedError
