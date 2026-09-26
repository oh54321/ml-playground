"""Base class for observations returned by games."""

import abc
from typing import Sequence

import torch


class State(abc.ABC):
    @abc.abstractmethod
    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self) -> "State":
        raise NotImplementedError

    @classmethod
    def batch_to_tensor(cls, states: Sequence["State"]) -> torch.Tensor:
        """Encode a homogeneous state batch, with a safe generic fallback."""
        if not states:
            raise ValueError("Cannot encode an empty state batch")
        return torch.stack([state.to_tensor() for state in states], dim=0)
