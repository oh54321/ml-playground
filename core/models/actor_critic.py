"""Architecture-independent actor-critic model contract used by PPO."""

import abc
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
from torch import nn


__all__ = ["ActorCriticModel", "ActorCriticOutput"]


@dataclass(frozen=True)
class ActorCriticOutput:
    action_logits: torch.Tensor
    value: torch.Tensor


class ActorCriticModel(nn.Module, abc.ABC):
    """A batch-in/batch-out policy and value model.

    PPO treats observations as tensors without depending on the architecture.
    """

    def prepared_inference(self) -> AbstractContextManager[None]:
        """Optionally reuse fixed-weight computations during rollout collection."""
        return nullcontext()

    @abc.abstractmethod
    def forward(self, observations: torch.Tensor) -> ActorCriticOutput:
        raise NotImplementedError
