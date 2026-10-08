"""Architecture-independent actor-critic model contract used by PPO."""

import abc
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


__all__ = ["ActorCriticModel", "ActorCriticOutput"]


@dataclass(frozen=True)
class ActorCriticOutput:
    action_logits: torch.Tensor
    value: torch.Tensor


class ActorCriticModel(nn.Module, abc.ABC):
    """A batch-in/batch-out policy and value model.

    PPO treats observations as tensors and does not depend on how a model
    extracts features. Implementations expose representation parameters so
    PPO can report policy/value gradient diagnostics without inspecting layers.
    """

    def prepared_inference(self) -> AbstractContextManager[None]:
        """Optionally reuse fixed-weight computations during rollout collection."""
        return nullcontext()

    @abc.abstractmethod
    def forward(self, observations: torch.Tensor) -> ActorCriticOutput:
        raise NotImplementedError

    @abc.abstractmethod
    def representation_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters shared by, or belonging to, the policy/value encoders."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def has_shared_representation(self) -> bool:
        """Whether policy and value objectives use any shared encoder."""
        raise NotImplementedError
