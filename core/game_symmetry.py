"""State and action mappings for one symmetry of a game."""

from dataclasses import dataclass
from typing import Callable

from core.state import State


@dataclass(frozen=True)
class GameSymmetry:
    """Transform a state and map actions between canonical and transformed views."""

    state_transform: Callable[[State], State]
    action_permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        action_count = len(self.action_permutation)
        if action_count == 0:
            raise ValueError("A symmetry must define at least one action")
        if tuple(sorted(self.action_permutation)) != tuple(range(action_count)):
            raise ValueError("A symmetry action mapping must be a permutation")

    def transform_state(self, state: State) -> State:
        return self.state_transform(state)

    def to_transformed_action(self, canonical_action: int) -> int:
        if not 0 <= canonical_action < len(self.action_permutation):
            raise ValueError(f"Action {canonical_action} is outside the action space")
        return self.action_permutation[canonical_action]

    def to_canonical_action(self, transformed_action: int) -> int:
        if not 0 <= transformed_action < len(self.action_permutation):
            raise ValueError(f"Action {transformed_action} is outside the action space")
        return self.action_permutation.index(transformed_action)
