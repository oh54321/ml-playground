import abc
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
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


@dataclass(frozen=True)
class GameSymmetry:
    """One game-specific state transform and its action-coordinate mapping.

    ``action_permutation[a]`` is the action in the transformed view that
    corresponds to canonical action ``a``.
    """

    name: str
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

    def transform_states(
        self,
        states: Sequence[State],
        symmetry_indices: Sequence[int],
    ) -> list[State]:
        """Transform a state batch; games may override to vectorize it."""
        if len(states) != len(symmetry_indices):
            raise ValueError("States and symmetry indices must have equal lengths")
        symmetries = self.symmetries()
        return [
            symmetries[symmetry_index].transform_state(state)
            for state, symmetry_index in zip(states, symmetry_indices, strict=True)
        ]

    @abc.abstractmethod
    def step(self, action: int) -> StepResult:
        raise NotImplementedError


class SymmetryAugmentedGame(Game):
    """Expose a randomly transformed view while keeping transitions canonical.

    The selected symmetry is stable for an entire decision.  In ``step`` mode,
    it is resampled only after the underlying game consumes the current action.
    This keeps state, legal actions, sampled action, and PPO log probability in
    one coordinate frame while changing that frame between moves.
    """

    _SYMMETRY_SEED_NAMESPACE = 0x53594D  # "SYM"

    def __init__(
        self,
        game: Game,
        seed: int | None = None,
        resample: Literal["step", "episode"] = "step",
    ) -> None:
        if resample not in ("step", "episode"):
            raise ValueError("resample must be 'step' or 'episode'")
        if not game.symmetries():
            raise ValueError(f"{type(game).__name__} does not define any symmetries")
        super().__init__(None)
        self.game = game
        self.resample = resample
        self.rng = self._new_symmetry_rng(seed)
        self.symmetry_index = self._sample_symmetry_index()

    @classmethod
    def _new_symmetry_rng(cls, seed: int | None) -> np.random.Generator:
        if seed is None:
            return np.random.default_rng()
        # This stream is deterministic but independent of the wrapped game's
        # tile/spawn RNG, so augmentation does not perturb environment randomness.
        seed_sequence = np.random.SeedSequence(
            [int(seed), cls._SYMMETRY_SEED_NAMESPACE]
        )
        return np.random.default_rng(seed_sequence)

    def _sample_symmetry_index(self) -> int:
        return int(self.rng.integers(len(self.game.symmetries())))

    @property
    def current_symmetry(self) -> GameSymmetry:
        return self.game.symmetries()[self.symmetry_index]

    @property
    def total_score(self) -> float:
        return self.game.total_score

    @property
    def done(self) -> bool:
        return self.game.done

    def symmetries(self) -> tuple[GameSymmetry, ...]:
        return self.game.symmetries()

    def reset(self, seed: int = None):
        self.game.reset(seed)
        if seed is not None:
            self.rng = self._new_symmetry_rng(seed)
        self.symmetry_index = self._sample_symmetry_index()

    def current_state(self) -> State:
        return self.current_symmetry.transform_state(self.game.current_state())

    def legal_actions(self) -> list:
        symmetry = self.current_symmetry
        return sorted(
            symmetry.to_transformed_action(action)
            for action in self.game.legal_actions()
        )

    def action_name(self, action: int) -> str:
        # Action indices retain their ordinary meaning in the transformed view.
        return self.game.action_name(action)

    def set_seed(self, seed: int):
        self.game.set_seed(seed)
        self.rng = self._new_symmetry_rng(seed)
        self.symmetry_index = self._sample_symmetry_index()

    def step(self, action: int) -> StepResult:
        canonical_action = self.current_symmetry.to_canonical_action(action)
        result = self.game.step(canonical_action)
        if result.action_applied and self.resample == "step":
            self.symmetry_index = self._sample_symmetry_index()
        return StepResult(
            next_state=self.current_state(),
            reward=result.reward,
            terminated=result.terminated,
            action_applied=result.action_applied,
        )


def batch_current_states(games: Sequence[Game]) -> list[State]:
    """Read current observations, batching symmetry transforms when possible."""
    if not games:
        return []
    if all(isinstance(game, SymmetryAugmentedGame) for game in games):
        augmented_games = [game for game in games if isinstance(game, SymmetryAugmentedGame)]
        wrapped_type = type(augmented_games[0].game)
        if all(type(game.game) is wrapped_type for game in augmented_games):
            return augmented_games[0].game.transform_states(
                [game.game.current_state() for game in augmented_games],
                [game.symmetry_index for game in augmented_games],
            )
    return [game.current_state() for game in games]
