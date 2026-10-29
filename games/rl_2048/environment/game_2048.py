"""2048 on rectangular boards, with immutable board snapshots and log2 observations."""

from enum import Enum
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from core.environments.game import Game, StepResult
from core.environments.state import State
from games.rl_2048.environment._game_2048_kernels import (
    is_terminal, legal_neighbor_flags, step_int64,
)


class ObservationEncoding(str, Enum):
    NUMERIC = "numeric"
    NUMERIC_ONEHOT = "numeric_onehot"

    def in_channels(self, height: int, width: int) -> int:
        return 1 if self is ObservationEncoding.NUMERIC else height * width + 3


class State2048(State):
    def __init__(self, grid: np.ndarray | None = None,
                 observation_encoding: ObservationEncoding | str = ObservationEncoding.NUMERIC,
                 *, height: int = 4, width: int = 4):
        self._observation_encoding = ObservationEncoding(observation_encoding)
        grid = np.zeros((height, width), dtype=np.int64) if grid is None else np.array(grid, copy=True, order="C")
        if grid.ndim != 2 or 0 in grid.shape:
            raise ValueError("Expected a nonempty 2D board")
        if not np.issubdtype(grid.dtype, np.integer) and not np.all(grid == np.floor(grid)):
            raise ValueError("Tiles must be integers")
        grid = grid.astype(np.int64, copy=False)
        valid = (grid == 0) | ((grid > 0) & ((grid & (grid - 1)) == 0))
        if not np.all(valid):
            raise ValueError("Tiles must be zero or positive powers of two")
        self._grid = grid
        self._grid.flags.writeable = False

    @classmethod
    def _from_validated_grid(cls, grid: np.ndarray,
                             observation_encoding: ObservationEncoding) -> "State2048":
        """Adopt an owned board produced by the game without copying it again."""
        state = cls.__new__(cls)
        state._grid = grid
        state._observation_encoding = observation_encoding
        grid.flags.writeable = False
        return state

    @property
    def observation_encoding(self) -> ObservationEncoding:
        return self._observation_encoding

    @property
    def height(self) -> int:
        return self._grid.shape[0]

    @property
    def width(self) -> int:
        return self._grid.shape[1]

    @property
    def grid(self) -> np.ndarray:
        return self._grid.view()

    def to_tensor(self) -> torch.Tensor:
        return self.batch_to_tensor([self])[0]

    @classmethod
    def batch_to_tensor(cls, states: Sequence["State2048"]) -> torch.Tensor:
        if not states:
            raise ValueError("Cannot encode an empty state batch")
        encoding = states[0].observation_encoding
        if any(state.observation_encoding != encoding for state in states):
            raise ValueError("Cannot mix observation encodings in one batch")
        tiles = torch.from_numpy(np.stack([state.grid for state in states])).float()
        # Empty cells map to 0; tiles 2, 4, 8, ... map to 1, 2, 3, ...
        tile_exponents = torch.where(tiles > 0, torch.log2(tiles), tiles)
        numeric_channel = tile_exponents.unsqueeze(1)
        if encoding is ObservationEncoding.NUMERIC:
            return numeric_channel

        # Exclude the numeric channel to count only the one-hot categories.
        num_classes = encoding.in_channels(states[0].height, states[0].width) - 1
        if torch.any((tile_exponents < 0) | (tile_exponents >= num_classes)):
            raise ValueError(f"One-hot observations support tile exponents from 0 through {num_classes - 1}")
        one_hot_channels = F.one_hot(
            tile_exponents.long(), num_classes=num_classes,
        ).float()
        # (batch, height, width, categories) -> (batch, categories, height, width)
        one_hot_channels = one_hot_channels.permute(0, 3, 1, 2)
        return torch.cat((numeric_channel, one_hot_channels), dim=1)

    def copy(self) -> "State2048":
        return self._from_validated_grid(self._grid.copy(order="C"), self.observation_encoding)

    def __repr__(self) -> str:
        return f"State2048(\n{self.grid}\n)"


class Game2048(Game):
    """Actions are ordered left, up, right, down."""

    def __init__(self, seed: int | None = None,
                 observation_encoding: ObservationEncoding | str = ObservationEncoding.NUMERIC,
                 *, height: int = 4, width: int = 4):
        super().__init__(seed)
        self._state = State2048(height=height, width=width, observation_encoding=observation_encoding)
        self.reset()

    @property
    def state(self) -> State2048:
        return self._state

    def current_state(self) -> State2048:
        return self.state

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        grid = np.zeros_like(self.state.grid)
        for _ in range(min(2, grid.size)):
            empty = np.argwhere(grid == 0)
            row, col = empty[self.rng.integers(len(empty))]
            grid[row, col] = 2 if self.rng.random() < 0.9 else 4
        self._state = State2048._from_validated_grid(grid, self.state.observation_encoding)
        self.total_score = 0
        self.done = bool(is_terminal(grid))

    def legal_actions(self) -> list[int]:
        return [action for action, legal in enumerate(legal_neighbor_flags(self.state.grid)) if legal]

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after game over")
        if not 0 <= action < 4:
            raise ValueError("Action must be between 0 and 3")
        grid = np.zeros_like(self.state.grid)
        reward, moved, terminated = step_int64(self.state.grid, action, self.rng, grid)
        self._state = State2048._from_validated_grid(grid, self.state.observation_encoding)
        self.total_score += int(reward)
        self.done = bool(terminated)
        return StepResult(self.state, reward, self.done, bool(moved))

    def replace_board(self, board: np.ndarray | State2048, *, total_score: int = 0) -> None:
        """Copy an explicit board without spawning tiles or consuming randomness."""
        grid = board.grid if isinstance(board, State2048) else board
        self._state = State2048(grid, self.state.observation_encoding)
        self.total_score = total_score
        self.done = bool(is_terminal(self.state.grid))
