from typing import Sequence, Tuple

import numpy as np
import torch

from core.environments.game import Game, StepResult
from core.environments.game_symmetry import GameSymmetry
from core.environments.state import State
from core.symmetries.d4_transforms import BoardTransform, transform_array
from games.rl_2048.environment.symmetry_2048 import (
    game_symmetries_for,
)


class State2048(State):
    def __init__(self, height: int = 4, width: int = 4, grid: np.ndarray = None):
        if grid is not None:
            self.grid = self._validate(grid)
            self.height, self.width = np.shape(grid)
        else:
            self.grid = np.zeros((height, width), dtype=np.int64)
            self.height, self.width = height, width

    def _validate(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(np.equal(np.mod(arr, 1), 0)):
                raise ValueError("grid contains non-integer values")
            arr = arr.astype(np.int64)
        mask = (arr == 0) | ((arr > 0) & ((arr & (arr - 1)) == 0))
        assert np.all(mask), f"non-power-of-2 values at {np.argwhere(~mask)}"
        return arr

    # 0 maps to 0, nonzero tiles map to their log2
    def to_tensor(self) -> torch.Tensor:
        grid = torch.tensor(self.grid, dtype=torch.float32)
        return torch.where(grid > 0, torch.log2(grid), grid)[None, :, :]

    @classmethod
    def batch_to_tensor(cls, states: Sequence[State]) -> torch.Tensor:
        """Vectorize array conversion and log encoding across the whole batch."""
        if not states:
            raise ValueError("Cannot encode an empty state batch")
        if not all(isinstance(state, cls) for state in states):
            raise TypeError("State2048 batches must contain only State2048 instances")
        grids = np.stack([state.grid for state in states], axis=0)
        tensor = torch.from_numpy(grids).to(dtype=torch.float32)
        encoded = torch.where(tensor > 0, torch.log2(tensor), tensor)
        return encoded.unsqueeze(1)

    def __repr__(self) -> str:
        return f"State2048(\n{self.grid}\n)"

    def copy(self) -> "State2048":
        return State2048(grid=self.grid.copy())


def _transform_state(state: State, transform: BoardTransform) -> State2048:
    if not isinstance(state, State2048):
        raise TypeError("2048 symmetries require State2048 instances")
    return State2048(grid=transform_array(state.grid, transform))


class Game2048(Game):
    ACTION_NAMES = {
        0: "left",
        1: "up",
        2: "right",
        3: "down",
    }

    def __init__(self, height: int = 4, width: int = 4, seed: int = None):
        super().__init__(seed)
        self._create_empty_board(height, width)

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._create_empty_board(self.state.height, self.state.width)

    def _create_empty_board(self, height: int = 4, width: int = 4):
        self.state = State2048(height, width)
        self.total_score = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after game over")
        assert 0 <= action <= 3
        new_grid, reward, moved = self._simulate_move(self.state.grid, action)
        new_grid = np.ascontiguousarray(new_grid)
        self.state = State2048(grid=new_grid)
        self.total_score += reward
        if moved:
            self._add_random_tile()
        self.done = self._is_game_over()
        return StepResult(
            next_state=self.state,
            reward=reward,
            terminated=self.done,
            action_applied=moved,
        )

    def current_state(self) -> State:
        return self.state

    def legal_actions(self) -> list:
        legal = []
        for a in range(4):
            _, _, moved = self._simulate_move(self.state.grid, a)
            if moved:
                legal.append(a)
        return legal

    def action_name(self, action: int) -> str:
        return self.ACTION_NAMES[action]

    def symmetries(self) -> tuple[GameSymmetry, ...]:
        # Quarter-turns and diagonal reflections swap dimensions, so rectangular
        # boards expose only the four transformations that preserve their shape.
        return game_symmetries_for(
            _transform_state,
            square_board=self.state.height == self.state.width,
        )

    def _simulate_move(
        self, grid: np.ndarray, action: int
    ) -> Tuple[np.ndarray, int, bool]:
        rotated = np.rot90(grid, k=action)
        new_rows = []
        reward = 0
        for row in rotated:
            merged_row, row_reward = self._slide_and_merge(row)
            new_rows.append(merged_row)
            reward += row_reward
        new_grid = np.rot90(np.array(new_rows), k=-action)
        moved = not np.array_equal(new_grid, grid)
        return new_grid, reward, moved

    def _slide_and_merge(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        nonzero = row[row != 0]
        merged = []
        reward = 0
        i = 0
        while i < len(nonzero):
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                merged_val = nonzero[i] * 2
                merged.append(merged_val)
                reward += merged_val
                i += 2  # skip both tiles that just merged
            else:
                merged.append(nonzero[i])
                i += 1
        padded = np.array(merged + [0] * (len(row) - len(merged)), dtype=row.dtype)
        return padded, reward

    def _is_game_over(self) -> bool:
        return len(self.legal_actions()) == 0

    def _add_random_tile(self) -> bool:
        empty_cells = np.argwhere(self.state.grid == 0)
        if len(empty_cells) == 0:
            return False
        row, col = empty_cells[self.rng.integers(len(empty_cells))]
        self.state.grid[row, col] = 2 if self.rng.random() < 0.9 else 4
        return True
