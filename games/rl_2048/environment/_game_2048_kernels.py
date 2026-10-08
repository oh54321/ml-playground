"""Small compiled helpers for the 2048 game."""

import numpy as np
from numba import njit


@njit(cache=True)
def _line_moves(rows):
    """Can any row slide/merge toward its start or toward its end?"""
    toward_start = False
    toward_end = False
    for row in rows:
        for col in range(len(row) - 1):
            first, second = row[col], row[col + 1]
            if first != 0 and first == second:
                return True, True
            toward_start |= first == 0 and second != 0
            toward_end |= first != 0 and second == 0
            if toward_start and toward_end:
                return True, True
    return toward_start, toward_end


@njit(cache=True)
def legal_neighbor_flags(grid):
    """Return left, up, right, down legality without simulating moves."""
    left, right = _line_moves(grid)
    up, down = _line_moves(grid.T)
    return left, up, right, down


@njit(cache=True)
def is_terminal(grid):
    left, right = _line_moves(grid)
    if left or right:
        return False
    up, down = _line_moves(grid.T)
    return not (up or down)


@njit(cache=True)
def _rows_toward_action(grid, action):
    """View each line from its destination edge; transposes/reversals don't copy."""
    if action == 0:  # left
        return grid
    if action == 1:  # up
        return grid.T
    if action == 2:  # right
        return grid[:, ::-1]
    if action == 3:  # down
        return grid[::-1, :].T
    raise ValueError("action must be in [0, 3]")


@njit(cache=True)
def move_int64(grid, action, output):
    """Merge into a caller-owned, zero-filled board; return reward/moved."""
    source = _rows_toward_action(grid, action)
    destination = _rows_toward_action(output, action)
    reward = 0
    moved = False
    nonzero = np.empty(source.shape[1], dtype=np.int64)
    for row in range(source.shape[0]):
        count = 0
        for value in source[row]:
            if value != 0:
                nonzero[count] = value
                count += 1
        index, target = 0, 0
        while index < count:
            value = nonzero[index]
            if index + 1 < count and value == nonzero[index + 1]:
                value *= 2
                reward += value
                index += 2
            else:
                index += 1
            destination[row, target] = value
            target += 1
        for col in range(source.shape[1]):
            moved |= destination[row, col] != source[row, col]
    return reward, moved


@njit(cache=True)
def step_int64(grid, action, rng, output):
    reward, moved = move_int64(grid, action, output)
    # Match State2048 validation before consuming randomness or publishing state.
    if np.any(output < 0):
        raise AssertionError("non-power-of-2 values")
    if moved:
        empty = np.flatnonzero(output == 0)
        if len(empty):
            cell = empty[rng.integers(0, len(empty))]
            output.flat[cell] = 2 if rng.random() < 0.9 else 4
    return reward, moved, is_terminal(output)
