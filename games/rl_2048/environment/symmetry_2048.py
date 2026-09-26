"""The 2048 board symmetries and their spatial transforms."""

from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable

import numpy as np
import torch

from core.environments.game_symmetry import GameSymmetry
from core.environments.state import State


class BoardTransform(Enum):
    IDENTITY = auto()
    ROTATE_CCW_90 = auto()
    ROTATE_180 = auto()
    ROTATE_CCW_270 = auto()
    FLIP_LEFT_RIGHT = auto()
    FLIP_UP_DOWN = auto()
    TRANSPOSE = auto()
    FLIP_ANTI_DIAGONAL = auto()


@dataclass(frozen=True)
class SymmetrySpec:
    """Declarative part of a symmetry shared by the game and D4 model."""

    transform: BoardTransform
    action_permutation: tuple[int, int, int, int]


# Keep this table in D4 order: the D4 feature layers use the same ordering.
SQUARE_SYMMETRY_SPECS = (
    SymmetrySpec(BoardTransform.IDENTITY, (0, 1, 2, 3)),
    SymmetrySpec(BoardTransform.ROTATE_CCW_90, (3, 0, 1, 2)),
    SymmetrySpec(BoardTransform.ROTATE_180, (2, 3, 0, 1)),
    SymmetrySpec(BoardTransform.ROTATE_CCW_270, (1, 2, 3, 0)),
    SymmetrySpec(BoardTransform.FLIP_LEFT_RIGHT, (2, 1, 0, 3)),
    SymmetrySpec(BoardTransform.FLIP_UP_DOWN, (0, 3, 2, 1)),
    SymmetrySpec(BoardTransform.TRANSPOSE, (1, 0, 3, 2)),
    SymmetrySpec(BoardTransform.FLIP_ANTI_DIAGONAL, (3, 2, 1, 0)),
)

_RECTANGULAR_TRANSFORMS = frozenset(
    {
        BoardTransform.IDENTITY,
        BoardTransform.ROTATE_180,
        BoardTransform.FLIP_LEFT_RIGHT,
        BoardTransform.FLIP_UP_DOWN,
    }
)
RECTANGULAR_SYMMETRY_SPECS = tuple(
    spec
    for spec in SQUARE_SYMMETRY_SPECS
    if spec.transform in _RECTANGULAR_TRANSFORMS
)


def game_symmetries_for(
    state_transform: Callable[..., State],
    square_board: bool,
) -> tuple[GameSymmetry, ...]:
    """Build the game-facing symmetries for a board shape."""
    specs = SQUARE_SYMMETRY_SPECS if square_board else RECTANGULAR_SYMMETRY_SPECS
    return tuple(
        GameSymmetry(
            state_transform=partial(state_transform, transform=spec.transform),
            action_permutation=spec.action_permutation,
        )
        for spec in specs
    )


def transform_array(array: np.ndarray, transform: BoardTransform) -> np.ndarray:
    """Apply a board transform to the final two NumPy array dimensions."""
    if transform is BoardTransform.IDENTITY:
        result = array
    elif transform is BoardTransform.ROTATE_CCW_90:
        result = np.rot90(array, 1, axes=(-2, -1))
    elif transform is BoardTransform.ROTATE_180:
        result = np.rot90(array, 2, axes=(-2, -1))
    elif transform is BoardTransform.ROTATE_CCW_270:
        result = np.rot90(array, 3, axes=(-2, -1))
    elif transform is BoardTransform.FLIP_LEFT_RIGHT:
        result = np.flip(array, axis=-1)
    elif transform is BoardTransform.FLIP_UP_DOWN:
        result = np.flip(array, axis=-2)
    elif transform is BoardTransform.TRANSPOSE:
        result = np.swapaxes(array, -2, -1)
    elif transform is BoardTransform.FLIP_ANTI_DIAGONAL:
        result = np.swapaxes(
            np.flip(np.flip(array, axis=-2), axis=-1), -2, -1
        )
    else:
        raise ValueError(f"Unsupported board transform: {transform}")
    return np.array(result, dtype=array.dtype, order="C", copy=True)


def transform_tensor(tensor: torch.Tensor, transform: BoardTransform) -> torch.Tensor:
    """Apply a board transform to the final two tensor dimensions."""
    if transform is BoardTransform.IDENTITY:
        return tensor
    if transform is BoardTransform.ROTATE_CCW_90:
        return torch.rot90(tensor, 1, dims=(-2, -1))
    if transform is BoardTransform.ROTATE_180:
        return torch.rot90(tensor, 2, dims=(-2, -1))
    if transform is BoardTransform.ROTATE_CCW_270:
        return torch.rot90(tensor, 3, dims=(-2, -1))
    if transform is BoardTransform.FLIP_LEFT_RIGHT:
        return torch.flip(tensor, dims=(-1,))
    if transform is BoardTransform.FLIP_UP_DOWN:
        return torch.flip(tensor, dims=(-2,))
    if transform is BoardTransform.TRANSPOSE:
        return tensor.transpose(-2, -1)
    if transform is BoardTransform.FLIP_ANTI_DIAGONAL:
        return torch.flip(tensor, dims=(-2, -1)).transpose(-2, -1)
    raise ValueError(f"Unsupported board transform: {transform}")
