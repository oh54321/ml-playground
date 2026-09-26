"""The 2048 board symmetries and their spatial transforms."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

from core.environments.game_symmetry import GameSymmetry
from core.environments.state import State
from core.symmetries.d4_transforms import BoardTransform


@dataclass(frozen=True)
class SymmetrySpec:
    """Declarative part of a symmetry shared by the game and D4 model."""

    transform: BoardTransform
    action_permutation: tuple[int, int, int, int]


# The D4 model consumes action permutations in this transform order.
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
