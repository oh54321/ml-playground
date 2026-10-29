"""Spatial transforms and stable element order for the square D4 group."""

from enum import Enum

import numpy as np
import torch


class BoardTransform(Enum):
    """D4 element encoded as ``(quarter_turns, reflected_left_to_right)``.

    A reflected transform applies the rotation first, then flips left to right.
    """

    IDENTITY = (0, False)
    ROTATE_CCW_90 = (1, False)
    ROTATE_180 = (2, False)
    ROTATE_CCW_270 = (3, False)
    FLIP_LEFT_RIGHT = (0, True)
    TRANSPOSE = (3, True)
    FLIP_UP_DOWN = (2, True)
    FLIP_ANTI_DIAGONAL = (1, True)

    @property
    def quarter_turns(self) -> int:
        return self.value[0]

    @property
    def reflected(self) -> bool:
        return self.value[1]


D4_TRANSFORMS = tuple(BoardTransform)
NUM_SYMMETRIES = len(D4_TRANSFORMS)
_TRANSFORM_INDICES = {
    transform: index for index, transform in enumerate(D4_TRANSFORMS)
}
_TRANSFORMS_BY_COORDINATES = {
    transform.value: transform for transform in D4_TRANSFORMS
}


def inverse_transform(transform: BoardTransform) -> BoardTransform:
    """Return the group element that undoes ``transform``."""
    if not isinstance(transform, BoardTransform):
        raise ValueError(f"Unsupported board transform: {transform}")
    quarter_turns = transform.quarter_turns
    if not transform.reflected:
        quarter_turns = -quarter_turns
    return _TRANSFORMS_BY_COORDINATES[quarter_turns % 4, transform.reflected]


def transform_index(transform: BoardTransform) -> int:
    """Return a transform's position in the canonical D4 ordering."""
    try:
        return _TRANSFORM_INDICES[transform]
    except KeyError as error:
        raise ValueError(f"Unsupported board transform: {transform}") from error


def transform_array(array: np.ndarray, transform: BoardTransform) -> np.ndarray:
    """Apply a D4 transform to the final two NumPy array dimensions."""
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
        result = np.swapaxes(np.flip(np.flip(array, axis=-2), axis=-1), -2, -1)
    else:
        raise ValueError(f"Unsupported board transform: {transform}")
    return np.array(result, dtype=array.dtype, order="C", copy=True)


def compose_transforms(
    outer: BoardTransform,
    inner: BoardTransform,
) -> BoardTransform:
    """Return the transform that applies ``inner`` first, then ``outer``."""
    if not isinstance(outer, BoardTransform) or not isinstance(inner, BoardTransform):
        raise ValueError("Both operands must be BoardTransform values")
    # With transforms represented as F^b R^k, R^k F = F R^-k.
    quarter_turns = (
        (-1 if inner.reflected else 1) * outer.quarter_turns
        + inner.quarter_turns
    ) % 4
    reflected = outer.reflected ^ inner.reflected
    return _TRANSFORMS_BY_COORDINATES[quarter_turns, reflected]


def relative_transform(
    reference: BoardTransform,
    orientation: BoardTransform,
) -> BoardTransform:
    """Return the orientation of ``orientation`` relative to ``reference``."""
    return compose_transforms(inverse_transform(reference), orientation)


def transform_tensor(tensor: torch.Tensor, transform: BoardTransform) -> torch.Tensor:
    """Apply a D4 transform to the final two tensor dimensions."""
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
