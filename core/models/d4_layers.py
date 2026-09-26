"""D4 transforms and equivariant layers for square-grid models.

Channels are ordered as (feature family, orientation).  All eight board
symmetries share weights, and transforming a board permutes the orientation
channels as well as the spatial positions.
"""

from math import sqrt
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from core.symmetries.d4_transforms import (
    BoardTransform,
    D4_TRANSFORMS,
    NUM_SYMMETRIES,
    relative_transform,
    transform_index,
    transform_tensor,
)

__all__ = [
    "RELATIVE_ORIENTATIONS",
    "transform_features",
    "D4LiftConv",
    "D4GroupConv",
]

RELATIVE_ORIENTATIONS = torch.tensor(
    [
        [transform_index(relative_transform(reference, orientation))
         for orientation in D4_TRANSFORMS]
        for reference in D4_TRANSFORMS
    ],
    dtype=torch.long,
)


def transform_features(
    x: torch.Tensor,
    transform: BoardTransform,
) -> torch.Tensor:
    """Act on both spatial cells and D4 orientation channels."""
    batch, channels, height, width = x.shape
    if channels % NUM_SYMMETRIES:
        raise ValueError("D4 feature channels must be divisible by 8")
    orientation_index = transform_index(transform)
    families = channels // NUM_SYMMETRIES
    orientation_indices = RELATIVE_ORIENTATIONS.to(x.device)[orientation_index]
    oriented = x.reshape(batch, families, NUM_SYMMETRIES, height, width)
    transformed = transform_tensor(
        oriented.index_select(2, orientation_indices), transform
    )
    return transformed.reshape(batch, channels, height, width)


class D4GroupNorm(nn.Module):
    """Normalize each feature family, tying affine parameters across poses."""

    def __init__(self, families: int) -> None:
        super().__init__()
        self.families = families
        self.norm = nn.GroupNorm(families, families * NUM_SYMMETRIES, affine=False)
        self.weight = nn.Parameter(torch.ones(families))
        self.bias = nn.Parameter(torch.zeros(families))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        normalized = self.norm(x).reshape(
            batch, self.families, NUM_SYMMETRIES, height, width
        )
        normalized = normalized * self.weight[None, :, None, None, None]
        normalized = normalized + self.bias[None, :, None, None, None]
        return normalized.reshape(batch, self.families * NUM_SYMMETRIES, height, width)


class D4LiftConv(nn.Module):
    """Lift a scalar board image to oriented feature maps with tied filters."""

    def __init__(self, in_channels: int, families: int, kernel_size: int = 3,
                 bias: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(families, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(families)) if bias else None
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / sqrt(in_channels * kernel_size**2),
                             1 / sqrt(in_channels * kernel_size**2))
        marker = torch.arange(kernel_size**2).reshape(kernel_size, kernel_size)
        spatial_indices = torch.stack([
            transform_tensor(marker, symmetry) for symmetry in D4_TRANSFORMS
        ])
        family_offsets = torch.arange(families)[:, None, None, None, None]
        channel_offsets = torch.arange(in_channels)[None, None, :, None, None]
        indices = (
            (family_offsets * in_channels + channel_offsets) * kernel_size**2
            + spatial_indices[None, :, None]
        )
        self.register_buffer("expanded_indices", indices.flatten(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.weight.flatten().index_select(
            0, self.expanded_indices
        ).reshape(-1, self.weight.shape[1], self.kernel_size, self.kernel_size)
        bias = self.bias.repeat_interleave(NUM_SYMMETRIES) if self.bias is not None else None
        return F.conv2d(x, weights, bias, padding=self.kernel_size // 2)


class D4GroupConv(nn.Module):
    """Convolve spatial positions and relative orientations with tied weights."""

    def __init__(self, in_families: int, out_families: int,
                 kernel_size: int = 3, bias: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_families = in_families
        self.out_families = out_families
        self.weight = nn.Parameter(torch.empty(
            out_families, in_families, NUM_SYMMETRIES, kernel_size, kernel_size
        ))
        self.bias = nn.Parameter(torch.empty(out_families)) if bias else None
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            fan_in = in_families * NUM_SYMMETRIES * kernel_size**2
            nn.init.uniform_(self.bias, -1 / sqrt(fan_in), 1 / sqrt(fan_in))
        marker = torch.arange(NUM_SYMMETRIES * kernel_size**2).reshape(
            NUM_SYMMETRIES, kernel_size, kernel_size
        )
        relative_spatial_indices = torch.stack([
            transform_tensor(
                marker.index_select(0, RELATIVE_ORIENTATIONS[index]), symmetry
            )
            for index, symmetry in enumerate(D4_TRANSFORMS)
        ])
        out_offsets = torch.arange(out_families)[:, None, None, None, None, None]
        in_offsets = torch.arange(in_families)[None, None, :, None, None, None]
        indices = (
            (out_offsets * in_families + in_offsets)
            * NUM_SYMMETRIES * kernel_size**2
            + relative_spatial_indices[None, :, None]
        )
        self.register_buffer("expanded_indices", indices.flatten(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.weight.flatten().index_select(
            0, self.expanded_indices
        ).reshape(
            self.out_families * NUM_SYMMETRIES,
            self.in_families * NUM_SYMMETRIES,
            self.kernel_size,
            self.kernel_size,
        )
        bias = self.bias.repeat_interleave(NUM_SYMMETRIES) if self.bias is not None else None
        return F.conv2d(x, weights, bias, padding=self.kernel_size // 2)


class D4ResBlock(nn.Module):
    def __init__(self, families: int, kernel_size: int = 3,
                 layers: int = 2, include_groupnorm: bool = True) -> None:
        super().__init__()
        modules = []
        for index in range(layers):
            modules.append(D4GroupConv(
                families, families, kernel_size, bias=not include_groupnorm
            ))
            if include_groupnorm:
                modules.append(D4GroupNorm(families))
            if index < layers - 1:
                modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.network(x))


class _D4ScalarProbe(nn.Module):
    """Evaluate all eight oriented versions of a feature map with one MLP."""

    def __init__(self, channels: int, height: int, width: int, hidden: int) -> None:
        super().__init__()
        marker = torch.arange(channels * height * width).reshape(
            1, channels, height, width
        )
        inverse_permutations = torch.stack([
            transform_features(marker, symmetry).flatten().argsort()
            for symmetry in D4_TRANSFORMS
        ])
        self.register_buffer(
            "inverse_permutations", inverse_permutations, persistent=False
        )
        self.network = nn.Sequential(
            nn.Linear(channels * height * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        first_layer = self.network[0]
        hidden = first_layer.out_features
        # Transform the weights once for the batch instead of materializing
        # eight large, transformed copies of every input feature map.
        weights = first_layer.weight[:, self.inverse_permutations]
        weights = weights.permute(1, 0, 2).reshape(
            NUM_SYMMETRIES * hidden, -1
        )
        biases = first_layer.bias.repeat(NUM_SYMMETRIES)
        activations = F.linear(x.flatten(1), weights, biases)
        activations = self.network[1](activations).reshape(
            batch * NUM_SYMMETRIES, hidden
        )
        scores = self.network[2](activations)
        return scores.reshape(batch, NUM_SYMMETRIES)


class D4ActionHead(_D4ScalarProbe):
    """Produce action-equivariant logits for a transitive D4 action set.

    Each ``BoardTransform`` in ``action_permutations`` maps canonical action
    indices. The policy averages scalar scores over transforms that carry an
    action to reference action zero.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        hidden: int,
        action_permutations: Mapping[BoardTransform, tuple[int, ...]],
    ) -> None:
        if set(action_permutations) != set(D4_TRANSFORMS):
            raise ValueError("D4 action mappings must define all eight transforms")
        action_count = len(action_permutations[D4_TRANSFORMS[0]])
        if action_count == 0 or any(
            tuple(sorted(action_permutations[symmetry])) != tuple(range(action_count))
            for symmetry in D4_TRANSFORMS
        ):
            raise ValueError("Each D4 action mapping must be a permutation")
        super().__init__(channels, height, width, hidden)
        self.action_count = action_count
        self.transforms_to_reference = tuple(
            tuple(
                index
                for index, symmetry in enumerate(D4_TRANSFORMS)
                if action_permutations[symmetry][action] == 0
            )
            for action in range(action_count)
        )
        if any(not indices for indices in self.transforms_to_reference):
            raise ValueError("Every action must belong to the reference action's D4 orbit")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = super().forward(x)
        return torch.stack(
            [scores[:, indices].mean(dim=-1) for indices in self.transforms_to_reference],
            dim=-1,
        )


class D4ValueHead(_D4ScalarProbe):
    """Produce exactly state-invariant values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).mean(dim=-1, keepdim=True)
