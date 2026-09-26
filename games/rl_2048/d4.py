"""D4-equivariant layers for square 2048 boards.

Channels are ordered as (feature family, orientation).  All eight board
symmetries share weights, and transforming a board permutes the orientation
channels as well as the spatial positions.
"""

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS, transform_tensor


NUM_SYMMETRIES = len(SQUARE_SYMMETRY_SPECS)


def transform(x: torch.Tensor, index: int) -> torch.Tensor:
    """Apply the spatial transform at the matching D4 table index."""
    if not 0 <= index < len(SQUARE_SYMMETRY_SPECS):
        raise ValueError(f"Unknown D4 symmetry index: {index}")
    return transform_tensor(x, SQUARE_SYMMETRY_SPECS[index].transform)


def inverse_transform(x: torch.Tensor, index: int) -> torch.Tensor:
    if index == 1:
        return transform(x, 3)
    if index == 3:
        return transform(x, 1)
    return transform(x, index)


def _group_tables() -> tuple[torch.Tensor, torch.Tensor]:
    marker = torch.arange(9).reshape(3, 3)
    images = [transform(marker, index) for index in range(NUM_SYMMETRIES)]
    composition = torch.empty(NUM_SYMMETRIES, NUM_SYMMETRIES, dtype=torch.long)
    for left in range(NUM_SYMMETRIES):
        for right in range(NUM_SYMMETRIES):
            image = transform(images[right], left)
            composition[left, right] = next(
                index for index, candidate in enumerate(images)
                if torch.equal(image, candidate)
            )
    inverses = torch.tensor([
        next(index for index in range(NUM_SYMMETRIES)
             if composition[g, index] == 0 and composition[index, g] == 0)
        for g in range(NUM_SYMMETRIES)
    ])
    return composition, inverses


COMPOSITION, INVERSES = _group_tables()
RELATIVE_ORIENTATIONS = torch.stack(
    [COMPOSITION[INVERSES[g]] for g in range(NUM_SYMMETRIES)]
)


def transform_features(
    x: torch.Tensor,
    index: int,
    relative_orientations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Act on both spatial cells and D4 orientation channels."""
    batch, channels, height, width = x.shape
    if channels % NUM_SYMMETRIES:
        raise ValueError("D4 feature channels must be divisible by 8")
    families = channels // NUM_SYMMETRIES
    if relative_orientations is None:
        relative_orientations = RELATIVE_ORIENTATIONS.to(x.device)
    orientation_indices = relative_orientations[index]
    oriented = x.reshape(batch, families, NUM_SYMMETRIES, height, width)
    transformed = transform(oriented.index_select(2, orientation_indices), index)
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
            transform(marker, index) for index in range(NUM_SYMMETRIES)
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
            transform(marker.index_select(0, RELATIVE_ORIENTATIONS[g]), g)
            for g in range(NUM_SYMMETRIES)
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
            transform_features(marker, index).flatten().argsort()
            for index in range(NUM_SYMMETRIES)
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
    """Produce exactly action-equivariant four-way logits."""

    def __init__(self, channels: int, height: int, width: int, hidden: int) -> None:
        super().__init__(channels, height, width, hidden)
        # Each action maps to LEFT in two orientations; average those scores.
        self.register_buffer("to_left", torch.tensor([
            [
                index
                for index, symmetry in enumerate(SQUARE_SYMMETRY_SPECS)
                if symmetry.action_permutation[action] == 0
            ]
            for action in range(4)
        ], dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = super().forward(x)
        return scores[:, self.to_left].mean(dim=-1)


class D4ValueHead(_D4ScalarProbe):
    """Produce exactly state-invariant values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).mean(dim=-1, keepdim=True)
