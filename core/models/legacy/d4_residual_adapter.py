"""Legacy D4 residual adapter retained for older checkpoints."""

import torch
from torch import nn

from core.models.conv_actor_critic import ConvActorCriticModel
from core.symmetries.d4_transforms import (
    D4_TRANSFORMS,
    inverse_transform,
    transform_tensor,
)


class D4ResidualStem(nn.Module):
    """Average aligned features from eight views of a spatial observation."""

    def __init__(self, shared_stem: nn.Module, n_channels: int) -> None:
        super().__init__()
        self.shared_stem = shared_stem
        self.residual_projection = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        views = torch.stack(
            [transform_tensor(x, symmetry) for symmetry in D4_TRANSFORMS], dim=1
        )
        view_features = self.shared_stem(views.flatten(0, 1))
        view_features = view_features.reshape(
            batch_size,
            len(D4_TRANSFORMS),
            view_features.shape[1],
            view_features.shape[2],
            view_features.shape[3],
        )
        aligned_features = torch.stack(
            [
                transform_tensor(
                    view_features[:, index],
                    inverse_transform(symmetry),
                )
                for index, symmetry in enumerate(D4_TRANSFORMS)
            ],
            dim=1,
        )
        mixed_features = aligned_features.mean(dim=1)
        return aligned_features[:, 0] + self.residual_projection(mixed_features)


class D4ResidualAdapterActorCriticModel(ConvActorCriticModel):
    """CNN with the legacy D4 residual adapter at its input stem."""

    def create_stem(self) -> nn.Module:
        return D4ResidualStem(super().create_stem(), self.config.n_channels)
