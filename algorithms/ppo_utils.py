"""Pure mathematical helpers used by PPO."""

from typing import Literal

import torch
import torch.nn.functional as F


ValueTransform = Literal["identity", "signed_log1p", "signed_sqrt1p"]


def action_log_probabilities(
    masked_logits: torch.Tensor,
    illegal_actions_mask: torch.Tensor,
    exploration_probability: float,
) -> torch.Tensor:
    """Log of the policy mixed with a uniform distribution over legal actions."""
    policy_log_probs = F.log_softmax(masked_logits, dim=-1)
    if exploration_probability == 0.0:
        return policy_log_probs
    legal = ~illegal_actions_mask
    uniform = legal.to(masked_logits.dtype) / legal.sum(dim=-1, keepdim=True)
    probabilities = (
        (1.0 - exploration_probability) * policy_log_probs.exp()
        + exploration_probability * uniform
    )
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log().masked_fill(
        illegal_actions_mask, -torch.inf
    )


def transform_value_target(value: torch.Tensor, value_transform: ValueTransform) -> torch.Tensor:
    """Map raw values to the space predicted by the value head."""
    if value_transform == "identity":
        return value
    magnitude = torch.abs(value)
    if value_transform == "signed_log1p":
        transformed_magnitude = torch.log1p(magnitude)
    else:
        # Smooth square-root compression with finite slope at zero.
        transformed_magnitude = torch.sqrt(1 + magnitude) - 1
    return torch.sign(value) * transformed_magnitude


def untransform_predicted_value(
    transformed_value: torch.Tensor,
    value_transform: ValueTransform,
) -> torch.Tensor:
    """Map a value-head output back to raw reward units for GAE."""
    if value_transform == "identity":
        return transformed_value
    magnitude = torch.abs(transformed_value)
    if value_transform == "signed_log1p":
        untransformed_magnitude = torch.expm1(magnitude)
    else:
        untransformed_magnitude = (1 + magnitude).square() - 1
    return torch.sign(transformed_value) * untransformed_magnitude


def forward_policy_kl(
    old_log_probabilities: torch.Tensor,
    new_log_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(old || new), ignoring actions masked in both policies."""
    old_probabilities = old_log_probabilities.exp()
    old_log_probabilities = old_log_probabilities.masked_fill(
        ~torch.isfinite(old_log_probabilities),
        0.0,
    )
    new_log_probabilities = new_log_probabilities.masked_fill(
        ~torch.isfinite(new_log_probabilities),
        0.0,
    )
    return (
        old_probabilities
        * (old_log_probabilities - new_log_probabilities)
    ).sum(dim=-1).mean()
