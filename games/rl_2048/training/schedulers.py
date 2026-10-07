"""Learning-rate schedules used by PPO training."""

from torch.optim.lr_scheduler import ExponentialLR


def clamp_restored_learning_rate(optimizer, scheduler, minimum: float | None) -> None:
    """Apply a newly configured floor before the next optimizer update."""
    if minimum is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"], minimum)
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


class FloorExponentialLR(ExponentialLR):
    """Exponential decay with an optional floor and resume-friendly state.

    Each step decays the optimizer's current LR. Checkpoints already store that
    LR with optimizer state, so restoring ``last_epoch`` preserves the absolute
    iteration while continuing from the restored value. ``min_learning_rate``
    may be omitted to retain the historical ExponentialLR behavior.
    """

    def __init__(self, optimizer, gamma: float, min_learning_rate: float | None = None):
        self.min_learning_rate = min_learning_rate
        super().__init__(optimizer, gamma=gamma)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        decayed = [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        if self.min_learning_rate is None:
            return decayed
        return [max(self.min_learning_rate, lr) for lr in decayed]
