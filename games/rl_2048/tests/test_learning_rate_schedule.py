"""Learning-rate decay keeps its iteration across warm starts and honors a floor."""

import unittest

import torch
from torch.optim.lr_scheduler import ExponentialLR

from games.rl_2048.training.schedulers import (
    FloorExponentialLR,
    clamp_restored_learning_rate,
)


class LearningRateScheduleTests(unittest.TestCase):
    def test_decay_reaches_floor_and_stays_there(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = FloorExponentialLR(optimizer, gamma=0.5, min_learning_rate=0.2)

        rates = []
        for _ in range(8):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])

        self.assertEqual(rates[:3], [0.5, 0.25, 0.2])
        self.assertEqual(rates[3:], [0.2] * 5)

    def test_omitted_floor_matches_exponential_lr(self):
        parameter_a = torch.nn.Parameter(torch.tensor(1.0))
        parameter_b = torch.nn.Parameter(torch.tensor(1.0))
        optimizer_a = torch.optim.SGD([parameter_a], lr=3e-4)
        optimizer_b = torch.optim.SGD([parameter_b], lr=3e-4)
        scheduler_a = FloorExponentialLR(optimizer_a, gamma=0.999)
        scheduler_b = ExponentialLR(optimizer_b, gamma=0.999)

        for _ in range(20):
            optimizer_a.step()
            optimizer_b.step()
            scheduler_a.step()
            scheduler_b.step()

        self.assertEqual(optimizer_a.param_groups[0]["lr"], optimizer_b.param_groups[0]["lr"])

    def test_resume_continues_from_restored_lr_and_epoch(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = FloorExponentialLR(optimizer, gamma=0.8, min_learning_rate=0.3)
        uninterrupted = []
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            uninterrupted.append(optimizer.param_groups[0]["lr"])
            if len(uninterrupted) == 4:
                split_lr = optimizer.param_groups[0]["lr"]
                split_epoch = scheduler.last_epoch

        resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
        resumed_optimizer = torch.optim.SGD(
            [resumed_parameter],
            lr=split_lr,
        )
        resumed_scheduler = FloorExponentialLR(
            resumed_optimizer,
            gamma=0.8,
            min_learning_rate=0.3,
        )
        resumed_scheduler.last_epoch = split_epoch
        resumed_scheduler._last_lr = [resumed_optimizer.param_groups[0]["lr"]]

        for expected in uninterrupted[4:]:
            resumed_optimizer.step()
            resumed_scheduler.step()
            self.assertEqual(resumed_optimizer.param_groups[0]["lr"], expected)

    def test_new_floor_clamps_old_checkpoint_lr_before_next_update(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
        optimizer.state[parameter]["momentum_buffer"] = torch.tensor(4.0)
        scheduler = FloorExponentialLR(optimizer, gamma=0.9, min_learning_rate=0.3)

        clamp_restored_learning_rate(optimizer, scheduler, 0.3)

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.3)
        self.assertEqual(scheduler._last_lr, [0.3])
        self.assertEqual(optimizer.state[parameter]["momentum_buffer"].item(), 4.0)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.3)


if __name__ == "__main__":
    unittest.main()
