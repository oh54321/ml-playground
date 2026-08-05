"""The value trainer should recover the analytic value function of a tiny MDP.

Run from the repo root so `core` is importable: `python -m pytest test/`
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from core.training.trajectory import Trajectory, TrajectoryStep
from core.training.trpo.value import ValueTrainer, ValueTrainerConfig

SEED = 0
GAMMA = 0.9
TERMINATION_PROBABILITY = 0.1
N_TRAJECTORIES = 200
N_ITERATIONS = 100
LEARNING_RATE = 1e-2
TRUST_REGION_EPSILON = 0.05
MAX_STEPS = 200
RELATIVE_TOLERANCE = 0.15


class TwoStateGame:
    """Two states plus an absorbing one. Named to avoid pytest collecting it."""

    rewards = [[3, -1], [8, 9]]

    def __init__(self, seed: int = SEED) -> None:
        self.rng = np.random.default_rng(seed)

    def reward(self, state: int, action: int) -> float:
        return self.rewards[state][action]

    def transition(self, state: int, action: int) -> Tuple[int, bool]:
        if self.rng.random() < TERMINATION_PROBABILITY:
            return (2, True)
        if action == 1:
            return (1 - state, False)
        return (state, False)

    def optimal_policy(self, state: int) -> int:
        if state == 0:
            return 1
        return 0

    def encode(self, state: int) -> torch.Tensor:
        return torch.eye(len(self.rewards))[state]

    def sample_optimal(self, state: int = 0) -> Trajectory:
        trajectory = Trajectory()
        done = False
        while not done:
            action = self.optimal_policy(state)
            reward = self.reward(state, action)
            next_state, done = self.transition(state, action)
            trajectory.append(
                TrajectoryStep(
                    state=self.encode(state),
                    action=action,
                    reward=reward,
                    next_state=None if done else self.encode(next_state),
                    done=done,
                )
            )
            state = next_state
        return trajectory


def expected_values() -> np.ndarray:
    """Closed form for the optimal policy.

    State 1 stays put for +8 a step, surviving with probability 1 - p:
        V(1) = 8 + gamma * (1 - p) * V(1)
    State 0 pays -1 to move to state 1:
        V(0) = -1 + gamma * (1 - p) * V(1)
    """
    survival = GAMMA * (1 - TERMINATION_PROBABILITY)
    value_one = 8 / (1 - survival)
    value_zero = -1 + survival * value_one
    return np.array([value_zero, value_one])


def current_values(value_network: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return value_network(torch.eye(2)).squeeze(-1).numpy()


def test_value_trainer_recovers_analytic_values() -> None:
    torch.manual_seed(SEED)
    game = TwoStateGame()
    trajectories = [game.sample_optimal() for _ in range(N_TRAJECTORIES)]

    value_network = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = Adam(value_network.parameters(), lr=LEARNING_RATE)
    trainer = ValueTrainer(
        gamma=GAMMA, epsilon=TRUST_REGION_EPSILON, max_steps=MAX_STEPS
    )
    config = ValueTrainerConfig(
        optimizer=optimizer,
        trajectories=trajectories,
        value_network=value_network,
    )

    # Each optimize() is a single trust-region step, so convergence needs a loop.
    for _ in range(N_ITERATIONS):
        trainer.optimize(config)

    actual = current_values(value_network)
    expected = expected_values()
    assert np.allclose(actual, expected, rtol=RELATIVE_TOLERANCE), (
        f"expected {expected}, got {actual}"
    )
