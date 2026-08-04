from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from core.game import Game, State

LEFT, RIGHT = 0, 1


class CartPoleState(State):
    def __init__(self, observation: np.ndarray):
        self.observation = np.asarray(observation, dtype=np.float32)

    @property
    def x(self) -> float:
        return float(self.observation[0])

    @property
    def x_dot(self) -> float:
        return float(self.observation[1])

    @property
    def theta(self) -> float:
        return float(self.observation[2])

    @property
    def theta_dot(self) -> float:
        return float(self.observation[3])

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.observation.copy())

    def copy(self) -> "CartPoleState":
        return CartPoleState(self.observation.copy())

    def __repr__(self) -> str:
        return (
            f"CartPoleState(x={self.x:+.4f}, x_dot={self.x_dot:+.4f}, "
            f"theta={self.theta:+.4f}, theta_dot={self.theta_dot:+.4f})"
        )


class CartPoleGame(Game):
    """
    Wraps gymnasium's CartPole-v1. `truncated` is kept as a separate attribute
    rather than folded into the Game.step contract, because a value function
    must bootstrap at a truncated step but not at a terminated one.
    """

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        observation, _ = self.env.reset(seed=seed)
        self.state = CartPoleState(observation)
        self.total_score = 0.0
        self.terminated = False
        self.truncated = False

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    # Returns the next state, the reward for the step, whether the episode is
    # over, and whether the action was legal (always true for CartPole)
    def step(self, action: int) -> Tuple[State, float, bool, bool]:
        if self.done:
            raise RuntimeError("step() called after game over")
        assert action in (LEFT, RIGHT)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.state = CartPoleState(observation)
        self.total_score += float(reward)
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        return self.state, float(reward), self.done, True

    def legal_actions(self) -> List[int]:
        return [LEFT, RIGHT]

    def render(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("render() requires render_mode to be set")
        return frame

    def close(self) -> None:
        self.env.close()
