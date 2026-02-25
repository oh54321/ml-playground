from collections import deque
import numpy as np
import torch

from core.environments.gym.trajectory import Trajectory


class FrameBuffer:
    def __init__(self, n: int):
        self.n = n
        self.buffer = deque(maxlen=n)
        self.frame_shape = None

    def reset(self):
        self.buffer = deque(maxlen=self.n)
        self.frame_shape = None

    def add(self, observation: np.ndarray):
        if len(observation.shape) != 3:
            raise ValueError(f"Observation must be 3D, got shape {observation.shape}")

        if self.frame_shape is None:
            self.frame_shape = observation.shape

        if observation.shape != self.frame_shape:
            raise ValueError(
                f"Observation shape {observation.shape} doesn't match frame shape {self.frame_shape}"
            )

        self.buffer.append(observation.copy())

    def size(self) -> int:
        return len(self.buffer)

    def empty(self) -> bool:
        return self.size() == 0

    def to_tensor(self) -> torch.Tensor:
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        frames = []
        dtype = self.buffer[0].dtype
        for i in range(self.n):
            if i < len(self.buffer):
                frames.append(self.buffer[i])
            else:
                frames.append(np.zeros(self.frame_shape, dtype=dtype))

        concatenated = np.concatenate(frames, axis=2)
        tensor = torch.from_numpy(concatenated).float()
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def is_full(self) -> bool:
        return self.size() >= self.n


class _ObservationReplayBuffer(FrameBuffer):
    def __init__(self, n: int, n_input: int):
        super().__init__(n)
        self.n_input = n_input

    def to_tensor(self):
        input_buffer = FrameBuffer(self.n_input)
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        tensors = []
        for frame in self.buffer:
            input_buffer.add(frame)
            tensors.append(input_buffer.to_tensor())
        return torch.stack(tensors)


class ReplayBuffer:
    def __init__(
        self,
        n: int,
        n_input: int,
    ):
        self.n = n
        self.replay_buffer = _ObservationReplayBuffer(n, n_input)
        self._actions = deque(maxlen=n)
        self._rewards = deque(maxlen=n)

    def rewards(self) -> torch.Tensor:
        return torch.tensor(self._rewards)

    def actions(self) -> torch.Tensor:
        return torch.tensor(self._actions)

    def inputs(self) -> torch.Tensor:
        return self.replay_buffer.to_tensor()

    def add(self, trajectory: Trajectory):
        for node in trajectory:
            self._rewards.append(node.reward)
            self._actions.append(node.action)
            self.replay_buffer.add(node.observation)

    def size(self) -> int:
        return self.replay_buffer.size()

    def empty(self) -> bool:
        return self.replay_buffer.empty()

    def is_full(self) -> bool:
        return self.replay_buffer.is_full()

    def reset(self):
        self.replay_buffer.reset()
        self._actions = deque(maxlen=self.n)
        self._rewards = deque(maxlen=self.n)
