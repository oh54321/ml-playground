from collections import deque
import numpy as np
import torch


class FrameBuffer:
    def __init__(self, n: int):
        self.n = n
        self.buffer = deque(maxlen=n)
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

        self.buffer.appendleft(observation.copy())

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
