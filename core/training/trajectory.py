from typing import Iterable, Iterator, List, Optional, Union, overload
import torch
from dataclasses import dataclass


@dataclass
class TrajectoryStep:
    state: torch.Tensor
    action: int
    reward: float
    next_state: Optional[torch.Tensor]
    done: bool


class Trajectory:
    def __init__(self, steps: Optional[List[TrajectoryStep]] = None) -> None:
        self.steps: List[TrajectoryStep] = [] if steps is None else steps

    def reset(self) -> None:
        self.steps = []

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def extend(self, steps: Iterable[TrajectoryStep]) -> None:
        self.steps.extend(steps)

    def select(self, indices: Iterable[int]) -> "Trajectory":
        return Trajectory([self.steps[i] for i in indices])

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[TrajectoryStep]:
        return iter(self.steps)

    @overload
    def __getitem__(self, index: int) -> TrajectoryStep: ...

    @overload
    def __getitem__(self, index: slice) -> "Trajectory": ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TrajectoryStep, "Trajectory"]:
        if isinstance(index, slice):
            return Trajectory(self.steps[index])
        return self.steps[index]
