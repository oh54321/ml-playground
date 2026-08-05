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

    def rewards(self) -> torch.Tensor:
        return torch.tensor([step.reward for step in self.steps])

    def states(self) -> torch.Tensor:
        return torch.stack([step.state for step in self.steps])

    def dones(self) -> torch.Tensor:
        return torch.tensor([step.done for step in self.steps])

    # next_state is None exactly when the step ended the episode, so these two
    # let a consumer bootstrap with zero on terminal steps and V(s') elsewhere.
    def has_next_state(self) -> torch.Tensor:
        return torch.tensor([step.next_state is not None for step in self.steps])

    def next_states(self) -> torch.Tensor:
        return torch.stack(
            [step.next_state for step in self.steps if step.next_state is not None]
        )

    def actions(self) -> List[int]:
        return [step.action for step in self.steps]
