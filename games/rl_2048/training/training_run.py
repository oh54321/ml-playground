import torch.nn as nn
from torch.optim.optimizer import Optimizer

from games.rl_2048.game import Game


class ActorTrainer:
    def __init__(self, critic: nn.Module) -> None:
        pass


class CriticTrainer:
    def __init__(self, critic: nn.Module) -> None:
        pass


class TrainingRun:
    def __init__(
        self,
        game: Game,
        actor: nn.Module,
        critic: nn.Module,
        actor_trainer: nn.Module,
        critic_trainer: nn.Module,
        optimizer: Optimizer
    ) -> None:
        self.actor = actor
        self.critic = critic

    def 