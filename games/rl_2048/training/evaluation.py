"""Evaluate greedy play on a fixed set of seeded games."""

from dataclasses import dataclass
from statistics import mean, median

import torch

from algorithms.ppo import PPO
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding


@dataclass(frozen=True)
class EvaluationResult:
    scores: list[int]
    moves: list[int]
    max_tiles: list[int]

    def metrics(self) -> dict[str, float]:
        return {
            "mean_score": mean(self.scores),
            "median_score": median(self.scores),
            "best_score": max(self.scores),
            "worst_score": min(self.scores),
            "mean_moves": mean(self.moves),
            "win_rate": sum(tile >= 2048 for tile in self.max_tiles) / len(self.max_tiles),
        }


@torch.no_grad()
def evaluate_games(
    ppo: PPO, seeds: list[int], *, height: int, width: int,
    observation_encoding: ObservationEncoding,
) -> EvaluationResult:
    if not seeds:
        raise ValueError("Must evaluate at least one game")
    games = [Game2048(seed=seed, height=height, width=width,
                      observation_encoding=observation_encoding) for seed in seeds]
    moves = [0] * len(games)
    active = list(range(len(games)))
    with ppo.model.prepared_inference():
        while active:
            decisions = ppo.rollout_collector.generate_next_actions(
                [games[index] for index in active], deterministic=True,
            )
            remaining = []
            for index, decision in zip(active, decisions, strict=True):
                result = games[index].step(decision.action)
                assert result.action_applied, "Evaluation selected an illegal action"
                moves[index] += 1
                if not result.terminated:
                    remaining.append(index)
            active = remaining
    return EvaluationResult(
        scores=[int(game.total_score) for game in games],
        moves=moves,
        max_tiles=[int(game.state.grid.max()) for game in games],
    )
