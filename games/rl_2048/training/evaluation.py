"""Run seeded 2048 games and collect their results."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from algorithms.ppo import PPO
from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.training.config import SymmetryAugmentation


__all__ = [
    "CompletedGame",
    "EvaluationResult",
    "EvaluationRun",
    "evaluate_games",
]


@dataclass(frozen=True)
class CompletedGame:
    game_number: int
    score: int
    steps: int
    seed: int
    trace: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    scores: list[int]
    moves: list[int]
    worst_game: CompletedGame
    best_game: CompletedGame


@dataclass
class EvaluationRun:
    output_dir: Path
    seed_base: int
    num_games: int
    save_traces: bool
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "mean": [],
            "median": [],
            "best": [],
            "worst": [],
            "mean_moves": [],
        }
    )
    iterations: list[int] = field(default_factory=list)
    symmetry_augmentation: SymmetryAugmentation = None
    exploration_schedule: dict | None = None
    best_mean_score: float = float("-inf")


@torch.no_grad()
def evaluate_games(
    ppo: PPO,
    seeds: list[int],
    *,
    save_traces: bool = False,
    trace_output_dir: Path | None = None,
) -> EvaluationResult:
    """Evaluate seeded games together, batching policy inference over active games."""
    if not seeds:
        raise ValueError("Must evaluate at least one game")

    game_numbers = list(range(1, len(seeds) + 1))
    games = {
        game_number: Game2048(seed=seed)
        for game_number, seed in zip(game_numbers, seeds, strict=True)
    }
    seeds_by_number = dict(zip(game_numbers, seeds, strict=True))
    active_game_numbers = game_numbers.copy()
    steps_by_game = dict.fromkeys(game_numbers, 0)
    traces_by_game = (
        {
            game_number: [f"Evaluation game seed: {seeds_by_number[game_number]}"]
            for game_number in game_numbers
        }
        if save_traces
        else {}
    )
    completed_games: dict[int, CompletedGame] = {}

    while active_game_numbers:
        active_games = [games[game_number] for game_number in active_game_numbers]
        decisions = ppo.generate_next_actions(active_games, deterministic=True)
        illegal_actions_by_game = torch.stack(
            [decision.illegal_actions_mask for decision in decisions]
        ).tolist()
        next_active_game_numbers = []

        for game_number, game, decision, illegal_actions in zip(
            active_game_numbers,
            active_games,
            decisions,
            illegal_actions_by_game,
            strict=True,
        ):
            state = game.current_state()
            result = game.step(decision.action)
            assert result.action_applied, "The policy selected an illegal action during evaluation"

            step_number = steps_by_game[game_number]
            if save_traces:
                legal_actions = [
                    action
                    for action, is_illegal in enumerate(illegal_actions)
                    if not is_illegal
                ]
                traces_by_game[game_number].extend(
                    [
                        f"Step {step_number}",
                        f"Board before move:\n{state}",
                        "Legal actions: " + ", ".join(
                            game.action_name(action) for action in legal_actions
                        ),
                        f"Selected action: {game.action_name(decision.action)}",
                        f"Reward: {result.reward}",
                    ]
                )
            steps_by_game[game_number] += 1

            if not result.terminated:
                next_active_game_numbers.append(game_number)
                continue

            score = int(game.total_score)
            trace = None
            if save_traces:
                traces_by_game[game_number].extend(
                    [
                        f"Final board:\n{game.current_state()}",
                        f"Final score: {score}",
                        f"Total steps: {steps_by_game[game_number]}",
                    ]
                )
                trace = "\n\n".join(traces_by_game.pop(game_number)) + "\n"
                if trace_output_dir is not None:
                    game_dir = trace_output_dir / f"game_{game_number:02d}"
                    game_dir.mkdir(parents=True, exist_ok=True)
                    (game_dir / "trace.txt").write_text(trace, encoding="utf-8")

            completed_games[game_number] = CompletedGame(
                game_number=game_number,
                score=score,
                steps=steps_by_game[game_number],
                seed=seeds_by_number[game_number],
                trace=trace,
            )
            del games[game_number]

        active_game_numbers = next_active_game_numbers

    games_in_order = [completed_games[number] for number in game_numbers]
    best_game = max(games_in_order, key=lambda game: (game.score, -game.game_number))
    worst_game = min(games_in_order, key=lambda game: (game.score, game.game_number))
    return EvaluationResult(
        scores=[game.score for game in games_in_order],
        moves=[game.steps for game in games_in_order],
        worst_game=worst_game,
        best_game=best_game,
    )
