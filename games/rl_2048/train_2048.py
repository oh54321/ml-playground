from torch.optim.lr_scheduler import ExponentialLR
from algorithms.ppo import EpochStatistics, GradientStatistics, PPO, PPOConfig
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from games.rl_2048.agent import ResNet, ResNetConfig
from games.rl_2048.plot_training_diagnostics import (
    save_evaluation_plots,
    save_training_diagnostic_plots,
)
from datetime import datetime
import io
import os
from pathlib import Path
from statistics import median
import sys
import tempfile

import torch
import torch.optim as optim

from games.rl_2048.game import Game2048


class _Tee(io.TextIOBase):
    """Write terminal output to both the terminal and a text file."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            # A stream can close before the Tee object is finalized during
            # interpreter shutdown.  Ignore that cleanup-only case.
            if getattr(stream, "closed", False):
                continue
            stream.flush()


def _sparkline(values: list[float]) -> str:
    """Render a compact terminal graph for one loss curve."""
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    low, high = min(values), max(values)
    if low == high:
        return blocks[0] * len(values)
    return "".join(blocks[round((value - low) / (high - low) * (len(blocks) - 1))] for value in values)


def print_loss_curves(epoch_stats: list[EpochStatistics]) -> None:
    metrics = (
        ("policy loss", "policy_loss"),
        ("value loss", "value_loss"),
        ("negative entropy", "negative_entropy"),
        ("approximate KL", "approximate_kl"),
        ("clip fraction", "clip_fraction"),
        ("explained variance", "explained_variance"),
    )
    for name, attribute in metrics:
        values = [getattr(stats, attribute) for stats in epoch_stats]
        formatted_values = ", ".join(f"{value:.4f}" for value in values)
        print(f"  {name:>16}: {_sparkline(values)}  [{formatted_values}]")


def print_value_diagnostics(epoch_stats: list[EpochStatistics]) -> None:
    """Print value-target scale once and value-fit diagnostics per update epoch."""
    if not epoch_stats:
        return
    first_stats = epoch_stats[0]
    print(
        "  value target scale: "
        f"mean={first_stats.value_target_mean:.4f}, "
        f"std={first_stats.value_target_std:.4f}"
    )
    metrics = (
        ("predicted value mean", "predicted_value_mean"),
        ("predicted value std", "predicted_value_std"),
        ("value error RMS", "value_error_rms"),
    )
    for name, attribute in metrics:
        values = [getattr(stats, attribute) for stats in epoch_stats]
        formatted_values = ", ".join(f"{value:.4f}" for value in values)
        print(f"  {name:>20}: {_sparkline(values)}  [{formatted_values}]")


def print_gradient_diagnostics(
    gradient_statistics: GradientStatistics | None,
) -> None:
    """Print loss-gradient diagnostics for the shared actor-critic trunk."""
    if gradient_statistics is None:
        return
    print(
        "  shared trunk gradient norms: "
        f"policy={gradient_statistics.policy_norm:.4f}, "
        f"value={gradient_statistics.value_norm:.4f}, "
        f"entropy={gradient_statistics.entropy_norm:.4f}"
    )
    print(
        "  value/policy gradient ratio: "
        f"{gradient_statistics.value_to_policy_ratio:.4f}; "
        "policy/value cosine similarity: "
        f"{gradient_statistics.policy_value_cosine_similarity:.4f}"
    )
    print(
        "  relative shared trunk parameter update: "
        f"{gradient_statistics.relative_parameter_update_size:.6e}"
    )


def print_outer_iteration_header(outer_iteration: int) -> None:
    print()
    print("=" * 72)
    print(f"Outer iteration {outer_iteration}")
    print("=" * 72)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_timestamp(message: str, enabled: bool) -> None:
    if enabled:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{timestamp}] {message}")


def save_checkpoint(
    output_path: Path,
    model: ResNet,
    optimizer: optim.Optimizer,
    ppo_config: PPOConfig,
    resnet_config: ResNetConfig,
    outer_iteration: int,
    mean_score: float,
    median_score: float,
    mean_moves: float,
) -> None:
    """Save everything needed to inspect or resume the best evaluated policy."""
    checkpoint = {
        "outer_iteration": outer_iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ppo_config": asdict(ppo_config),
        "resnet_config": asdict(resnet_config),
        "evaluation": {
            "mean_score": mean_score,
            "median_score": median_score,
            "mean_moves": mean_moves,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(checkpoint, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@torch.no_grad()
def save_completed_games(
    ppo: PPO,
    seeds: list[int],
    trace_output_dir: Path | None = None,
) -> tuple[
    list[int],
    list[int],
    tuple[int, int, int, str],
    tuple[int, int, int, str],
]:
    """Evaluate seeded games together, batching policy inference over active games."""
    if not seeds:
        raise ValueError("Must evaluate at least one game")

    game_numbers = list(range(1, len(seeds) + 1))
    seeds_by_number = dict(zip(game_numbers, seeds))
    games = {
        game_number: Game2048(seed=seed)
        for game_number, seed in seeds_by_number.items()
    }
    active_game_numbers = game_numbers.copy()
    steps_by_game = {game_number: 0 for game_number in game_numbers}
    traces_by_game = {
        game_number: [f"Evaluation game seed: {seeds_by_number[game_number]}"]
        for game_number in game_numbers
    }
    results_by_game: dict[int, tuple[int, int]] = {}
    best_game: tuple[int, int, int, str] | None = None
    worst_game: tuple[int, int, int, str] | None = None
    best_game_number: int | None = None
    worst_game_number: int | None = None

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

            steps = steps_by_game[game_number]
            score = int(game.total_score)
            traces_by_game[game_number].extend(
                [
                    f"Final board:\n{game.current_state()}",
                    f"Final score: {score}",
                    f"Total steps: {steps}",
                ]
            )
            trace_text = "\n\n".join(traces_by_game.pop(game_number)) + "\n"
            seed = seeds_by_number[game_number]
            game_result = (score, steps, seed, trace_text)
            results_by_game[game_number] = (score, steps)

            if trace_output_dir is not None:
                game_dir = trace_output_dir / f"game_{game_number:02d}"
                game_dir.mkdir(parents=True, exist_ok=True)
                (game_dir / "trace.txt").write_text(trace_text, encoding="utf-8")

            if (
                best_game is None
                or score > best_game[0]
                or (
                    score == best_game[0]
                    and best_game_number is not None
                    and game_number < best_game_number
                )
            ):
                best_game = game_result
                best_game_number = game_number
            if (
                worst_game is None
                or score < worst_game[0]
                or (
                    score == worst_game[0]
                    and worst_game_number is not None
                    and game_number < worst_game_number
                )
            ):
                worst_game = game_result
                worst_game_number = game_number
            del games[game_number]

        active_game_numbers = next_active_game_numbers

    assert best_game is not None and worst_game is not None
    scores = [results_by_game[game_number][0] for game_number in game_numbers]
    moves = [results_by_game[game_number][1] for game_number in game_numbers]
    return scores, moves, worst_game, best_game


@dataclass
class EvaluationRun:
    output_dir: Path
    seed_base: int
    num_games: int
    save_traces: bool
    history: dict[str, list[float]]
    iterations: list[int]
    best_mean_score: float = float("-inf")


def evaluate_and_report(
    ppo: PPO,
    model: ResNet,
    optimizer: optim.Optimizer,
    ppo_config: PPOConfig,
    resnet_config: ResNetConfig,
    outer_iteration: int,
    evaluation: EvaluationRun,
) -> None:
    """Run one evaluation pass and update its plots, summaries, and checkpoint."""
    print_timestamp("Starting evaluation games.", ppo_config.print_timestamps)
    was_training = model.training
    model.eval()
    try:
        outer_evaluation_dir = evaluation.output_dir / f"outer_iteration_{outer_iteration:04d}"
        scores, moves, worst_game, best_game = save_completed_games(
            ppo,
            [
                evaluation.seed_base + game_number
                for game_number in range(1, evaluation.num_games + 1)
            ],
            trace_output_dir=outer_evaluation_dir if evaluation.save_traces else None,
        )
        scores.sort()
        outer_evaluation_dir.mkdir(parents=True, exist_ok=True)
        (outer_evaluation_dir / "worst_game.txt").write_text(worst_game[3], encoding="utf-8")
        (outer_evaluation_dir / "best_game.txt").write_text(best_game[3], encoding="utf-8")
        mean_score = sum(scores) / len(scores)
        median_score = median(scores)
        mean_moves = sum(moves) / len(moves)
        evaluation.history["mean"].append(mean_score)
        evaluation.history["median"].append(median_score)
        evaluation.history["best"].append(best_game[0])
        evaluation.history["worst"].append(worst_game[0])
        evaluation.history["mean_moves"].append(mean_moves)
        evaluation.iterations.append(outer_iteration)
        save_evaluation_plots(
            evaluation.history,
            evaluation.iterations,
            evaluation.output_dir,
        )

        if mean_score > evaluation.best_mean_score:
            evaluation.best_mean_score = mean_score
            save_checkpoint(
                evaluation.output_dir / "best.pt",
                model,
                optimizer,
                ppo_config,
                resnet_config,
                outer_iteration,
                mean_score,
                median_score,
                mean_moves,
            )
            print(f"  Saved new best checkpoint: {evaluation.output_dir / 'best.pt'}")

        summary = (
            f"Outer iteration {outer_iteration}\n"
            f"Mean score: {mean_score:.1f}\n"
            f"Median score: {median_score:.1f}\n"
            f"Mean moves: {mean_moves:.1f}\n"
            f"Raw scores: {scores}\n"
            f"Best game: score {best_game[0]}, steps {best_game[1]}, seed {best_game[2]}\n"
            f"Worst game: score {worst_game[0]}, steps {worst_game[1]}, seed {worst_game[2]}\n"
            f"Traces: {outer_evaluation_dir}\n\n"
        )
        with (evaluation.output_dir / "summary.txt").open("a", encoding="utf-8") as summary_file:
            summary_file.write(summary)
        print(
            f"  Evaluation mean score: {mean_score:.1f}; "
            f"median score: {median_score:.1f}; mean moves: {mean_moves:.1f}"
        )
        print(f"  Evaluation raw scores: {scores}")
        print(
            f"  Best game: {best_game[0]} points; worst game: {worst_game[0]} points "
            f"({outer_evaluation_dir})"
        )
        print(f"  Evaluation traces: {outer_evaluation_dir}; run summary: {evaluation.output_dir / 'summary.txt'}")
        print_timestamp("Evaluation games complete.", ppo_config.print_timestamps)
    finally:
        model.train(was_training)


def main() -> None:
    # Larger experiment settings to revisit after the smoke test succeeds:
    # ppo_config = PPOConfig(
    #     gamma=0.9,
    #     gae_lambda=0.9,
    #     clip_epsilon=0.25,
    #     value_coefficient=0.5,
    #     entropy_coefficient=0.1,
    #     min_update_epochs=100,
    #     max_update_epochs=100,
    #     target_kl=0.02,
    #     num_parallel_games=20,
    #     steps_per_game_per_rollout=250,
    #     max_episode_steps=2**20,
    #     mini_batch_size=500,
    #     gradient_clip_norm=None,
    #     seed=12345,
    #     print_moves=False,
    #     print_timestamps=True,
    # )
    ppo_config = PPOConfig(
        gamma=1.0,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.003,
        entropy_coefficient=0.01,
        min_update_epochs=1,
        max_update_epochs=8,
        target_kl=0.02,
        num_parallel_games=32,
        steps_per_game_per_rollout=1024,
        max_episode_steps=2**20,
        mini_batch_size=4096,
        seed=12345,
        reward_multiplier=0.01,
        value_transform="identity",
        denominator_epsilon=1e-8,
        gradient_clip_norm=None,
        print_moves=False,
        print_timestamps=True,
    )
    num_outer_iterations = 1000
    evaluation_interval = 10  # Evaluate every k outer iterations; set to 5 for longer runs.
    evaluation_seed = 67890  # Independent, fixed seed stream for evaluation games.
    run_suffix = "bigger_model_lr_3e-4_minibatch_4096_parallel_32_adaptive"  # Optional experiment label appended after the timestamp.
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be at least 1")
    device = select_device()
    torch.manual_seed(ppo_config.seed)
    resnet_config = ResNetConfig(
        in_channels=1,
        n_channels=32,
        blocks=8,
        height=4,
        width=4,
        output_dim=4,
        head_channels=32,
        kernel_size=3,
        block_layers=2,
        include_groupnorm=True,
    )
    model = ResNet(resnet_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = ExponentialLR(optimizer, 0.999)
    ppo = PPO(
        model,
        optimizer,
        ppo_config,
        game_factory=lambda seed: Game2048(seed=seed),
    )
    evaluation_output_root = Path("artifacts") / "ppo_2048"
    run_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"run_{run_timestamp}"
    if run_suffix:
        run_name += f"_{run_suffix}"
    evaluation_output_dir = evaluation_output_root / run_name
    evaluation = EvaluationRun(
        output_dir=evaluation_output_dir,
        seed_base=evaluation_seed,
        num_games=100,
        save_traces=False,
        history={
            "mean": [],
            "median": [],
            "best": [],
            "worst": [],
            "mean_moves": [],
        },
        iterations=[],
    )
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = evaluation_output_dir / "summary.txt"
    run_summary_path.write_text(
        "PPO 2048 evaluation summary\n"
        f"Training seed: {ppo_config.seed}\n"
        f"Evaluation seed base: {evaluation_seed}\n\n",
        encoding="utf-8",
    )
    raw_summary_path = evaluation_output_dir / "raw_summary.txt"
    with raw_summary_path.open("w", encoding="utf-8") as raw_summary_file:
        with redirect_stdout(_Tee(sys.stdout, raw_summary_file)):
            print(f"Training smoke test on {device}.")
            for outer_iteration in range(1, num_outer_iterations + 1):
                print_outer_iteration_header(outer_iteration)
                epoch_stats = ppo.run_outer_iteration()
                scheduler.step()
                print(f"  learning rate: {optimizer.param_groups[0]['lr']:.8e}")
                print(f"  PPO update epochs used: {len(epoch_stats)}")
                print_loss_curves(epoch_stats)
                print_value_diagnostics(epoch_stats)
                print_gradient_diagnostics(ppo.last_gradient_statistics)
                print(
                    "  policy KL vs previous outer iteration: "
                    f"{ppo.last_outer_iteration_policy_kl:.6f}"
                )
                if outer_iteration % evaluation_interval == 0:
                    evaluate_and_report(
                        ppo,
                        model,
                        optimizer,
                        ppo_config,
                        resnet_config,
                        outer_iteration,
                        evaluation,
                    )
                else:
                    print(
                        f"Skipping evaluation at outer iteration {outer_iteration}; "
                        f"evaluation interval is {evaluation_interval}."
                    )
    if evaluation.iterations:
        save_checkpoint(
            evaluation_output_dir / "final.pt",
            model,
            optimizer,
            ppo_config,
            resnet_config,
            evaluation.iterations[-1],
            evaluation.history["mean"][-1],
            evaluation.history["median"][-1],
            evaluation.history["mean_moves"][-1],
        )
    save_training_diagnostic_plots(
        raw_summary_path,
        evaluation_output_dir,
        run_summary_path,
    )


if __name__ == "__main__":
    main()
