from algorithms.ppo import EpochStatistics, PPO, PPOConfig
from contextlib import redirect_stdout
from dataclasses import asdict
from games.rl_2048.agent import ResNet, ResNetConfig
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
def save_completed_game(ppo: PPO, seed: int, output_path: Path | None = None) -> tuple[int, int, str]:
    """Play a separate game greedily to completion and return its raw board trace."""
    game = Game2048(seed=seed)
    trace = [f"Evaluation game seed: {seed}"]
    steps = 0

    while not game.done:
        state = game.current_state()
        legal_actions = game.legal_actions()
        decision = ppo.generate_next_actions([game], deterministic=True)[0]
        result = game.step(decision.action)
        assert result.action_applied, "The policy selected an illegal action during evaluation"

        trace.extend(
            [
                f"Step {steps}",
                f"Board before move:\n{state}",
                "Legal actions: " + ", ".join(game.action_name(legal_action) for legal_action in legal_actions),
                f"Selected action: {game.action_name(decision.action)}",
                f"Reward: {result.reward}",
            ]
        )
        steps += 1

    trace.extend(
        [
            f"Final board:\n{game.current_state()}",
            f"Final score: {game.total_score}",
            f"Total steps: {steps}",
        ]
    )
    trace_text = "\n\n".join(trace) + "\n"
    if output_path is not None:
        output_path.write_text(trace_text, encoding="utf-8")
    return int(game.total_score), steps, trace_text


def main() -> None:
    # Larger experiment settings to revisit after the smoke test succeeds:
    # ppo_config = PPOConfig(
    #     gamma=0.9,
    #     gae_lambda=0.9,
    #     clip_epsilon=0.25,
    #     value_coefficient=0.5,
    #     entropy_coefficient=0.1,
    #     update_epochs=100,
    #     num_parallel_games=20,
    #     steps_per_game_per_rollout=250,
    #     mini_batch_size=500,
    #     seed=12345,
    #     print_moves=False,
    #     print_timestamps=True,
    # )
    ppo_config = PPOConfig(
        gamma=1.0,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.1,
        entropy_coefficient=0.01,
        update_epochs=10,
        num_parallel_games=16,
        steps_per_game_per_rollout=1024,
        mini_batch_size=1024,
        seed=12345,
        reward_multiplier=0.01,
        value_transform="identity",
        denominator_epsilon=1e-8,
        print_moves=False,
        print_timestamps=True,
    )
    num_outer_iterations = 5
    device = select_device()
    torch.manual_seed(ppo_config.seed)
    resnet_config = ResNetConfig(
        in_channels=1,
        n_channels=16,
        blocks=3,
        height=4,
        width=4,
        output_dim=4,
        head_channels=16,
        kernel_size=3,
        block_layers=2,
        include_groupnorm=True,
    )
    model = ResNet(resnet_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ppo = PPO(
        model,
        optimizer,
        ppo_config,
        game_factory=lambda seed: Game2048(seed=seed),
    )
    evaluation_output_root = Path("artifacts") / "ppo_2048"
    run_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    evaluation_output_dir = evaluation_output_root / f"run_{run_timestamp}"
    checkpoint_path = evaluation_output_dir / "best.pt"
    num_evaluation_games = 100
    save_evaluation_traces = False
    best_mean_score = float("-inf")
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = evaluation_output_dir / "summary.txt"
    run_summary_path.write_text("PPO 2048 evaluation summary\n\n", encoding="utf-8")
    raw_summary_path = evaluation_output_dir / "raw_summary.txt"
    with raw_summary_path.open("w", encoding="utf-8") as raw_summary_file:
        with redirect_stdout(_Tee(sys.stdout, raw_summary_file)):
            print(f"Training smoke test on {device}.")
            for outer_iteration in range(1, num_outer_iterations + 1):
                print()
                print("=" * 72)
                print(f"Outer iteration {outer_iteration}")
                print("=" * 72)
                epoch_stats = ppo.run_outer_iteration()
                print_loss_curves(epoch_stats)
                print_timestamp("Starting evaluation games.", ppo_config.print_timestamps)
                was_training = model.training
                model.eval()
                outer_evaluation_dir = evaluation_output_dir / f"outer_iteration_{outer_iteration:04d}"
                scores = []
                moves = []
                worst_game: tuple[int, int, int, str] | None = None
                best_game: tuple[int, int, int, str] | None = None
                for game_number in range(1, num_evaluation_games + 1):
                    evaluation_seed = ppo_config.seed + game_number
                    trace_path = None
                    if save_evaluation_traces:
                        game_dir = outer_evaluation_dir / f"game_{game_number:02d}"
                        game_dir.mkdir(parents=True, exist_ok=True)
                        trace_path = game_dir / "trace.txt"
                    score, steps, trace_text = save_completed_game(ppo, evaluation_seed, trace_path)
                    scores.append(score)
                    moves.append(steps)
                    game_result = (score, steps, evaluation_seed, trace_text)
                    if worst_game is None or score < worst_game[0]:
                        worst_game = game_result
                    if best_game is None or score > best_game[0]:
                        best_game = game_result

                scores.sort()
                assert worst_game is not None and best_game is not None
                outer_evaluation_dir.mkdir(parents=True, exist_ok=True)
                (outer_evaluation_dir / "worst_game.txt").write_text(worst_game[3], encoding="utf-8")
                (outer_evaluation_dir / "best_game.txt").write_text(best_game[3], encoding="utf-8")
                mean_score = sum(scores) / len(scores)
                median_score = median(scores)
                mean_moves = sum(moves) / len(moves)
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        ppo_config,
                        resnet_config,
                        outer_iteration,
                        mean_score,
                        median_score,
                        mean_moves,
                    )
                    print(f"  Saved new best checkpoint: {checkpoint_path}")
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
                with run_summary_path.open("a", encoding="utf-8") as summary_file:
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
                print(f"  Evaluation traces: {outer_evaluation_dir}; run summary: {run_summary_path}")
                print_timestamp("Evaluation games complete.", ppo_config.print_timestamps)
                model.train(was_training)


if __name__ == "__main__":
    main()
