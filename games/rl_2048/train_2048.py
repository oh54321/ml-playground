from algorithms.ppo import PPO, PPOConfig
from contextlib import redirect_stdout
from games.rl_2048.agent import ResNet, ResNetConfig
from datetime import datetime
import io
from pathlib import Path
from statistics import median
import sys

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


def print_loss_curves(epoch_stats: list[tuple[float, float, float]]) -> None:
    names = ("policy loss", "value loss", "negative entropy")
    for index, name in enumerate(names):
        values = [stats[index] for stats in epoch_stats]
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


@torch.no_grad()
def save_completed_game(ppo: PPO, seed: int, output_path: Path | None = None) -> tuple[int, int]:
    """Play a separate game greedily to completion, optionally saving its raw board trace."""
    game = Game2048(seed=seed)
    trace = [f"Evaluation game seed: {seed}"]
    steps = 0

    while not game.done:
        state = game.current_state()
        legal_actions = game.legal_actions()
        action, _, _, _ = ppo.select_action(state, legal_actions, deterministic=True)
        next_state, reward, done, moved = game.step(action)
        assert moved, "The policy selected an illegal action during evaluation"

        trace.extend(
            [
                f"Step {steps}",
                f"Board before move:\n{state}",
                "Legal actions: " + ", ".join(game.action_name(legal_action) for legal_action in legal_actions),
                f"Selected action: {game.action_name(action)}",
                f"Reward: {reward}",
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
    if output_path is not None:
        output_path.write_text("\n\n".join(trace) + "\n", encoding="utf-8")
    return int(game.total_score), steps


def main() -> None:
    # Larger experiment settings to revisit after the smoke test succeeds:
    # ppo_config = PPOConfig(
    #     gamma=0.9,
    #     gae_lambda=0.9,
    #     clip_epsilon=0.25,
    #     value_coefficient=0.5,
    #     entropy_coefficient=0.1,
    #     update_epochs=100,
    #     num_policy_rollout=5000,
    #     max_game_steps=2500,
    #     mini_batch_size=500,
    #     seed=12345,
    #     print_moves=False,
    #     print_timestamps=True,
    # )
    ppo_config = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=1e-6,
        entropy_coefficient=1, # 10
        update_epochs=50,
        num_policy_rollout=16384,
        max_game_steps=1024,
        mini_batch_size=1024,
        seed=12345,
        print_moves=False,
        print_timestamps=True,
    )
    num_outer_iterations = 5
    device = select_device()
    torch.manual_seed(ppo_config.seed)
    resnet_config = ResNetConfig(
        in_channels=1,
        n_channels=4,
        blocks=1,
        height=4,
        width=4,
        output_dim=4,
        head_channels=4,
        kernel_size=3,
        block_layers=2,
        include_groupnorm=True,
    )
    model = ResNet(resnet_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    ppo = PPO(model, optimizer, ppo_config)
    game = Game2048(seed=ppo_config.seed)
    evaluation_output_dir = Path("artifacts") / "ppo_2048"
    num_evaluation_games = 100
    save_evaluation_traces = False
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
                epoch_stats = ppo.run_outer_iteration(game)
                print_loss_curves(epoch_stats)
                print_timestamp("Starting evaluation games.", ppo_config.print_timestamps)
                was_training = model.training
                model.eval()
                outer_evaluation_dir = evaluation_output_dir / f"outer_iteration_{outer_iteration:04d}"
                scores = []
                moves = []
                for game_number in range(1, num_evaluation_games + 1):
                    evaluation_seed = ppo_config.seed + outer_iteration * num_evaluation_games + game_number
                    trace_path = None
                    if save_evaluation_traces:
                        game_dir = outer_evaluation_dir / f"game_{game_number:02d}"
                        game_dir.mkdir(parents=True, exist_ok=True)
                        trace_path = game_dir / "trace.txt"
                    score, steps = save_completed_game(ppo, evaluation_seed, trace_path)
                    scores.append(score)
                    moves.append(steps)

                scores.sort()
                mean_score = sum(scores) / len(scores)
                median_score = median(scores)
                mean_moves = sum(moves) / len(moves)
                summary = (
                    f"Outer iteration {outer_iteration}\n"
                    f"Mean score: {mean_score:.1f}\n"
                    f"Median score: {median_score:.1f}\n"
                    f"Mean moves: {mean_moves:.1f}\n"
                    f"Raw scores: {scores}\n"
                    f"Traces: {outer_evaluation_dir if save_evaluation_traces else 'not saved'}\n\n"
                )
                with run_summary_path.open("a", encoding="utf-8") as summary_file:
                    summary_file.write(summary)
                print(
                    f"  Evaluation mean score: {mean_score:.1f}; "
                    f"median score: {median_score:.1f}; mean moves: {mean_moves:.1f}"
                )
                print(f"  Evaluation raw scores: {scores}")
                trace_status = outer_evaluation_dir if save_evaluation_traces else "not saved"
                print(f"  Evaluation traces: {trace_status}; run summary: {run_summary_path}")
                print_timestamp("Evaluation games complete.", ppo_config.print_timestamps)
                model.train(was_training)


if __name__ == "__main__":
    main()
