"""Terminal and file reporting for PPO 2048 training runs."""

from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import sys
from typing import Iterator

from algorithms.ppo import EpochStatistics, GradientStatistics


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
            if not getattr(stream, "closed", False):
                stream.flush()


@contextmanager
def tee_stdout(output_path: Path) -> Iterator[None]:
    """Mirror this run's console output into its raw training log."""
    with output_path.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(_Tee(sys.stdout, output_file)):
            yield


def _sparkline(values: list[float]) -> str:
    """Render a compact terminal graph for one metric."""
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    low, high = min(values), max(values)
    if low == high:
        return blocks[0] * len(values)
    return "".join(
        blocks[round((value - low) / (high - low) * (len(blocks) - 1))]
        for value in values
    )


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
    """Print value-target scale and value-fit diagnostics per update epoch."""
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
    """Print loss-gradient diagnostics over the model's trunk parameters."""
    if gradient_statistics is None:
        return
    trunk_label = "shared trunk" if gradient_statistics.shared_trunk else "separate trunks"
    print(
        f"  {trunk_label} gradient norms: "
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
        f"  relative {trunk_label} parameter update: "
        f"{gradient_statistics.relative_parameter_update_size:.6e}"
    )
    if not gradient_statistics.shared_trunk:
        print("  Policy/value cosine is structurally zero: no shared parameters.")


def print_outer_iteration_header(outer_iteration: int) -> None:
    print()
    print("=" * 72)
    print(f"Outer iteration {outer_iteration}")
    print("=" * 72)


def print_timestamp(message: str, enabled: bool = True) -> None:
    if enabled:
        from datetime import datetime

        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{timestamp}] {message}")


def report_run_start(
    device: str,
    model_name: str,
    model_config: dict[str, object],
    symmetry_augmentation: str | None,
    exploration_schedule: dict[str, object],
    parameter_count: int,
    warm_start_from: Path | None,
    first_iteration: int,
) -> None:
    """Print the settings that identify this training run."""
    print(f"Training on {device}.")
    print(f"Model: {model_name}; config: {model_config}")
    print(f"Symmetry augmentation: {symmetry_augmentation}")
    print(f"Exploration schedule: {exploration_schedule}")
    print(f"Total trainable parameters: {parameter_count:,}")
    if warm_start_from is not None:
        print(
            f"Warm-started from {warm_start_from} "
            f"at iteration {first_iteration - 1}; "
            "game and RNG streams start fresh."
        )


def report_training_iteration_start(
    iteration: int,
    exploration_probability: float,
) -> None:
    """Print the outer-iteration heading and active exploration setting."""
    print_outer_iteration_header(iteration)
    print(f"  rollout exploration probability: {exploration_probability:.6f}")


def report_training_iteration(
    learning_rate: float,
    epoch_stats: list[EpochStatistics],
    gradient_statistics: GradientStatistics | None,
    policy_kl: float,
) -> None:
    """Print the PPO and optimizer statistics produced by one outer iteration."""
    print(f"  learning rate: {learning_rate:.8e}")
    print(f"  PPO update epochs used: {len(epoch_stats)}")
    print_loss_curves(epoch_stats)
    print_value_diagnostics(epoch_stats)
    print_gradient_diagnostics(gradient_statistics)
    print(f"  policy KL vs previous outer iteration: {policy_kl:.6f}")


def report_evaluation_skipped(iteration: int, evaluation_interval: int) -> None:
    """Explain why an outer iteration did not run scheduled evaluation."""
    print(
        f"Skipping evaluation at outer iteration {iteration}; "
        f"evaluation interval is {evaluation_interval}."
    )


__all__ = [
    "print_loss_curves",
    "print_value_diagnostics",
    "print_gradient_diagnostics",
    "print_outer_iteration_header",
    "print_timestamp",
    "report_run_start",
    "report_training_iteration_start",
    "report_training_iteration",
    "report_evaluation_skipped",
    "tee_stdout",
]
