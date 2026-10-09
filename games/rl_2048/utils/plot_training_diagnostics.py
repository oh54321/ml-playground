"""Save ordinary PNG plots for PPO 2048 training diagnostics.

The project already records terminal-friendly summaries.  This module keeps
the plotting dependency small and turns those summaries into the same kind of
standalone PNG files as the evaluation score plots.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Iterable, Sequence


_COLORS = ((31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14))
_WIDTH = 1200
_HEIGHT = 700


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 10000:
        return f"{value:.0f}"
    if absolute >= 1000:
        return f"{value:,.0f}"
    if absolute >= 10:
        return f"{value:.1f}"
    if absolute >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _nice_ticks(low: float, high: float, count: int = 6) -> list[float]:
    if high <= low:
        return [low]
    raw_step = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    first = math.floor(low / step) * step
    ticks: list[float] = []
    value = first
    while value <= high + step * 0.01 and len(ticks) < 20:
        if value >= low - step * 0.01:
            ticks.append(value)
        value += step
    return ticks or [low, high]


def _draw_vertical_label(image, text: str, center_y: float, font) -> None:
    from PIL import Image, ImageDraw

    bounds = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    layer = Image.new("RGBA", (bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((4 - bounds[0], 4 - bounds[1]), text, fill=(45, 45, 45, 255), font=font)
    rotated = layer.rotate(90, expand=True)
    image.paste(rotated, (14, int(center_y - rotated.height / 2)), rotated)


def save_line_plot(
    output_path: Path,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], tuple[int, int, int]]],
    title: str,
    x_label: str = "Outer training iteration",
    y_label: str = "Value",
    y_min: float | None = None,
    y_max: float | None = None,
    reference_lines: Sequence[tuple[float, str, tuple[int, int, int]]] = (),
) -> None:
    """Save a readable standalone line chart without requiring matplotlib."""
    from PIL import Image, ImageDraw

    if not x_values or not series:
        return
    values = [float(value) for _, values_for_series, _ in series for value in values_for_series]
    values.extend(float(value) for value, _, _ in reference_lines)
    data_min, data_max = min(values), max(values)
    if y_min is not None:
        data_min = min(data_min, y_min)
    if data_min == data_max:
        padding = max(abs(data_min) * 0.1, 1.0)
    else:
        padding = (data_max - data_min) * 0.08
    plot_low = y_min if y_min is not None else data_min - padding
    plot_high = y_max if y_max is not None else data_max + padding
    if plot_low == plot_high:
        plot_high = plot_low + 1.0

    image = Image.new("RGB", (_WIDTH, _HEIGHT), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    label_font = _font(17)
    tick_font = _font(14)
    draw.text((70, 22), title, fill=(25, 25, 25), font=title_font)

    left, top, right, bottom = 105, 78, 34, 92
    x0, x1 = left, _WIDTH - right
    y0, y1 = top, _HEIGHT - bottom

    def x_pixel(value: float) -> float:
        if x_values[-1] == x_values[0]:
            return (x0 + x1) / 2
        return x0 + (value - x_values[0]) / (x_values[-1] - x_values[0]) * (x1 - x0)

    def y_pixel(value: float) -> float:
        return y1 - (value - plot_low) / (plot_high - plot_low) * (y1 - y0)

    y_ticks = _nice_ticks(plot_low, plot_high)
    for tick in y_ticks:
        y = y_pixel(tick)
        draw.line((x0, y, x1, y), fill=(220, 220, 220), width=1)
        text = _format_number(tick)
        bbox = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((x0 - (bbox[2] - bbox[0]) - 10, y - (bbox[3] - bbox[1]) / 2), text, fill=(75, 75, 75), font=tick_font)

    x_ticks = list(dict.fromkeys(round(value) for value in _evenly_spaced(x_values[0], x_values[-1], 7)))
    for tick in x_ticks:
        x = x_pixel(tick)
        draw.line((x, y0, x, y1), fill=(235, 235, 235), width=1)
        text = str(tick)
        bbox = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y1 + 10), text, fill=(75, 75, 75), font=tick_font)

    draw.line((x0, y0, x0, y1), fill=(70, 70, 70), width=2)
    draw.line((x0, y1, x1, y1), fill=(70, 70, 70), width=2)
    draw.text(((x0 + x1) / 2 - 100, _HEIGHT - 40), x_label, fill=(45, 45, 45), font=label_font)
    _draw_vertical_label(image, y_label, (y0 + y1) / 2, label_font)

    for reference, label, color in reference_lines:
        y = y_pixel(reference)
        for start in range(int(x0), int(x1), 16):
            draw.line((start, y, min(start + 8, x1), y), fill=color, width=2)
        draw.text((x1 - 145, y - 22), label, fill=color, font=tick_font)

    for index, (label, values_for_series, color) in enumerate(series):
        points = [(x_pixel(x), y_pixel(float(value))) for x, value in zip(x_values, values_for_series)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3, joint="curve")
        for point in points[:: max(1, len(points) // 40)]:
            radius = 3
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
        legend_x = x0 + index * 205
        legend_y = 48
        draw.line((legend_x, legend_y + 9, legend_x + 26, legend_y + 9), fill=color, width=3)
        draw.text((legend_x + 34, legend_y), label, fill=(45, 45, 45), font=tick_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _evenly_spaced(low: float, high: float, count: int) -> Iterable[float]:
    if count <= 1 or low == high:
        yield low
        return
    for index in range(count):
        yield low + (high - low) * index / (count - 1)


def save_evaluation_plots(
    evaluation_history: dict[str, list[float]],
    evaluation_iterations: list[int],
    output_dir: Path,
) -> None:
    """Save the standard score plots used during evaluation."""
    plot_details = (
        ("mean", "Mean evaluation score", "evaluation_mean_score.png", "Score (points)"),
        ("median", "Median evaluation score", "evaluation_median_score.png", "Score (points)"),
        ("best", "Best evaluation game score", "evaluation_best_score.png", "Score (points)"),
        ("worst", "Worst evaluation game score", "evaluation_worst_score.png", "Score (points)"),
        ("mean_moves", "Mean evaluation game length", "evaluation_mean_moves.png", "Moves"),
        (
            "win_rate",
            "Evaluation win rate (2048+)",
            "evaluation_win_rate.png",
            "Games reaching 2048 (%)",
        ),
    )
    for metric, title, filename, y_label in plot_details:
        if metric not in evaluation_history:
            continue
        save_line_plot(
            output_dir / filename,
            evaluation_iterations,
            [
                (
                    metric.replace("_", " ").title(),
                    [value * 100 for value in evaluation_history[metric]]
                    if metric == "win_rate"
                    else evaluation_history[metric],
                    _COLORS[0],
                )
            ],
            title,
            y_label=y_label,
            y_min=0,
            y_max=100 if metric == "win_rate" else None,
        )


def _numbers_in_brackets(line: str) -> list[float]:
    match = re.search(r"\[([^]]*)\]\s*$", line)
    if not match:
        return []
    return [float(value.strip()) for value in match.group(1).split(",") if value.strip()]


def _parse_raw_summary(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        iteration_match = re.match(r"^Outer iteration (\d+)$", line)
        if iteration_match:
            if current is not None:
                records.append(current)
            current = {"iteration": int(iteration_match.group(1))}
            continue
        if current is None:
            continue
        for name, key in (
            ("policy loss", "policy_loss"),
            ("value loss", "value_loss"),
            ("negative entropy", "negative_entropy"),
            ("approximate KL", "approximate_kl"),
            ("clip fraction", "clip_fraction"),
            ("explained variance", "explained_variance"),
            ("value error RMS", "value_error_rms"),
        ):
            if line.strip().startswith(f"{name}:"):
                current[key] = _numbers_in_brackets(line)
                break
        else:
            match = re.search(r"value target scale: mean=([-+0-9.eE]+), std=([-+0-9.eE]+)", line)
            if match:
                current["value_target_mean"] = float(match.group(1))
                current["value_target_std"] = float(match.group(2))
                continue
            match = re.search(
                r"Rollout advantages: mean=([-+0-9.eE]+); std=([-+0-9.eE]+)", line
            )
            if match:
                current["advantage_std"] = float(match.group(2))
                continue
            match = re.search(
                r"(?:shared trunk|separate trunks) gradient norms: policy=([-+0-9.eE]+), value=([-+0-9.eE]+), entropy=([-+0-9.eE]+)",
                line,
            )
            if match:
                current["policy_gradient"] = float(match.group(1))
                current["value_gradient"] = float(match.group(2))
                current["entropy_gradient"] = float(match.group(3))
                continue
            match = re.search(r"value/policy gradient ratio: ([-+0-9.eE]+);", line)
            if match:
                current["value_policy_ratio"] = float(match.group(1))
            match = re.search(r"policy/value cosine similarity: ([-+0-9.eE]+)", line)
            if match:
                current["gradient_cosine"] = float(match.group(1))
            if match:
                continue
            match = re.search(r"relative (?:shared trunk|separate trunks) parameter update: ([-+0-9.eE]+)", line)
            if match:
                current["parameter_update"] = float(match.group(1))
                continue
            match = re.search(r"policy KL vs previous outer iteration: ([-+0-9.eE]+)", line)
            if match:
                current["policy_kl_between_iterations"] = float(match.group(1))
                continue
            match = re.search(r"learning rate: ([-+0-9.eE]+)", line)
            if match:
                current["learning_rate"] = float(match.group(1))
    if current is not None:
        records.append(current)
    return records


def _parse_evaluation_summary(path: Path) -> tuple[list[int], list[list[float]]]:
    iterations: list[int] = []
    scores: list[list[float]] = []
    current_iteration: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^Outer iteration (\d+)$", line)
        if match:
            current_iteration = int(match.group(1))
            continue
        if current_iteration is None or not line.startswith("Raw scores:"):
            continue
        try:
            values = ast.literal_eval(line.removeprefix("Raw scores: ").strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(values, list) and values:
            iterations.append(current_iteration)
            scores.append([float(value) for value in values])
    return iterations, scores


def _parse_evaluation_win_rates(path: Path) -> tuple[list[int], list[float]]:
    iterations: list[int] = []
    rates: list[float] = []
    current_iteration: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^Outer iteration (\d+)$", line)
        if match:
            current_iteration = int(match.group(1))
            continue
        rate_match = re.match(r"^Win rate \(2048\+\): ([0-9.]+)%", line)
        if current_iteration is not None and rate_match:
            iterations.append(current_iteration)
            rates.append(float(rate_match.group(1)))
    return iterations, rates


def _series(records: list[dict[str, object]], key: str, end: bool = False) -> list[float]:
    result: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, list):
            result.append(float(value[-1 if end else 0]))
        elif isinstance(value, (int, float)):
            result.append(float(value))
    return result


def save_training_diagnostic_plots(
    raw_summary_path: Path,
    output_dir: Path,
    evaluation_summary_path: Path | None = None,
) -> None:
    """Save one PNG for each non-duplicate metric in raw_summary.txt."""
    records = _parse_raw_summary(raw_summary_path)
    if not records:
        return
    iterations = [int(record["iteration"]) for record in records]
    paired = (
        ("training_policy_loss.png", "Policy loss", "Policy loss", "policy_loss", None),
        ("training_value_loss.png", "Value loss", "Value loss", "value_loss", 0),
        ("training_entropy.png", "Entropy", "Entropy (nats)", "negative_entropy", 0),
        ("training_approximate_kl.png", "Approximate KL divergence", "KL divergence", "approximate_kl", 0),
        ("training_clip_fraction.png", "Clip fraction", "Fraction", "clip_fraction", 0),
        ("training_explained_variance.png", "Explained variance", "Explained variance", "explained_variance", None),
        ("training_value_error_rms.png", "Value error RMS", "Error in transformed value units", "value_error_rms", 0),
    )
    for filename, title, y_label, key, y_min in paired:
        values = _series(records, key)
        final_values = _series(records, key, end=True)
        if len(values) == len(iterations) and len(final_values) == len(iterations):
            if key == "negative_entropy":
                values = [-value for value in values]
                final_values = [-value for value in final_values]
            save_line_plot(
                output_dir / filename,
                iterations,
                [("start", values, _COLORS[0]), ("end", final_values, _COLORS[1])],
                title,
                y_label=y_label,
                y_min=y_min,
                reference_lines=((0.02, "target 0.02", (120, 120, 120)),) if key == "approximate_kl" else (),
            )

    for filename, title, y_label, key in (
        ("training_value_target_mean.png", "Value target mean", "Transformed target", "value_target_mean"),
        ("training_value_target_std.png", "Value target standard deviation", "Transformed target", "value_target_std"),
        ("training_advantage_std.png", "Raw advantage standard deviation", "Advantage", "advantage_std"),
        ("training_value_policy_gradient_ratio.png", "Value-to-policy gradient ratio", "Value / policy", "value_policy_ratio"),
        ("training_gradient_cosine_similarity.png", "Policy/value gradient cosine similarity", "Cosine similarity", "gradient_cosine"),
        ("training_parameter_update.png", "Relative trunk parameter update", "Relative update", "parameter_update"),
        ("training_policy_kl_between_iterations.png", "Policy KL between consecutive outer iterations", "KL divergence", "policy_kl_between_iterations"),
    ):
        values = _series(records, key)
        if len(values) == len(iterations):
            save_line_plot(
                output_dir / filename,
                iterations,
                [(key.replace("_", " "), values, _COLORS[0])],
                title,
                y_label=y_label,
                y_min=0,
                reference_lines=((0.02, "target 0.02", (120, 120, 120)),) if key == "policy_kl_between_iterations" else (),
            )

    gradient_series = []
    for label, key, color in (
        ("policy", "policy_gradient", _COLORS[0]),
        ("value", "value_gradient", _COLORS[1]),
        ("entropy", "entropy_gradient", _COLORS[2]),
    ):
        values = _series(records, key)
        if len(values) == len(iterations):
            gradient_series.append((label, values, color))
    if gradient_series:
        save_line_plot(output_dir / "training_gradient_norms.png", iterations, gradient_series, "Trunk gradient norms", y_label="Gradient norm")

    learning_rates = _series(records, "learning_rate")
    if len(learning_rates) != len(iterations):
        learning_rates = [1e-3 * 0.995**iteration for iteration in iterations]
    save_line_plot(output_dir / "training_learning_rate.png", iterations, [("Adam learning rate", learning_rates, _COLORS[0])], "Learning rate", y_label="Learning rate", y_min=0)

    if evaluation_summary_path is not None and evaluation_summary_path.exists():
        evaluation_iterations, score_sets = _parse_evaluation_summary(evaluation_summary_path)
        if evaluation_iterations:
            quantile_series = []
            for label, quantile, color in (
                ("p10", 0.10, _COLORS[0]),
                ("p25", 0.25, _COLORS[1]),
                ("median", 0.50, _COLORS[2]),
                ("p75", 0.75, _COLORS[3]),
                ("p90", 0.90, _COLORS[4]),
                ("p99", 0.99, (0, 128, 128)),
            ):
                quantile_series.append((label, [_quantile(scores, quantile) for scores in score_sets], color))
            save_line_plot(output_dir / "evaluation_score_quantiles.png", evaluation_iterations, quantile_series, "Evaluation score distribution", y_label="Score (points)", y_min=0)
        win_iterations, win_rates = _parse_evaluation_win_rates(evaluation_summary_path)
        if win_iterations:
            save_line_plot(
                output_dir / "evaluation_win_rate.png",
                win_iterations,
                [("Win rate", win_rates, _COLORS[0])],
                "Evaluation win rate (2048+)",
                y_label="Games reaching 2048 (%)",
                y_min=0,
                y_max=100,
            )


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
