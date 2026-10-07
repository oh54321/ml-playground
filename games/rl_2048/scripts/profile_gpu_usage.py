"""Capture and summarize GPU telemetry alongside timestamped PPO phase logs.

Run ``collect`` on a Runpod GPU and ``analyze`` on the resulting CSV plus the
training log. See ``games/rl_2048/documents/GPU_PROFILING.md`` for the repeatable
workflow and interpretation notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


CSV_HEADER = [
    "timestamp_utc",
    "gpu_util_pct",
    "memory_controller_util_pct",
    "gpu_memory_used_mib",
    "power_w",
]
NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
    "--format=csv,noheader,nounits",
    "--loop=1",
]
TIMESTAMP_RE = re.compile(r"^\[(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\]")
PHASE_MARKERS = (
    ("rollout", "Outer iteration starting; collecting rollouts."),
    ("ppo_update", "Rollouts collected; starting PPO updates."),
    ("other", "Outer iteration complete."),
    ("evaluation", "Starting evaluation games."),
    ("other", "Evaluation games complete."),
)


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Unsupported UTC timestamp: {value!r}")


def collect(output: Path, seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        csv_file.flush()
        process = subprocess.Popen(
            NVIDIA_SMI_QUERY,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            while time.monotonic() - started < seconds:
                line = process.stdout.readline()
                if not line:
                    error = process.stderr.read() if process.stderr else ""
                    raise RuntimeError(f"nvidia-smi stopped unexpectedly: {error.strip()}")
                fields = next(csv.reader([line], skipinitialspace=True))
                if len(fields) != 5:
                    continue
                # Keep the source timestamp's sub-second precision while
                # normalizing to an explicit UTC marker for new captures.
                stamp = parse_timestamp(fields[0]).isoformat(timespec="milliseconds")
                writer.writerow([stamp, *(field.strip() for field in fields[1:])])
                csv_file.flush()
                count += 1
        except KeyboardInterrupt:
            print("Capture interrupted; keeping samples already written.", file=sys.stderr)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(f"Saved {count} samples over {time.monotonic() - started:.1f}s to {output}")


def read_samples(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            stamp = row.get("timestamp_utc") or row.get("timestamp")
            if not stamp:
                continue
            rows.append({
                "time": parse_timestamp(stamp),
                "gpu": float(row["gpu_util_pct"]),
                "memory_util": float(row["memory_controller_util_pct"]),
                "memory_mib": float(row["gpu_memory_used_mib"]),
                "power_w": float(row["power_w"]),
            })
    if not rows:
        raise ValueError(f"No telemetry samples found in {path}")
    return rows


def read_markers(path: Path) -> list[tuple[datetime, str]]:
    markers = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            match = TIMESTAMP_RE.match(line)
            if not match:
                continue
            for phase, marker in PHASE_MARKERS:
                if marker in line:
                    markers.append((datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc), phase))
                    break
    # Log events have second precision, so multiple phase boundaries can share
    # a timestamp. Preserve their order in the file instead of using the phase
    # name as a tie-breaker.
    markers.sort(key=lambda item: item[0])
    return markers


def summarize(samples: list[dict], markers: list[tuple[datetime, str]]) -> dict:
    by_phase = defaultdict(lambda: {"seconds": 0.0, "samples": []})
    intervals = list(zip(markers, markers[1:]))
    if markers and markers[-1][0] < samples[-1]["time"]:
        intervals.append((markers[-1], (samples[-1]["time"], markers[-1][1])))
    for (start, phase), (end, _) in intervals:
        lo = max(start, samples[0]["time"])
        hi = min(end, samples[-1]["time"])
        seconds = (hi - lo).total_seconds()
        if seconds <= 0:
            continue
        by_phase[phase]["seconds"] += seconds

    # Assign samples to the phase active at that timestamp. Phase changes are
    # the timestamped boundary event, so this also works for old CSV captures.
    phase_index = 0
    current_phase = None
    for sample in samples:
        while phase_index < len(markers) and markers[phase_index][0] <= sample["time"]:
            current_phase = markers[phase_index][1]
            phase_index += 1
        if current_phase:
            by_phase[current_phase]["samples"].append(sample)

    whole_seconds = (samples[-1]["time"] - samples[0]["time"]).total_seconds()
    total_gpu = sum(s["gpu"] for s in samples) / len(samples)
    phases = {}
    classified_seconds = sum(item["seconds"] for item in by_phase.values())
    for phase in ("rollout", "ppo_update", "evaluation", "other"):
        item = by_phase[phase]
        phase_samples = item["samples"]
        values = [s["gpu"] for s in phase_samples]
        phases[phase] = {
            "elapsed_seconds": round(item["seconds"], 3),
            "elapsed_pct": round(100 * item["seconds"] / classified_seconds, 2) if classified_seconds else None,
            "sample_count": len(values),
            "gpu_util_mean_pct": round(sum(values) / len(values), 2) if values else None,
            "gpu_util_median_pct": round(median(values), 2) if values else None,
            "gpu_util_at_least_50_pct_samples_pct": round(100 * sum(v >= 50 for v in values) / len(values), 2) if values else None,
            "memory_util_mean_pct": round(sum(s["memory_util"] for s in phase_samples) / len(phase_samples), 2) if phase_samples else None,
            "gpu_memory_mean_mib": round(sum(s["memory_mib"] for s in phase_samples) / len(phase_samples), 1) if phase_samples else None,
            "power_mean_w": round(sum(s["power_w"] for s in phase_samples) / len(phase_samples), 2) if phase_samples else None,
        }
    return {
        "capture_start_utc": samples[0]["time"].isoformat(),
        "capture_end_utc": samples[-1]["time"].isoformat(),
        "capture_elapsed_seconds": round(whole_seconds, 3),
        "sample_count": len(samples),
        "sample_interval_mean_seconds": round(whole_seconds / max(1, len(samples) - 1), 3),
        "rollout_starts_in_capture": sum(
            samples[0]["time"] <= t <= samples[-1]["time"] and phase == "rollout"
            for t, phase in markers
        ),
        "rollout_starts_per_hour": round(
            3600
            * sum(
                samples[0]["time"] <= t <= samples[-1]["time"] and phase == "rollout"
                for t, phase in markers
            )
            / whole_seconds,
            2,
        ) if whole_seconds else None,
        "gpu_util_mean_pct": round(total_gpu, 2),
        "gpu_util_median_pct": round(median(s["gpu"] for s in samples), 2),
        "memory_util_mean_pct": round(sum(s["memory_util"] for s in samples) / len(samples), 2),
        "gpu_memory_mean_mib": round(sum(s["memory_mib"] for s in samples) / len(samples), 1),
        "power_mean_w": round(sum(s["power_w"] for s in samples) / len(samples), 2),
        "phase_markers_in_capture": sum(samples[0]["time"] <= t <= samples[-1]["time"] for t, _ in markers),
        "phases": phases,
    }


def analyze(csv_path: Path, log_path: Path, output: Path | None) -> None:
    report = summarize(read_samples(csv_path), read_markers(log_path))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output:
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect", help="sample nvidia-smi once per second")
    collect_parser.add_argument("--seconds", type=float, default=3600)
    collect_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze", help="align a GPU CSV with the timestamped training log")
    analyze_parser.add_argument("--csv", type=Path, required=True)
    analyze_parser.add_argument("--log", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, args.seconds)
    else:
        analyze(args.csv, args.log, args.output)


if __name__ == "__main__":
    main()
