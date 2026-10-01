"""Verify locked frozen-controller outputs and compute paired game-level stats."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from games.rl_2048.scripts.evaluate_ppo_controllers import (
    EXPECTED_SOURCE_SHA, OUT, ROOT, canonical_sha, load_inputs, sha256, write_json,
)


CONTROLLERS = ("sampled_actor", "greedy_actor", "greedy_critic")
BOOTSTRAPS = 20_000


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for McNemar discordant pairs."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, statistic: str = "mean") -> list[float]:
    n = len(values)
    if statistic == "mean":
        reps = np.empty(BOOTSTRAPS, dtype=np.float64)
        for start in range(0, BOOTSTRAPS, 256):
            size = min(256, BOOTSTRAPS - start)
            indexes = rng.integers(0, n, size=(size, n))
            reps[start:start + size] = values[indexes].mean(axis=1)
    elif statistic == "median":
        reps = np.empty(BOOTSTRAPS, dtype=np.float64)
        for start in range(0, BOOTSTRAPS, 256):
            size = min(256, BOOTSTRAPS - start)
            indexes = rng.integers(0, n, size=(size, n))
            reps[start:start + size] = np.median(values[indexes], axis=1)
    else:
        raise ValueError(statistic)
    return [float(x) for x in np.quantile(reps, [0.025, 0.975], method="linear")]


def _paired(rows_a: list[dict], rows_b: list[dict], rng: np.random.Generator) -> dict:
    by_a = {int(row["environment_seed"]): row for row in rows_a}
    by_b = {int(row["environment_seed"]): row for row in rows_b}
    if len(by_a) != len(rows_a) or by_a.keys() != by_b.keys():
        raise ValueError("Controllers do not have the same unique environment seeds")
    seeds = sorted(by_a)
    a, b = [by_a[s] for s in seeds], [by_b[s] for s in seeds]
    for row_a, row_b in zip(a, b, strict=True):
        if row_a["initial_board"] != row_b["initial_board"]:
            raise ValueError("Paired controllers do not share the same initial board")
        if not row_a["terminated"] or not row_b["terminated"]:
            raise ValueError("Every main comparison game must terminate naturally")
    reach_a = np.asarray([r["reached_2048"] for r in a], dtype=np.int8)
    reach_b = np.asarray([r["reached_2048"] for r in b], dtype=np.int8)
    reach_diff = reach_a - reach_b
    a_wins = int(np.sum((reach_a == 1) & (reach_b == 0)))
    b_wins = int(np.sum((reach_a == 0) & (reach_b == 1)))
    score_diff = np.asarray([x["score"] - y["score"] for x, y in zip(a, b, strict=True)], dtype=np.float64)
    score_a = np.asarray([r["score"] for r in a], dtype=np.float64)
    score_b = np.asarray([r["score"] for r in b], dtype=np.float64)
    tile_diff = np.asarray([math.log2(x["max_tile"]) - math.log2(y["max_tile"]) for x, y in zip(a, b, strict=True)], dtype=np.float64)
    # Use identical seed-resample indices for each statistic in this contrast.
    index_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
    reach_ci = _bootstrap_interval(reach_diff.astype(np.float64), index_rng, "mean")
    score_mean_ci = _bootstrap_interval(score_diff, index_rng, "mean")
    # Paired bootstrap interval for the difference between controller medians.
    n = len(a)
    median_reps = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 256):
        size = min(256, BOOTSTRAPS - start)
        idx = index_rng.integers(0, n, size=(size, n))
        median_reps[start:start + size] = np.median(score_a[idx], axis=1) - np.median(score_b[idx], axis=1)
    score_median_ci = [float(x) for x in np.quantile(median_reps, [0.025, 0.975], method="linear")]
    tile_ci = _bootstrap_interval(tile_diff, index_rng, "mean")
    return {
        "n_paired_games": n,
        "reach_rate_a": float(reach_a.mean()), "reach_rate_b": float(reach_b.mean()),
        "reach_difference_a_minus_b": float(reach_diff.mean()), "reach_difference_ci95": reach_ci,
        "reach_discordant_a_reached_b_not": a_wins, "reach_discordant_b_reached_a_not": b_wins,
        "mcnemar_exact_p_two_sided": _binom_two_sided(a_wins, a_wins + b_wins),
        "mean_score_a": float(score_a.mean()), "mean_score_b": float(score_b.mean()),
        "mean_score_difference_a_minus_b": float(score_diff.mean()), "mean_score_difference_ci95": score_mean_ci,
        "median_score_a": float(np.median(score_a)), "median_score_b": float(np.median(score_b)),
        "median_score_difference_a_minus_b": float(np.median(score_a) - np.median(score_b)),
        "median_score_difference_ci95": score_median_ci,
        "mean_log2_max_tile_a": float(np.log2([r["max_tile"] for r in a]).mean()),
        "mean_log2_max_tile_b": float(np.log2([r["max_tile"] for r in b]).mean()),
        "mean_log2_max_tile_difference_a_minus_b": float(tile_diff.mean()),
        "mean_log2_max_tile_difference_ci95": tile_ci,
        "max_tile_counts_a": {str(tile): int(sum(r["max_tile"] == tile for r in a)) for tile in sorted({r["max_tile"] for r in a})},
        "max_tile_counts_b": {str(tile): int(sum(r["max_tile"] == tile for r in b)) for tile in sorted({r["max_tile"] for r in b})},
        "scores_a": [int(r["score"]) for r in a], "scores_b": [int(r["score"]) for r in b],
        "seed_rows": [{"environment_seed": s, "reached_a": int(x["reached_2048"]), "reached_b": int(y["reached_2048"]), "score_a": int(x["score"]), "score_b": int(y["score"]), "max_tile_a": int(x["max_tile"]), "max_tile_b": int(y["max_tile"])} for s, x, y in zip(seeds, a, b, strict=True)],
    }


def summarize(args) -> None:
    root = args.audit
    plan, manifest = load_inputs(root)
    inputs = {}
    for controller in CONTROLLERS:
        path = (root / f"{controller}_results.json") if controller == "greedy_critic" else (root / "actor_baselines" / f"{controller}_results.json")
        data = json.loads(path.read_text())
        if data.get("protocol_sha256") != plan["protocol_sha256"] or data.get("seed_manifest_sha256") != manifest["seed_manifest_sha256"]:
            raise ValueError(f"{controller}: protocol/seed-manifest mismatch")
        is_actor = controller != "greedy_critic"
        row_count = data.get("summary", {}).get("games") if is_actor else data.get("games")
        if data.get("checkpoint_sha256") != EXPECTED_SOURCE_SHA or row_count != 256:
            raise ValueError(f"{controller}: checkpoint or count mismatch")
        rows = data.get("games") if is_actor else data.get("rows")
        if rows is None or len(rows) != 256:
            raise ValueError(f"{controller}: missing per-game rows")
        if is_actor:
            rows = [dict(row, environment_seed=row["game_seed"]) for row in rows]
        rows = [dict(row, moves=row.get("moves", row.get("steps"))) for row in rows]
        expected_seeds = set(manifest["environment_seeds"])
        if {int(row["environment_seed"]) for row in rows} != expected_seeds:
            raise ValueError(f"{controller}: game seed set differs from the lock")
        if data.get("model_state_immutable", data.get("model_state_unchanged")) is not True:
            raise ValueError(f"{controller}: model state changed")
        elapsed = data.get("summary", {}).get("wall_runtime_seconds") if is_actor else data.get("elapsed_seconds")
        data = dict(data, rows=rows, elapsed_seconds=elapsed, games_per_second=256 / elapsed)
        inputs[controller] = {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path), "data": data}
    rng = np.random.default_rng(int(plan["design"]["analysis_seed"]))
    contrasts = {
        "primary_greedy_critic_minus_greedy_actor": _paired(inputs["greedy_critic"]["data"]["rows"], inputs["greedy_actor"]["data"]["rows"], rng),
        "exploratory_sampled_actor_minus_greedy_actor": _paired(inputs["sampled_actor"]["data"]["rows"], inputs["greedy_actor"]["data"]["rows"], rng),
        "exploratory_sampled_actor_minus_greedy_critic": _paired(inputs["sampled_actor"]["data"]["rows"], inputs["greedy_critic"]["data"]["rows"], rng),
    }
    out = {
        "protocol_sha256": plan["protocol_sha256"], "seed_manifest_sha256": manifest["seed_manifest_sha256"],
        "checkpoint_sha256": EXPECTED_SOURCE_SHA, "games_per_controller": 256,
        "controllers": {name: {"results_path": item["path"], "results_sha256": item["sha256"],
                                "wall_runtime_seconds": item["data"]["elapsed_seconds"],
                                "games_per_second": item["data"]["games_per_second"],
                                "mean_score": float(np.mean([r["score"] for r in item["data"]["rows"]])),
                                "median_score": float(np.median([r["score"] for r in item["data"]["rows"]])),
                                "score_percentiles_25_75": [float(x) for x in np.quantile([r["score"] for r in item["data"]["rows"]], [0.25, 0.75])],
                                "mean_moves": float(np.mean([r["moves"] for r in item["data"]["rows"]])),
                                "median_moves": float(np.median([r["moves"] for r in item["data"]["rows"]])),
                                "reached_2048_count": int(sum(r["reached_2048"] for r in item["data"]["rows"])),
                                "max_tile_counts": {str(tile): int(sum(r["max_tile"] == tile for r in item["data"]["rows"])) for tile in sorted({r["max_tile"] for r in item["data"]["rows"]})}}
                       for name, item in inputs.items()},
        "contrasts": contrasts,
        "limits": ["one frozen checkpoint and one evaluation seed distribution", "greedy critic has extra one-step value lookahead computation; this is controller comparison, not equal-compute policy quality", "continuation trajectories diverge so common environment seed pairs initial/spawn RNG streams, not identical realized future tile placements", "paired bootstrap is conditional on 256 seeds; secondary controller contrasts are exploratory and not multiplicity adjusted", "the pilot is excluded and has no outcome fields"],
    }
    write_json(root / "summary.json", out)
    print(json.dumps({k: contrasts["primary_greedy_critic_minus_greedy_actor"][k] for k in ("reach_rate_a", "reach_rate_b", "reach_difference_a_minus_b", "reach_difference_ci95", "reach_discordant_a_reached_b_not", "reach_discordant_b_reached_a_not", "mcnemar_exact_p_two_sided", "mean_score_difference_a_minus_b", "mean_score_difference_ci95")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=OUT)
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
