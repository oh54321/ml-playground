"""Trace saved action sampling and GAE evidence to actor probability changes.

This consumes the existing iteration-100 fixed-state PPO updates. It does not
run environments or optimizer steps. See the output report for estimands and
limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[3]
BALANCED_ROOT = HERE / "artifacts/ppo_2048/ppo_balanced_sampling_iter100"
OUTPUT_ROOT = HERE / "artifacts/ppo_2048/ppo_action_probability_audit_iter100"
CONFIRMED_IDS = (0, 1, 2, 3, 5)
PROBABILITY_TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_boards(reference_boards, audit_boards):
    """Match exact saved board and critic action; numeric IDs are insufficient."""
    result = {}
    for ref in reference_boards:
        ref_board = ref["board"]
        ref_action = int(ref["critic_action"])
        matches = [
            b for b in audit_boards
            if b["board"] == ref_board and int(b["critic_action"]) == ref_action
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one exact board/action match for reference id {ref.get('id')}; found {len(matches)}")
        match = matches[0]
        if int(match["id"]) in result.values():
            raise ValueError("Two reference pairs matched the same audit board")
        result[int(ref["id"])] = int(match["id"])
    return result


def direction(before: float, after: float, tolerance: float = PROBABILITY_TOLERANCE) -> str:
    delta = float(after) - float(before)
    if delta > tolerance:
        return "increased"
    if delta < -tolerance:
        return "decreased"
    return "unchanged"


def count_samples_by_board(samples, boards):
    """Reconstruct per-board action counts from manifest [state, action, trial] rows."""
    return [
        {str(a): sum(1 for state, action, _ in samples if int(state) == int(board["id"]) and int(action) == int(a)) for a in board["legal_actions"]}
        for board in boards
    ]


def summarize_advantages(values, mean, std):
    arr = np.asarray(values, dtype=np.float64)
    normalized = (arr - float(mean)) / (float(std) + 1e-8)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "positive_fraction": float(np.mean(arr > 0)) if arr.size else None,
        "normalized_mean_fixed_full_rollout": float(normalized.mean()) if arr.size else None,
        "normalized_positive_fraction_fixed_full_rollout": float(np.mean(normalized > 0)) if arr.size else None,
        "available": bool(arr.size),
    }


def _pool(root: Path, board_id: int, action: int):
    path = root / f"training_board_{board_id:02d}_action_{action}.json"
    records = json.loads(path.read_text())["records"]
    return records


def analyze(balanced_root: Path = BALANCED_ROOT, output_root: Path = OUTPUT_ROOT):
    balanced_root = balanced_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    original_selection = json.loads((balanced_root / "selection/selection.json").read_text())
    largegap_selection = json.loads((HERE / "artifacts/ppo_2048/ppo_large_delta_q_iter100/selection.json").read_text())
    replicate_data = json.loads((balanced_root / "conservative_sensitivity_replicates.json").read_text())
    original_replicates = json.loads((balanced_root / "replicates.json").read_text())
    draw_manifest = json.loads((balanced_root / "sensitivity_training_draw_manifest.json").read_text())
    mean = float(original_selection["advantage_mean"])
    std = float(original_selection["advantage_std"])
    boards = original_selection["boards"]
    ref_by_id = {int(b["id"]): b for b in largegap_selection["boards"]}
    map_ids = match_boards([ref_by_id[i] for i in CONFIRMED_IDS], boards)
    # Cross-check all six locked pairs; only five have the prior familywise-positive
    # evidence, but the remaining board stays in the exploratory output.
    all_map_ids = match_boards(largegap_selection["boards"], boards)
    if any(all_map_ids[k] != k for k in all_map_ids):
        raise ValueError(f"Locked six-board bank no longer shares expected IDs: {all_map_ids}")

    # The draw manifest's list/hash and counts must agree with the saved replicate
    # objects and source pool hashes before any advantage summary is emitted.
    draw_rows = {int(r["replicate"]): r for r in draw_manifest["replicates"]}
    if len(draw_rows) != 12 or len(replicate_data["replicates"]) != 12:
        raise ValueError("Expected exactly 12 saved paired update replicates")
    for board in boards:
        for action in board["legal_actions"]:
            filename = f"training_board_{board['id']:02d}_action_{action}.json"
            if sha256(balanced_root / filename) != draw_manifest["pool_sha256"][filename]:
                raise ValueError(f"Training pool hash mismatch: {filename}")
    for i, (rep, original_rep) in enumerate(zip(replicate_data["replicates"], original_replicates["replicates"])):
        if int(rep["replicate"]) != i or int(original_rep["replicate"]) != i:
            raise ValueError("Replicate order changed")
        for arm in ("natural", "balanced"):
            manifest_samples = draw_rows[i]["arms"][arm]["samples"]
            samples = [[int(s), int(a), int(t)] for s, a, t in manifest_samples]
            digest = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
            if samples != manifest_samples or digest != draw_rows[i]["arms"][arm]["sha256"]:
                raise ValueError(f"Training draw manifest mismatch for replicate {i} {arm}")
            reconstructed = count_samples_by_board(samples, boards)
            if reconstructed != original_rep["action_counts"][arm]:
                raise ValueError(f"Saved action counts mismatch for replicate {i} {arm}")
            if reconstructed != rep["action_counts"][arm]:
                raise ValueError(f"Sensitivity action counts mismatch for replicate {i} {arm}")

    old_behavior = np.asarray(original_replicates["source_behavior_probabilities"], dtype=np.float64)
    old_actor = np.asarray(original_replicates["source_base_policy_probabilities"], dtype=np.float64)
    epsilon = float(original_selection["exploration_probability"])
    if old_behavior.shape != (24, 4) or old_actor.shape != (24, 4):
        raise ValueError("Unexpected source probability array shape")
    expected_behavior = np.zeros_like(old_behavior)
    for i, board in enumerate(boards):
        legal = [int(a) for a in board["legal_actions"]]
        expected_behavior[i, legal] = old_actor[i, legal] * (1 - epsilon) + epsilon / len(legal)
    if not np.allclose(old_behavior, expected_behavior, atol=1e-7):
        raise ValueError("Saved behavior probabilities do not match legal-action actor/exploration mixture")

    rows = []
    for rep in replicate_data["replicates"]:
        rid = int(rep["replicate"])
        for arm in ("natural", "balanced"):
            details = rep[arm]
            after_behavior = np.asarray(details["behavior_probabilities"], dtype=np.float64)
            after_actor = np.asarray(details["base_policy_probabilities"], dtype=np.float64)
            for ref_id, audit_id in all_map_ids.items():
                board = boards[audit_id]
                legal = [int(a) for a in board["legal_actions"]]
                if not np.isclose(after_behavior[audit_id, legal].sum(), 1.0, atol=1e-10):
                    raise ValueError(f"Behavior probabilities fail legal-action sum check on board {audit_id}")
                if not np.isclose(after_actor[audit_id, legal].sum(), 1.0, atol=1e-10):
                    raise ValueError(f"Actor probabilities fail legal-action sum check on board {audit_id}")
                manifest_samples = draw_rows[rid]["arms"][arm]["samples"]
                by_action = {}
                for action in legal:
                    all_pool = _pool(balanced_root, audit_id, action)
                    sampled_indices = [int(trial) for state, sampled_action, trial in manifest_samples if int(state) == audit_id and int(sampled_action) == action]
                    sampled_values = [float(all_pool[j]["gae"]) for j in sampled_indices]
                    pool_values = [float(r["gae"]) for r in all_pool]
                    count = len(sampled_values)
                    weight = (1 / (24 * 32)) if arm == "natural" else (
                        float(old_behavior[audit_id, action]) / (24 * count) if count else None
                    )
                    adv = summarize_advantages(sampled_values, mean, std)
                    pool_adv = summarize_advantages(pool_values, mean, std)
                    by_action[str(action)] = {
                        "count": count,
                        "expected_natural_count": float(32 * old_behavior[audit_id, action]),
                        "sample_weight_per_trajectory": weight,
                        "sampled_advantage": adv,
                        "finite_training_pool_advantage_estimate": pool_adv,
                        "sampled_training_pool_trial_indices": sampled_indices,
                        "sampled_raw_advantages": sampled_values,
                        "sampled_normalized_advantages_fixed_full_rollout": [float((v - mean) / (std + 1e-8)) for v in sampled_values],
                        "weighted_normalized_advantage_sum": (float(sum((v - mean) / (std + 1e-8) for v in sampled_values) * weight) if count else None),
                        "mean_weighted_normalized_advantage_per_state": (float(sum((v - mean) / (std + 1e-8) for v in sampled_values) * weight * 24) if count else None),
                        "old_behavior_probability": float(old_behavior[audit_id, action]),
                        "old_actor_probability": float(old_actor[audit_id, action]),
                        "after_behavior_probability": float(after_behavior[audit_id, action]),
                        "after_actor_probability": float(after_actor[audit_id, action]),
                        "behavior_delta_probability_points": float((after_behavior[audit_id, action] - old_behavior[audit_id, action]) * 100),
                        "actor_delta_probability_points": float((after_actor[audit_id, action] - old_actor[audit_id, action]) * 100),
                        "behavior_direction": direction(old_behavior[audit_id, action], after_behavior[audit_id, action]),
                        "actor_direction": direction(old_actor[audit_id, action], after_actor[audit_id, action]),
                    }
                old_policy_weighted_pool_mean = sum(
                    float(old_behavior[audit_id, a]) * by_action[str(a)]["finite_training_pool_advantage_estimate"]["normalized_mean_fixed_full_rollout"]
                    for a in legal
                )
                for action in legal:
                    by_action[str(action)]["finite_pool_relative_to_old_behavior_weighted_legal_mean"] = float(
                        by_action[str(action)]["finite_training_pool_advantage_estimate"]["normalized_mean_fixed_full_rollout"]
                        - old_policy_weighted_pool_mean
                    )
                ref = ref_by_id[ref_id]
                target = int(ref["critic_action"])
                rows.append({
                    "replicate": rid,
                    "arm": arm,
                    "reference_board_id": ref_id,
                    "audit_board_id": audit_id,
                    "board": board["board"],
                    "target_action": target,
                    "actor_action": int(ref["actor_action"]),
                    "legal_actions": legal,
                    "confirmed_prior_return": (ref_id in CONFIRMED_IDS),
                    "target_action_details": by_action[str(target)],
                    "per_legal_action": by_action,
                    "behavior_legal_probability_sum_before": float(old_behavior[audit_id, legal].sum()),
                    "behavior_legal_probability_sum_after": float(after_behavior[audit_id, legal].sum()),
                    "training_kl": float(details["behavior_kl_from_old"]),
                })

    summary = summarize(rows, old_behavior, old_actor, map_ids)
    result = {
        "metadata": {
            "source": "saved conservative-LR paired actor-only updates; no new game simulations or optimizer updates",
            "primary_sensitivity_lr_divisor": replicate_data["settings"]["learning_rate_divisor"],
            "replicates": 12,
            "epochs_per_update": 8,
            "samples_per_state_arm": 32,
            "fixed_full_rollout_advantage_mean": mean,
            "fixed_full_rollout_advantage_std": std,
            "behavior_exploration_probability": epsilon,
            "probability_direction_tolerance": PROBABILITY_TOLERANCE,
            "board_matching": "exact 4x4 board and critic action across saved large-gap and balanced-sampling selections",
            "confirmed_reference_board_ids": list(CONFIRMED_IDS),
            "all_six_reference_to_sensitivity_bank_ids": all_map_ids,
            "training_draw_hashes_and_counts_verified": True,
            "training_pool_hashes_verified": True,
        },
        "summary": summary,
        "per_replicate_board_arm": rows,
    }
    (output_root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(output_root, result)
    return result


def summarize(rows, old_behavior, old_actor, board_map):
    summary = {"confirmed_boards": {}, "confirmed_subset": {}, "exploratory_boards": {}}
    for ref_id, audit_id in board_map.items():
        if ref_id not in CONFIRMED_IDS:
            continue
        board_rows = [r for r in rows if r["reference_board_id"] == ref_id]
        target = board_rows[0]["target_action"]
        per_arm = {}
        for arm in ("natural", "balanced"):
            current = [r for r in board_rows if r["arm"] == arm]
            changes = np.asarray([r["target_action_details"]["behavior_delta_probability_points"] for r in current])
            actor_changes = np.asarray([r["target_action_details"]["actor_delta_probability_points"] for r in current])
            counts = np.asarray([r["target_action_details"]["count"] for r in current], dtype=int)
            per_arm[arm] = {
                "target_action": target,
                "old_behavior_probability": current[0]["target_action_details"]["old_behavior_probability"],
                "old_actor_probability": current[0]["target_action_details"]["old_actor_probability"],
                "mean_behavior_delta_probability_points": float(changes.mean()),
                "range_behavior_delta_probability_points": [float(changes.min()), float(changes.max())],
                "behavior_direction_counts": {d: int(sum(r["target_action_details"]["behavior_direction"] == d for r in current)) for d in ("increased", "decreased", "unchanged")},
                "mean_actor_delta_probability_points": float(actor_changes.mean()),
                "range_actor_delta_probability_points": [float(actor_changes.min()), float(actor_changes.max())],
                "target_action_sample_count_per_replicate": {"mean": float(counts.mean()), "min": int(counts.min()), "max": int(counts.max()), "zero_count_replicates": int(np.sum(counts == 0))},
                "mean_natural_expected_count": float(current[0]["target_action_details"]["expected_natural_count"]),
                "sampled_mean_normalized_advantage": float(np.mean([r["target_action_details"]["sampled_advantage"]["normalized_mean_fixed_full_rollout"] for r in current if r["target_action_details"]["sampled_advantage"]["n"] > 0])) if any(r["target_action_details"]["sampled_advantage"]["n"] > 0 for r in current) else None,
                "sampled_positive_fraction_mean": float(np.mean([r["target_action_details"]["sampled_advantage"]["positive_fraction"] for r in current if r["target_action_details"]["sampled_advantage"]["n"] > 0])) if any(r["target_action_details"]["sampled_advantage"]["n"] > 0 for r in current) else None,
                "finite_pool_normalized_mean_advantage": current[0]["target_action_details"]["finite_training_pool_advantage_estimate"]["normalized_mean_fixed_full_rollout"],
                "finite_pool_advantage_relative_to_old_behavior_weighted_legal_mean": current[0]["target_action_details"]["finite_pool_relative_to_old_behavior_weighted_legal_mean"],
            }
        balanced_minus_natural = np.asarray([
            next(r for r in board_rows if r["replicate"] == rep and r["arm"] == "balanced")["target_action_details"]["behavior_delta_probability_points"] -
            next(r for r in board_rows if r["replicate"] == rep and r["arm"] == "natural")["target_action_details"]["behavior_delta_probability_points"]
            for rep in range(12)
        ])
        summary["confirmed_boards"][str(ref_id)] = {
            "audit_bank_board_id": audit_id,
            "target_action": target,
            "actor_action": board_rows[0]["actor_action"],
            "per_arm": per_arm,
            "paired_balanced_minus_natural_behavior_delta_probability_points": {
                "mean": float(balanced_minus_natural.mean()),
                "range": [float(balanced_minus_natural.min()), float(balanced_minus_natural.max())],
                "replicate_values": balanced_minus_natural.tolist(),
            },
        }
    for arm in ("natural", "balanced"):
        arm_rows = [r for r in rows if r["confirmed_prior_return"] and r["arm"] == arm]
        change_by_rep = {}
        for rep in range(12):
            change_by_rep[rep] = float(np.mean([r["target_action_details"]["behavior_delta_probability_points"] for r in arm_rows if r["replicate"] == rep]))
        direction_totals = {d: sum(v == d for v in [r["target_action_details"]["behavior_direction"] for r in arm_rows]) for d in ("increased", "decreased", "unchanged")}
        summary["confirmed_subset"][arm] = {
            "n_board_replicate_pairs": len(arm_rows),
            "mean_target_behavior_probability_delta_points": float(np.mean([r["target_action_details"]["behavior_delta_probability_points"] for r in arm_rows])),
            "direction_counts_over_board_replicates_descriptive": direction_totals,
            "mean_replicate_average_delta_points": float(np.mean(list(change_by_rep.values()))),
            "replicate_average_deltas_points": [change_by_rep[i] for i in range(12)],
        }
    for audit_id in range(24):
        if audit_id in CONFIRMED_IDS:
            continue
        board_rows = [r for r in rows if r["audit_board_id"] == audit_id]
        if not board_rows:
            continue
        target = board_rows[0]["target_action"]
        exploratory = {"target_action": target, "per_arm": {}}
        for arm in ("natural", "balanced"):
            current = [r for r in board_rows if r["arm"] == arm]
            changes = np.asarray([r["target_action_details"]["behavior_delta_probability_points"] for r in current])
            counts = np.asarray([r["target_action_details"]["count"] for r in current], dtype=int)
            exploratory["per_arm"][arm] = {
                "mean_behavior_delta_probability_points": float(changes.mean()),
                "range_behavior_delta_probability_points": [float(changes.min()), float(changes.max())],
                "behavior_direction_counts": {d: int(sum(r["target_action_details"]["behavior_direction"] == d for r in current)) for d in ("increased", "decreased", "unchanged")},
                "target_action_sample_count_per_replicate": {"mean": float(counts.mean()), "min": int(counts.min()), "max": int(counts.max()), "zero_count_replicates": int(np.sum(counts == 0))},
                "finite_pool_normalized_mean_advantage": current[0]["target_action_details"]["finite_training_pool_advantage_estimate"]["normalized_mean_fixed_full_rollout"],
                "finite_pool_advantage_relative_to_old_behavior_weighted_legal_mean": current[0]["target_action_details"]["finite_pool_relative_to_old_behavior_weighted_legal_mean"],
            }
        summary["exploratory_boards"][str(audit_id)] = exploratory
    return summary


def write_report(out: Path, result):
    summary = result["summary"]
    lines = [
        "# PPO action-probability audit at iteration 100",
        "",
        "## Why this audit",
        "",
        "Earlier forced-first-action simulations found critic-preferred moves with better frozen-policy continuations, even though the actor assigned them low probabilities. Balanced and ordinary sampling had similar average held-out gains. Those averages did not reveal whether the specific better actions were learned or whether sparse/noisy examples explained failures. This audit follows the saved first-action sample counts, the PPO advantage signal, and the actor's measured probability change.",
        "",
        "## Design and matching",
        "",
        "This reuses the existing 12 paired, eight-epoch actor-only updates at original learning rate divided by 128. The 12 updates resample a shared finite continuation pool; they are conditional optimizer-resampling replicates, not 12 independent training runs. The five primary pairs were fixed from the prior corrected frozen-policy analysis: exact board and critic-action matches for reference boards 0, 1, 2, 3, and 5. Board 4 is retained as exploratory because its familywise return interval did not confirm a positive effect. The other 18 states in the 24-board bank are also exploratory because they lack the prior forced-action confirmation. All six prior pairs were matched by full board and action, then checked against saved IDs. No new games were simulated.",
        "",
        "For each arm, board, and replicate, the audit reports before/after actor and behavior probabilities, legal-action probability sums, sample counts and source indices, raw and fixed-full-rollout-normalized GAE summaries, and the exact per-trajectory sample weights. Balanced weights are the saved old behavior probability divided by state count and action count. Natural weights are the saved equal sample weights. Missing samples are marked as unavailable; they are not treated as zero. The normalization uses the original full-rollout mean and standard deviation. The primary table uses behavior-probability changes; actor-softmax changes and full detail are in `results.json`.",
        "",
        "## Confirmed board results",
        "",
        "Changes are in probability points (one point = 0.01 probability). Direction counts are descriptive over 12 conditional update resamples per board.",
        "",
        "| Board | Better action | Old behavior p | Natural mean Δ (range) | Natural direction | Balanced mean Δ (range) | Balanced direction | Natural count / 32p | Balanced count | Target pool μnorm / relative to legal mean |",
        "|---:|---:|---:|---:|:---|---:|:---|---:|---:|:---|",
    ]
    for ref_id in CONFIRMED_IDS:
        b = summary["confirmed_boards"][str(ref_id)]
        n = b["per_arm"]["natural"]
        z = b["per_arm"]["balanced"]
        target_signal = f"{n['finite_pool_normalized_mean_advantage']:+.2f} / {n['finite_pool_advantage_relative_to_old_behavior_weighted_legal_mean']:+.2f} SD"
        lines.append(
            f"| {ref_id} | {b['target_action']} | {n['old_behavior_probability']:.4g} | "
            f"{n['mean_behavior_delta_probability_points']:+.5g} ({n['range_behavior_delta_probability_points'][0]:+.5g}, {n['range_behavior_delta_probability_points'][1]:+.5g}) | "
            f"↑{n['behavior_direction_counts']['increased']} ↓{n['behavior_direction_counts']['decreased']} ={n['behavior_direction_counts']['unchanged']} | "
            f"{z['mean_behavior_delta_probability_points']:+.5g} ({z['range_behavior_delta_probability_points'][0]:+.5g}, {z['range_behavior_delta_probability_points'][1]:+.5g}) | "
            f"↑{z['behavior_direction_counts']['increased']} ↓{z['behavior_direction_counts']['decreased']} ={z['behavior_direction_counts']['unchanged']} | "
            f"{n['target_action_sample_count_per_replicate']['mean']:.2f}/{n['mean_natural_expected_count']:.2f} | "
            f"{z['target_action_sample_count_per_replicate']['mean']:.2f} | "
            + target_signal
        )
    exploratory4 = summary["exploratory_boards"]["4"]
    lines += [
        "",
        f"Exploratory board 4 (critic-preferred action {exploratory4['target_action']}): natural mean behavior change {exploratory4['per_arm']['natural']['mean_behavior_delta_probability_points']:+.5g} points; balanced {exploratory4['per_arm']['balanced']['mean_behavior_delta_probability_points']:+.5g} points. This board remains exploratory because prior familywise return evidence did not confirm the forced action.",
    ]
    lines += ["", "Across the five confirmed states:"]
    for arm in ("natural", "balanced"):
        s = summary["confirmed_subset"][arm]
        lines.append(f"- {arm}: mean target-action behavior change {s['mean_target_behavior_probability_delta_points']:+.4f} points; direction counts across 60 board/update pairs: {s['direction_counts_over_board_replicates_descriptive']}; mean of the 12 replicate-level five-board averages {s['mean_replicate_average_delta_points']:+.4f} points.")
    lines += [
        "",
        "Per-board paired balanced-minus-natural mean changes are in the JSON artifact; they are conditional descriptive contrasts, not causal estimates of sampling. Individual update values can be small because the sensitivity learning rate is 1/128 of the original. Board 1 had a favorable action-relative pool estimate (+3.31 SD), but ordinary batches missed the action in 4/12 updates and its mean probability change was only +0.0017 points; balanced batches covered it every time, yet the direction split 5 increases to 7 decreases and the mean change remained +0.0021 points. Board 3's target was likewise covered in every balanced batch versus 2/12 ordinary batches, but its actor probability was about 0.00034% and the measured mean actor increase was only 0.0000021 points. These cases show better coverage alone did not ensure a visible actor change. Board 0 increased substantially under both arms even though the target action's average GAE was negative, which is why action-level signs must be read relative to other legal actions and the shared PPO objective.",
        "",
        "The per-action finite-pool advantage is a conditional estimate from that saved continuation pool, not an independent oracle. The relative column subtracts the old-behavior-weighted average normalized advantage across that board's legal actions. For balanced batches, raw pooled GAE alone is not the objective: PPO uses per-trajectory normalized advantages and the saved importance weights.",
        "",
        "## Interpretation and limits",
        "",
        "These results answer whether the already-saved small PPO updates changed the probability of the previously confirmed better first action, on the exact fixed states. They do not identify sampling as the cause of a change. Parameters are shared across all 24 states and actions, and Adam history, entropy, clipping, and gradient interactions can move an action indirectly, including when that action was not sampled. A favorable action-relative advantage estimate followed by a flat or falling probability is a reason to inspect those mechanisms, not proof of interference. Conversely, probability increases on these chosen states do not establish full-game gains or population-wide improvement.",
        "",
        "All uncertainty is conditional on the five selected states, the common saved pools, and 12 resamples. Direction totals are not independent game counts. Board selection used critic gaps, so the subset is selected and does not estimate how often such states occur. The other 19 states remain available in the per-replicate JSON and exploratory summaries, without treating critic argmax as automatically correct. The base actor and the behavior mixture are both recorded so exploration epsilon is not confused with the learned actor probability.",
        "",
        "## Reproducibility",
        "",
        "Run `python -m games.rl_2048.scripts.audit_ppo_action_probability`. It verifies checkpoint-derived pool hashes, draw-manifest hashes, exact reconstructed sample indices, and saved counts before producing this report and `results.json`. The script reads the saved JSON artifacts and performs no training or game simulation.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-root", type=Path, default=BALANCED_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    analyze(args.balanced_root, args.output_root)
    print(f"Wrote action-probability audit to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
