"""Summarize locked iteration-1000 return, PPO update, and coverage outputs."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/ppo_2048/ppo_longer_run_post_refactor_iter1000_audit")
N_BOOT = 5000
SEED = 20482215
SEED_MATCHED = 20482216


def episode_cluster_replicate_ci(matrix: np.ndarray, episode_ids: list[str], seed: int) -> dict:
    """Matrix is states × shared update replicates; resample both dimensions."""
    by_episode: dict[str, list[int]] = {}
    for i, episode in enumerate(episode_ids):
        by_episode.setdefault(episode, []).append(i)
    episodes = sorted(by_episode)
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        reps = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draw_eps = rng.choice(episodes, size=len(episodes), replace=True)
        states = [i for ep in draw_eps for i in by_episode[str(ep)]]
        means[b] = matrix[np.ix_(states, reps)].mean()
    return {"lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975))}


def _metric_matrix(updates, arm, field, project=None):
    out = np.empty((96, 12), dtype=np.float64)
    for rep in range(12):
        u = updates["replicates"][rep]["arms"][arm]
        for sid, row in enumerate(u["final"]):
            if field == "heldout_value_change":
                board = row["actor_probabilities"]
                init = u["initial"][sid]["actor_probabilities"]
                out[sid, rep] = sum((board[a] - init[a]) * project[sid][a] for a in project[sid])
            else:
                out[sid, rep] = row[field]
    return out


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    ROOT = parser.parse_args().audit
    plan = json.loads((ROOT / "locked_plan.json").read_text())
    protocol = json.loads((ROOT / "prebranch_protocol_v1.json").read_text())
    uncertainty = protocol["heldout_analysis"]["uncertainty"]
    N_BOOT = int(uncertainty["draws"])
    SEED = int(uncertainty["paired_update_seed"])
    SEED_MATCHED = int(uncertainty["matched_seed"])
    episode_ids = [b["source_episode_id"] for b in plan["boards"]]
    outcome = json.loads((ROOT / "heldout_action_outcomes.json").read_text())
    heldout_rows = outcome["paired_max_t"]["rows"]
    supported = [int(r["id"]) for r in heldout_rows if r["familywise_lower"] > 0]
    updates = json.loads((ROOT / "paired_natural_balanced_updates.json").read_text())
    heldout_means = [{int(a): float(v) for a, v in r["action_mean_return"].items()} for r in heldout_rows]
    arm_summary = {}
    matrices = {}
    for arm in ("natural", "balanced"):
        logodds = _metric_matrix(updates, arm, "target_preferred_log_odds_change")
        targetp = _metric_matrix(updates, arm, "target_actor_probability_change")
        value = _metric_matrix(updates, arm, "heldout_value_change", heldout_means)
        matrices[arm] = {"logodds": logodds, "targetp": targetp, "value": value}
        ep_first, ep_last, cos, kls = [], [], [], []
        for rep in range(12):
            u = updates["replicates"][rep]["arms"][arm]
            initial = [r["target_actor_probability"] for r in u["initial"]]
            first = np.asarray(u["epochs"][0]["target_actor_probabilities"]) - np.asarray(initial)
            last = np.asarray(u["epochs"][-1]["target_actor_probabilities"]) - np.asarray(initial)
            ep_first.extend(first.tolist()); ep_last.extend(last.tolist())
            cos.extend(e["adam_displacement_cosine_with_negative_total_objective_gradient"] for e in u["epochs"])
            kls.extend(e["exact_forward_kl_old_to_current"] for e in u["epochs"])
        arm_summary[arm] = {
            "mean_target_preferred_logodds_change": float(logodds.mean()),
            "target_preferred_logodds_positive_negative": [int((logodds > 1e-10).sum()), int((logodds < -1e-10).sum())],
            "target_behavior_probability_mean_change": float(targetp.mean()),
            "target_behavior_probability_positive_negative": [int((targetp > 1e-10).sum()), int((targetp < -1e-10).sum())],
            "episode_cluster_replicate_ci_logodds": episode_cluster_replicate_ci(logodds, episode_ids, SEED),
            "episode_cluster_replicate_ci_heldout_value_alignment": episode_cluster_replicate_ci(value, episode_ids, SEED),
            "mean_heldout_value_alignment_points": float(value.mean()),
            "one_step_target_probability_positive_negative": [int((np.asarray(ep_first) > 1e-10).sum()), int((np.asarray(ep_first) < -1e-10).sum())],
            "eight_step_target_probability_positive_negative": [int((np.asarray(ep_last) > 1e-10).sum()), int((np.asarray(ep_last) < -1e-10).sum())],
            "first_to_eighth_step_target_probability_sign_reversals": int(np.sum(((np.asarray(ep_first)>1e-10)&(np.asarray(ep_last)<-1e-10))|((np.asarray(ep_first)<-1e-10)&(np.asarray(ep_last)>1e-10)))),
            "mean_cosine_actual_Adam_displacement_vs_negative_total_objective_gradient": float(np.mean(cos)),
            "mean_exact_forward_KL_over_all_steps": float(np.mean(kls)),
            "maximum_probability_reconstruction_error": max(updates["replicates"][rep]["arms"][arm]["maximum_saved_vs_reconstructed_behavior_probability_error"] for rep in range(12)),
        }
    difference = matrices["balanced"]["logodds"] - matrices["natural"]["logodds"]
    arm_summary["balanced_minus_natural_logodds"] = {
        "mean": float(difference.mean()),
        "episode_cluster_replicate_ci": episode_cluster_replicate_ci(difference, episode_ids, SEED),
    }
    value_difference = matrices["balanced"]["value"] - matrices["natural"]["value"]
    arm_summary["balanced_minus_natural_heldout_value_alignment"] = {
        "mean_points": float(value_difference.mean()),
        "episode_cluster_replicate_ci": episode_cluster_replicate_ci(value_difference, episode_ids, SEED),
    }

    matched = json.loads((ROOT / "matched_action_coverage_results.json").read_text())
    rows = matched["rows"]
    fields = ["paired_replacement_minus_baseline_focal_logodds", "paired_replacement_minus_baseline_bank_mean_logodds",
              "paired_replacement_minus_baseline_focal_heldout_value_alignment", "paired_replacement_minus_baseline_bank_mean_heldout_value_alignment"]
    matched_summary = {}
    for field in fields:
        m = np.empty((96, 12), dtype=np.float64)
        for row in rows:
            m[int(row["focal_state"]), int(row["replicate"])] = float(row[field])
        matched_summary[field] = {"mean": float(m.mean()), "positive_negative": [int((m > 0).sum()), int((m < 0).sum())],
                                  "episode_cluster_replicate_ci": episode_cluster_replicate_ci(m, episode_ids, SEED_MATCHED)}
    missing = [r for r in rows if r["critic_target_missing"]]
    present = [r for r in rows if not r["critic_target_missing"]]
    matched_summary["coverage_descriptive"] = {
        "critic_target_missing_rows": len(missing), "all_rows": len(rows),
        "target_missing_mean_paired_logodds_change": float(np.mean([r["paired_replacement_minus_baseline_focal_logodds"] for r in missing])),
        "target_covered_mean_paired_logodds_change": float(np.mean([r["paired_replacement_minus_baseline_focal_logodds"] for r in present])),
        "interpretation": "Descriptive strata of the natural draw; replacement changes every focal action contribution to its all-action finite-pool reference, not only missing-target cases.",
    }
    support_rows = [r for r in rows if int(r["focal_state"]) in supported]
    own = json.loads((ROOT / "gradient_decomposition.json").read_text())
    own_by_id = {int(r["id"]): r for r in own["results"]}
    plan_by_id = {int(b["id"]): b for b in plan["boards"]}
    mechanisms = {"supported_states": supported, "count": len(supported),
                  "own_target_direction_positive": sum(own_by_id[i]["own_state"]["target_behavior_probability_directional_derivative"] > 0 for i in supported),
                  "own_target_logodds_positive": sum(own_by_id[i]["own_state"]["target_base_logodds_directional_derivative"] > 0 for i in supported),
                  "other_states_opposing_positive_own": sum(own_by_id[i]["own_state"]["target_behavior_probability_directional_derivative"] > 0 and own_by_id[i]["other_states"]["target_behavior_probability_projection"] < 0 for i in supported),
                  "other_states_reverse_own_total_sign": 0,
                  "focal_intervention_rows": len(support_rows),
                  "focal_intervention_mean_logodds_effect": float(np.mean([r["paired_replacement_minus_baseline_focal_logodds"] for r in support_rows]))}
    for i in supported:
        own_v = own_by_id[i]["own_state"]["target_behavior_probability_directional_derivative"]
        other_v = own_by_id[i]["other_states"]["target_behavior_probability_projection"]
        mechanisms["other_states_reverse_own_total_sign"] += int(own_v > 0 and own_v + other_v < 0)

    output = {
        "protocol_sha256": updates["protocol_sha256"], "locked_plan_sha256": plan["lock_sha256"],
        "paired_update_bootstrap": {"draws": N_BOOT, "seed": SEED, "unit": f"global resample of 12 shared reps and {len(set(episode_ids))} actual source episodes; equal state weighting"},
        "matched_bootstrap": {"draws": N_BOOT, "seed": SEED_MATCHED, "unit": f"global resample of 12 shared reps and {len(set(episode_ids))} actual source episodes; equal state weighting"},
        "critic_supported_state_ids_from_paired_max_t": supported,
        "update_summary": arm_summary, "matched_summary": matched_summary,
        "supported_subset_mechanisms": mechanisms,
    }
    (ROOT / "analysis_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"supported": supported, "update": arm_summary, "matched": matched_summary, "mechanisms": mechanisms}, indent=2))


if __name__ == "__main__":
    main()
