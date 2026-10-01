"""Analyze paired held-out forced-first-action pools for the locked 1000 bank."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/ppo_2048/ppo_longer_run_post_refactor_iter1000_audit")


def raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    ROOT = parser.parse_args().audit
    plan = json.loads((ROOT / "locked_plan.json").read_text())
    protocol = json.loads((ROOT / "prebranch_protocol_v1.json").read_text())
    protocol_sha = raw_sha(ROOT / "prebranch_protocol_v1.json")
    pool_manifest = json.loads((ROOT / "heldout_branch_pool_manifest.json").read_text())
    if (pool_manifest.get("protocol_sha256") != protocol_sha
            or pool_manifest.get("locked_plan_sha256") != plan["lock_sha256"]
            or pool_manifest.get("checkpoint_sha256") != plan["checkpoint_sha256"]):
        raise ValueError("held-out pool manifest does not match this protocol/bank/checkpoint")
    expected_trials = int(protocol["branch_rng_pairing"]["trials_per_legal_action_per_pool"])
    expected_names = {
        f"heldout_board_{b['id']:02d}_action_{action}.json"
        for b in plan["boards"] for action in b["legal_actions"]
    }
    hashes = pool_manifest["pool_sha256"]
    if set(hashes) != expected_names or pool_manifest.get("pool_file_count") != len(expected_names):
        raise ValueError("held-out manifest file list does not match locked legal action pools")
    if pool_manifest.get("settings", {}).get("protocol_sha256") != protocol_sha:
        raise ValueError("held-out settings do not match the locked protocol")
    if pool_manifest.get("settings", {}).get("training_per_state_action") != expected_trials or pool_manifest.get("settings", {}).get("heldout_per_state_action") != expected_trials:
        raise ValueError("held-out manifest trial count differs from the locked protocol")
    uncertainty = protocol["heldout_analysis"]["uncertainty"]
    SEED_MAXT = int(uncertainty["statewise_seed"])
    SEED_CLUSTER = int(uncertainty["episode_cluster_seed"])
    BOOTSTRAPS = int(uncertainty["draws"])
    states = plan["boards"]
    returns: list[dict[int, np.ndarray]] = []
    rows = []
    for state in states:
        i = state["id"]
        means = {}
        by_action = {}
        for a in state["legal_actions"]:
            p = ROOT / f"heldout_board_{i:02d}_action_{a}.json"
            payload = json.loads(p.read_text())
            if raw_sha(p) != hashes[p.name]:
                raise ValueError(f"held-out pool hash mismatch: {p.name}")
            if (payload["count"] != expected_trials or payload["board_id"] != i
                    or payload["action"] != a or payload.get("rng_namespace") != protocol["branch_rng_pairing"]["heldout_namespace"]):
                raise ValueError(f"held-out pool metadata mismatch: {p.name}")
            arr = np.array([r["game_score"] for r in payload["records"]], dtype=np.float64)
            if len(arr) != expected_trials:
                raise ValueError(f"held-out pool record count mismatch: {p.name}")
            by_action[a] = arr
            means[a] = float(arr.mean())
        returns.append(by_action)
        target, preferred = state["critic_action"], state["actor_action"]
        paired = by_action[target] - by_action[preferred]
        rows.append({
            "id": i,
            "source_episode_id": state["source_episode_id"],
            "legal_actions": state["legal_actions"],
            "actor_preferred_action": preferred,
            "critic_target_action": target,
            "target_base_actor_probability": state["actor_probabilities"][target],
            "target_behavior_probability": state["behavior_probabilities"][target],
            "gap_stratum": state["gap_stratum"],
            "probability_stratum": state["probability_stratum"],
            "action_mean_return": {str(a): means[a] for a in means},
            "paired_target_minus_preferred_mean": float(paired.mean()),
            "paired_target_minus_preferred_sd": float(paired.std(ddof=1)),
            "paired_target_minus_preferred_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
            "paired_target_minus_preferred_trial_differences": paired.tolist(),
        })

    observed = np.array([r["paired_target_minus_preferred_mean"] for r in rows])
    ses = np.array([r["paired_target_minus_preferred_se"] for r in rows])
    n = len(rows)
    rng = np.random.default_rng(SEED_MAXT)
    max_abs_t = np.empty(BOOTSTRAPS)
    # Paired trial resampling: a single trial index is reused across every action
    # within a board, preserving the locked common-random-number pairing.
    for b in range(BOOTSTRAPS):
        draws = rng.integers(0, expected_trials, size=(n, expected_trials))
        means_b = np.array([np.mean(np.array(r["paired_target_minus_preferred_trial_differences"])[draws[j]]) for j, r in enumerate(rows)])
        max_abs_t[b] = np.max(np.abs((means_b - observed) / ses))
    critical = float(np.quantile(max_abs_t, 0.95, method="higher"))
    lower = observed - critical * ses
    upper = observed + critical * ses

    # Episode-cluster bootstrap: draw actual rollout episodes, include all selected
    # states in each sampled episode, resample each state's paired trial rows, then
    # give each included state equal weight in that draw.
    episode_to_ids: dict[str, list[int]] = {}
    for j, r in enumerate(rows):
        episode_to_ids.setdefault(r["source_episode_id"], []).append(j)
    episodes = sorted(episode_to_ids)
    rng = np.random.default_rng(SEED_CLUSTER)
    cluster_means = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        state_means = []
        for ep in sampled:
            for j in episode_to_ids[ep]:
                d = np.array(rows[j]["paired_target_minus_preferred_trial_differences"])
                state_means.append(float(d[rng.integers(0, expected_trials, size=expected_trials)].mean()))
        cluster_means[b] = np.mean(state_means)

    # Full policy-local held-out value alignment uses the held-out action means as
    # fixed per-state Q estimates, not post-update game outcomes.
    bank_mean = float(np.mean(observed))
    output = {
        "status": "complete",
        "estimand": "equal-state mean heldout game-score difference for critic-Q-preferred first action minus actor-preferred first action, conditional on this locked bank and frozen continuation policy",
        "n_states": n,
        "n_actual_source_episodes": len(episodes),
        "trials_per_legal_action": int(protocol["branch_rng_pairing"]["trials_per_legal_action_per_pool"]),
        "paired_max_t": {
            "bootstrap_seed": SEED_MAXT,
            "bootstrap_draws": BOOTSTRAPS,
            "critical_95": critical,
            "observed_bank_mean": bank_mean,
            "equal_state_se": float(np.sqrt(np.mean(ses ** 2) / n)),
            "positive_interval_count": int(np.sum(lower > 0)),
            "negative_interval_count": int(np.sum(upper < 0)),
            "intervals_spanning_zero": int(np.sum((lower <= 0) & (upper >= 0))),
            "mean_ci_familywise_lower": float(bank_mean - critical * np.sqrt(np.mean(ses ** 2) / n)),
            "mean_ci_familywise_upper": float(bank_mean + critical * np.sqrt(np.mean(ses ** 2) / n)),
            "rows": [dict(r, familywise_lower=float(lower[j]), familywise_upper=float(upper[j])) for j, r in enumerate(rows)],
        },
        "episode_cluster_bootstrap": {
            "seed": SEED_CLUSTER,
            "draws": BOOTSTRAPS,
            "unit": f"{len(episodes)} actual source episodes; include all selected states in sampled episodes; within-state paired trial resampling; equal state mean per bootstrap draw",
            "lower_95": float(np.quantile(cluster_means, .025)),
            "upper_95": float(np.quantile(cluster_means, .975)),
        },
        "provenance": {
            "protocol_sha256": protocol_sha,
            "locked_plan_sha256": plan["lock_sha256"],
            "heldout_pool_manifest_sha256": raw_sha(ROOT / "heldout_branch_pool_manifest.json"),
            "source_checkpoint_sha256": protocol["checkpoint_sha256"],
        },
    }
    (ROOT / "heldout_action_outcomes.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"heldout bank mean={bank_mean:.6f}; max-t critical={critical:.6f}; intervals +/−/zero={output['paired_max_t']['positive_interval_count']}/{output['paired_max_t']['negative_interval_count']}/{output['paired_max_t']['intervals_spanning_zero']}; cluster 95%=[{output['episode_cluster_bootstrap']['lower_95']:.3f},{output['episode_cluster_bootstrap']['upper_95']:.3f}]")


if __name__ == "__main__":
    main()
