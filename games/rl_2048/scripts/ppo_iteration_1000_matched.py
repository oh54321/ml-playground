"""Paired one-state full-action gradient replacement on the locked 96-state bank."""
from __future__ import annotations

import copy
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from games.rl_2048.environment.game_2048 import State2048
from games.rl_2048.scripts.audit_ppo_update import load_runtime, sha256, tensor_tree_sha256, write_json
from games.rl_2048.scripts.balanced_sampling_ppo import clipped_policy_loss, _read_pool
from games.rl_2048.scripts.ppo_iteration_1000_updates import (
    DEFAULT_AUDIT, PROTOCOL_SHA, _legal_masks, _probability_metrics, _read_inputs,
)


def _validate_pool_manifest(audit, kind, plan, protocol, protocol_sha):
    # Phase-B v3 stores immutable per-kind pools under training/ and heldout/
    # and records basename hashes in their raw manifests. The older matched
    # reader uses root-level aliases, so validate the v3 source first and then
    # verify/create only deterministic symlinks to those exact files.
    raw_path = audit / f"{kind}_manifest.json"
    raw = json.loads(raw_path.read_text())
    continuation_path = audit.parent / "continuation_protocol_parallel_v3.json"
    continuation = json.loads(continuation_path.read_text())
    continuation_copy = dict(continuation)
    claimed_continuation_digest = continuation_copy.pop("protocol_sha256")
    calculated_continuation_digest = hashlib.sha256(
        json.dumps(continuation_copy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed_continuation_digest != calculated_continuation_digest or protocol.get("continuation_protocol_sha256") != claimed_continuation_digest:
        raise ValueError("pool continuation protocol digest mismatch")
    expected_source_names = {f"board_{b['id']:02d}_action_{int(a)}.json"
                             for b in plan["boards"] for a in b["legal_actions"]}
    expected_names = {f"{kind}_{name}" for name in expected_source_names}
    raw_pool_dir = audit / kind
    trials = int(continuation["trials_per_action"][kind])
    namespace = int(continuation["rng_namespaces"][kind])
    if (raw.get("pool_kind") != kind
            or raw.get("checkpoint_sha256") != plan["checkpoint_sha256"]
            or raw.get("capture_sha256") != plan["capture_sha256"]
            or raw.get("bank_sha256") != plan["selected_bank_sha256"]
            or raw.get("bank_protocol_sha256") != continuation.get("bank_protocol_sha256")
            or raw.get("protocol_sha256") != (
                continuation.get("training_pools_precollected_under_protocol_sha256")
                if kind == "training" else claimed_continuation_digest)
            or raw.get("plan_sha256") != continuation.get("plan_sha256")
            or raw.get("pool_file_count") != len(expected_source_names)
            or set(raw.get("pool_sha256", {})) != expected_source_names
            or raw.get("trajectory_count") != len(expected_source_names) * trials
            or raw.get("trials_per_action") != trials
            or raw.get("rng_seed") != continuation["rng_seed"]
            or raw.get("rng_namespace") != namespace
            or raw.get("max_episode_steps") != continuation["max_episode_steps"]):
        raise ValueError(f"{kind} v3 manifest provenance or expected file list mismatch")
    alias_hashes = {}
    for source_name, expected_hash in raw["pool_sha256"].items():
        source_path = raw_pool_dir / source_name
        if sha256(source_path) != expected_hash:
            raise ValueError(f"{kind} v3 pool file hash mismatch: {source_name}")
        payload = json.loads(source_path.read_text())
        if (payload.get("count") != trials or len(payload.get("records", [])) != trials
                or payload.get("rng_namespace") != namespace):
            raise ValueError(f"{kind} v3 pool metadata mismatch: {source_name}")
        alias_name = f"{kind}_{source_name}"
        alias = audit / alias_name
        if alias.exists() or alias.is_symlink():
            if not alias.is_symlink() or alias.resolve() != source_path.resolve():
                raise ValueError(f"pool alias does not resolve to locked v3 file: {alias_name}")
        else:
            alias.symlink_to(Path(kind) / source_name)
        alias_hashes[alias_name] = expected_hash
    # Synthesize the legacy in-memory fields from the locked v3 manifest.
    # Training draw provenance still uses its existing immutable legacy
    # training manifest SHA; heldout provenance uses the raw v3 manifest SHA.
    settings = {
        "protocol_sha256": protocol_sha,
        "lock_sha256": plan["lock_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "rng_seed": continuation["rng_seed"],
        "max_moves": continuation["max_episode_steps"],
        f"{kind}_namespace": namespace,
        "forced_first_action": True,
        "frozen_policy": True,
        "training_per_state_action": int(continuation["trials_per_action"]["training"]),
        "heldout_per_state_action": int(continuation["trials_per_action"]["heldout"]),
    }
    manifest = {
        "pool_kind": kind,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "protocol_sha256": protocol_sha,
        "locked_plan_sha256": plan["lock_sha256"],
        "pool_file_count": len(expected_names),
        "pool_sha256": alias_hashes,
        "settings": settings,
    }
    if kind == "training":
        legacy_manifest_path = audit / "training_branch_pool_manifest.json"
        legacy = json.loads(legacy_manifest_path.read_text())
        if (legacy.get("pool_sha256") != alias_hashes
                or legacy.get("checkpoint_sha256") != plan["checkpoint_sha256"]
                or legacy.get("locked_plan_sha256") != plan["lock_sha256"]
                or legacy.get("protocol_sha256") != protocol_sha
                or legacy.get("continuation_protocol_sha256") != claimed_continuation_digest):
            raise ValueError("legacy training manifest differs from verified v3 pool files")
        manifest_path = legacy_manifest_path
    else:
        manifest_path = raw_path
    return sha256(manifest_path), manifest


def _validate_draw_manifest(audit, plan, protocol, protocol_sha, train_pool_manifest_sha):
    path = audit / "training_draw_manifest.json"
    draws = json.loads(path.read_text())
    digest = draws.get("manifest_sha256")
    copy = dict(draws); copy.pop("manifest_sha256", None)
    calculated = hashlib.sha256(json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (digest != calculated or draws.get("protocol_sha256") != protocol_sha
            or draws.get("lock_sha256") != plan["lock_sha256"]
            or draws.get("training_pool_manifest_sha256") != train_pool_manifest_sha
            or draws.get("replicates") != protocol["replicates"]
            or draws.get("samples_per_state_arm") != protocol["samples_per_state_arm"]):
        raise ValueError("training draw manifest provenance/digest mismatch")
    for filename, expected_hash in draws["training_pool_sha256"].items():
        if sha256(audit / filename) != expected_hash:
            raise ValueError(f"training draw pool hash mismatch: {filename}")
    return sha256(path), draws


def _one_step(boards, capture, model, optimizer, ppo, rows, pools, heldout_means, lr, focal=None):
    plan_hash = tensor_tree_sha256(capture["model_state_dict"])
    opt_hash = tensor_tree_sha256(capture["optimizer_state_dict"])
    model.load_state_dict(copy.deepcopy(capture["model_state_dict"]))
    optimizer.load_state_dict(copy.deepcopy(capture["optimizer_state_dict"]))
    if tensor_tree_sha256(model.state_dict()) != plan_hash or tensor_tree_sha256(optimizer.state_dict()) != opt_hash:
        raise RuntimeError("pristine model/Adam restore failed")
    for group in optimizer.param_groups:
        group["lr"] = lr
    actor_params = [p for p in model.parameters() if p.requires_grad]
    starting_steps = [int(optimizer.state[p]["step"].item()) for p in actor_params if "step" in optimizer.state.get(p, {})]
    if not starting_steps or min(starting_steps) != max(starting_steps):
        raise RuntimeError(f"actor Adam counter did not restore uniformly: {starting_steps[:4]}")
    adam_start = starting_steps[0]
    states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in boards]
    masks = _legal_masks(boards)
    model.train()
    state_ids = torch.tensor([int(r["state_id"]) for r in rows], dtype=torch.long)
    actions = torch.tensor([int(r["action"]) for r in rows], dtype=torch.long)
    gaes = torch.tensor([float(r["gae"]) for r in rows], dtype=torch.float32)
    old_probs = torch.tensor([float(r["old_probability"]) for r in rows], dtype=torch.float32)
    old_logs = old_probs.log()
    weights = torch.tensor([float(r["sample_weight"]) for r in rows], dtype=torch.float32)
    mean, std = float(capture["advantage_mean"]), float(capture["advantage_std"])
    eps = float(ppo.config.denominator_epsilon)
    normalized = (gaes - mean) / (std + eps)
    logits = model(ppo._state_batch_tensor(states)).action_logits
    all_logs = ppo._action_log_probabilities(logits.masked_fill(masks, -torch.inf), masks)
    selected = all_logs[state_ids, actions]
    ratios = torch.exp(selected - old_logs)
    per_loss = clipped_policy_loss(ratios, normalized, ppo.config.clip_epsilon)
    row_loss = weights * per_loss
    if focal is None:
        policy = row_loss.sum()
    else:
        keep = state_ids != focal
        terms = [row_loss[keep].sum()]
        board = boards[focal]
        for action in board["legal_actions"]:
            action = int(action)
            records = pools[(focal, action)]
            action_gae = torch.tensor([float(x["gae"]) for x in records], dtype=torch.float32)
            action_adv = (action_gae - mean) / (std + eps)
            prob = float(board["behavior_probabilities"][action])
            action_old = torch.full_like(action_adv, math.log(prob))
            action_ratio = torch.exp(all_logs[focal, action].expand_as(action_old) - action_old)
            terms.append((prob / (len(boards) * len(records))) * clipped_policy_loss(
                action_ratio, action_adv, ppo.config.clip_epsilon).sum())
        policy = torch.stack(terms).sum()
    entropy = Categorical(logits=all_logs).entropy().mean()
    objective = policy - ppo.config.entropy_coefficient * entropy
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    grad_before = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in actor_params if p.grad is not None))
    if ppo.config.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(actor_params, ppo.config.gradient_clip_norm)
    grad_after = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in actor_params if p.grad is not None))
    optimizer.step()
    ending_steps = [int(optimizer.state[p]["step"].item()) for p in actor_params if "step" in optimizer.state.get(p, {})]
    if min(ending_steps) != max(ending_steps) or ending_steps[0] != adam_start + 1:
        raise RuntimeError(f"Adam counter did not advance exactly once from {adam_start}: {ending_steps[:4]}")
    model.eval()
    after = _probability_metrics(model, ppo, boards)
    old = [{
        "actor_probabilities": [float(x) for x in b["behavior_probabilities"]],
        "target_actor_probability": float(b["behavior_probabilities"][int(b["critic_action"])]),
        "preferred_actor_probability": float(b["behavior_probabilities"][int(b["actor_action"])]),
        "target_over_preferred_log_odds": float(math.log(b["behavior_probabilities"][int(b["critic_action"])]) - math.log(b["behavior_probabilities"][int(b["actor_action"])])),
    } for b in boards]
    behavior0 = np.asarray([x["actor_probabilities"] for x in old], dtype=np.float64)
    behavior1 = np.asarray([x["actor_probabilities"] for x in after], dtype=np.float64)
    odds0 = np.asarray([x["target_over_preferred_log_odds"] for x in old])
    odds1 = np.asarray([x["target_over_preferred_log_odds"] for x in after])
    value_by_state = [float(sum((behavior1[i, int(a)] - behavior0[i, int(a)]) * heldout_means[i][int(a)]
                                 for a in b["legal_actions"])) for i, b in enumerate(boards)]
    if tensor_tree_sha256(capture["model_state_dict"]) != plan_hash or tensor_tree_sha256(capture["optimizer_state_dict"]) != opt_hash:
        raise RuntimeError("source model/Adam capture mutated during one-step replay")
    return {
        "objective": float(objective.detach()), "policy_loss": float(policy.detach()), "entropy": float(entropy.detach()),
        "gradient_norm_before_clip": grad_before, "gradient_norm_after_clip": grad_after,
        "adam_start": adam_start, "adam_end": adam_start + 1,
        "target_probabilities": [float(x["target_actor_probability"]) for x in after],
        "target_logodds": odds1.tolist(), "behavior_probabilities": behavior1.tolist(),
        "logodds_change": (odds1 - odds0).tolist(),
        "target_probability_change": (behavior1[np.arange(len(boards)), [int(b["critic_action"]) for b in boards]] - behavior0[np.arange(len(boards)), [int(b["critic_action"]) for b in boards]]).tolist(),
        "heldout_value_change_by_state": value_by_state,
        "heldout_value_change_bank_mean": float(np.mean(value_by_state)),
        "optimizer_hash_after": tensor_tree_sha256(optimizer.state_dict()),
    }


def run(args):
    global PROTOCOL_SHA
    audit = args.audit.resolve()
    PROTOCOL_SHA = sha256(audit / "prebranch_protocol_v1.json")
    prior_gate_hashes = None
    if args.mode == "interventions":
        gate_paths = [audit / "matched_baseline_parity.json", audit / "matched_arm_order_gate.json"]
        if not all(p.exists() for p in gate_paths):
            raise FileNotFoundError("Run --mode preflight and obtain independent review before interventions")
        prior_gate_hashes = {p.name: sha256(p) for p in gate_paths}
    plan, protocol, native_manifest, capture, model, optimizer, ppo, boards = _read_inputs(audit)
    capture_design = json.loads((audit.parent.parent / "phase_b_capture_protocol.json").read_text())
    if "one-step matched replay" not in capture_design["diagnostics"]["matched_action_coverage"]:
        raise ValueError("capture protocol does not authorize one-step matched replay")
    protocol["matched_action_coverage"] = {"updates_per_focal_state": 1}
    training_pool_manifest_sha, training_pool_manifest = _validate_pool_manifest(
        audit, "training", plan, protocol, PROTOCOL_SHA)
    heldout_pool_manifest_sha, heldout_pool_manifest = _validate_pool_manifest(
        audit, "heldout", plan, protocol, PROTOCOL_SHA)
    draw_manifest_sha, draw_manifest = _validate_draw_manifest(
        audit, plan, protocol, PROTOCOL_SHA, training_pool_manifest_sha)
    data_provenance = {
        "training_pool_manifest_sha256": training_pool_manifest_sha,
        "heldout_pool_manifest_sha256": heldout_pool_manifest_sha,
        "training_draw_manifest_raw_sha256": draw_manifest_sha,
        "training_draw_manifest_digest": draw_manifest["manifest_sha256"],
    }
    lr_lock = json.loads((audit / "learning_rate_lock.json").read_text())
    if lr_lock["protocol_sha256"] != PROTOCOL_SHA or lr_lock["lock_sha256"] != plan["lock_sha256"]:
        raise ValueError("LR lock does not match bank/protocol")
    calibration_path = audit / "learning_rate_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    draw_digest = draw_manifest.get("manifest_sha256")
    if (lr_lock.get("calibration_file_sha256") != sha256(calibration_path)
            or lr_lock.get("training_draw_manifest_sha256") != draw_digest
            or calibration.get("protocol_sha256") != PROTOCOL_SHA
            or calibration.get("lock_sha256") != plan["lock_sha256"]
            or calibration.get("training_draw_manifest_sha256") != draw_digest
            or calibration.get("training_pool_manifest_sha256") != training_pool_manifest_sha
            or calibration.get("capture_sha256") != plan["capture_sha256"]
            or calibration.get("checkpoint_sha256") != plan["checkpoint_sha256"]
            or calibration.get("selected_divisor") != lr_lock.get("selected_divisor")
            or calibration.get("selected_learning_rate") != lr_lock.get("selected_learning_rate")
            or lr_lock.get("selected_learning_rate") != calibration.get("original_learning_rate") / calibration.get("selected_divisor")):
        raise ValueError("LR lock/calibration/training draw provenance is inconsistent")
    lr = float(lr_lock["selected_learning_rate"])
    pools = {(i, int(a)): _read_pool(audit, "training", i, int(a)) for i, b in enumerate(boards) for a in b["legal_actions"]}
    heldout_means = [{int(a): float(np.mean([x["game_score"] for x in _read_pool(audit, "heldout", i, int(a))]))
                      for a in b["legal_actions"]} for i, b in enumerate(boards)]
    # Reconstruct rows exactly from the saved sample draw manifest and saved GAE pools.
    samples_by_rep = []
    mean, std = float(capture["advantage_mean"]), float(capture["advantage_std"])
    eps = float(ppo.config.denominator_epsilon)
    for draw in draw_manifest["draws"]:
        samples = []
        for r in draw["arms"]["natural"]["rows"]:
            pool = pools[(int(r["state_id"]), int(r["action"]))]
            trial = int(r["training_pool_trial_index"])
            if abs(float(pool[trial]["gae"]) - float(r["gae"])) > 1e-12:
                raise ValueError("draw manifest GAE differs from immutable training pool")
            samples.append({**r, "normalized_gae": (float(r["gae"]) - mean) / (std + eps)})
        if len(samples) != 96 * 32:
            raise ValueError("expected exactly 32 natural samples per board")
        samples_by_rep.append(samples)
    baseline_rows = []
    trace_path = audit / "paired_natural_balanced_updates.json"
    corrected_trace = json.loads(trace_path.read_text())
    if (corrected_trace.get("protocol_sha256") != PROTOCOL_SHA
            or corrected_trace.get("lock_sha256") != plan["lock_sha256"]
            or corrected_trace.get("learning_rate_lock_sha256") != sha256(audit / "learning_rate_lock.json")):
        raise ValueError("paired update trace is not bound to this protocol/bank/LR lock")
    parity_rows = []
    for rep, rows in enumerate(samples_by_rep):
        baseline = _one_step(boards, capture, model, optimizer, ppo, rows, pools, heldout_means, lr)
        expected = corrected_trace["replicates"][rep]["arms"]["natural"]["epochs"][0]
        bp = np.asarray(baseline["target_probabilities"])
        ep = np.asarray(expected["target_actor_probabilities"])
        blog = np.asarray(baseline["target_logodds"])
        elog = np.asarray(expected["target_over_preferred_log_odds"])
        start_counter = int(expected["adam_actor_step_min"]) - 1
        parity = {"replicate": rep, "adam_start": start_counter,
                  "adam_end": int(expected["adam_actor_step_max"]),
                  "max_target_probability_abs_error": float(np.max(np.abs(bp-ep))),
                  "max_logodds_abs_error": float(np.max(np.abs(blog-elog))),
                  "scalar_objective_abs_error": abs(baseline["objective"] - expected["weighted_policy_loss"] + float(ppo.config.entropy_coefficient) * expected["entropy"])}
        if baseline["adam_start"] != start_counter or baseline["adam_end"] != start_counter + 1:
            raise RuntimeError(f"captured Adam step gate failed on replicate {rep}")
        if parity["max_target_probability_abs_error"] > 5e-5 or parity["max_logodds_abs_error"] > 5e-5 or parity["scalar_objective_abs_error"] > 5e-5:
            raise RuntimeError(f"one-step baseline does not match paired trace epoch 1: {parity}")
        parity_rows.append(parity)
        baseline_rows.append(baseline)
    write_json(audit / "matched_baseline_parity.json", {
        "protocol_sha256": PROTOCOL_SHA, "locked_plan_sha256": plan["lock_sha256"],
        "learning_rate_lock_sha256": sha256(audit / "learning_rate_lock.json"),
        **data_provenance, "draw_manifest_sha256": draw_manifest["manifest_sha256"], "replicates": parity_rows,
        "max_target_probability_abs_error": max(r["max_target_probability_abs_error"] for r in parity_rows),
        "max_logodds_abs_error": max(r["max_logodds_abs_error"] for r in parity_rows),
        "all_passed_at_locked_5e-5": True,
    })
    # Restore-order gate for a baseline and a focal replacement. This verifies
    # neither arm inherits model buffers or Adam moments from the preceding arm.
    rows0 = samples_by_rep[0]
    ab_base = _one_step(boards, capture, model, optimizer, ppo, rows0, pools, heldout_means, lr)
    ab_replacement = _one_step(boards, capture, model, optimizer, ppo, rows0, pools, heldout_means, lr, focal=0)
    ba_replacement = _one_step(boards, capture, model, optimizer, ppo, rows0, pools, heldout_means, lr, focal=0)
    ba_base = _one_step(boards, capture, model, optimizer, ppo, rows0, pools, heldout_means, lr)
    def endpoint_error(x, y):
        values = [
            np.max(np.abs(np.asarray(x["target_probabilities"]) - np.asarray(y["target_probabilities"]))),
            np.max(np.abs(np.asarray(x["target_logodds"]) - np.asarray(y["target_logodds"]))),
            abs(x["objective"] - y["objective"]),
            abs(x["gradient_norm_before_clip"] - y["gradient_norm_before_clip"]),
            abs(x["gradient_norm_after_clip"] - y["gradient_norm_after_clip"]),
            abs(x["heldout_value_change_bank_mean"] - y["heldout_value_change_bank_mean"]),
        ]
        if x["optimizer_hash_after"] != y["optimizer_hash_after"]:
            return float("inf")
        return float(max(values))
    order_base_error = endpoint_error(ab_base, ba_base)
    order_replacement_error = endpoint_error(ab_replacement, ba_replacement)
    write_json(audit / "matched_arm_order_gate.json", {
        "protocol_sha256": PROTOCOL_SHA, "locked_plan_sha256": plan["lock_sha256"],
        **data_provenance,
        "replicate": 0, "focal_state": 0,
        "orders": {"baseline_then_replacement": ["baseline", "replacement"], "replacement_then_baseline": ["replacement", "baseline"]},
        "max_baseline_endpoint_error": order_base_error,
        "max_replacement_endpoint_error": order_replacement_error,
        "passed": order_base_error == 0.0 and order_replacement_error == 0.0,
    })
    if order_base_error != 0.0 or order_replacement_error != 0.0:
        raise RuntimeError("baseline/replacement arm order changed one-step outcomes")
    if args.mode == "preflight":
        print("Baseline parity and A/B order gates saved; no focal interventions were run.", flush=True)
        return
    if prior_gate_hashes is not None:
        for filename, expected_hash in prior_gate_hashes.items():
            if sha256(audit / filename) != expected_hash:
                raise RuntimeError(f"Independent-review gate artifact changed before intervention: {filename}")
            gate = json.loads((audit / filename).read_text())
            if any(gate.get(key) != value for key, value in data_provenance.items()):
                raise RuntimeError(f"Independent-review gate has stale input provenance: {filename}")
    interventions = []
    for rep, rows in enumerate(samples_by_rep):
        for focal in range(96):
            result = _one_step(boards, capture, model, optimizer, ppo, rows, pools, heldout_means, lr, focal=focal)
            base = baseline_rows[rep]
            board = boards[focal]
            legal = [int(a) for a in board["legal_actions"]]
            focal_target = int(board["critic_action"])
            target_prob_delta = result["target_probability_change"][focal] - base["target_probability_change"][focal]
            focal_logodds_delta = result["logodds_change"][focal] - base["logodds_change"][focal]
            heldout_delta = result["heldout_value_change_by_state"][focal] - base["heldout_value_change_by_state"][focal]
            row_for_state = [r for r in rows if int(r["state_id"]) == focal]
            action_counts = {a: sum(int(r["action"]) == a for r in row_for_state) for a in legal}
            sample_adv = {a: [float(r["normalized_gae"]) for r in row_for_state if int(r["action"]) == a] for a in legal}
            interventions.append({
                "replicate": rep, "focal_state": focal, "source_episode_id": board["source_episode_id"],
                "critic_target_action": focal_target, "actor_preferred_action": int(board["actor_action"]),
                "critic_target_missing": action_counts[focal_target] == 0,
                "critic_target_count": action_counts[focal_target], "sample_action_counts": {str(a): action_counts[a] for a in legal},
                "sample_mean_normalized_advantage_by_action": {str(a): (float(np.mean(sample_adv[a])) if sample_adv[a] else None) for a in legal},
                "baseline_focal_logodds_change": base["logodds_change"][focal],
                "replacement_focal_logodds_change": result["logodds_change"][focal],
                "paired_replacement_minus_baseline_focal_logodds": focal_logodds_delta,
                "baseline_focal_target_probability_change": base["target_probability_change"][focal],
                "replacement_focal_target_probability_change": result["target_probability_change"][focal],
                "paired_replacement_minus_baseline_focal_target_probability": target_prob_delta,
                "baseline_focal_heldout_value_alignment": base["heldout_value_change_by_state"][focal],
                "replacement_focal_heldout_value_alignment": result["heldout_value_change_by_state"][focal],
                "paired_replacement_minus_baseline_focal_heldout_value_alignment": heldout_delta,
                "baseline_bank_mean_logodds_change": float(np.mean(base["logodds_change"])),
                "replacement_bank_mean_logodds_change": float(np.mean(result["logodds_change"])),
                "paired_replacement_minus_baseline_bank_mean_logodds": float(np.mean(np.asarray(result["logodds_change"])-np.asarray(base["logodds_change"]))),
                "baseline_bank_mean_heldout_value_alignment": base["heldout_value_change_bank_mean"],
                "replacement_bank_mean_heldout_value_alignment": result["heldout_value_change_bank_mean"],
                "paired_replacement_minus_baseline_bank_mean_heldout_value_alignment": result["heldout_value_change_bank_mean"]-base["heldout_value_change_bank_mean"],
                "replacement_gradient_norm_before_clip": result["gradient_norm_before_clip"],
                "replacement_gradient_norm_after_clip": result["gradient_norm_after_clip"],
                "adam_start_end": [result["adam_start"], result["adam_end"]],
            })
            if rep == 0 and focal % 16 == 15:
                print(f"completed matched coverage {focal+1}/96 states for replicate 1", flush=True)
    write_json(audit / "matched_action_coverage_results.json", {
        "protocol_sha256": PROTOCOL_SHA, "locked_plan_sha256": plan["lock_sha256"],
        **data_provenance,
        "learning_rate_lock_sha256": sha256(audit / "learning_rate_lock.json"),
        "training_draw_manifest_sha256": draw_manifest["manifest_sha256"],
        "matched_baseline_parity_sha256": sha256(audit / "matched_baseline_parity.json"),
        "design": "For each 12 natural 96-state draws, restore identical captured model/Adam and perform paired one-step baseline or single-focal replacement. Replace only that focal sampled PPO policy-gradient contribution with all-action finite training pool weighted by locked old behavior probabilities and 1/96; retain other 95 states and entropy; clip the combined gradient globally. Each focal intervention is separate, not a joint policy update.",
        "rows": interventions,
    })
    print(f"Saved {len(interventions)} matched focal interventions", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "interventions"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
