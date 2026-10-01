"""Draw and replay paired natural/balanced PPO updates on the locked 96-state bank."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from games.rl_2048.environment.game_2048 import State2048
from games.rl_2048.scripts.audit_ppo_update import load_runtime, sha256, tensor_tree_sha256, write_json
from games.rl_2048.scripts.balanced_sampling_ppo import balanced_action_counts, clipped_policy_loss, _read_pool
from games.rl_2048.scripts.inspect_advantages import stream_seed


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = ROOT / "artifacts/ppo_2048/ppo_longer_run_post_refactor_iter1000_audit"
PROTOCOL_SHA = ""
DRAW_SEED = 20482201
CANDIDATE_DIVISORS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _read_inputs(audit: Path):
    global PROTOCOL_SHA
    plan = json.loads((audit / "locked_plan.json").read_text())
    protocol_path = audit / "prebranch_protocol_v1.json"
    PROTOCOL_SHA = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if "protocol_sha256" in protocol:
        protocol_copy = dict(protocol)
        claimed_protocol_digest = protocol_copy.pop("protocol_sha256")
        calculated_protocol_digest = hashlib.sha256(
            json.dumps(protocol_copy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if claimed_protocol_digest != calculated_protocol_digest:
            raise ValueError("prebranch protocol self-digest mismatch")
    for relative, expected_hash in protocol.get("implementation_sha256", {}).items():
        implementation_path = ROOT / relative
        if not implementation_path.is_file() or sha256(implementation_path) != expected_hash:
            raise ValueError(f"Locked implementation hash mismatch: {relative}")
    if (protocol["locked_plan_sha256"] != sha256(audit / "locked_plan.json")
            or protocol.get("lock_digest") != plan["lock_sha256"]
            or protocol["checkpoint_sha256"] != plan["checkpoint_sha256"]
            or protocol["capture_sha256"] != plan["capture_sha256"]):
        raise ValueError("protocol refers to a different bank lock")
    if protocol["train_only_learning_rate_calibration"]["candidate_lr_divisors_in_test_order"] != CANDIDATE_DIVISORS:
        raise ValueError("code LR candidates differ from the locked protocol")
    native = audit / "native"
    manifest = json.loads((native / "selection.json").read_text())
    capture_path = native / "capture.pt"
    source_checkpoint = Path(manifest["source_checkpoint"])
    if sha256(source_checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("native source checkpoint differs from locked plan")
    if sha256(capture_path) != plan["capture_sha256"]:
        raise ValueError("native capture differs from bank lock")
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    torch.set_num_threads(int(manifest["threads"]))
    _, model, optimizer, ppo, _ = load_runtime(
        Path(manifest["source_checkpoint"]), int(manifest["seed"]),
        bool(manifest["current_d4_order_acknowledged"]),
    )
    model.load_state_dict(copy.deepcopy(capture["model_state_dict"]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("critic" not in name and "value" not in name)
    boards = plan["boards"]
    return plan, protocol, manifest, capture, model, optimizer, ppo, boards


def _legal_masks(boards):
    masks = torch.ones((len(boards), 4), dtype=torch.bool)
    for row, board in enumerate(boards):
        masks[row, board["legal_actions"]] = False
    return masks


def _probability_metrics(model, ppo, boards):
    rows = []
    with torch.no_grad():
        for start in range(0, len(boards), 256):
            group = boards[start:start + 256]
            states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in group]
            logits = model(ppo._state_batch_tensor(states)).action_logits
            for i, board in enumerate(group):
                legal = board["legal_actions"]
                mask = torch.ones_like(logits[i], dtype=torch.bool)
                mask[legal] = False
                probs = torch.softmax(logits[i].masked_fill(mask, -torch.inf), -1).cpu().numpy()
                target, preferred = int(board["critic_action"]), int(board["actor_action"])
                rows.append({
                    "actor_probabilities": [float(x) for x in probs],
                    "target_actor_probability": float(probs[target]),
                    "preferred_actor_probability": float(probs[preferred]),
                    "target_over_preferred_log_odds": float(
                        np.log(max(float(probs[target]), 1e-300))
                        - np.log(max(float(probs[preferred]), 1e-300))
                    ),
                })
    return rows


def _locked_old_probability_metrics(boards):
    """Old policy metrics from the exact probabilities in the pre-outcome lock."""
    rows = []
    for board in boards:
        probabilities = [float(x) for x in board["behavior_probabilities"]]
        target, preferred = int(board["critic_action"]), int(board["actor_action"])
        rows.append({
            "actor_probabilities": probabilities,
            "target_actor_probability": probabilities[target],
            "preferred_actor_probability": probabilities[preferred],
            "target_over_preferred_log_odds": float(
                np.log(max(probabilities[target], 1e-300))
                - np.log(max(probabilities[preferred], 1e-300))
            ),
        })
    return rows


def _exact_bank_kl(model, ppo, boards, old_logs):
    with torch.no_grad():
        states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in boards]
        logits = model(ppo._state_batch_tensor(states)).action_logits
        masked = logits.masked_fill(_legal_masks(boards), -torch.inf)
        new_logs = ppo._action_log_probabilities(masked, _legal_masks(boards))
        old_probs = old_logs.exp()
        old_safe = old_logs.masked_fill(~torch.isfinite(old_logs), 0.)
        new_safe = new_logs.masked_fill(~torch.isfinite(new_logs), 0.)
        per_state = (old_probs * (old_safe - new_safe)).sum(-1)
        return float(per_state.mean()), [float(x) for x in per_state.cpu().tolist()], new_logs


def _build_draws(audit: Path, boards, capture, replicates=12, n_per_state=32, seed=DRAW_SEED):
    pool_hashes = {}
    for board in boards:
        for action in board["legal_actions"]:
            path = audit / f"training_board_{board['id']:02d}_action_{action}.json"
            pool_hashes[path.name] = sha256(path)
    rows = []
    for rep in range(replicates):
        draw = {"replicate": rep, "arms": {}}
        for arm_id, arm in enumerate(("natural", "balanced")):
            arm_rows = []
            state_counts = []
            for board in boards:
                legal = [int(a) for a in board["legal_actions"]]
                probs = {a: float(board["behavior_probabilities"][a]) for a in legal}
                rng = np.random.default_rng(stream_seed(seed, 701 + arm_id, int(board["id"]), rep, 0))
                if arm == "natural":
                    actions = rng.choice(legal, size=n_per_state, p=[probs[a] for a in legal]).tolist()
                    counts = {a: actions.count(a) for a in legal}
                else:
                    counts = balanced_action_counts(legal, n_per_state, (int(board["id"]) + rep) % len(legal))
                    actions = [a for a in legal for _ in range(counts[a])]
                    rng.shuffle(actions)
                state_counts.append({str(a): int(counts[a]) for a in legal})
                for action in actions:
                    pool = _read_pool(audit, "training", int(board["id"]), int(action))
                    trial = int(rng.integers(len(pool)))
                    weight = (1. / (len(boards) * n_per_state)) if arm == "natural" else probs[action] / (len(boards) * counts[action])
                    arm_rows.append({
                        "state_id": int(board["id"]), "action": int(action),
                        "training_pool_trial_index": trial, "gae": float(pool[trial]["gae"]),
                        "old_probability": probs[action], "sample_weight": float(weight),
                    })
            if len(arm_rows) != len(boards) * n_per_state:
                raise AssertionError("unexpected sample count")
            draw["arms"][arm] = {"rows": arm_rows, "action_counts": state_counts}
        rows.append(draw)
    manifest = {
        "protocol_sha256": PROTOCOL_SHA,
        "lock_sha256": json.loads((audit / "locked_plan.json").read_text())["lock_sha256"],
        "candidate_screen_manifest_sha256": sha256(audit / "candidate_screen_manifest.json"),
        "training_pool_manifest_sha256": sha256(audit / "training_branch_pool_manifest.json"),
        "seed": seed, "replicates": replicates, "samples_per_state_arm": n_per_state,
        "training_pool_sha256": pool_hashes, "draws": rows,
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest


def _run_update(boards, capture, model, optimizer, ppo, batch_rows, learning_rate):
    capture_model_hash = tensor_tree_sha256(capture["model_state_dict"])
    capture_optimizer_hash = tensor_tree_sha256(capture["optimizer_state_dict"])
    model.load_state_dict(copy.deepcopy(capture["model_state_dict"]))
    optimizer.load_state_dict(copy.deepcopy(capture["optimizer_state_dict"]))
    if tensor_tree_sha256(model.state_dict()) != capture_model_hash:
        raise RuntimeError("Model restore does not match captured parameters/buffers")
    if tensor_tree_sha256(optimizer.state_dict()) != capture_optimizer_hash:
        raise RuntimeError("Optimizer restore does not match captured Adam state")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    old_metrics = _probability_metrics(model, ppo, boards)
    old_metrics = _locked_old_probability_metrics(boards)
    old_prob_rows = np.asarray([row["actor_probabilities"] for row in old_metrics])
    old_log_odds = np.asarray([row["target_over_preferred_log_odds"] for row in old_metrics])
    states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in boards]
    masks = _legal_masks(boards)
    # The exact stored behavior probabilities are the PPO ratio denominators.
    # Recomputed model probabilities are separately checked because batched
    # float32 kernels may differ slightly from the pre-lock inference batch.
    old_probability_tensor = torch.tensor(old_prob_rows, dtype=torch.float32)
    old_logs = old_probability_tensor.log().masked_fill(masks, -torch.inf)
    max_probability_reconstruction_error = 0.0
    for i, board in enumerate(boards):
        for action in board["legal_actions"]:
            error = abs(float(old_prob_rows[i, action]) - float(board["behavior_probabilities"][action]))
            max_probability_reconstruction_error = max(max_probability_reconstruction_error, error)
            if error != 0.0:
                raise ValueError("Locked old probability row changed unexpectedly")
    reconstructed_metrics = _probability_metrics(model, ppo, boards)
    for i, board in enumerate(boards):
        for action in board["legal_actions"]:
            error = abs(float(reconstructed_metrics[i]["actor_probabilities"][action]) - float(board["behavior_probabilities"][action]))
            max_probability_reconstruction_error = max(max_probability_reconstruction_error, error)
    if max_probability_reconstruction_error > 5e-5:
        raise ValueError(f"Checkpoint probability reconstruction exceeds locked 5e-5 tolerance: {max_probability_reconstruction_error}")
    state_ids = torch.tensor([int(r["state_id"]) for r in batch_rows], dtype=torch.long)
    actions = torch.tensor([int(r["action"]) for r in batch_rows], dtype=torch.long)
    gaes = torch.tensor([float(r["gae"]) for r in batch_rows], dtype=torch.float32)
    weights = torch.tensor([float(r["sample_weight"]) for r in batch_rows], dtype=torch.float32)
    old_row_logs = old_logs[state_ids, actions].detach()
    mean = float(capture["advantage_mean"])
    std = float(capture["advantage_std"])
    traces = []
    actor_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    starts = []
    for parameter in actor_parameters:
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        if step is not None:
            starts.append(int(step.item()) if torch.is_tensor(step) else int(step))
    if not starts or min(starts) != max(starts):
        raise RuntimeError(f"Captured actor Adam counters are not uniform: {starts[:4]}")
    start_step = starts[0]
    if any(start_step != int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
           for parameter in actor_parameters for state in [optimizer.state.get(parameter, {})] if "step" in state):
        raise RuntimeError("Actor optimizer state has inconsistent starting steps")
    model.train()
    for epoch in range(8):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ppo._state_batch_tensor(states)).action_logits
        all_logs = ppo._action_log_probabilities(logits.masked_fill(masks, -torch.inf), masks)
        selected = all_logs[state_ids, actions]
        normalized = (gaes - mean) / (std + ppo.config.denominator_epsilon)
        ratios = torch.exp(selected - old_row_logs)
        per_row_loss = clipped_policy_loss(ratios, normalized, ppo.config.clip_epsilon)
        policy_loss = torch.sum(weights * per_row_loss)
        entropy = Categorical(logits=all_logs).entropy().mean()
        objective = policy_loss - ppo.config.entropy_coefficient * entropy
        objective.backward()
        # Preserve the gradient of the full current objective (policy loss minus
        # entropy bonus) before optional global clipping, then compare it with the
        # displacement the actual Adam step makes from this exact state.
        objective_gradient = [
            p.grad.detach().clone() for p in actor_parameters if p.grad is not None
        ]
        parameters_before = [p.detach().clone() for p in actor_parameters]
        raw_grad_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in objective_gradient))
        if ppo.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.config.gradient_clip_norm)
        grad_sq = sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        displacement = [p.detach() - before for p, before in zip(actor_parameters, parameters_before)]
        negative_gradient = [-g for g in objective_gradient]
        displacement_norm = math.sqrt(sum(float(d.pow(2).sum()) for d in displacement))
        negative_gradient_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in negative_gradient))
        dot = sum(float((d * g).sum()) for d, g in zip(displacement, negative_gradient))
        cosine = dot / (displacement_norm * negative_gradient_norm) if displacement_norm and negative_gradient_norm else 0.0
        actor_steps = []
        for parameter in actor_parameters:
            value = optimizer.state.get(parameter, {}).get("step")
            if value is not None:
                actor_steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
        expected_step = start_step + epoch + 1
        if not actor_steps or min(actor_steps) != max(actor_steps) or actor_steps[0] != expected_step:
            raise RuntimeError(f"Adam step counter mismatch at epoch {epoch + 1}: {actor_steps[:4]}, expected {expected_step}")
        model.eval()
        exact_kl, per_state_kl, new_logs = _exact_bank_kl(model, ppo, boards, old_logs)
        new_metrics = _probability_metrics(model, ppo, boards)
        model.train()
        traces.append({
            "epoch": epoch + 1, "adam_actor_step_min": min(actor_steps), "adam_actor_step_max": max(actor_steps),
            "weighted_policy_loss": float(policy_loss.detach()), "entropy": float(entropy.detach()),
            "total_objective": float(objective.detach()),
            "raw_total_objective_gradient_norm_before_clipping": raw_grad_norm,
            "gradient_norm_after_configured_clipping": math.sqrt(grad_sq),
            "actor_parameter_displacement_l2": displacement_norm,
            "adam_displacement_cosine_with_negative_total_objective_gradient": cosine,
            "mean_sample_ratio": float(ratios.detach().mean()),
            "sample_clip_fraction": float((torch.abs(ratios.detach() - 1) > ppo.config.clip_epsilon).float().mean()),
            "exact_forward_kl_old_to_current": exact_kl, "per_state_exact_kl": per_state_kl,
            "target_actor_probabilities": [float(r["target_actor_probability"]) for r in new_metrics],
            "preferred_actor_probabilities": [float(r["preferred_actor_probability"]) for r in new_metrics],
            "target_over_preferred_log_odds": [float(r["target_over_preferred_log_odds"]) for r in new_metrics],
        })
    model.eval()
    final_metrics = _probability_metrics(model, ppo, boards)
    for metric, initial in zip(final_metrics, old_metrics):
        metric["target_actor_probability_change"] = metric["target_actor_probability"] - initial["target_actor_probability"]
        metric["target_preferred_log_odds_change"] = metric["target_over_preferred_log_odds"] - initial["target_over_preferred_log_odds"]
    if tensor_tree_sha256(capture["model_state_dict"]) != capture_model_hash:
        raise RuntimeError("Captured model state was mutated by an update replay")
    if tensor_tree_sha256(capture["optimizer_state_dict"]) != capture_optimizer_hash:
        raise RuntimeError("Captured optimizer state was mutated by an update replay")
    return {"initial": old_metrics, "final": final_metrics, "epochs": traces,
            "maximum_saved_vs_reconstructed_behavior_probability_error": max_probability_reconstruction_error,
            "model_state_sha256": tensor_tree_sha256(model.state_dict()),
            "optimizer_state_sha256": tensor_tree_sha256(optimizer.state_dict()),
            "capture_model_unchanged": True, "capture_optimizer_unchanged": True,
            "adam_start_step": start_step, "adam_end_step": start_step + 8}


def prepare(args):
    audit = args.audit.resolve()
    plan, protocol, manifest, capture, model, optimizer, ppo, boards = _read_inputs(audit)
    train_pool_manifest = audit / "training_branch_pool_manifest.json"
    if not train_pool_manifest.exists():
        raise FileNotFoundError("Training pools are incomplete; held-out pools are not used here")
    stamped = json.loads(train_pool_manifest.read_text())
    stamped["protocol_sha256"] = PROTOCOL_SHA
    stamped["locked_plan_sha256"] = plan["lock_sha256"]
    stamped["candidate_screen_manifest_sha256"] = sha256(audit / "candidate_screen_manifest.json")
    for filename, expected in stamped["pool_sha256"].items():
        if sha256(audit / filename) != expected:
            raise ValueError(f"Training pool hash mismatch: {filename}")
    write_json(train_pool_manifest, stamped)
    draws = _build_draws(audit, boards, capture)
    write_json(audit / "training_draw_manifest.json", draws)
    print(f"Locked 12×2 training draws; digest {draws['manifest_sha256']}")


def calibrate(args):
    audit = args.audit.resolve()
    plan, protocol, manifest, capture, model, optimizer, ppo, boards = _read_inputs(audit)
    if not (audit / "training_draw_manifest.json").exists():
        raise FileNotFoundError("Run prepare before calibration")
    draws = json.loads((audit / "training_draw_manifest.json").read_text())
    if draws["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("training draws use a different protocol")
    source_model_hash = tensor_tree_sha256(capture["model_state_dict"])
    source_optimizer_hash = tensor_tree_sha256(capture["optimizer_state_dict"])
    original_lr = float(capture["optimizer_state_dict"]["param_groups"][0]["lr"])
    probes = []
    selected = None
    for divisor in CANDIDATE_DIVISORS:
        lr = original_lr / divisor
        row = {"divisor": divisor, "learning_rate": lr, "replicate": 0, "arms": {}}
        for arm in ("natural", "balanced"):
            result = _run_update(boards, capture, model, optimizer, ppo, draws["draws"][0]["arms"][arm]["rows"], lr)
            row["arms"][arm] = {
                "final_exact_forward_kl_old_to_current": result["epochs"][-1]["exact_forward_kl_old_to_current"],
                "per_epoch_exact_forward_kl": [x["exact_forward_kl_old_to_current"] for x in result["epochs"]],
                "actor_step_start": result["epochs"][0]["adam_actor_step_min"] - 1,
                "actor_step_end": result["epochs"][-1]["adam_actor_step_max"],
            }
        probes.append(row)
        if all(row["arms"][arm]["final_exact_forward_kl_old_to_current"] <= ppo.config.target_kl for arm in ("natural", "balanced")):
            selected = row
            break
    if selected is None:
        write_json(audit / "learning_rate_calibration.json", {
            "status": "no candidate passed; heldout pools must remain uncollected",
            "protocol_sha256": PROTOCOL_SHA, "original_learning_rate": original_lr,
            "target_kl": ppo.config.target_kl, "candidate_divisors": CANDIDATE_DIVISORS,
            "probes": probes, "capture_optimizer_sha256_before": source_optimizer_hash,
            "capture_optimizer_sha256_after": tensor_tree_sha256(capture["optimizer_state_dict"]),
        })
        raise RuntimeError("No predeclared common LR passed training-only KL calibration; do not inspect heldout")
    # Exact repeat and A/B order checks use only replicate-0 training rows.
    lr = float(selected["learning_rate"])
    rows0 = draws["draws"][0]["arms"]
    forward_n = _run_update(boards, capture, model, optimizer, ppo, rows0["natural"]["rows"], lr)
    forward_b = _run_update(boards, capture, model, optimizer, ppo, rows0["balanced"]["rows"], lr)
    reverse_b = _run_update(boards, capture, model, optimizer, ppo, rows0["balanced"]["rows"], lr)
    reverse_n = _run_update(boards, capture, model, optimizer, ppo, rows0["natural"]["rows"], lr)
    repeat_n = _run_update(boards, capture, model, optimizer, ppo, rows0["natural"]["rows"], lr)
    def result_error(left, right):
        if left["model_state_sha256"] != right["model_state_sha256"] or left["optimizer_state_sha256"] != right["optimizer_state_sha256"]:
            return float("inf")
        error = 0.
        for a, b in zip(left["final"], right["final"]):
            error = max(error, abs(a["target_actor_probability"] - b["target_actor_probability"]))
            error = max(error, abs(a["target_over_preferred_log_odds"] - b["target_over_preferred_log_odds"]))
            error = max(error, max(abs(x-y) for x,y in zip(a["actor_probabilities"], b["actor_probabilities"])))
        for a, b in zip(left["epochs"], right["epochs"]):
            for key in ("weighted_policy_loss", "entropy", "gradient_norm_after_configured_clipping",
                        "mean_sample_ratio", "sample_clip_fraction", "exact_forward_kl_old_to_current",
                        "total_objective", "raw_total_objective_gradient_norm_before_clipping",
                        "actor_parameter_displacement_l2",
                        "adam_displacement_cosine_with_negative_total_objective_gradient"):
                error = max(error, abs(a[key] - b[key]))
            for key in ("per_state_exact_kl", "target_actor_probabilities",
                        "preferred_actor_probabilities", "target_over_preferred_log_odds"):
                error = max(error, max(abs(x-y) for x,y in zip(a[key], b[key])))
        return error
    repeat_error = result_error(forward_n, repeat_n)
    order_n_error = result_error(forward_n, reverse_n)
    order_b_error = result_error(forward_b, reverse_b)
    if repeat_error != 0.0 or order_n_error != 0.0 or order_b_error != 0.0:
        raise RuntimeError(f"Training-only deterministic restore/order gate failed: repeat={repeat_error}, natural order={order_n_error}, balanced order={order_b_error}")
    cal = {
        "status": "passed; heldout pools remain prohibited until learning_rate_lock.json is saved",
        "protocol_sha256": PROTOCOL_SHA, "lock_sha256": plan["lock_sha256"],
        "training_draw_manifest_sha256": draws["manifest_sha256"],
        "training_pool_manifest_sha256": draws["training_pool_manifest_sha256"],
        "capture_sha256": plan["capture_sha256"], "checkpoint_sha256": plan["checkpoint_sha256"],
        "original_learning_rate": original_lr, "target_kl": float(ppo.config.target_kl),
        "candidate_divisors": CANDIDATE_DIVISORS, "probes": probes,
        "selected_divisor": int(selected["divisor"]), "selected_learning_rate": lr,
        "restore_gate": {"capture_model_unchanged": tensor_tree_sha256(capture["model_state_dict"]) == source_model_hash,
                         "capture_optimizer_unchanged": tensor_tree_sha256(capture["optimizer_state_dict"]) == source_optimizer_hash,
                         "repeat_probability_max_abs_error": repeat_error,
                         "natural_order_probability_max_abs_error": order_n_error,
                         "balanced_order_probability_max_abs_error": order_b_error,
                         "all_interventions_start_from_captured_adam_state": True},
    }
    write_json(audit / "learning_rate_calibration.json", cal)
    lock_lr = {
        "status": "locked before any heldout pools were collected",
        "protocol_sha256": PROTOCOL_SHA, "lock_sha256": plan["lock_sha256"],
        "calibration_file_sha256": sha256(audit / "learning_rate_calibration.json"),
        "training_draw_manifest_sha256": draws["manifest_sha256"],
        "original_learning_rate": original_lr, "selected_divisor": int(selected["divisor"]),
        "selected_learning_rate": lr, "target_kl": float(ppo.config.target_kl),
        "natural_rep0_final_exact_kl": float(selected["arms"]["natural"]["final_exact_forward_kl_old_to_current"]),
        "balanced_rep0_final_exact_kl": float(selected["arms"]["balanced"]["final_exact_forward_kl_old_to_current"]),
    }
    write_json(audit / "learning_rate_lock.json", lock_lr)
    print(f"Locked common LR {lr:g} (÷{selected['divisor']}); natural/balanced exact KL {lock_lr['natural_rep0_final_exact_kl']:.5g}/{lock_lr['balanced_rep0_final_exact_kl']:.5g}")


def updates(args):
    audit = args.audit.resolve()
    plan, protocol, manifest, capture, model, optimizer, ppo, boards = _read_inputs(audit)
    lr_lock_path = audit / "learning_rate_lock.json"
    if not lr_lock_path.exists():
        raise FileNotFoundError("No LR lock; cannot run paired updates")
    lr_lock = json.loads(lr_lock_path.read_text())
    if lr_lock["protocol_sha256"] != PROTOCOL_SHA or lr_lock["lock_sha256"] != plan["lock_sha256"]:
        raise ValueError("learning-rate lock provenance mismatch")
    draws = json.loads((audit / "training_draw_manifest.json").read_text())
    output = {"protocol_sha256": PROTOCOL_SHA, "lock_sha256": plan["lock_sha256"],
              "learning_rate_lock_sha256": sha256(lr_lock_path), "learning_rate": lr_lock["selected_learning_rate"],
              "replicates": []}
    source_model_hash = tensor_tree_sha256(capture["model_state_dict"])
    source_opt_hash = tensor_tree_sha256(capture["optimizer_state_dict"])
    for draw in draws["draws"]:
        row = {"replicate": draw["replicate"], "arms": {}}
        for arm in ("natural", "balanced"):
            row["arms"][arm] = _run_update(
                boards, capture, model, optimizer, ppo, draw["arms"][arm]["rows"],
                float(lr_lock["selected_learning_rate"]),
            )
        output["replicates"].append(row)
    if tensor_tree_sha256(capture["model_state_dict"]) != source_model_hash or tensor_tree_sha256(capture["optimizer_state_dict"]) != source_opt_hash:
        raise RuntimeError("Native capture changed during paired update replay")
    output["capture_integrity"] = {"model_unchanged": True, "optimizer_unchanged": True,
                                    "adam_start": min(r["arms"]["natural"]["epochs"][0]["adam_actor_step_min"] - 1 for r in output["replicates"]),
                                    "adam_end": max(r["arms"]["natural"]["epochs"][-1]["adam_actor_step_max"] for r in output["replicates"])}
    write_json(audit / "paired_natural_balanced_updates.json", output)
    print(f"Saved {len(output['replicates'])} paired updates per arm under locked LR {output['learning_rate']:g}")


def native_probabilities(args):
    audit = args.audit.resolve()
    plan, protocol, manifest, capture, model, optimizer, ppo, boards = _read_inputs(audit)
    update_path = audit / "native" / "update_state.pt"
    update_state = torch.load(update_path, map_location="cpu", weights_only=False)
    if update_state.get("source_sha256") != plan["checkpoint_sha256"]:
        raise ValueError("native update state belongs to a different source checkpoint")
    model.eval()
    before = _probability_metrics(model, ppo, boards)
    model.load_state_dict(copy.deepcopy(update_state["model_state_dict"]))
    model.eval()
    after = _probability_metrics(model, ppo, boards)
    rollout = capture["rollout"]
    counts = []
    for board in boards:
        target = int(board["critic_action"])
        exact_rows = []
        grid = np.asarray(board["board"], dtype=np.int64)
        for row_index, step in enumerate(rollout):
            if np.array_equal(np.asarray(step["board"]), grid):
                exact_rows.append({"rollout_index": row_index, "action": int(step["action"]),
                                   "raw_advantage": float(step["advantage"]),
                                   "normalized_advantage": float((float(step["advantage"]) - float(capture["advantage_mean"])) /
                                                                  (float(capture["advantage_std"]) + ppo.config.denominator_epsilon))})
        counts.append(exact_rows)
    rows = []
    for board, pre, post, exact in zip(boards, before, after, counts):
        rows.append({
            "id": int(board["id"]), "source_episode_id": board["source_episode_id"],
            "rollout_index": int(board["rollout_index"]),
            "actor_preferred_action": int(board["actor_action"]), "critic_target_action": int(board["critic_action"]),
            "exact_board_occurrences_in_native_rollout": len(exact),
            "exact_board_native_samples": exact,
            "pre_update": pre, "post_update": post,
            "target_probability_change": post["target_actor_probability"] - pre["target_actor_probability"],
            "preferred_probability_change": post["preferred_actor_probability"] - pre["preferred_actor_probability"],
            "target_over_preferred_log_odds_change": post["target_over_preferred_log_odds"] - pre["target_over_preferred_log_odds"],
        })
    changes = np.asarray([r["target_over_preferred_log_odds_change"] for r in rows])
    target_changes = np.asarray([r["target_probability_change"] for r in rows])
    preferred_changes = np.asarray([r["preferred_probability_change"] for r in rows])
    summary = {
        "n_states": len(rows), "mean_pre_update_actor_preferred_probability": float(np.mean([r["pre_update"]["preferred_actor_probability"] for r in rows])),
        "mean_post_update_actor_preferred_probability": float(np.mean([r["post_update"]["preferred_actor_probability"] for r in rows])),
        "mean_pre_update_critic_target_probability": float(np.mean([r["pre_update"]["target_actor_probability"] for r in rows])),
        "mean_post_update_critic_target_probability": float(np.mean([r["post_update"]["target_actor_probability"] for r in rows])),
        "mean_target_probability_change": float(target_changes.mean()),
        "mean_preferred_probability_change": float(preferred_changes.mean()),
        "mean_target_over_preferred_log_odds_change": float(changes.mean()),
        "median_target_over_preferred_log_odds_change": float(np.median(changes)),
        "log_odds_toward_target_count": int(np.sum(changes > 1e-8)),
        "log_odds_toward_preferred_count": int(np.sum(changes < -1e-8)),
        "log_odds_unchanged_count": int(np.sum(np.abs(changes) <= 1e-8)),
        "target_probability_increased_count": int(np.sum(target_changes > 1e-8)),
        "preferred_probability_increased_count": int(np.sum(preferred_changes > 1e-8)),
        "states_with_target_action_observed_at_exact_board": int(sum(any(x["action"] == r["critic_target_action"] for x in r["exact_board_native_samples"]) for r in rows)),
        "exact_board_native_transition_total": int(sum(len(x) for x in counts)),
    }
    write_json(audit / "native_update_bank_probabilities.json", {
        "protocol_sha256": PROTOCOL_SHA, "locked_plan_sha256": plan["lock_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"], "capture_sha256": plan["capture_sha256"],
        "update_state_sha256": sha256(update_path), "update_json_sha256": sha256(audit / "native" / "update.json"),
        "estimand": "Pre/post probability changes from one saved ordinary full-rollout PPO update, evaluated on each selected frozen checkpoint state; action labels are critic Q argmax and actor pre-update argmax.",
        "normalization": {"advantage_mean": capture["advantage_mean"], "advantage_std": capture["advantage_std"], "epsilon": ppo.config.denominator_epsilon},
        "summary": summary, "per_state": rows,
    })
    print(f"Saved native full-update probability changes: target ↑ {summary['target_probability_increased_count']}/96; target log odds ↑ {summary['log_odds_toward_target_count']}/96")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "calibrate", "updates", "native-probabilities"):
        p = sub.add_parser(name)
        p.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "calibrate": calibrate, "updates": updates,
     "native-probabilities": native_probabilities}[args.command](args)


if __name__ == "__main__":
    main()
