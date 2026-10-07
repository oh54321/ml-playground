"""Broad, fixed-state PPO2048 gradient and update audit.

Selection is locked from the saved rollout before any new continuations are run.
See artifacts/ppo_2048/ppo_broad_gradient_audit_iter100/lockedplan.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.scripts.audit_ppo_update import (
    ACTION_NAMES, counterfactual_gae_horizon, deserialize_step, load_runtime,
    mixture_probabilities, sha256, write_json,
)
from games.rl_2048.scripts.balanced_sampling_ppo import (
    _evaluate_probabilities, _make_replicate_batches, _read_pool,
    clipped_policy_loss,
)

HERE = Path(__file__).resolve().parents[3]
SOURCE = HERE / "artifacts/ppo_2048/ppo_update_audit_iter100"
OLD = HERE / "artifacts/ppo_2048/ppo_balanced_sampling_iter100"
OUT = HERE / "artifacts/ppo_2048/ppo_broad_gradient_audit_iter100"
NEW_IDS = range(24, 96)
SEED = 20482091


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_tree_sha(obj):
    digest=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            tensor=value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):digest.update(str(key).encode());visit(value[key])
        elif isinstance(value,(list,tuple)):
            for item in value:visit(item)
        else:digest.update(repr(value).encode())
    visit(obj);return digest.hexdigest()


def lock(args):
    """Screen the pre-existing 2,048-state rollout sample and lock 72 states."""
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((SOURCE / "selection.json").read_text())
    source = Path(prior["source_checkpoint"])
    if sha256(source) != prior["source_sha256"]:
        raise ValueError("checkpoint hash differs from rollout manifest")
    cap_path = SOURCE / "capture.pt"
    capture = torch.load(cap_path, map_location="cpu", weights_only=False)
    if capture.get("source_sha256") != prior["source_sha256"]:
        raise ValueError("capture/checkpoint hash mismatch")
    torch.set_num_threads(int(prior["threads"]))
    _, model, _, ppo, epsilon = load_runtime(source, prior["seed"], prior["current_d4_order_acknowledged"])
    model.load_state_dict(capture["model_state_dict"])
    model.eval().requires_grad_(False)
    steps = [deserialize_step(row) for row in capture["rollout"]]
    per_game = int(ppo.config.steps_per_game_per_rollout)
    eligible = [i for i, s in enumerate(steps) if int((~s.legal_actions_mask).sum()) >= 2]
    positions = np.unique(np.linspace(0, len(eligible) - 1, min(args.candidate_limit, len(eligible)), dtype=int))
    rows = []
    actor_states = []
    successor_states = []
    action_meta = []
    for pos in positions:
        index = eligible[int(pos)]; step = steps[index]
        game = Game2048(seed=0); game.replace_board(step.state)
        legal = game.legal_actions()
        ci = len(rows); rows.append((index, step, legal)); actor_states.append(step.state)
        for action in legal:
            moved, reward, applied = game._simulate_move(step.state.grid, action)
            if not applied: raise ValueError("legal move did not apply")
            empty = np.argwhere(moved == 0); terms = []
            if len(empty):
                for r, c in empty:
                    for tile, probability in ((2, .9), (4, .1)):
                        spawned = moved.copy(); spawned[r, c] = tile
                        nxt = Game2048(seed=0); nxt.replace_board(spawned)
                        if nxt.legal_actions():
                            terms.append((len(successor_states), probability / len(empty)))
                            successor_states.append(nxt.state)
                        else: terms.append((None, probability / len(empty)))
            action_meta.append((ci, action, float(reward), terms))
    actor_logits = []
    with torch.no_grad():
        for start in range(0, len(actor_states), 256):
            actor_logits.append(model(ppo._state_batch_tensor(actor_states[start:start + 256])).action_logits.cpu())
    actor_logits = torch.cat(actor_logits)
    successor_values = []
    with torch.no_grad():
        for start in range(0, len(successor_states), 256):
            pred = model(ppo.rollout_collector.state_batch_tensor(successor_states[start:start + 256])).value
            successor_values.extend((ppo._untransform_predicted_value(pred) / ppo.config.reward_multiplier).cpu().tolist())
    qs = [{} for _ in rows]
    for ci, action, reward, terms in action_meta:
        qs[ci][int(action)] = reward + ppo.config.gamma * sum((0. if j is None else float(successor_values[j])) * w for j, w in terms)
    candidates = []
    for ci, (index, step, legal) in enumerate(rows):
        logits = actor_logits[ci].masked_fill(step.legal_actions_mask, -torch.inf)
        pi = torch.softmax(logits, -1).numpy()
        behavior = mixture_probabilities(pi, legal, epsilon)
        actor_action = int(np.argmax(pi)); critic_action = max(qs[ci], key=qs[ci].get)
        if actor_action == critic_action: continue
        qgap = float(qs[ci][critic_action] - qs[ci][actor_action])
        if qgap < args.minimum_gap: continue
        slot, offset = divmod(index, per_game)
        episode_start = slot * per_game
        for prior_index in range(index - 1, slot * per_game - 1, -1):
            if steps[prior_index].done or steps[prior_index].truncated:
                episode_start = prior_index + 1; break
        episode = sum(1 for s in steps[slot * per_game:episode_start] if s.done or s.truncated)
        ep_id = f"slot{slot:02d}_episode{episode:03d}"
        horizon = 0; boundary = "rollout_segment"
        for following in steps[index:index + per_game - offset]:
            horizon += 1
            if following.done: boundary = "termination"; break
            if following.truncated: boundary = "truncation"; break
        age = index - episode_start
        horizon_limit = counterfactual_gae_horizon(per_game - offset, age, ppo.config.max_episode_steps)
        candidates.append({
            "rollout_index": index, "game_index": slot, "game_step": offset,
            "episode_index_within_env_slot": episode, "source_episode_id": ep_id,
            "board": step.state.grid.tolist(), "legal_actions": legal,
            "actor_action": actor_action, "critic_action": critic_action,
            "actor_probabilities": pi.tolist(), "behavior_probabilities": behavior.tolist(),
            "exact_q": {str(a): float(v) for a, v in qs[ci].items()}, "delta_q": qgap,
            "raw_advantage": float(step.advantage), "sampled_action": int(step.action),
            "old_log_prob": float(step.log_prob), "captured_episode_age": age,
            "counterfactual_gae_horizon_limit": horizon_limit, "remaining_rollout_horizon": horizon,
            "boundary_after": boundary, "empty_cells": int((step.state.grid == 0).sum()),
            "largest_tile": int(step.state.grid.max()),
        })
    existing = json.loads((OLD / "selection/selection.json").read_text())["boards"]
    existing_keys = {(tuple(tuple(r) for r in b["board"]), int(b["critic_action"])) for b in existing}
    existing_episode_ids = set()
    # Derive true episodes for prior selected transitions using captured done/truncated markers.
    for b in existing:
        idx = int(b["rollout_index"]); slot = idx // per_game; start = slot * per_game
        for j in range(idx - 1, start - 1, -1):
            if steps[j].done or steps[j].truncated: start = j + 1; break
        ep = sum(1 for s in steps[slot * per_game:start] if s.done or s.truncated)
        b["source_episode_id"] = f"slot{slot:02d}_episode{ep:03d}"
        existing_episode_ids.add(b["source_episode_id"])
    # Predeclared 3x3 gap/probability strata, rotating deterministic ranks across cells.
    for c in candidates:
        c["target_actor_probability"] = float(c["actor_probabilities"][c["critic_action"]])
        c["gap_stratum"] = int(np.digitize(c["delta_q"], [750, 1250]))
        c["probability_stratum"] = int(np.digitize(c["target_actor_probability"], [0.01, 0.10]))
    candidates.sort(key=lambda c: (-c["delta_q"], c["rollout_index"]))
    cells = {(g, p): [] for g in range(3) for p in range(3)}
    for c in candidates: cells[(c["gap_stratum"], c["probability_stratum"])].append(c)
    selected = []
    counts = {}
    # Round-robin over all nine cells, selecting highest-gap remaining row in each.
    while len(selected) < args.boards:
        progressed = False
        for cell in sorted(cells):
            choices = cells[cell]
            while choices:
                c = choices.pop(0)
                if (tuple(tuple(r) for r in c["board"]), c["critic_action"]) in existing_keys: continue
                # Existing states count toward the same per-episode cap.
                n = counts.get(c["source_episode_id"], sum(1 for b in existing if b.get("source_episode_id") == c["source_episode_id"])) + 1
                if n > args.max_per_episode: continue
                selected.append(c); counts[c["source_episode_id"]] = n; progressed = True
                break
            if len(selected) >= args.boards: break
        if not progressed: break
    if len(selected) < args.boards:
        raise ValueError(f"Only locked {len(selected)} of {args.boards} requested new states under eligibility/diversity rules; expand candidate screen before relaxing constraints")
    all_episode_ids = existing_episode_ids | set(counts)
    if len(all_episode_ids) < args.minimum_episodes:
        raise ValueError(f"Combined bank covers only {len(all_episode_ids)} source episodes")
    for i, c in enumerate(selected, 24): c["id"] = i
    locked_existing = []
    for i, b in enumerate(existing):
        item = dict(b); item["id"] = i
        slot = int(item["rollout_index"]) // per_game; start = slot * per_game
        for j in range(int(item["rollout_index"]) - 1, start - 1, -1):
            if steps[j].done or steps[j].truncated: start = j + 1; break
        ep = sum(1 for s in steps[slot * per_game:start] if s.done or s.truncated)
        item["source_episode_id"] = f"slot{slot:02d}_episode{ep:03d}"
        locked_existing.append(item)
    plan = {
        "title": "Broad fixed-state PPO gradient and update audit",
        "status": "locked before new continuation outcomes",
        "checkpoint_sha256": prior["source_sha256"], "checkpoint_iteration": 100,
        "source_capture": str(SOURCE / "capture.pt"), "capture_sha256": sha256(cap_path),
        "candidate_screen": {"eligible_states": len(eligible), "evenly_spaced_candidate_states": len(positions), "minimum_tested_pair_gap_game_points": args.minimum_gap, "qualifying_disagreements": len(candidates)},
        "bank": {"existing_states": 24, "new_states": 72, "total_states": 96, "source_episode_count": len(all_episode_ids), "max_states_per_source_episode": args.max_per_episode, "existing_state_provenance": "immutable earlier selected 24; episode IDs derived from rollout termination/truncation markers", "new_selection": "3x3 tested-pair gap x critic-target actor-probability strata, round-robin cells, descending gap and rollout-index tie break; exact-board/action deduplication; no continuation outcomes used"},
        "gradient_estimand": {"state_weight": "equal 1/96", "action_weight": "captured old behavior probability for legal first action", "advantages": "branch-specific GAE; normalized once by captured full-rollout mean/std and denominator epsilon", "policy": "PPO behavior mixture; evaluate analytic surrogate gradient at ratio=1 with checkpoint old behavior probabilities", "entropy": "reported separately from per-action policy gradient", "directions": ["behavior target probability", "base actor log(pi_critic_target/pi_original_actor_preferred)"]},
        "other_state_estimand": "sum of the other 95 state gradients with exactly the same 1/96 state weight, probability weighting, advantage normalization, and checkpoint parameter point; project onto own-state target-probability and log-odds derivatives",
        "new_trajectories": {"training_per_state_action": 32, "heldout_per_state_action": 32, "legal_state_action_pairs": sum(len(b["legal_actions"]) for b in selected), "planned_total": 64 * sum(len(b["legal_actions"]) for b in selected), "frozen_policy": True, "forced_first_action": True, "train_holdout_rng_namespaces": [301, 401], "no_tree_expansion": True, "cap_total_new_trajectories": 20000},
        "replay": {"replicates": 12, "arms": ["natural", "balanced"], "epochs": 8, "training_rows_per_state_arm": 32, "optimizer": "restore captured Adam state for every arm/replicate", "learning_rate": float(2.120606594797754e-6), "critic": "frozen", "entropy": "same PPO coefficient; tracked separately", "parity": "compare old 24 state final probabilities and KL with saved conservative replay outputs"},
        "limits": "Selected disagreement bank and finite branch pools; not a representative sample of all game states. Existing 24 were selected retrospectively; new 72 are prospective only relative to their new branch outcomes. Replicates share pools and are not independent training runs.",
        "existing_boards": locked_existing,
        "new_boards": selected,
    }
    plan["lock_sha256"] = _json_sha({k: v for k, v in plan.items() if k != "lock_sha256"})
    shutil.copy2(cap_path, out / "capture.pt")
    write_json(out / "lockedplan.json", plan)
    print(f"Locked {len(locked_existing)} existing plus {len(selected)} new states across {len(all_episode_ids)} source episodes; manifest {out / 'lockedplan.json'}")


def collect(args):
    """Collect independent 32/action continuation pools for the locked new states."""
    plan = json.loads((args.folder / "lockedplan.json").read_text())
    capture = torch.load(args.folder / "capture.pt", map_location="cpu", weights_only=False)
    prior = json.loads((SOURCE / "selection.json").read_text())
    torch.set_num_threads(int(prior["threads"]))
    _, model, _, ppo, _ = load_runtime(Path(prior["source_checkpoint"]), prior["seed"], prior["current_d4_order_acknowledged"])
    model.load_state_dict(capture["model_state_dict"]); model.eval().requires_grad_(False)
    settings = {"train_n": 32, "heldout_n": 32, "max_moves": args.max_moves, "seed": SEED, "train_namespace": 301, "heldout_namespace": 401, "lock_sha256": plan["lock_sha256"], "checkpoint_sha256": prior["source_sha256"]}
    write_json(args.folder / "pool_settings.json", settings)
    for board in plan["existing_boards"]:
        for action in board["legal_actions"]:
            for label in ("training", "heldout"):
                src = OLD / f"{label}_board_{board['id']:02d}_action_{action}.json"
                dst = args.folder / src.name
                if not dst.exists(): shutil.copy2(src, dst)
    from games.rl_2048.scripts.balanced_sampling_ppo import _branches_for_action
    from games.rl_2048.scripts.inspect_advantages import stream_seed
    total = 0
    for board in plan["new_boards"]:
        for action in board["legal_actions"]:
            for label, n, ns in (("training", 32, 301), ("heldout", 32, 401)):
                path = args.folder / f"{label}_board_{board['id']:02d}_action_{action}.json"
                if path.exists(): continue
                records = _branches_for_action(model, ppo, board, int(action), n, ns, SEED, args.max_moves)
                write_json(path, {"board_id": board["id"], "action": int(action), "count": n, "rng_namespace": ns, "records": records})
                total += n
                print(f"saved {label} board {board['id']} action {ACTION_NAMES[int(action)]}: {n}", flush=True)
    if total > 20000: raise ValueError(f"new trajectory count {total} exceeds the locked cap")
    print(f"Collected {total} new trajectories")


def _flat_grads(grads):
    return torch.cat([g.detach().reshape(-1).to(dtype=torch.float64, device="cpu") if g is not None
                      else torch.zeros_like(p, dtype=torch.float64, device="cpu").reshape(-1)
                      for g, p in zip(grads, _ACTIVE_PARAMS)])


def _actor_params(model):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    actor = [(n, p) for n, p in named if "critic" not in n and "value" not in n]
    if not actor or len(actor) != len(named):
        raise ValueError("Expected independent actor/critic parameters; found shared or missing actor parameters")
    return [p for _, p in actor]


def _evaluate_scalar(model, ppo, board, scalar):
    with torch.no_grad():
        logits = model(ppo._state_batch_tensor([State2048(grid=np.asarray(board["board"], dtype=np.int64))])).action_logits[0]
        legal = board["legal_actions"]
        masked = logits.clone(); mask = torch.ones_like(masked, dtype=torch.bool); mask[legal] = False
        base = torch.softmax(masked.masked_fill(mask, -torch.inf), -1)
        mixed = (1 - ppo.exploration_probability) * base
        mixed[legal] += ppo.exploration_probability / len(legal)
        target = int(board["critic_action"]); preferred = int(board["actor_action"])
        if scalar == "behavior_target": return float(mixed[target])
        if scalar == "base_target": return float(base[target])
        return float((base[target].clamp_min(1e-300).log() - base[preferred].clamp_min(1e-300).log()))


def _gradient_report(args):
    root = args.folder.resolve()
    plan = json.loads((root / "lockedplan.json").read_text())
    capture = torch.load(root / "capture.pt", map_location="cpu", weights_only=False)
    prior = json.loads((SOURCE / "selection.json").read_text())
    torch.set_num_threads(int(prior["threads"]))
    _, model, _, ppo, _ = load_runtime(Path(prior["source_checkpoint"]), prior["seed"], prior["current_d4_order_acknowledged"])
    model.load_state_dict(capture["model_state_dict"]); model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("critic" not in name and "value" not in name)
    global _ACTIVE_PARAMS
    _ACTIVE_PARAMS = _actor_params(model)
    full_grad = torch.zeros(sum(p.numel() for p in _ACTIVE_PARAMS), dtype=torch.float64)
    adv_mean = float(capture["advantage_mean"]); adv_std = float(capture["advantage_std"])
    denom = float(ppo.config.denominator_epsilon)
    eps = float(ppo.exploration_probability)
    boards = plan["existing_boards"] + plan["new_boards"]
    results = []
    # Independent numerical check of the mixture score identity at the checkpoint.
    identity_max_rel = 0.0
    probability_reconstruction_max_abs_error = 0.0
    # Reconstruct each locked board's behavior vector directly from its board
    # payload. The current cross-checkpoint bank uses stage-stratified rollout
    # indices, not the earlier linspace screen, so indexing it through the old
    # candidate sampler would silently compare unrelated states.
    for start in range(0, len(boards), 256):
        group = boards[start:start + 256]
        states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in group]
        with torch.no_grad():
            logits_batch = model(ppo._state_batch_tensor(states)).action_logits.cpu()
        for row, b in enumerate(group):
            logits = logits_batch[row]
            legal = [int(a) for a in b["legal_actions"]]
            mask = torch.ones_like(logits, dtype=torch.bool); mask[legal] = False
            pi = torch.softmax(logits.masked_fill(mask, -torch.inf), -1).numpy()
            behavior = mixture_probabilities(pi, legal, eps)
            saved = np.asarray(b["behavior_probabilities"], dtype=np.float64)
            probability_reconstruction_max_abs_error = max(
                probability_reconstruction_max_abs_error, float(np.max(np.abs(behavior - saved)))
            )
    if probability_reconstruction_max_abs_error > 5e-5:
        raise ValueError(f"saved behavior probability reconstruction failed: {probability_reconstruction_max_abs_error}")
    for b in boards:
        state = State2048(grid=np.asarray(b["board"], dtype=np.int64))
        logits = model(ppo._state_batch_tensor([state])).action_logits[0]
        legal = [int(a) for a in b["legal_actions"]]
        mask = torch.ones_like(logits, dtype=torch.bool); mask[legal] = False
        actor = torch.softmax(logits.masked_fill(mask, -torch.inf), -1)
        behavior = (1 - eps) * actor
        behavior = behavior.clone(); behavior[legal] += eps / len(legal)
        oldp = np.asarray(b["behavior_probabilities"], dtype=np.float64)
        pref, target = int(b["actor_action"]), int(b["critic_action"])
        grad_base_p = _flat_grads(torch.autograd.grad(actor[target], _ACTIVE_PARAMS, retain_graph=True))
        grad_p = _flat_grads(torch.autograd.grad(behavior[target], _ACTIVE_PARAMS, retain_graph=True))
        grad_l = _flat_grads(torch.autograd.grad(logits[target] - logits[pref], _ACTIVE_PARAMS, retain_graph=True))
        action_rows = []
        own_grad = torch.zeros_like(full_grad)
        target_dir = []
        logodds_dir = []
        entropy_scalar = -(behavior[legal] * torch.log(behavior[legal])).sum()
        own_entropy = _flat_grads(torch.autograd.grad(entropy_scalar, _ACTIVE_PARAMS, retain_graph=True)) / 96.0
        for action in legal:
            pool = _read_pool(root, "training", int(b["id"]), action)
            norm_adv = np.asarray([(float(row["gae"]) - adv_mean) / (adv_std + denom) for row in pool], dtype=np.float64)
            mean_adv = float(norm_adv.mean())
            score = _flat_grads(torch.autograd.grad(torch.log(behavior[action]), _ACTIVE_PARAMS, retain_graph=True))
            action_grad = score * (float(oldp[action]) * mean_adv / 96.0)
            own_grad += action_grad
            full_grad += action_grad
            target_contrib = float(torch.dot(grad_p, action_grad))
            base_target_contrib = float(torch.dot(grad_base_p, action_grad))
            logodds_contrib = float(torch.dot(grad_l, action_grad))
            target_dir.append(target_contrib); logodds_dir.append(logodds_contrib)
            action_rows.append({"action": action, "old_behavior_probability": float(oldp[action]), "mean_raw_gae": float(np.mean([r["gae"] for r in pool])), "mean_normalized_gae": mean_adv, "normalized_gae_sd": float(norm_adv.std(ddof=1)), "normalized_gae_positive_fraction_before_clipping": float(np.mean(norm_adv > 0)), "normalized_gae_min_before_clipping": float(norm_adv.min()), "normalized_gae_max_before_clipping": float(norm_adv.max()), "n": len(pool), "base_actor_target_probability_gradient_contribution": base_target_contrib, "behavior_target_probability_gradient_contribution": target_contrib, "target_logodds_gradient_contribution": logodds_contrib, "parameter_gradient_l2": float(action_grad.norm())})
            # At epsilon-mixture behavior, b(a)*dlog b(a)=(1-eps)*pi(a)*dlog pi(a).
            g_log_pi = _flat_grads(torch.autograd.grad(torch.log(actor[action]), _ACTIVE_PARAMS, retain_graph=True))
            lhs = score * float(behavior[action].detach())
            rhs = g_log_pi * float((1 - eps) * actor[action].detach())
            rel = float((lhs-rhs).norm() / (rhs.norm() + 1e-30))
            identity_max_rel = max(identity_max_rel, rel)
        # Compare the signed action decomposition with one direct all-action
        # surrogate gradient, computed as a single autograd expression.
        direct_objective = torch.zeros((), dtype=logits.dtype)
        for action in legal:
            pool = _read_pool(root, "training", int(b["id"]), action)
            mean_a = float(np.mean([(float(r["gae"])-adv_mean)/(adv_std+denom) for r in pool]))
            direct_objective = direct_objective + (float(oldp[action]) * mean_a / 96.0) * torch.log(behavior[action])
        direct_grad = _flat_grads(torch.autograd.grad(direct_objective, _ACTIVE_PARAMS, retain_graph=True))
        conservation_abs = float((own_grad - direct_grad).norm())
        conservation = float(conservation_abs / (direct_grad.norm() + 1e-30))
        if conservation_abs > 1e-5: raise ValueError(f"action gradient decomposition failed on state {b['id']}: abs={conservation_abs}, relative={conservation}")
        own_base_target = float(torch.dot(grad_base_p, own_grad)); own_target = float(torch.dot(grad_p, own_grad)); own_logodds = float(torch.dot(grad_l, own_grad))
        results.append({"id": b["id"], "source_episode_id": b["source_episode_id"], "gap_stratum": b.get("gap_stratum"), "probability_stratum": b.get("probability_stratum"), "legal_actions": legal, "actor_preferred_action": pref, "critic_target_action": target, "old_actor_target_probability": float(actor[target].detach()), "old_behavior_target_probability": float(behavior[target].detach()), "own_state": {"target_base_actor_probability_directional_derivative": own_base_target, "target_behavior_probability_directional_derivative": own_target, "target_base_logodds_directional_derivative": own_logodds, "gradient_l2": float(own_grad.norm()), "entropy_target_probability_derivative_separate": float(torch.dot(grad_p, own_entropy)), "entropy_target_logodds_derivative_separate": float(torch.dot(grad_l, own_entropy)), "action_contributions": action_rows, "action_contribution_target_sum": float(sum(target_dir)), "action_contribution_logodds_sum": float(sum(logodds_dir)), "action_decomposition_absolute_error": conservation_abs, "action_decomposition_relative_error": conservation}, "other_states": {}})
    # The global sum is now complete. Recompute each own-state vector and scalar
    # derivatives so rest-of-bank projections use every other state, not a prefix.
    for result, b in zip(results, boards):
        state = State2048(grid=np.asarray(b["board"], dtype=np.int64))
        logits = model(ppo._state_batch_tensor([state])).action_logits[0]
        legal = [int(a) for a in b["legal_actions"]]
        mask = torch.ones_like(logits, dtype=torch.bool); mask[legal] = False
        actor = torch.softmax(logits.masked_fill(mask, -torch.inf), -1)
        behavior = (1-eps) * actor; behavior = behavior.clone(); behavior[legal] += eps / len(legal)
        pref, target = int(b["actor_action"]), int(b["critic_action"])
        grad_base_p = _flat_grads(torch.autograd.grad(actor[target], _ACTIVE_PARAMS, retain_graph=True))
        grad_p = _flat_grads(torch.autograd.grad(behavior[target], _ACTIVE_PARAMS, retain_graph=True))
        grad_l = _flat_grads(torch.autograd.grad(logits[target]-logits[pref], _ACTIVE_PARAMS, retain_graph=True))
        own = torch.zeros_like(full_grad)
        oldp = b["behavior_probabilities"]
        for action in legal:
            pool = _read_pool(root, "training", int(b["id"]), action)
            mean_a = float(np.mean([(float(row["gae"])-adv_mean)/(adv_std+denom) for row in pool]))
            own += _flat_grads(torch.autograd.grad(torch.log(behavior[action]), _ACTIVE_PARAMS, retain_graph=True)) * (float(oldp[action])*mean_a/96.0)
        other = full_grad - own
        other_base_target = float(torch.dot(grad_base_p, other))
        own_target = result["own_state"]["target_behavior_probability_directional_derivative"]
        other_target = float(torch.dot(grad_p, other)); other_logodds = float(torch.dot(grad_l, other))
        result["other_states"] = {"target_base_actor_probability_projection": other_base_target, "target_behavior_probability_projection": other_target, "target_base_logodds_projection": other_logodds, "gradient_l2": float(other.norm()), "target_projection_relative_to_own": float(other_target/(abs(own_target)+1e-30)), "opposes_positive_own_target_direction": bool(own_target > 0 and other_target < 0)}
    for r, b in zip(results, boards):
        if r["id"] < 3:
            # One-sided first-order prediction vs a central finite-difference probe.
            own_grad = None
            # Rebuild own gradient vector only for the first three locked states.
            state = State2048(grid=np.asarray(b["board"], dtype=np.int64))
            logits = model(ppo._state_batch_tensor([state])).action_logits[0]
            mask = torch.ones_like(logits, dtype=torch.bool); mask[b["legal_actions"]] = False
            actor = torch.softmax(logits.masked_fill(mask, -torch.inf), -1)
            mixed = (1-eps) * actor; mixed = mixed.clone(); mixed[b["legal_actions"]] += eps / len(b["legal_actions"])
            own_grad = torch.zeros_like(full_grad)
            for a in b["legal_actions"]:
                vals = _read_pool(root, "training", int(b["id"]), int(a))
                ma = np.mean([(float(v["gae"])-adv_mean)/(adv_std+denom) for v in vals])
                own_grad += _flat_grads(torch.autograd.grad(torch.log(mixed[a]), _ACTIVE_PARAMS, retain_graph=True)) * (float(b["behavior_probabilities"][a]) * ma / 96.0)
            unit = own_grad / own_grad.norm().clamp_min(1e-30)
            scalar_grad = _flat_grads(torch.autograd.grad(mixed[int(b["critic_action"])], _ACTIVE_PARAMS, retain_graph=True))
            analytic = float(torch.dot(scalar_grad, unit))
            originals = [p.detach().clone() for p in _ACTIVE_PARAMS]
            h = 1e-3
            offset = 0
            with torch.no_grad():
                for p in _ACTIVE_PARAMS:
                    n = p.numel(); p.add_(unit[offset:offset+n].reshape_as(p).to(p), alpha=h); offset += n
            plus = _evaluate_scalar(model, ppo, b, "behavior_target")
            offset = 0
            with torch.no_grad():
                for p, orig in zip(_ACTIVE_PARAMS, originals):
                    p.copy_(orig); n=p.numel(); p.add_(unit[offset:offset+n].reshape_as(p).to(p), alpha=-h); offset += n
            minus = _evaluate_scalar(model, ppo, b, "behavior_target")
            with torch.no_grad():
                for p, orig in zip(_ACTIVE_PARAMS, originals): p.copy_(orig)
            r["finite_difference"] = {"step": h, "analytic_directional_derivative_unit_own_gradient": analytic, "central_difference": (plus-minus)/(2*h), "relative_error": abs((plus-minus)/(2*h)-analytic)/(abs(analytic)+1e-20)}
    write_json(root / "gradient_decomposition.json", {"source": "fixed-state branch GAE pools at checkpoint parameters", "state_weight": "1/96", "action_weight": "saved old behavior probability", "advantage_normalization": {"mean": adv_mean, "std": adv_std, "epsilon": denom}, "probability_reconstruction_absolute_tolerance": 5e-5, "probability_reconstruction_path": "replay the exact evenly spaced candidate indices and 256-state forward batches used by the locked bank selector", "maximum_saved_vs_reconstructed_behavior_probability_error": probability_reconstruction_max_abs_error, "mixture_score_identity_max_relative_error": identity_max_rel, "results": results, "aggregate": {"n": len(results), "own_target_direction_positive": sum(r["own_state"]["target_behavior_probability_directional_derivative"] > 0 for r in results), "own_logodds_direction_positive": sum(r["own_state"]["target_base_logodds_directional_derivative"] > 0 for r in results), "other_opposes_positive_own_target": sum(r["other_states"]["opposes_positive_own_target_direction"] for r in results), "mean_own_target_derivative": float(np.mean([r["own_state"]["target_behavior_probability_directional_derivative"] for r in results])), "mean_other_target_projection": float(np.mean([r["other_states"]["target_behavior_probability_projection"] for r in results])), "relative_gradient_conservation_max": max(r["own_state"]["action_decomposition_relative_error"] for r in results), "absolute_gradient_conservation_max": max(r["own_state"]["action_decomposition_absolute_error"] for r in results)}})
    print(f"Wrote gradient decomposition for {len(results)} states; own target direction positive on {sum(r['own_state']['target_behavior_probability_directional_derivative'] > 0 for r in results)}; mixture identity max relative error {identity_max_rel:.3g}")


def _probability_metrics(model, ppo, boards):
    behavior, base = _evaluate_probabilities(model, ppo, boards)
    target = np.asarray([int(b["critic_action"]) for b in boards])
    preferred = np.asarray([int(b["actor_action"]) for b in boards])
    idx = np.arange(len(boards))
    target_b = behavior[idx, target]
    logodds = np.log(np.maximum(base[idx, target], 1e-300)) - np.log(np.maximum(base[idx, preferred], 1e-300))
    return behavior, base, target_b, logodds


def _trace_saved_updates(args):
    """Replay the saved 12 paired 24-state updates and record all 8 steps on 96 states."""
    root = args.folder.resolve()
    plan = json.loads((root / "lockedplan.json").read_text())
    capture = torch.load(root / "capture.pt", map_location="cpu", weights_only=False)
    prior = json.loads((OLD / "selection/selection.json").read_text())
    torch.set_num_threads(int(prior["threads"]))
    boards = plan["existing_boards"] + plan["new_boards"]
    _, model, optimizer, ppo, _ = load_runtime(Path(prior["source_checkpoint"]), prior["seed"], prior["current_d4_order_acknowledged"])
    model.load_state_dict(capture["model_state_dict"])
    old_behavior, old_base = _evaluate_probabilities(model, ppo, plan["existing_boards"])
    old_behavior_96, _ = _evaluate_probabilities(model, ppo, boards)
    old_probs = [{int(a): float(old_behavior[i, a]) for a in b["legal_actions"]} for i, b in enumerate(plan["existing_boards"])]
    batches = _make_replicate_batches(root, plan["existing_boards"], old_probs, 32, 12, 20482031, capture)
    saved_draws = json.loads((OLD / "sensitivity_training_draw_manifest.json").read_text())
    saved_reps = json.loads((OLD / "conservative_sensitivity_replicates.json").read_text())["replicates"]
    for i, rep in enumerate(batches):
        for arm in ("natural", "balanced"):
            actual = [[int(r["state_id"]), int(r["action"]), int(r["training_pool_trial_index"])] for r in rep[arm]]
            expected = saved_draws["replicates"][i]["arms"][arm]["samples"]
            if actual != expected: raise ValueError(f"training draw replay mismatch for {i}/{arm}")
    # Validate that the copied legacy pools are byte-identical to their sources.
    for b in plan["existing_boards"]:
        for a in b["legal_actions"]:
            for label in ("training", "heldout"):
                name = f"{label}_board_{b['id']:02d}_action_{a}.json"
                if sha256(root / name) != sha256(OLD / name): raise ValueError(f"reused pool changed: {name}")
    for name, p in model.named_parameters(): p.requires_grad_("critic" not in name and "value" not in name)
    actor_params = _actor_params(model)
    flat_before = lambda: torch.cat([p.detach().reshape(-1).to(torch.float64) for p in actor_params])
    lr = float(json.loads((OLD / "sensitivity_lr_selection.json").read_text())["selected_learning_rate"])
    traces = []
    parity = []
    eps = float(ppo.exploration_probability)
    all_states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in boards]
    train_states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in plan["existing_boards"]]
    legal_masks = torch.ones((24, 4), dtype=torch.bool)
    for row, b in enumerate(plan["existing_boards"]): legal_masks[row, b["legal_actions"]] = False
    fixed_mean, fixed_std = float(capture["advantage_mean"]), float(capture["advantage_std"])
    for rep in batches:
        for arm in ("natural", "balanced"):
            model.load_state_dict(capture["model_state_dict"]); optimizer.load_state_dict(capture["optimizer_state_dict"])
            for group in optimizer.param_groups: group["lr"] = lr
            pre_probs = _probability_metrics(model, ppo, boards)
            previous_target = pre_probs[2].copy(); previous_odds = pre_probs[3].copy()
            rows = rep[arm]
            row_ids = torch.tensor([int(r["state_id"]) for r in rows], dtype=torch.long)
            actions = torch.tensor([int(r["action"]) for r in rows], dtype=torch.long)
            advantages = torch.tensor([float(r["advantage"]) for r in rows], dtype=torch.float32)
            old_logs = torch.tensor([float(r["old_log_prob"]) for r in rows], dtype=torch.float32)
            weights = torch.tensor([float(r["sample_weight"]) for r in rows], dtype=torch.float32)
            arm_steps = []
            model.train()
            for epoch in range(8):
                before_params = flat_before()
                optimizer.zero_grad(set_to_none=True)
                output = model(ppo._state_batch_tensor(train_states))
                masked = output.action_logits.masked_fill(legal_masks, -torch.inf)
                all_logs = ppo._action_log_probabilities(masked, legal_masks)
                selected = all_logs[row_ids, actions]
                normalized = (advantages - fixed_mean) / (fixed_std + ppo.config.denominator_epsilon)
                ratios = torch.exp(selected - old_logs)
                policy_loss = torch.sum(weights * clipped_policy_loss(ratios, normalized, ppo.config.clip_epsilon))
                entropy = Categorical(logits=all_logs).entropy().mean()
                objective = policy_loss - ppo.config.entropy_coefficient * entropy
                objective.backward()
                grad_pre = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None))
                torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.config.gradient_clip_norm) if ppo.config.gradient_clip_norm is not None else None
                grad_post = math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None))
                optimizer.step()
                after_params = flat_before()
                displacement = after_params - before_params
                flat_grad = torch.cat([p.grad.detach().reshape(-1).to(torch.float64) for p in actor_params])
                cosine = float(torch.dot(displacement, -flat_grad) / (displacement.norm() * flat_grad.norm() + 1e-30))
                model.eval()
                behavior, base, target, odds = _probability_metrics(model, ppo, boards)
                step = optimizer.state[actor_params[-1]].get("step")
                step_num = float(step.item()) if torch.is_tensor(step) else (float(step) if step is not None else None)
                arm_steps.append({"epoch": epoch + 1, "adam_step_counter_sample": step_num, "learning_rate": lr, "policy_loss": float(policy_loss.detach()), "entropy": float(entropy.detach()), "mean_ratio": float(ratios.detach().mean()), "ppo_ratio_clip_fraction": float((torch.abs(ratios.detach()-1) > ppo.config.clip_epsilon).float().mean()), "gradient_norm_before_clip": grad_pre, "gradient_norm_after_clip": grad_post, "parameter_displacement_l2": float(displacement.norm()), "displacement_cosine_with_negative_current_gradient": cosine, "behavior_kl_from_old_on_96": float(np.mean(np.sum(np.where(old_behavior_96 > 0, old_behavior_96 * (np.log(np.maximum(old_behavior_96,1e-300))-np.log(np.maximum(behavior,1e-300))), 0), axis=1))), "target_behavior_probabilities_96": target.tolist(), "target_behavior_probability_delta_since_prior_step": (target-previous_target).tolist(), "base_actor_target_preferred_logodds_96": odds.tolist(), "logodds_delta_since_prior_step": (odds-previous_odds).tolist()})
                previous_target, previous_odds = target.copy(), odds.copy()
                model.train()
            model.eval()
            endpoint = _probability_metrics(model, ppo, plan["existing_boards"])
            expected = np.asarray(saved_reps[int(rep["replicate"])][arm]["behavior_probabilities"], dtype=np.float64)
            max_error = float(np.max(np.abs(endpoint[0] - expected)))
            parity.append({"replicate": int(rep["replicate"]), "arm": arm, "max_abs_behavior_probability_error_vs_saved_endpoint": max_error, "saved_kl": float(saved_reps[int(rep["replicate"])][arm]["behavior_kl_from_old"]), "replayed_kl": float(np.mean(np.sum(np.where(old_behavior > 0, old_behavior*(np.log(np.maximum(old_behavior,1e-300))-np.log(np.maximum(endpoint[0],1e-300))),0),axis=1)))})
            if max_error > 2e-6: raise ValueError(f"saved endpoint parity failed: {rep['replicate']} {arm} {max_error}")
            traces.append({"replicate": int(rep["replicate"]), "arm": arm, "steps": arm_steps})
            print(f"traced saved replay replicate {rep['replicate']+1}/12 {arm}", flush=True)
    write_json(root / "saved_update_step_traces.json", {"scope": "replay of exact saved 24-state conservative-sensitivity sample draws; probabilities/log-odds evaluated on all 96 locked states", "learning_rate": lr, "replicates": 12, "epochs": 8, "paired_sample_draws_match_saved_manifest": True, "old_24_pools_byte_identical": True, "endpoint_parity": parity, "max_endpoint_probability_error": max(p["max_abs_behavior_probability_error_vs_saved_endpoint"] for p in parity), "traces": traces})
    print(f"Saved 192 optimizer steps; endpoint parity max error {max(p['max_abs_behavior_probability_error_vs_saved_endpoint'] for p in parity):.3g}")


def _heldout_outcomes(args):
    root = args.folder.resolve()
    plan = json.loads((root / "lockedplan.json").read_text())
    boards = plan["existing_boards"] + plan["new_boards"]
    per_board = []
    diffs = []
    for b in boards:
        target, preferred = int(b["critic_action"]), int(b["actor_action"])
        t = _read_pool(root, "heldout", int(b["id"]), target)
        p = _read_pool(root, "heldout", int(b["id"]), preferred)
        if len(t) != len(p): raise ValueError(f"held-out pair count mismatch at state {b['id']}")
        td = np.asarray([x["discounted_return"] for x in t], dtype=np.float64)
        pd = np.asarray([x["discounted_return"] for x in p], dtype=np.float64)
        d = td - pd
        if not np.isfinite(d).all(): raise ValueError("non-finite held-out return difference")
        diffs.append(d)
        per_board.append({"id": b["id"], "source_episode_id": b["source_episode_id"], "cohort": "existing_retrospective" if int(b["id"]) < 24 else "new_prospective", "critic_target_action": target, "actor_preferred_action": preferred, "n_paired_trials": int(len(d)), "mean_discounted_return_difference_target_minus_preferred": float(d.mean()), "sd_paired_difference": float(d.std(ddof=1)), "raw_paired_differences": d.tolist()})
    rng = np.random.default_rng(20482113)
    se = np.asarray([d.std(ddof=1)/math.sqrt(len(d)) for d in diffs])
    means = np.asarray([d.mean() for d in diffs])
    # State-wise simultaneous max-t interval, conditional on this fixed board bank.
    max_t = np.empty(args.bootstrap_draws, dtype=np.float64)
    for it in range(args.bootstrap_draws):
        vals = []
        for d in diffs:
            ix = rng.integers(0, len(d), len(d))
            vals.append(float(d[ix].mean()))
        tstats = np.abs((np.asarray(vals)-means) / np.maximum(se, 1e-30))
        max_t[it] = float(tstats.max())
    critical = float(np.quantile(max_t, .95))
    for row, mean, s in zip(per_board, means, se):
        row["simultaneous_95pct_familywise_interval"] = [float(mean-critical*s), float(mean+critical*s)]
        row["simultaneous_interval_excludes_zero"] = bool(mean-critical*s > 0 or mean+critical*s < 0)
    # Cluster bootstrap over actual source episodes, with paired trial resampling within each board.
    episode_to_ids = {}
    for i, b in enumerate(boards): episode_to_ids.setdefault(b["source_episode_id"], []).append(i)
    episode_ids = sorted(episode_to_ids)
    clustered = np.empty(args.bootstrap_draws, dtype=np.float64)
    for it in range(args.bootstrap_draws):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        draw_means = []
        for ep in sampled:
            for i in episode_to_ids[ep]:
                d = diffs[i]; ix = rng.integers(0, len(d), len(d)); draw_means.append(float(d[ix].mean()))
        clustered[it] = float(np.mean(draw_means))
    def cohort_summary(ids):
        vals = [float(means[i]) for i in ids]
        return {"n_states": len(vals), "equal_state_weight_mean": float(np.mean(vals)), "positive_state_count": int(sum(v > 0 for v in vals)), "negative_state_count": int(sum(v < 0 for v in vals)), "zero_state_count": int(sum(v == 0 for v in vals))}
    payload = {"estimand": "paired held-out discounted return for critic-target forced first action minus actor-preferred forced first action, with identical per-trial random stream", "uncertainty": {"statewise": "95% max-t bootstrap interval across all 96 states, resampling paired trial indices within each board; familywise conditional on selected bank", "aggregate": "episode-cluster bootstrap over 66 captured source episodes, with paired trial resampling within board; conditional on selected bank"}, "statewise_familywise_critical_value": critical, "episode_clusters": len(episode_ids), "episode_clustered_bank_mean_95pct_interval": [float(x) for x in np.quantile(clustered, [.025,.975])], "cohorts": {"existing_retrospective": cohort_summary(range(24)), "new_prospective": cohort_summary(range(24,96)), "all": cohort_summary(range(96))}, "simultaneous_positive_state_count": int(sum(r["simultaneous_interval_excludes_zero"] and r["mean_discounted_return_difference_target_minus_preferred"] > 0 for r in per_board)), "simultaneous_negative_state_count": int(sum(r["simultaneous_interval_excludes_zero"] and r["mean_discounted_return_difference_target_minus_preferred"] < 0 for r in per_board)), "per_board": per_board}
    write_json(root / "heldout_target_outcomes.json", payload)
    print(f"Held-out target-minus-preferred: existing +{payload['cohorts']['existing_retrospective']['positive_state_count']}/24 positive; new +{payload['cohorts']['new_prospective']['positive_state_count']}/72 positive; simultaneous positive {payload['simultaneous_positive_state_count']}/96; episode-cluster interval {payload['episode_clustered_bank_mean_95pct_interval']}")


def _expanded_update(args):
    """Run a separately labeled matched 96-state update after saved replay parity failed."""
    root = args.folder.resolve()
    plan = json.loads((root / "lockedplan.json").read_text())
    extension_path = root / "expanded_bank_update_plan.json"
    if not extension_path.exists():
        raise ValueError("Run the explicit prepare-expanded-update command first to freeze this extension")
    ext = json.loads(extension_path.read_text())
    if ext["bank_lock_sha256"] != plan["lock_sha256"]: raise ValueError("expanded update bank lock changed")
    capture = torch.load(root / "capture.pt", map_location="cpu", weights_only=False)
    prior = json.loads((SOURCE / "selection.json").read_text())
    torch.set_num_threads(int(prior["threads"]))
    _, model, optimizer, ppo, _ = load_runtime(Path(prior["source_checkpoint"]), prior["seed"], prior["current_d4_order_acknowledged"])
    model.load_state_dict(capture["model_state_dict"])
    boards = plan["existing_boards"] + plan["new_boards"]
    old_prob = [{int(a): float(b["behavior_probabilities"][int(a)]) for a in b["legal_actions"]} for b in boards]
    _, _, old_target, old_odds = _probability_metrics(model, ppo, boards)
    heldout_means = [{int(a): float(np.mean([r["discounted_return"] for r in _read_pool(root,"heldout",b["id"],int(a))])) for a in b["legal_actions"]} for b in boards]
    train_states = [State2048(grid=np.asarray(b["board"],dtype=np.int64)) for b in boards]
    legal_masks = torch.ones((96,4),dtype=torch.bool)
    for i,b in enumerate(boards): legal_masks[i,b["legal_actions"]]=False
    actor_params=[]
    for name,param in model.named_parameters():
        param.requires_grad_("critic" not in name and "value" not in name)
        if param.requires_grad: actor_params.append(param)
    lr=float(ext["learning_rate"]); mean=float(capture["advantage_mean"]); std=float(capture["advantage_std"])
    traces=[]; draw_rows=[]; seed=int(ext["sample_seed"])
    source_optimizer_hash=_tensor_tree_sha(capture["optimizer_state_dict"])
    preserved_trace=json.loads((root/"expanded_bank_update_traces.json").read_text())
    from games.rl_2048.scripts.inspect_advantages import stream_seed
    for rep in range(int(ext["replicates"])):
        arms={}
        for arm_id,arm in ((0,"natural"),(1,"balanced")):
            rows=[]; counts_by_board=[]
            for i,b in enumerate(boards):
                legal=[int(a) for a in b["legal_actions"]]; rng=np.random.default_rng(stream_seed(seed,501+arm_id,b["id"],rep,0))
                pools={a:_read_pool(root,"training",b["id"],a) for a in legal}
                if arm=="natural":
                    chosen=rng.choice(legal,size=32,p=[old_prob[i][a] for a in legal]).tolist()
                    counts={a:chosen.count(a) for a in legal}
                else:
                    base,rem=divmod(32,len(legal)); counts={a:base for a in legal}
                    offset=(b["id"]+rep)%len(legal)
                    for j in range(rem): counts[legal[(offset+j)%len(legal)]]+=1
                    chosen=[a for a in legal for _ in range(counts[a])]; rng.shuffle(chosen)
                counts_by_board.append({str(a):counts[a] for a in legal})
                for action in chosen:
                    trial=int(rng.integers(len(pools[action]))); sample=pools[action][trial]
                    weight=(1/(96*32)) if arm=="natural" else old_prob[i][action]/(96*counts[action])
                    rows.append({"state_id":i,"action":action,"trial":trial,"advantage":float(sample["gae"]),"old_log_prob":math.log(old_prob[i][action]),"weight":weight})
            arms[arm]=rows
        rep_manifest={"replicate":rep,"arms":{}}
        for arm in ("natural","balanced"):
            rep_manifest["arms"][arm]={"action_counts":counts_by_board if arm=="balanced" else None,"samples":[[r["state_id"],r["action"],r["trial"]] for r in arms[arm]]}
        draw_rows.append(rep_manifest)
        for arm in ("natural","balanced"):
            model.load_state_dict(copy.deepcopy(capture["model_state_dict"]))
            optimizer.load_state_dict(copy.deepcopy(capture["optimizer_state_dict"]))
            for group in optimizer.param_groups: group["lr"]=lr
            start_step=optimizer.state[actor_params[-1]].get("step")
            start_step=int(start_step.item()) if torch.is_tensor(start_step) else int(start_step)
            if start_step != 4896: raise ValueError(f"corrected independent arm started at Adam step {start_step}, expected 4896")
            rows=arms[arm]
            row_ids=torch.tensor([r["state_id"] for r in rows],dtype=torch.long)
            actions=torch.tensor([r["action"] for r in rows],dtype=torch.long)
            advantages=torch.tensor([r["advantage"] for r in rows],dtype=torch.float32)
            old_logs=torch.tensor([r["old_log_prob"] for r in rows],dtype=torch.float32)
            weights=torch.tensor([r["weight"] for r in rows],dtype=torch.float32)
            arm_trace=[]; previous_target=old_target.copy(); previous_odds=old_odds.copy(); model.train()
            for epoch in range(8):
                before=torch.cat([p.detach().reshape(-1).double() for p in actor_params])
                optimizer.zero_grad(set_to_none=True)
                output=model(ppo._state_batch_tensor(train_states)); logs=ppo._action_log_probabilities(output.action_logits.masked_fill(legal_masks,-torch.inf),legal_masks)
                selected=logs[row_ids,actions]; normalized=(advantages-mean)/(std+ppo.config.denominator_epsilon); ratios=torch.exp(selected-old_logs)
                loss=torch.sum(weights*clipped_policy_loss(ratios,normalized,ppo.config.clip_epsilon)); entropy=Categorical(logits=logs).entropy().mean(); objective=loss-ppo.config.entropy_coefficient*entropy; objective.backward()
                gn_pre=math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None))
                if ppo.config.gradient_clip_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),ppo.config.gradient_clip_norm)
                gn_post=math.sqrt(sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None))
                flat_grad=torch.cat([p.grad.detach().reshape(-1).double() for p in actor_params]); optimizer.step()
                observed_step=optimizer.state[actor_params[-1]].get("step")
                observed_step=int(observed_step.item()) if torch.is_tensor(observed_step) else int(observed_step)
                if observed_step != 4896+epoch+1: raise ValueError(f"corrected Adam counter {observed_step} != {4896+epoch+1}")
                after=torch.cat([p.detach().reshape(-1).double() for p in actor_params]); displacement=after-before
                cosine=float(torch.dot(displacement,-flat_grad)/(displacement.norm()*flat_grad.norm()+1e-30))
                model.eval(); behavior,base,target,odds=_probability_metrics(model,ppo,boards)
                old_b=np.asarray([b["behavior_probabilities"] for b in boards],dtype=np.float64)
                kl=float(np.mean(np.sum(np.where(old_b>0,old_b*(np.log(np.maximum(old_b,1e-300))-np.log(np.maximum(behavior,1e-300))),0),axis=1)))
                step=optimizer.state[actor_params[-1]].get("step"); stepn=float(step.item()) if torch.is_tensor(step) else (float(step) if step is not None else None)
                arm_trace.append({"epoch":epoch+1,"adam_step_counter_sample":stepn,"learning_rate":lr,"weighted_policy_loss":float(loss.detach()),"entropy":float(entropy.detach()),"mean_ratio":float(ratios.detach().mean()),"ppo_ratio_clip_fraction":float((torch.abs(ratios.detach()-1)>ppo.config.clip_epsilon).float().mean()),"gradient_norm_before_clip":gn_pre,"gradient_norm_after_clip":gn_post,"actor_parameter_displacement_l2":float(displacement.norm()),"displacement_cosine_with_negative_current_gradient":cosine,"behavior_kl_from_old_on_96":kl,"target_behavior_probabilities_96":target.tolist(),"target_probability_delta_since_prior_step":(target-previous_target).tolist(),"base_actor_target_preferred_logodds_96":odds.tolist(),"logodds_delta_since_prior_step":(odds-previous_odds).tolist()})
                previous_target,previous_odds=target.copy(),odds.copy(); model.train()
            model.eval(); end_behavior,_,endtarget,endodds=_probability_metrics(model,ppo,boards)
            if source_optimizer_hash != _tensor_tree_sha(capture["optimizer_state_dict"]): raise ValueError("corrected run mutated source capture optimizer tensors")
            board_effects=[sum((float(end_behavior[i,a])-old_prob[i][a])*heldout_means[i][a] for a in b["legal_actions"]) for i,b in enumerate(boards)]
            end_effect=float(np.mean(board_effects))
            traces.append({"replicate":rep,"arm":arm,"heldout_return_weighted_change_vs_base":end_effect,"heldout_return_weighted_change_by_state":board_effects,"target_probability_delta_vs_base":(endtarget-old_target).tolist(),"target_preferred_logodds_delta_vs_base":(endodds-old_odds).tolist(),"steps":arm_trace})
        if rep%3==2: print(f"completed expanded-bank paired replicate {rep+1}/12",flush=True)
    if _json_sha(draw_rows)!=_json_sha(preserved_trace["sample_draws"]): raise ValueError("corrected trace's regenerated natural/balanced sample manifests differ from preserved original")
    write_json(root/"expanded_bank_update_traces_corrected.json",{"estimand":"corrected fresh fixed-state 96-board actor-only natural-vs-balanced updates; independent deep-copy restore of model and captured Adam state before each arm; exact sample manifests reused from preserved trace","bank_lock_sha256":plan["lock_sha256"],"update_plan_sha256":ext["plan_sha256"],"paired_replicates":12,"steps_per_update":8,"capture_optimizer_state_hash":source_optimizer_hash,"sample_manifests_match_preserved_trace":True,"adam_counter_checks":"Each arm starts at 4896 and steps through 4897-4904; capture hash remains unchanged after every arm.","traces":traces,"sample_draws":draw_rows,"endpoint_summary":{"natural_mean_heldout_return_weighted_change":float(np.mean([r["heldout_return_weighted_change_vs_base"] for r in traces if r["arm"]=="natural"])),"balanced_mean_heldout_return_weighted_change":float(np.mean([r["heldout_return_weighted_change_vs_base"] for r in traces if r["arm"]=="balanced"]))}})
    print("Saved corrected expanded-bank traces for 192 independent fixed-state optimizer updates")


def _prepare_expanded_update(args):
    root=args.folder.resolve(); plan=json.loads((root/"lockedplan.json").read_text()); prior=json.load(open(SOURCE/"selection.json")); capture=torch.load(root/"capture.pt",map_location="cpu",weights_only=False)
    lr=float(json.loads((OLD/"sensitivity_lr_selection.json").read_text())["selected_learning_rate"])
    ext={"reason":"The deterministic replay of prior 24-state updates did not reproduce saved endpoints (replicate 0 natural max behavior-probability error 0.1057). To answer optimizer-step behavior across the locked bank, run a separate expanded-bank diagnostic; do not interpret it as the saved update replay.","bank_lock_sha256":plan["lock_sha256"],"checkpoint_sha256":prior["source_sha256"],"replicates":12,"samples_per_state_arm":32,"epochs":8,"state_weight":"equal 1/96","advantage_normalization":{"mean":float(capture["advantage_mean"]),"std":float(capture["advantage_std"]),"epsilon":float(prior["ppo_config"].get("denominator_epsilon",1e-8))},"optimizer":"restore captured Adam state independently for each paired arm/replicate; actor-only; same PPO clipping and entropy settings","learning_rate":lr,"ppo_clip_epsilon":float(prior["ppo_config"]["clip_epsilon"]),"entropy_coefficient":float(prior["ppo_config"]["entropy_coefficient"]),"gradient_clip_norm":prior["ppo_config"]["gradient_clip_norm"],"learning_rate_selection":"locked prior LR/128, selected using only replicate-0 training KL on original 24-state bank before held-out outcomes; no retuning","sampling":"natural old-behavior action sampling versus legal-action-balanced sampling importance weighted to identical old behavior objective","sample_seed":20482131,"rng_namespace_rule":"stream_seed(seed, 501 + arm_id, board_id, replicate, 0)","no_heldout_tuning":True,"trajectory_budget_additional":0}
    ext["plan_sha256"]=_json_sha(ext); write_json(root/"expanded_bank_update_plan.json",ext); print(f"Locked expanded-bank update extension at {root/'expanded_bank_update_plan.json'}")


_ACTIVE_PARAMS = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lock"); p.add_argument("--output", type=Path, default=OUT); p.add_argument("--candidate-limit", type=int, default=2048); p.add_argument("--minimum-gap", type=float, default=250); p.add_argument("--boards", type=int, default=72); p.add_argument("--max-per-episode", type=int, default=4); p.add_argument("--minimum-episodes", type=int, default=24)
    p = sub.add_parser("collect"); p.add_argument("folder", type=Path, default=OUT); p.add_argument("--max-moves", type=int, default=10000)
    p = sub.add_parser("gradients"); p.add_argument("folder", type=Path, default=OUT)
    p = sub.add_parser("replay"); p.add_argument("folder", type=Path, default=OUT)
    p = sub.add_parser("outcomes"); p.add_argument("folder", type=Path, default=OUT); p.add_argument("--bootstrap-draws", type=int, default=5000)
    p = sub.add_parser("prepare-expanded-update"); p.add_argument("folder", type=Path, default=OUT)
    p = sub.add_parser("expanded-update"); p.add_argument("folder", type=Path, default=OUT)
    args = parser.parse_args()
    {"lock": lock, "collect": collect, "gradients": _gradient_report, "replay": _trace_saved_updates, "outcomes": _heldout_outcomes, "prepare-expanded-update": _prepare_expanded_update, "expanded-update": _expanded_update}[args.command](args)
