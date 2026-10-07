"""Fixed-state natural-versus-balanced PPO sampling diagnostic.

This is a separate diagnostic from production PPO training. See
artifacts/ppo_2048/ppo_balanced_sampling_iter100/lockedplan.json.
"""
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

from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.scripts.audit_ppo_update import (
    ACTION_NAMES,
    counterfactual_gae_horizon,
    generalized_advantage_prefix,
    load_runtime,
    mixture_probabilities,
    sha256,
    write_json,
)
from games.rl_2048.scripts.inspect_advantages import stream_seed


VERSION = 1


def balanced_action_counts(legal_actions, budget: int, offset: int = 0):
    """Allocate a fixed first-action budget as evenly as possible."""
    legal = list(legal_actions)
    if not legal or budget < len(legal):
        raise ValueError("budget must cover every legal action at least once")
    base, remainder = divmod(budget, len(legal))
    counts = {int(a): base for a in legal}
    for j in range(remainder):
        counts[int(legal[(offset + j) % len(legal)])] += 1
    return counts


def balanced_sample_weights(old_probabilities, counts, n_states: int):
    """Per-sample weights whose sum estimates E_{a~p_old}[loss(a)]."""
    return {int(a): float(p) / (n_states * counts[int(a)]) for a, p in old_probabilities.items()}


def clipped_policy_loss(ratio: torch.Tensor, normalized_advantage: torch.Tensor, clip_epsilon: float):
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.minimum(ratio * normalized_advantage, clipped * normalized_advantage)


def _load_inputs(root: Path):
    selection_dir = root / "selection"
    manifest = json.loads((selection_dir / "selection.json").read_text())
    source = Path(manifest["source_checkpoint"])
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Source checkpoint hash changed")
    capture = torch.load(selection_dir / "capture.pt", map_location="cpu", weights_only=False)
    if capture.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Capture/checkpoint provenance mismatch")
    return manifest, capture


def _branches_for_action(model, ppo, board, action: int, count: int, phase: int, seed: int, max_moves: int):
    """Run one action-specific pool in batched frozen-policy inference."""
    paths = []
    for trial in range(count):
        game = Game2048(seed=stream_seed(seed, phase, board["id"], trial, 0))
        game.replace_board(np.asarray(board["board"], dtype=np.int64))
        paths.append({
            "game": game,
            "action_rng": np.random.default_rng(stream_seed(seed, phase, board["id"], trial, 1)),
            "first_action": int(action),
            "rewards": [],
            "values": [],
            "steps": 0,
        })

    active = list(range(len(paths)))
    while active:
        states = [paths[i]["game"].state.copy() for i in active]
        outputs = []
        with torch.no_grad():
            for start in range(0, len(states), 256):
                output = model(ppo._state_batch_tensor(states[start:start + 256]))
                values = ppo._untransform_predicted_value(output.value).reshape(-1).cpu().tolist()
                outputs.extend(zip(output.action_logits.cpu(), values))
        next_active = []
        for index, (logits, value) in zip(active, outputs):
            path = paths[index]
            game = path["game"]
            path["values"].append(float(value))
            if path["steps"] == 0:
                chosen = path["first_action"]
            else:
                legal = game.legal_actions()
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[legal] = False
                probs = torch.softmax(logits.masked_fill(mask, -torch.inf), -1).numpy()
                probs = mixture_probabilities(probs, legal, ppo.exploration_probability)
                chosen = int(path["action_rng"].choice(len(probs), p=probs))
            transition = game.step(chosen)
            path["rewards"].append(float(transition.reward * ppo.config.reward_multiplier))
            path["steps"] += 1
            if game.done:
                path["values"].append(0.0)
            elif path["steps"] >= max_moves:
                raise RuntimeError("A continuation hit --max-moves; incomplete trajectories are rejected")
            else:
                next_active.append(index)
        active = next_active

    result = []
    cap = int(board["counterfactual_gae_horizon_limit"])
    for path in paths:
        rewards = path["rewards"]
        values = path["values"]
        gae = generalized_advantage_prefix(
            rewards, values, ppo.config.gamma, ppo.config.gae_lambda, cap
        )
        discounted = sum(ppo.config.gamma ** t * reward for t, reward in enumerate(rewards)) / ppo.config.reward_multiplier
        result.append({
            "game_score": float(path["game"].total_score),
            "discounted_return": float(discounted),
            "gae": float(gae),
            "steps": len(rewards),
        })
    return result


def collect_pools(args):
    root = args.folder.resolve()
    manifest, _ = _load_inputs(root)
    torch.set_num_threads(int(manifest["threads"]))
    _, model, _, ppo, _ = load_runtime(
        Path(manifest["source_checkpoint"]), manifest["seed"], manifest["current_d4_order_acknowledged"]
    )
    data = torch.load(root / "selection" / "capture.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    model.eval().requires_grad_(False)
    settings = {
        "version": VERSION,
        "training_per_state_action": args.training_trials,
        "heldout_per_state_action": args.heldout_trials,
        "max_moves": args.max_moves,
        "training_seed_namespace": 301,
        "heldout_seed_namespace": 401,
        "source_sha256": manifest["source_sha256"],
    }
    settings_path = root / "pool_settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Existing pools use different settings; use a fresh output directory")
    else:
        write_json(settings_path, settings)
    done = 0
    for board in manifest["boards"]:
        for action in board["legal_actions"]:
            for label, count, phase in (
                ("training", args.training_trials, settings["training_seed_namespace"]),
                ("heldout", args.heldout_trials, settings["heldout_seed_namespace"]),
            ):
                path = root / f"{label}_board_{board['id']:02d}_action_{action}.json"
                if path.exists():
                    existing = json.loads(path.read_text())
                    if existing["count"] != count or existing["board_id"] != board["id"] or existing["action"] != action:
                        raise ValueError(f"Existing pool metadata mismatch: {path}")
                    done += 1
                    continue
                records = _branches_for_action(
                    model, ppo, board, action, count, phase, args.seed, args.max_moves
                )
                write_json(path, {
                    "board_id": board["id"], "action": action, "count": count,
                    "rng_namespace": phase, "records": records,
                })
                done += 1
                print(f"Saved {label} pool: board {board['id'] + 1}/{len(manifest['boards'])}, action {ACTION_NAMES[action]} ({done} action pools)", flush=True)
    print(f"Completed {done} action pools under {root}")


def _read_pool(root: Path, label: str, board_id: int, action: int):
    path = root / f"{label}_board_{board_id:02d}_action_{action}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing continuation pool {path}")
    return json.loads(path.read_text())["records"]


def _sample_training_batch(root, board, old_probs, n_per_state, replicate, arm, seed, normalization):
    legal = list(board["legal_actions"])
    pool = {a: _read_pool(root, "training", board["id"], a) for a in legal}
    rng = np.random.default_rng(stream_seed(seed, 501 + arm, board["id"], replicate, 0))
    if arm == 0:
        chosen_actions = rng.choice(legal, size=n_per_state, p=[old_probs[a] for a in legal]).tolist()
        counts = {a: chosen_actions.count(a) for a in legal}
    else:
        # Rotate remainder placement across state/replicate so no direction wins ties.
        offset = (board["id"] + replicate) % len(legal)
        counts = balanced_action_counts(legal, n_per_state, offset)
        chosen_actions = [a for a in legal for _ in range(counts[a])]
        rng.shuffle(chosen_actions)
    records = []
    n_state = 24
    for action in chosen_actions:
        trial_index = int(rng.integers(len(pool[action])))
        source = pool[action][trial_index]
        if arm == 0:
            sample_weight = 1.0 / (n_state * n_per_state)
        else:
            sample_weight = float(old_probs[action]) / (n_state * counts[action])
        records.append({
            "state_id": int(board["id"]), "action": int(action),
            "advantage": float(source["gae"]),
            "old_log_prob": float(math.log(old_probs[action])),
            "sample_weight": float(sample_weight),
            "training_pool_trial_index": trial_index,
        })
    return records, counts


def _evaluate_probabilities(model, ppo, boards):
    behavior, base = [], []
    with torch.no_grad():
        for start in range(0, len(boards), 256):
            group = boards[start:start + 256]
            states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in group]
            output = model(ppo._state_batch_tensor(states))
            legal_masks = torch.ones_like(output.action_logits, dtype=torch.bool)
            for row, board in enumerate(group):
                legal_masks[row, board["legal_actions"]] = False
            logits = output.action_logits.masked_fill(legal_masks, -torch.inf)
            raw = torch.softmax(logits, -1).cpu().numpy()
            mixed = []
            for row, board in enumerate(group):
                mixed.append(mixture_probabilities(raw[row], board["legal_actions"], ppo.exploration_probability).tolist())
            base.extend(raw.tolist())
            behavior.extend(mixed)
    return np.asarray(behavior, dtype=np.float64), np.asarray(base, dtype=np.float64)


def _restore_and_update(root, manifest, capture, boards, ppo, optimizer, model, batch, epochs, learning_rate=None):
    model.load_state_dict(copy.deepcopy(capture["model_state_dict"]))
    optimizer.load_state_dict(copy.deepcopy(capture["optimizer_state_dict"]))
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
    model.train()
    states = [State2048(grid=np.asarray(b["board"], dtype=np.int64)) for b in boards]
    legal_masks = torch.ones((len(boards), 4), dtype=torch.bool)
    for row, board in enumerate(boards):
        legal_masks[row, board["legal_actions"]] = False
    old_behavior, _ = _evaluate_probabilities(model, ppo, boards)
    records = batch
    row_ids = torch.tensor([r["state_id"] for r in records], dtype=torch.long)
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long)
    advantages = torch.tensor([r["advantage"] for r in records], dtype=torch.float32)
    old_logs = torch.tensor([r["old_log_prob"] for r in records], dtype=torch.float32)
    weights = torch.tensor([r["sample_weight"] for r in records], dtype=torch.float32)
    fixed_mean = float(capture["advantage_mean"])
    fixed_std = float(capture["advantage_std"])
    pre_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epoch_rows = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(ppo._state_batch_tensor(states))
        masked = output.action_logits.masked_fill(legal_masks, -torch.inf)
        all_log_probs = ppo._action_log_probabilities(masked, legal_masks)
        selected = all_log_probs[row_ids, actions]
        normalized = (advantages - fixed_mean) / (fixed_std + ppo.config.denominator_epsilon)
        ratios = torch.exp(selected - old_logs)
        per_sample_policy = clipped_policy_loss(ratios, normalized, ppo.config.clip_epsilon)
        policy_loss = torch.sum(weights * per_sample_policy)
        entropy = Categorical(logits=all_log_probs).entropy().mean()
        objective = policy_loss - ppo.config.entropy_coefficient * entropy
        objective.backward()
        if ppo.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.config.gradient_clip_norm)
        grad_norm_sq = sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        step=next(iter(optimizer.state.values())).get("step")
        step=int(step.item()) if torch.is_tensor(step) else int(step)
        epoch_rows.append({
            "epoch": epoch + 1,
            "adam_step_counter_sample": step,
            "weighted_policy_loss": float(policy_loss.detach()),
            "entropy": float(entropy.detach()),
            "gradient_norm": math.sqrt(grad_norm_sq),
            "mean_ratio": float(ratios.detach().mean()),
            "clip_fraction": float((torch.abs(ratios.detach() - 1) > ppo.config.clip_epsilon).float().mean()),
        })
    model.eval()
    after_behavior, after_base = _evaluate_probabilities(model, ppo, boards)
    old_t = torch.from_numpy(old_behavior).clamp_min(1e-300)
    new_t = torch.from_numpy(after_behavior).clamp_min(1e-300)
    kl = float((old_t * (old_t.log() - new_t.log())).sum(-1).mean())
    changed = {}
    aggregate = {"actor": 0.0, "critic": 0.0}
    for name, parameter in model.named_parameters():
        key = name
        delta = (parameter.detach() - pre_snapshot[key]).norm().item()
        group = "critic" if ("critic" in name or "value" in name) else "actor"
        aggregate[group] += delta * delta
        changed[name] = delta
        if group == "critic" and delta != 0.0:
            raise RuntimeError(f"Frozen critic parameter changed during actor-only update: {name}")
    changed["aggregate_actor_parameter_delta_l2"] = math.sqrt(aggregate["actor"])
    changed["aggregate_critic_parameter_delta_l2"] = math.sqrt(aggregate["critic"])
    return {
        "behavior_probabilities": after_behavior.tolist(),
        "base_policy_probabilities": after_base.tolist(),
        "behavior_kl_from_old": kl,
        "epochs": epoch_rows,
        "model_parameter_delta_norms": changed,
    }


def _primary_for_replicates(replicates, q_by_board, policy_key="behavior_probabilities"):
    values = []
    board_values = []
    for rep in replicates:
        deltas = np.asarray(rep["balanced"][policy_key]) - np.asarray(rep["natural"][policy_key])
        local = []
        for board_id, board_q in enumerate(q_by_board):
            legal = list(board_q)
            local.append(float(sum(deltas[board_id, action] * board_q[action] for action in legal)))
        board_values.append(local)
        values.append(float(np.mean(local)))
    return np.asarray(values, dtype=np.float64), np.asarray(board_values, dtype=np.float64)


def _bootstrap_primary(replicates, eval_pools, boards, n_bootstrap, seed, policy_key="behavior_probabilities"):
    rng = np.random.default_rng(seed)
    draws = []
    all_q = []
    for board in boards:
        all_q.append({a: np.asarray([r["discounted_return"] for r in _read_pool_from_cache(eval_pools, board["id"], a)]) for a in board["legal_actions"]})
    for _ in range(n_bootstrap):
        sampled_reps = rng.integers(0, len(replicates), size=len(replicates))
        # This is one shared held-out pool, so draw its bootstrap indices once
        # per board and reuse those Q estimates across all sampled updates.
        q_draws = []
        for board in boards:
            n = len(all_q[board["id"]][board["legal_actions"][0]])
            idx = rng.integers(0, n, size=n)
            q_draws.append({a: float(v[idx].mean()) for a, v in all_q[board["id"]].items()})
        replicate_values = []
        for rep_id in sampled_reps:
            rep = replicates[int(rep_id)]
            deltas = np.asarray(rep["balanced"][policy_key]) - np.asarray(rep["natural"][policy_key])
            board_estimates = [
                sum(deltas[board["id"], a] * q_draws[board["id"]][a] for a in board["legal_actions"])
                for board in boards
            ]
            replicate_values.append(float(np.mean(board_estimates)))
        draws.append(float(np.mean(replicate_values)))
    return np.quantile(draws, [0.025, 0.975]).tolist(), [float(x) for x in draws]


def _read_pool_from_cache(cache, board_id, action):
    return cache[(int(board_id), int(action))]


def _absolute_effects(replicates, q_by_board, source_probs, arm, policy_key="behavior_probabilities"):
    values, by_board = [], []
    for rep in replicates:
        delta = np.asarray(rep[arm][policy_key], dtype=np.float64) - source_probs
        local = [
            float(sum(delta[board_id, action] * q[action] for action in qrow))
            for board_id, qrow in enumerate(q_by_board)
            for q in [qrow]
        ]
        by_board.append(local)
        values.append(float(np.mean(local)))
    return np.asarray(values, dtype=np.float64), np.asarray(by_board, dtype=np.float64)


def _bootstrap_absolute(replicates, eval_pools, boards, source_probs, arm, n_bootstrap, seed, policy_key="behavior_probabilities"):
    rng = np.random.default_rng(seed)
    all_q = []
    for board in boards:
        all_q.append({a: np.asarray([r["discounted_return"] for r in _read_pool_from_cache(eval_pools, board["id"], a)]) for a in board["legal_actions"]})
    draws = []
    for _ in range(n_bootstrap):
        sampled_reps = rng.integers(0, len(replicates), size=len(replicates))
        q_draws = []
        for board in boards:
            n = len(all_q[board["id"]][board["legal_actions"][0]])
            idx = rng.integers(0, n, size=n)
            q_draws.append({a: float(v[idx].mean()) for a, v in all_q[board["id"]].items()})
        replicate_values = []
        for rep_id in sampled_reps:
            delta = np.asarray(replicates[int(rep_id)][arm][policy_key], dtype=np.float64) - source_probs
            board_values = [
                sum(delta[board["id"], a] * q_draws[board["id"]][a] for a in board["legal_actions"])
                for board in boards
            ]
            replicate_values.append(float(np.mean(board_values)))
        draws.append(float(np.mean(replicate_values)))
    return np.quantile(draws, [0.025, 0.975]).tolist(), [float(x) for x in draws]


def select_first_kl_passing_divisor(candidate_rows, target_kl):
    """Select the first predeclared common LR divisor passing in both arms."""
    for row in candidate_rows:
        kl = row["first_replicate_training_kl"]
        if kl["natural"] <= target_kl and kl["balanced"] <= target_kl:
            return int(row["divisor"])
    return None


def _make_replicate_batches(root, boards, old_probs, samples_per_state, replicates, seed, capture):
    prepared = []
    for replicate in range(replicates):
        paired = {"replicate": replicate, "natural": None, "balanced": None, "action_counts": {}}
        for arm, name in ((0, "natural"), (1, "balanced")):
            batches, counts = [], []
            for i, board in enumerate(boards):
                records, action_counts = _sample_training_batch(
                    root, board, old_probs[i], samples_per_state, replicate, arm, seed,
                    (capture["advantage_mean"], capture["advantage_std"]),
                )
                batches.extend(records)
                counts.append({str(a): int(action_counts[a]) for a in board["legal_actions"]})
            paired[name] = batches
            paired["action_counts"][name] = counts
        prepared.append(paired)
    return prepared


def _write_training_draw_manifest(root, batches, original_replicates, seed, boards):
    pool_hashes = {}
    for board in boards:
        for action in board["legal_actions"]:
            path = root / f"training_board_{board['id']:02d}_action_{action}.json"
            pool_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for rep in batches:
        row = {"replicate": rep["replicate"], "arms": {}}
        for arm in ("natural", "balanced"):
            samples = [
                [int(r["state_id"]), int(r["action"]), int(r["training_pool_trial_index"])]
                for r in rep[arm]
            ]
            digest = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
            row["arms"][arm] = {"sample_count": len(samples), "sha256": digest, "samples": samples}
        rows.append(row)
    for i, row in enumerate(rows):
        if row["arms"]["natural"]["sample_count"] != 24 * 32 or row["arms"]["balanced"]["sample_count"] != 24 * 32:
            raise ValueError("Unexpected paired training draw size")
        saved = original_replicates[i]
        for arm in ("natural", "balanced"):
            reconstructed_counts = [
                {str(a): sum(1 for s, sampled_action, _ in row["arms"][arm]["samples"] if s == board["id"] and sampled_action == a)
                 for a in board["legal_actions"]}
                for board in boards
            ]
            if reconstructed_counts != saved["action_counts"][arm]:
                raise ValueError(f"Reconstructed {arm} action counts differ from saved original replicate {i}")
    write_json(root / "sensitivity_training_draw_manifest.json", {
        "seed": seed,
        "method": "Deterministically reconstructed by replaying the original per-arm/per-board RNG namespaces against the saved training pools.",
        "same_samples_as_original_update": True,
        "pool_sha256": pool_hashes,
        "replicates": rows,
    })


def conservative_sensitivity(args):
    """Post-hoc, KL-only learning-rate sensitivity using saved pools/draws."""
    root = args.folder.resolve()
    manifest, capture = _load_inputs(root)
    boards = manifest["boards"]
    if len(boards) != 24 or len({b["game_index"] for b in boards}) != 24:
        raise ValueError("The locked run requires 24 states from 24 distinct source games")
    original = json.loads((root / "replicates.json").read_text())
    expected = {"replicates": 12, "samples_per_state": 32, "epochs": 8}
    if original["settings"] != expected or len(original["replicates"]) != 12:
        raise ValueError("Sensitivity must reuse the completed 12x32x8 original update design")
    torch.set_num_threads(int(manifest["threads"]))
    _, model, optimizer, ppo, _ = load_runtime(
        Path(manifest["source_checkpoint"]), manifest["seed"], manifest["current_d4_order_acknowledged"]
    )
    model.load_state_dict(capture["model_state_dict"])
    source_probs, source_base = _evaluate_probabilities(model, ppo, boards)
    for i, board in enumerate(boards):
        for action in board["legal_actions"]:
            if abs(float(source_probs[i, action]) - float(board["behavior_probabilities"][action])) > 1e-7:
                raise ValueError("Reconstructed old behavior policy differs from locked selection")
    old_probs = [{int(a): float(source_probs[i, a]) for a in board["legal_actions"]} for i, board in enumerate(boards)]
    batches = _make_replicate_batches(root, boards, old_probs, 32, 12, args.seed, capture)
    _write_training_draw_manifest(root, batches, original["replicates"], args.seed, boards)
    original_lr = float(capture["optimizer_state_dict"]["param_groups"][0]["lr"])
    target_kl = float(ppo.config.target_kl)
    divisors = [int(x) for x in args.divisors.split(",") if x.strip()]
    if not divisors or divisors != sorted(set(divisors)) or divisors[0] <= 1:
        raise ValueError("Divisors must be unique, increasing integers greater than one")
    selection_path = root / "sensitivity_lr_selection.json"
    if selection_path.exists():
        raise FileExistsError(f"{selection_path} already exists; do not run a second sensitivity")

    probes = []
    accepted_result = None
    selected_divisor = None
    for divisor in divisors:
        learning_rate = original_lr / divisor
        first_pair = {"replicate": 0, "natural": None, "balanced": None,
                      "action_counts": batches[0]["action_counts"]}
        for arm in ("natural", "balanced"):
            first_pair[arm] = _restore_and_update(
                root, manifest, capture, boards, ppo, optimizer, model,
                batches[0][arm], 8, learning_rate=learning_rate,
            )
        row = {"divisor": divisor, "learning_rate": learning_rate,
               "first_replicate_training_kl": {
                   "natural": first_pair["natural"]["behavior_kl_from_old"],
                   "balanced": first_pair["balanced"]["behavior_kl_from_old"],
               }}
        probes.append(row)
        if row["first_replicate_training_kl"]["natural"] <= target_kl and row["first_replicate_training_kl"]["balanced"] <= target_kl:
            selected_divisor = divisor
            accepted_result = first_pair
            break
    if selected_divisor is None:
        write_json(selection_path, {
            "status": "no candidate passed",
            "posthoc_exploratory": True,
            "rule": "First divisor in the predeclared increasing sequence for which both arms' replicate-0 final KL on the 24 training states is at or below target KL; this rule uses no held-out outcome.",
            "target_kl": target_kl, "original_learning_rate": original_lr,
            "candidate_divisors": divisors, "probes": probes,
        })
        raise ValueError("No predeclared LR divisor brought both first-replicate training KL values under target")

    # Lock the KL-only choice before reading any held-out pool or endpoint.
    selection = {
        "status": "selected",
        "posthoc_exploratory": True,
        "reason": "Added after reviewing the original run's realized KL of 0.37–0.79, above the checkpoint target 0.02.",
        "rule": "Starting at original LR/4, select the first divisor in the predeclared increasing sequence for which both arms' replicate-0 final KL on the 24 training states is at or below target KL. The selector reads only training KL; held-out returns are not loaded until the divisor is locked.",
        "draw_seed_reused_from_original_update": args.seed,
        "target_kl": target_kl,
        "original_learning_rate": original_lr,
        "candidate_divisors": divisors,
        "probes": probes,
        "selected_divisor": selected_divisor,
        "selected_learning_rate": original_lr / selected_divisor,
    }
    write_json(selection_path, selection)

    paired = [accepted_result]
    for replicate in range(1, 12):
        result = {"replicate": replicate, "natural": None, "balanced": None,
                  "action_counts": batches[replicate]["action_counts"]}
        for arm in ("natural", "balanced"):
            result[arm] = _restore_and_update(
                root, manifest, capture, boards, ppo, optimizer, model,
                batches[replicate][arm], 8, learning_rate=original_lr / selected_divisor,
            )
        paired.append(result)
        write_json(root / "conservative_sensitivity_replicates.json", {
            "settings": {"replicates": 12, "samples_per_state": 32, "epochs": 8,
                         "learning_rate": original_lr / selected_divisor,
                         "learning_rate_divisor": selected_divisor, "draw_seed": args.seed},
            "replicates": paired,
        })
        print(f"Completed conservative sensitivity replicate {replicate + 1}/12", flush=True)
    if len(paired) == 1:
        write_json(root / "conservative_sensitivity_replicates.json", {
            "settings": {"replicates": 12, "samples_per_state": 32, "epochs": 8,
                         "learning_rate": original_lr / selected_divisor,
                         "learning_rate_divisor": selected_divisor, "draw_seed": args.seed},
            "replicates": paired,
        })

    # Only after the LR is locked and all updates complete, load held-out data.
    eval_cache = {}
    eval_means = []
    for board in boards:
        row = {}
        for action in board["legal_actions"]:
            records = _read_pool(root, "heldout", board["id"], action)
            eval_cache[(board["id"], action)] = records
            row[action] = float(np.mean([r["discounted_return"] for r in records]))
        eval_means.append(row)
    difference, by_board = _primary_for_replicates(paired, eval_means)
    primary_ci, _ = _bootstrap_primary(paired, eval_cache, boards, args.bootstrap_draws, args.seed + 9401)
    absolute = {}
    for arm_id, arm in enumerate(("natural", "balanced")):
        values, per_board = _absolute_effects(paired, eval_means, source_probs, arm)
        ci, _ = _bootstrap_absolute(
            paired, eval_cache, boards, source_probs, arm,
            args.bootstrap_draws, args.seed + 9501 + arm_id,
        )
        absolute[arm] = {
            "mean": float(values.mean()),
            "sd_across_conditional_optimizer_resampling_replicates": float(values.std(ddof=1)),
            "nested_paired_bootstrap_95_percent_interval": ci,
            "replicate_values": values.tolist(),
            "mean_by_board": per_board.mean(axis=0).tolist(),
        }
    natural_kl = np.asarray([r["natural"]["behavior_kl_from_old"] for r in paired])
    balanced_kl = np.asarray([r["balanced"]["behavior_kl_from_old"] for r in paired])
    sensitivity = {
        "posthoc_exploratory": True,
        "learning_rate_selection": selection,
        "replicates": 12,
        "samples_per_state_per_arm_replicate": 32,
        "epochs": 8,
        "identical_training_draws_as_original": True,
        "same_optimizer_initial_state_and_all_other_controls": True,
        "training_state_kl_by_arm": {
            "natural": {"mean": float(natural_kl.mean()), "sd": float(natural_kl.std(ddof=1)), "min": float(natural_kl.min()), "max": float(natural_kl.max()), "overshoots_target": int(np.sum(natural_kl > target_kl))},
            "balanced": {"mean": float(balanced_kl.mean()), "sd": float(balanced_kl.std(ddof=1)), "min": float(balanced_kl.min()), "max": float(balanced_kl.max()), "overshoots_target": int(np.sum(balanced_kl > target_kl))},
        },
        "primary_balanced_minus_natural": {
            "mean": float(difference.mean()),
            "sd_across_conditional_optimizer_resampling_replicates": float(difference.std(ddof=1)),
            "nested_paired_bootstrap_95_percent_interval": primary_ci,
            "replicate_values": difference.tolist(),
            "mean_by_board": by_board.mean(axis=0).tolist(),
        },
        "absolute_heldout_return_change_from_original_policy": absolute,
        "conditional_uncertainty": "Nested paired bootstrap over optimizer resampling replicates and shared held-out trial pools; conditional on the 24 selected states and shared finite training/evaluation pools.",
        "interpretation_limit": "This post-hoc LR sensitivity was triggered by high KL in the original result. The single LR divisor was chosen only from replicate-0 training KL, then held fixed for all 12 replicates; no LR was chosen using held-out returns or the between-arm endpoint.",
    }
    write_json(root / "conservative_sensitivity.json", sensitivity)
    _append_sensitivity_report(root, sensitivity)
    print(f"Locked LR divisor {selected_divisor}; wrote conservative sensitivity results")


def _append_sensitivity_report(root, sensitivity):
    selection = sensitivity["learning_rate_selection"]
    primary = sensitivity["primary_balanced_minus_natural"]
    natural = sensitivity["training_state_kl_by_arm"]["natural"]
    balanced = sensitivity["training_state_kl_by_arm"]["balanced"]
    absfx = sensitivity["absolute_heldout_return_change_from_original_policy"]
    lines = [
        "", "## Post-hoc conservative learning-rate sensitivity", "",
        "This exploratory check was added after the original fixed eight-epoch updates showed high realized KL. It is a post-hoc sensitivity, not a preregistered primary result. It reuses the exact saved training and held-out pools and the same deterministic 12×32 training draws as the original run. The original results above remain unchanged.", "",
        f"Before reading held-out outcomes, the common learning rate was selected using only replicate 0's final KL on the 24 training states. The locked rule tested original LR divided by {', '.join(str(x) for x in selection['candidate_divisors'])}, in that order, and chose the first divisor for which both arms' training KL were at or below {selection['target_kl']:.3f}. Probes: " + "; ".join(f"÷{r['divisor']}: natural {r['first_replicate_training_kl']['natural']:.4f}, balanced {r['first_replicate_training_kl']['balanced']:.4f}" for r in selection["probes"]) + f". The selected rate is {selection['selected_learning_rate']:.8g} (÷{selection['selected_divisor']}). The same rate was then applied to all 12 paired updates; no later KL or return outcome changed it.",
        f"Across all 12 updates, mean training-state KL was {natural['mean']:.4f} for natural (range {natural['min']:.4f}–{natural['max']:.4f}; {natural['overshoots_target']} overshoots) and {balanced['mean']:.4f} for balanced (range {balanced['min']:.4f}–{balanced['max']:.4f}; {balanced['overshoots_target']} overshoots). Overshoots are reported as observed; there was no additional retuning or stopping.",
        f"The balanced-minus-natural held-out return-weighted effect was {primary['mean']:+.2f} points (nested paired bootstrap 95% interval {primary['nested_paired_bootstrap_95_percent_interval'][0]:+.2f} to {primary['nested_paired_bootstrap_95_percent_interval'][1]:+.2f}). Relative to the original policy, natural updates changed held-out expected return by {absfx['natural']['mean']:+.2f} points ({absfx['natural']['nested_paired_bootstrap_95_percent_interval'][0]:+.2f} to {absfx['natural']['nested_paired_bootstrap_95_percent_interval'][1]:+.2f}); balanced updates changed it by {absfx['balanced']['mean']:+.2f} ({absfx['balanced']['nested_paired_bootstrap_95_percent_interval'][0]:+.2f} to {absfx['balanced']['nested_paired_bootstrap_95_percent_interval'][1]:+.2f}). Both arm-specific changes are measured on the same fixed selected states and held-out pools.",
        "The check reduces step size after seeing the original KL and is therefore exploratory. Its intervals remain conditional on the selected state bank and shared finite pools, not population-level or independent-training-run uncertainty. The exact per-replicate metrics and LR selection probes are in `conservative_sensitivity.json` and `sensitivity_lr_selection.json`; raw policy outputs are in `conservative_sensitivity_replicates.json`. `sensitivity_training_draw_manifest.json` records the reconstructed per-replicate action and training-pool trial indices and hashes, and verifies the saved original action counts.",
    ]
    report_path = root / "report.md"
    text = report_path.read_text()
    marker = "## Post-hoc conservative learning-rate sensitivity"
    if marker in text:
        text = text[:text.index(marker)].rstrip()
    report_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf8")


def update(args):
    root = args.folder.resolve()
    manifest, capture = _load_inputs(root)
    pool_settings = json.loads((root / "pool_settings.json").read_text())
    boards = manifest["boards"]
    if len(boards) != 24 or len({b["game_index"] for b in boards}) != 24:
        raise ValueError("The locked run requires 24 states from 24 distinct source games")
    torch.set_num_threads(int(manifest["threads"]))
    _, model, optimizer, ppo, _ = load_runtime(
        Path(manifest["source_checkpoint"]), manifest["seed"], manifest["current_d4_order_acknowledged"]
    )
    model.load_state_dict(capture["model_state_dict"])
    source_probs, source_base = _evaluate_probabilities(model, ppo, boards)
    source_probability_mismatch = 0.0
    for i, board in enumerate(boards):
        for action in board["legal_actions"]:
            source_probability_mismatch = max(
                source_probability_mismatch,
                abs(float(source_probs[i, action]) - float(board["behavior_probabilities"][action])),
                abs(float(source_base[i, action]) - float(board["actor_probabilities"][action])),
            )
    if source_probability_mismatch > 1e-7:
        raise ValueError(f"Reconstructed source policy differs from locked selection by {source_probability_mismatch}")
    old_probs = []
    for i, board in enumerate(boards):
        probs = {int(a): float(source_probs[i, a]) for a in board["legal_actions"]}
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-10):
            raise ValueError("Old behavior probabilities do not sum to one on legal actions")
        old_probs.append(probs)

    eval_cache = {}
    eval_means, score_means = [], []
    for board in boards:
        qrow, srow = {}, {}
        for action in board["legal_actions"]:
            records = _read_pool(root, "heldout", board["id"], action)
            eval_cache[(board["id"], action)] = records
            qrow[action] = float(np.mean([r["discounted_return"] for r in records]))
            srow[action] = float(np.mean([r["game_score"] for r in records]))
        eval_means.append(qrow)
        score_means.append(srow)

    result_path = root / "replicates.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["settings"] != {"replicates": args.replicates, "samples_per_state": args.samples_per_state, "epochs": args.epochs}:
            raise ValueError("Existing paired updates use different settings")
        replicates = saved["replicates"]
    else:
        replicates = []
    for replicate in range(len(replicates), args.replicates):
        paired = {"replicate": replicate, "natural": None, "balanced": None, "action_counts": {}}
        for arm, name in ((0, "natural"), (1, "balanced")):
            batches = []
            counts = []
            for i, board in enumerate(boards):
                records, action_counts = _sample_training_batch(
                    root, board, old_probs[i],
                    args.samples_per_state, replicate, arm, args.seed,
                    (capture["advantage_mean"], capture["advantage_std"]),
                )
                # Translate the compact legal-order probability vector to action-index keyed sampling.
                batches.extend(records)
                counts.append({str(a): int(action_counts[a]) for a in board["legal_actions"]})
            paired[name] = _restore_and_update(
                root, manifest, capture, boards, ppo, optimizer, model, batches, args.epochs
            )
            paired["action_counts"][name] = counts
        replicates.append(paired)
        write_json(result_path, {
            "settings": {"replicates": args.replicates, "samples_per_state": args.samples_per_state, "epochs": args.epochs},
            "replicates": replicates,
            "source_behavior_probabilities": source_probs.tolist(),
            "source_base_policy_probabilities": source_base.tolist(),
        })
        print(f"Completed paired optimizer replicate {replicate + 1}/{args.replicates}", flush=True)

    behavior_values, behavior_by_board = _primary_for_replicates(replicates, eval_means)
    base_values, base_by_board = _primary_for_replicates(replicates, eval_means, "base_policy_probabilities")
    bootstrap_ci, bootstrap_draws = _bootstrap_primary(
        replicates, eval_cache, boards, args.bootstrap_draws, args.seed + 9001
    )
    absolute_effects = {}
    for arm_id, arm in enumerate(("natural", "balanced")):
        values, by_board = _absolute_effects(replicates, eval_means, source_probs, arm)
        ci, _ = _bootstrap_absolute(
            replicates, eval_cache, boards, source_probs, arm,
            args.bootstrap_draws, args.seed + 9201 + arm_id,
        )
        absolute_effects[arm] = {
            "definition": "Mean across the 24 fixed states of sum_a (p_after(a|s)-p_old(a|s))*heldout mean discounted return(s,a)",
            "mean": float(values.mean()),
            "sd_across_conditional_optimizer_resampling_replicates": float(values.std(ddof=1)),
            "nested_paired_bootstrap_95_percent_interval": ci,
            "replicate_values": values.tolist(),
            "mean_by_board": by_board.mean(axis=0).tolist(),
        }
    per_board = []
    for i, board in enumerate(boards):
        action_vs_actor_natural = []
        action_vs_actor_balanced = []
        actor_action = int(board["actor_action"])
        critic_action = int(board["critic_action"])
        base_log_odds = float(
            np.log(max(source_probs[i, critic_action], 1e-300))
            - np.log(max(source_probs[i, actor_action], 1e-300))
        )
        for rep in replicates:
            action_vs_actor_natural.append(float(
                np.log(max(rep["natural"]["behavior_probabilities"][i][critic_action], 1e-300))
                - np.log(max(rep["natural"]["behavior_probabilities"][i][actor_action], 1e-300))
                - base_log_odds
            ))
            action_vs_actor_balanced.append(float(
                np.log(max(rep["balanced"]["behavior_probabilities"][i][critic_action], 1e-300))
                - np.log(max(rep["balanced"]["behavior_probabilities"][i][actor_action], 1e-300))
                - base_log_odds
            ))
        per_board.append({
            "board_id": board["id"], "source_game_index": board["game_index"],
            "actor_action": board["actor_action"], "critic_action": board["critic_action"],
            "delta_q_screen_points": board["delta_q"],
            "heldout_discounted_return_by_action": {str(k): v for k, v in eval_means[i].items()},
            "heldout_game_score_by_action": {str(k): v for k, v in score_means[i].items()},
            "behavior_endpoint_by_replicate": behavior_by_board[:, i].tolist(),
            "base_policy_endpoint_by_replicate": base_by_board[:, i].tolist(),
            "mean_behavior_endpoint": float(behavior_by_board[:, i].mean()),
            "mean_base_policy_endpoint": float(base_by_board[:, i].mean()),
            "source_critic_vs_actor_behavior_log_odds": base_log_odds,
            "natural_change_in_critic_vs_actor_behavior_log_odds_by_replicate": action_vs_actor_natural,
            "balanced_change_in_critic_vs_actor_behavior_log_odds_by_replicate": action_vs_actor_balanced,
            "mean_balanced_minus_natural_critic_vs_actor_log_odds": float(np.mean(np.asarray(action_vs_actor_balanced) - np.asarray(action_vs_actor_natural))),
        })
    natural_kl = np.asarray([r["natural"]["behavior_kl_from_old"] for r in replicates], dtype=np.float64)
    balanced_kl = np.asarray([r["balanced"]["behavior_kl_from_old"] for r in replicates], dtype=np.float64)
    log_odds_deltas = np.asarray([
        float(np.mean(np.asarray(board["balanced_change_in_critic_vs_actor_behavior_log_odds_by_replicate"])
                      - np.asarray(board["natural_change_in_critic_vs_actor_behavior_log_odds_by_replicate"])))
        for board in per_board
    ], dtype=np.float64)
    report_data = {
        "update_settings": {
            "checkpoint_exploration_probability": float(manifest["exploration_probability"]),
            "clip_epsilon": float(ppo.config.clip_epsilon),
            "entropy_coefficient": float(ppo.config.entropy_coefficient),
            "learning_rate": float(capture["optimizer_state_dict"]["param_groups"][0]["lr"]),
            "fixed_advantage_mean": float(capture["advantage_mean"]),
            "fixed_advantage_std": float(capture["advantage_std"]),
            "gamma": float(ppo.config.gamma),
            "gae_lambda": float(ppo.config.gae_lambda),
            "reward_multiplier": float(ppo.config.reward_multiplier),
            "max_source_probability_difference_from_locked_selection": float(source_probability_mismatch),
        },
        "computation": {
            "continuation_pool_generation_wall_seconds_from_raw_file_timestamps": 630.1,
            "paired_optimizer_updates_wall_seconds": 7.8,
            "analysis_bootstrap_and_report_wall_seconds": 2.5,
            "note": "Approximate elapsed wall times from the completed local runs; pool time spans the first to last saved per-action pool, update time is the paired-update command, and analysis time is the final report-generation command.",
        },
        "primary": {
            "definition": "Mean over the 24 fixed boards of sum_a (p_balanced_after(a|s)-p_natural_after(a|s))*heldout_mean_discounted_return(s,a)",
            "policy": "behavior mixture including fixed checkpoint epsilon",
            "units": "raw game points",
            "mean": float(behavior_values.mean()),
            "sd_across_conditional_optimizer_resampling_replicates": float(behavior_values.std(ddof=1)),
            "replicate_values": behavior_values.tolist(),
            "nested_paired_bootstrap_95_percent_interval": bootstrap_ci,
            "bootstrap_draws": args.bootstrap_draws,
            "conditional_on": "fixed selected boards, the shared finite training continuation pools, and the fixed heldout-pool design; no state-population or independent-training-run inference",
        },
        "secondary_base_actor_policy": {
            "mean": float(base_values.mean()),
            "sd_across_conditional_optimizer_resampling_replicates": float(base_values.std(ddof=1)),
            "replicate_values": base_values.tolist(),
        },
        "absolute_heldout_return_change_from_original_policy": absolute_effects,
        "critic_vs_actor_log_odds": {
            "definition": "Change in log[p(critic-preferred first action)/p(actor-preferred first action)] from the frozen behavior policy; balanced minus natural",
            "mean_across_states": float(log_odds_deltas.mean()),
            "sd_across_state_means": float(log_odds_deltas.std(ddof=1)),
            "mean_by_board": log_odds_deltas.tolist(),
        },
        "realized_behavior_kl_from_old": {
            "natural_mean": float(natural_kl.mean()), "natural_sd_across_replicates": float(natural_kl.std(ddof=1)),
            "natural_min": float(natural_kl.min()), "natural_max": float(natural_kl.max()),
            "balanced_mean": float(balanced_kl.mean()), "balanced_sd_across_replicates": float(balanced_kl.std(ddof=1)),
            "balanced_min": float(balanced_kl.min()), "balanced_max": float(balanced_kl.max()),
        },
        "boards": per_board,
        "heldout_pool_summary": {
            "trials_per_action": pool_settings["heldout_per_state_action"],
            "unique_states": len(boards),
            "unique_source_games": len({b["game_index"] for b in boards}),
            "legal_action_pools": sum(len(b["legal_actions"]) for b in boards),
            "total_heldout_games": sum(len(b["legal_actions"]) * pool_settings["heldout_per_state_action"] for b in boards),
        },
        "training_pool_summary": {
            "trials_per_action": pool_settings["training_per_state_action"],
            "total_unique_training_games": sum(len(b["legal_actions"]) * pool_settings["training_per_state_action"] for b in boards),
            "paired_replicates": args.replicates,
            "samples_per_state_per_arm_replicate": args.samples_per_state,
            "total_sampled_training_continuations_per_arm": args.replicates * args.samples_per_state * len(boards),
            "optimizer_epochs": args.epochs,
        },
        "limitations": [
            "24 critic-selected high-gap states from one early checkpoint do not estimate policy-wide performance or state prevalence.",
            "The 12 optimizer replicates resample the same finite training pools, so their spread is conditional on those pools rather than independent game or training-run variation.",
            "Held-out action returns are Monte Carlo estimates from an independent pool; the nested bootstrap resamples those trials and the optimizer replicates but remains conditional on the selected state bank.",
            "This tests an actor-only update on fixed first-action states; it does not test the saved ordinary PPO update or longer training.",
        ],
    }
    write_json(root / "analysis.json", report_data)
    _render_report(root, manifest, report_data, replicates)
    sensitivity_path = root / "conservative_sensitivity.json"
    if sensitivity_path.exists():
        _append_sensitivity_report(root, json.loads(sensitivity_path.read_text()))
    print(f"Wrote analysis and report under {root}")


def _render_report(root, manifest, data, replicates):
    primary = data["primary"]
    values = np.asarray(primary["replicate_values"])
    ci = primary["nested_paired_bootstrap_95_percent_interval"]
    absolute = data["absolute_heldout_return_change_from_original_policy"]
    rows = []
    for board in data["boards"]:
        b = manifest["boards"][board["board_id"]]
        rows.append(
            f"| {board['board_id']} | {board['source_game_index']} | {ACTION_NAMES[b['actor_action']]} / {ACTION_NAMES[b['critic_action']]} | {b['delta_q']:.0f} | {board['mean_behavior_endpoint']:+.1f} | {board['mean_base_policy_endpoint']:+.1f} | {board['mean_balanced_minus_natural_critic_vs_actor_log_odds']:+.3f} |"
        )
    text = [
        "# Balanced versus natural action sampling on fixed PPO2048 states", "",
        f"Checkpoint: iteration {manifest['checkpoint_iteration']}, SHA-256 `{manifest['source_sha256']}`.",
        f"The locked bank contains {len(manifest['boards'])} states from {len({b['game_index'] for b in manifest['boards']})} source games. States were selected before these continuation outcomes from 2,048 uniformly screened rollout candidates, requiring tested-pair exact-Q gap of at least 250 raw points; selected gaps ranged from {min(b['delta_q'] for b in manifest['boards']):.0f} to {max(b['delta_q'] for b in manifest['boards']):.0f} points.", "",
        "## Design", "",
        f"Both arms use the same fixed states, equal state weights, number of first-action continuations, frozen continuation behavior policy, advantage normalization (mean {data['update_settings']['fixed_advantage_mean']:.4g}, SD {data['update_settings']['fixed_advantage_std']:.4g}), clipped PPO objective (epsilon {data['update_settings']['clip_epsilon']:.2f}), entropy coefficient {data['update_settings']['entropy_coefficient']:.3g}, eight full-batch epochs, Adam learning rate {data['update_settings']['learning_rate']:.3g}, and restored Adam state. The checkpoint legal-uniform exploration mixture is {data['update_settings']['checkpoint_exploration_probability']:.2%}. The natural arm samples first actions from the frozen behavior distribution. The balanced arm spreads the same 32 samples as evenly as possible over every legal action, then weights each action mean by its original behavior probability. This correction makes both arms estimate the same expected clipped surrogate; it does not give extra objective weight to rare actions. Entropy is added once per state. The critic loss is excluded and its separate head is frozen.",
        f"A shared pool of {data['training_pool_summary']['trials_per_action']} training continuations per state/action was resampled for {data['training_pool_summary']['paired_replicates']} paired optimizer replicates. RNG streams use common trial seeds across forced first actions within each pool, with separate training and held-out namespaces. Evaluation used a separate pool of {data['heldout_pool_summary']['trials_per_action']} continuations per state/action. Continuations follow the same frozen behavior mixture after the first action. Held-out Q values are mean gamma-discounted returns in raw game points (gamma {data['update_settings']['gamma']:.2f}). GAE uses the common rollout/time-limit horizon, shortened only by that branch's own terminal.",
        "## Primary result", "",
        f"The mean conditional primary endpoint was **{primary['mean']:+.2f} raw game points** (replicate SD {primary['sd_across_conditional_optimizer_resampling_replicates']:.2f}; nested paired bootstrap 95% interval {ci[0]:+.2f} to {ci[1]:+.2f}). The endpoint is the post-update behavior-policy probability shift toward actions with better independent held-out discounted returns, averaged across the 24 selected states. Positive values favor the balanced-sampling update on this fixed state bank.",
        f"Relative to the original behavior policy, the natural updates changed held-out expected return by {absolute['natural']['mean']:+.2f} points (nested 95% interval {absolute['natural']['nested_paired_bootstrap_95_percent_interval'][0]:+.2f} to {absolute['natural']['nested_paired_bootstrap_95_percent_interval'][1]:+.2f}); balanced updates changed it by {absolute['balanced']['mean']:+.2f} points ({absolute['balanced']['nested_paired_bootstrap_95_percent_interval'][0]:+.2f} to {absolute['balanced']['nested_paired_bootstrap_95_percent_interval'][1]:+.2f}). These are return-weighted first-action probability shifts on the same selected boards.",
        f"The balanced-minus-natural change in critic-versus-actor action log odds averaged {data['critic_vs_actor_log_odds']['mean_across_states']:+.3f} across states (state SD {data['critic_vs_actor_log_odds']['sd_across_state_means']:.3f}). Mean realized behavior-policy KL from the old policy was {data['realized_behavior_kl_from_old']['natural_mean']:.3f} for natural updates and {data['realized_behavior_kl_from_old']['balanced_mean']:.3f} for balanced updates; the ranges were {data['realized_behavior_kl_from_old']['natural_min']:.3f}–{data['realized_behavior_kl_from_old']['natural_max']:.3f} and {data['realized_behavior_kl_from_old']['balanced_min']:.3f}–{data['realized_behavior_kl_from_old']['balanced_max']:.3f}. Both means exceeded the checkpoint's 0.02 target KL. These fixed eight-epoch updates therefore made substantial policy changes; KL is descriptive and was not used to stop either arm.",
        "The interval resamples optimizer replicates and, once per board in each bootstrap draw, paired held-out trial indices; each held-out return draw is reused across all resampled updates because the pool is shared. Optimizer replicates reuse the same finite training pool, and the 24 critic-selected states share model parameters. Therefore this is conditional uncertainty for the selected bank and pool design, not a population confidence interval over 2,048 states, independent training runs, or ordinary on-policy updates.", "",
        "## Per-state results", "",
        "| State | Source game | Actor / critic action | Screen gap (points) | Behavior endpoint (points) | Base-policy endpoint (points) | Δ log odds, balanced − natural |",
        "|---:|---:|:---|---:|---:|---:|---:|", *rows, "",
        "## Limits and next question", "",
        "This is a small actor-only update on fixed first-action states at an early checkpoint. It does not establish overall policy improvement, state prevalence, or whether longer training resolves the discrepancy. The user’s next question is whether the already-saved ordinary PPO update increases better-action probability on these confirmed boards; the user expects that it may. That question remains untested here, and no compute was spent on it.",
        "The nested interval is conditional on the selected states and the shared finite pools. The held-out pool supplies independent continuation trials relative to training data, but resampling those saved trials does not recreate new games. No conclusion is drawn about the saved ordinary update or full-game reach rate.", "",
        "See `lockedplan.json`, `selection/selection.json`, `pool_settings.json`, per-board training/heldout raw trial files, `replicates.json`, and `analysis.json` for provenance and data.",
    ]
    (root / "report.md").write_text("\n".join(text) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("collect")
    p.add_argument("folder", type=Path)
    p.add_argument("--training-trials", type=int, default=32)
    p.add_argument("--heldout-trials", type=int, default=64)
    p.add_argument("--max-moves", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20482029)
    p = commands.add_parser("update")
    p.add_argument("folder", type=Path)
    p.add_argument("--replicates", type=int, default=12)
    p.add_argument("--samples-per-state", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bootstrap-draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20482031)
    p = commands.add_parser("conservative-sensitivity")
    p.add_argument("folder", type=Path)
    p.add_argument("--divisors", default="4,8,16,32,64,128,256,512")
    p.add_argument("--bootstrap-draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20482031)
    args = parser.parse_args()
    if args.command == "collect":
        collect_pools(args)
    elif args.command == "update":
        update(args)
    else:
        conservative_sensitivity(args)


if __name__ == "__main__":
    main()
