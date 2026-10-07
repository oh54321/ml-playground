"""Audit PPO GAE targets at rollout boundaries for a frozen 2048 checkpoint.

Example:
  python -m games.rl_2048.scripts.audit_rollout_boundaries capture CHECKPOINT \
    --output artifacts/ppo_2048/boundary_500
  python -m games.rl_2048.scripts.audit_rollout_boundaries analyze artifacts/ppo_2048/boundary_500

Capture data are immutable inputs to analyze. Each selected cutoff has
independently seeded sampled continuations under the frozen actor policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from algorithms.ppo_types import PolicyRolloutInformation

from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.scripts.audit_ppo_update import load_runtime, tensor_tree_sha256

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gradient_code_sha256() -> str:
    digest = hashlib.sha256()
    for function in (gae_vector, policy_gradient, batch_gradient_audit):
        digest.update(inspect.getsource(function).encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def step_dict(step):
    return {"board": step.state.grid.tolist(), "next_board": step.next_state.grid.tolist(),
            "action": int(step.action), "log_prob": float(step.log_prob),
            "value": float(step.value), "reward": float(step.reward),
            "done": bool(step.done), "truncated": bool(step.truncated),
            "legal_mask": step.legal_actions_mask.cpu().tolist(), "advantage": float(step.advantage)}


def value(model, ppo, board) -> float:
    state = State2048(grid=np.asarray(board, dtype=np.int64))
    with torch.no_grad():
        out = model(ppo._state_batch_tensor([state])).value
        return float(ppo._untransform_predicted_value(out).item())


def capture(args) -> None:
    capture_started = perf_counter()
    source = args.checkpoint.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checkpoint_sha = sha256(source)
    checkpoint, model, _, ppo, epsilon = load_runtime(source, args.seed, args.current_d4_order)
    model.eval().requires_grad_(False)
    # One full warm-up rollout is discarded so selected cutoffs come from a
    # fresh steady-state batch; subsequent batches retain the real collector.
    ppo.collect_batched_information()
    burn_in_seconds = perf_counter() - capture_started
    rollout_started = perf_counter()
    captured = []
    for _ in range(args.batches):
        steps, _, _ = ppo.collect_batched_information()
        captured.append(steps)
    rollout_seconds = perf_counter() - rollout_started
    # Retain every transition, including terminal episodes, for full-batch
    # weighting and later reanalysis.
    write_json(out / "rollouts.json", {"batches": [[step_dict(s) for s in batch] for batch in captured]})
    if sha256(source) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed while capture was running")
    per_slot = ppo.config.steps_per_game_per_rollout
    candidates = []
    for batch_index, batch in enumerate(captured):
        for slot in range(ppo.config.num_parallel_games):
            segment = batch[slot * per_slot:(slot + 1) * per_slot]
            cutoff = segment[-1]
            if not cutoff.done and not cutoff.truncated:
                candidates.append({"batch": batch_index, "slot": slot,
                                   "prefix": [step_dict(s) for s in segment],
                                   "cutoff_board": cutoff.next_state.grid.tolist()})
    # Use one cutoff per slot, selected by a fixed slot order, so independent
    # continuation replication does not inflate the number of sampled episodes.
    selected = []
    seen = set()
    selection_rng = np.random.default_rng(args.seed + 17)
    for candidate_index in selection_rng.permutation(len(candidates)):
        row = candidates[int(candidate_index)]
        if row["slot"] not in seen:
            selected.append(row)
            seen.add(row["slot"])
        if len(selected) >= args.episodes:
            break
    if len(selected) < args.episodes:
        raise RuntimeError(f"Only {len(selected)} distinct slot cutoffs found; requested {args.episodes}. Increase --batches.")

    # Deterministic independent branch seeds are fixed before observing returns.
    seed_rng = np.random.default_rng(args.seed + 1)
    branch_seeds = seed_rng.integers(1, 2**32 - 1, size=(len(selected), args.branches), dtype=np.uint32)
    branch_records = []
    active_games = []
    active_rows = []
    branch_symmetries = []
    branch_symmetry_rngs = []
    for episode_id, row in enumerate(selected):
        branches = []
        for branch_id, branch_seed in enumerate(branch_seeds[episode_id]):
            game = Game2048(seed=int(branch_seed))
            game.replace_board(np.asarray(row["cutoff_board"], dtype=np.int64))
            branch = {"seed": int(branch_seed), "transitions": [], "censored": False,
                      "bootstrap_value": 0.0}
            branches.append(branch)
            symmetries = game.symmetries()
            symmetry_rng = np.random.default_rng(int(branch_seed) + 29)
            symmetry = (symmetries[int(symmetry_rng.integers(len(symmetries)))]
                        if checkpoint.get("symmetry_augmentation") == "episode" and symmetries else None)
            if not game.done:
                active_games.append(game)
                active_rows.append(branch)
                branch_symmetries.append(symmetry)
                branch_symmetry_rngs.append(symmetry_rng)
        branch_records.append({"batch": row["batch"], "slot": row["slot"],
                               "prefix": row["prefix"], "cutoff_board": row["cutoff_board"],
                               "branches": branches})
    # Batch all live continuation states at each depth. This preserves each
    # environment's independent RNG while avoiding one network call per tail.
    torch.manual_seed(args.seed + 2)
    branches_started = perf_counter()
    for _ in range(args.max_moves):
        live_indices = [i for i, game in enumerate(active_games) if not game.done]
        if not live_indices:
            break
        live_games = [active_games[i] for i in live_indices]
        live_symmetries = []
        live_states = []
        live_legal = []
        for index, game in zip(live_indices, live_games, strict=True):
            symmetry = branch_symmetries[index]
            if checkpoint.get("symmetry_augmentation") == "step":
                available = game.symmetries()
                rng = branch_symmetry_rngs[index]
                symmetry = available[int(rng.integers(len(available)))] if available else None
            live_symmetries.append(symmetry)
            state = game.current_state()
            legal = game.legal_actions()
            if symmetry is not None:
                state = symmetry.transform_state(state)
                legal = sorted(symmetry.to_transformed_action(a) for a in legal)
            live_states.append(state)
            live_legal.append(legal)
        decisions = ppo.generate_next_actions(live_games, deterministic=False,
                                              states=live_states, legal_actions=live_legal)
        for index, decision, symmetry in zip(live_indices, decisions, live_symmetries, strict=True):
            game, branch = active_games[index], active_rows[index]
            action = decision.action if symmetry is None else symmetry.to_canonical_action(decision.action)
            result = game.step(action)
            branch["transitions"].append({"reward": float(result.reward * ppo.config.reward_multiplier),
                                          "value": float(decision.value), "next_board": result.next_state.grid.tolist(),
                                          "done": bool(result.terminated)})
    uncensored = [(game, branch) for game, branch in zip(active_games, active_rows, strict=True) if not game.done]
    if uncensored:
        for game, branch in uncensored:
            branch["censored"] = True
        cutoff_values = ppo.rollout_collector.untransform_predicted_value(
            model(ppo._state_batch_tensor([g.current_state() for g, _ in uncensored])).value
        ).detach().cpu().tolist()
        for (_, branch), bootstrap in zip(uncensored, cutoff_values, strict=True):
            branch["bootstrap_value"] = float(bootstrap)
    branch_seconds = perf_counter() - branches_started

    protocol = {"checkpoint": str(source), "checkpoint_sha256": checkpoint_sha,
                "iteration": int(checkpoint["outer_iteration"]), "model_config": checkpoint["resnet_config"],
                "ppo_config": asdict(ppo.config), "symmetry_augmentation": checkpoint.get("symmetry_augmentation"),
                "current_d4_order_acknowledged": args.current_d4_order,
                "exploration_probability": epsilon, "seed": args.seed, "burn_in_batches": 1,
                "captured_batches": args.batches, "episodes": args.episodes, "branches_per_episode": args.branches,
                "selection_method": "seeded permutation of eligible nonterminal batch/slot cutoffs, capped at one cutoff per slot",
                "selected_cutoffs": [{"batch": row["batch"], "slot": row["slot"]} for row in selected],
                "branch_policy": f"frozen checkpoint actor sampling the legal-action mixture with epsilon={epsilon}",
                "branch_rng": "independent environment seeds assigned before outcomes; batched actor samples use a fixed torch seed",
                "max_moves": args.max_moves, "threads": 1,
                "timings_seconds": {"burn_in": burn_in_seconds, "rollout_capture": rollout_seconds,
                                    "continuations": branch_seconds, "capture_total": perf_counter() - capture_started},
                "script_sha256": sha256(Path(__file__).resolve())}
    protocol["protocol_sha256"] = digest_json(protocol)
    write_json(out / "manifest.json", protocol)
    write_json(out / "branches.json", {"episodes": branch_records})
    protocol["cached_data_sha256"] = {
        "rollouts.json": sha256(out / "rollouts.json"),
        "branches.json": sha256(out / "branches.json"),
    }
    protocol["protocol_sha256"] = digest_json({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    write_json(out / "manifest.json", protocol)
    (out / "README.md").write_text(
        "# Rollout boundary audit\n\n"
        "`manifest.json` locks the checkpoint, configuration, seeds, and capture settings. `branches.json` caches every selected PPO segment and all frozen-policy continuations. Run `python -m games.rl_2048.scripts.audit_rollout_boundaries analyze .` to regenerate the analysis without rerunning games.\n",
        encoding="utf-8")
    print(f"Captured {len(selected)} cutoff episodes with {args.branches} branches each at {out}")


def gae_vector(rewards, values, terminal, gamma, lam, bootstrap, truncated=None,
               truncation_bootstrap=None):
    truncated = truncated or [False] * len(rewards)
    truncation_bootstrap = truncation_bootstrap or [0.0] * len(rewards)
    acc = 0.0
    result = [0.0] * len(rewards)
    for i in reversed(range(len(rewards))):
        nv = (0.0 if terminal[i] else truncation_bootstrap[i] if truncated[i]
              else values[i + 1] if i + 1 < len(values) else bootstrap)
        trace = 0.0 if terminal[i] or truncated[i] else acc
        acc = rewards[i] + gamma * nv - values[i] + gamma * lam * trace
        result[i] = acc
    return result


def gae(rewards, values, terminal, gamma, lam, bootstrap):
    """Return the first GAE value; retained as a compact math-test helper."""
    return gae_vector(rewards, values, terminal, gamma, lam, bootstrap)[0]


def policy_gradient(ppo, steps, advantages, chunk_size=512):
    """Full-batch normalized PPO surrogate gradient, accumulated in chunks."""
    advantages = np.asarray(advantages, dtype=np.float64)
    mean, std = float(advantages.mean()), float(advantages.std())
    params = tuple(ppo.model.parameters())
    accum = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
    for start in range(0, len(steps), chunk_size):
        stop = min(len(steps), start + chunk_size)
        minibatch = [__import__("dataclasses").replace(s, advantage=float(a))
                     for s, a in zip(steps[start:stop], advantages[start:stop], strict=True)]
        objective = ppo.updater.compute_objective(minibatch, mean, std).policy.mean() * ((stop - start) / len(steps))
        grads = torch.autograd.grad(objective, params, allow_unused=True)
        for dest, grad in zip(accum, grads, strict=True):
            if grad is not None:
                dest.add_(grad.detach())
        del objective, grads, minibatch
    vector = torch.cat([g.reshape(-1) for g in accum])
    return vector, {"advantage_mean": mean, "advantage_std": std}


def row_step(row):
    return PolicyRolloutInformation(
        State2048(grid=np.asarray(row["board"], dtype=np.int64)), int(row["action"]),
        State2048(grid=np.asarray(row["next_board"], dtype=np.int64)), float(row["log_prob"]),
        float(row["value"]), float(row["reward"]), bool(row["done"]), bool(row["truncated"]),
        torch.tensor(row["legal_mask"], dtype=torch.bool), float(row["advantage"]))


def batch_gradient_audit(ppo, manifest, branch_data, rollout_data, lambdas):
    """Compare gradients over a pooled batch of one complete slot prefix per cutoff."""
    gamma = float(manifest["ppo_config"]["gamma"])
    del rollout_data  # Full rollouts remain cached; this estimand deliberately pools 32 cutoff prefixes.
    pooled_steps = []
    episode_data = []
    for episode in branch_data["episodes"]:
        prefix = episode["prefix"]
        prefix_steps = [row_step(r) for r in prefix]
        with torch.no_grad():
            cutoff_state = State2048(grid=np.asarray(episode["cutoff_board"], dtype=np.int64))
            cutoff_value = float(ppo.rollout_collector.untransform_predicted_value(
                ppo.model(ppo.rollout_collector.state_batch_tensor([cutoff_state])).value).item())
        rewards = [r["reward"] for r in prefix]
        vals = [r["value"] for r in prefix]
        done = [r["done"] for r in prefix]
        truncated = [r["truncated"] for r in prefix]
        trunc_values = [value(ppo.model, ppo, r["next_board"]) if r["truncated"] else 0.0 for r in prefix]
        episode_data.append({"episode": episode, "steps": prefix_steps,
                             "cutoff_value": cutoff_value, "trunc_values": trunc_values})
        pooled_steps.extend(prefix_steps)
    if len(episode_data) != 32:
        raise ValueError(f"Expected 32 selected prefixes for the pooled gradient, found {len(episode_data)}")
    per_lambda = {}
    for lam in lambdas:
        ordinary_parts, branch_parts = [], []
        for datum in episode_data:
            ep = datum["episode"]
            prefix = ep["prefix"]
            rewards = [r["reward"] for r in prefix]
            vals = [r["value"] for r in prefix]
            done = [r["done"] for r in prefix]
            trunc = [r["truncated"] for r in prefix]
            ordinary = gae_vector(rewards, vals, done, gamma, lam, datum["cutoff_value"],
                                  trunc, datum["trunc_values"])
            branch_advs = []
            for branch in ep["branches"]:
                suffix = branch["transitions"]
                ext = gae_vector(rewards + [r["reward"] for r in suffix],
                    vals + [r["value"] for r in suffix], done + [r["done"] for r in suffix],
                    gamma, lam, float(branch["bootstrap_value"]), trunc + [False] * len(suffix),
                    datum["trunc_values"] + [0.0] * len(suffix))[:len(prefix)]
                branch_advs.append(ext)
            ordinary_parts.extend(ordinary)
            branch_parts.append(branch_advs)
        corrected_mean = np.concatenate([np.mean(a, axis=0) for a in branch_parts])
        split_index = len(branch_parts[0]) // 2
        if split_index == 0:
            raise ValueError("At least two continuations are needed for branch-split stability")
        corrected_first = np.concatenate([np.mean(a[:split_index], axis=0) for a in branch_parts])
        corrected_second = np.concatenate([np.mean(a[split_index:], axis=0) for a in branch_parts])
        ordinary_all = np.asarray(ordinary_parts)
        base, norm = policy_gradient(ppo, pooled_steps, ordinary_all)
        corr, _ = policy_gradient(ppo, pooled_steps, corrected_mean)
        split_a, _ = policy_gradient(ppo, pooled_steps, corrected_first)
        split_b, _ = policy_gradient(ppo, pooled_steps, corrected_second)
        per_lambda[str(lam)] = {"normalization": norm,
            "pooled_cutoff_prefixes": 32, "pooled_transitions": len(pooled_steps),
            "ordinary_gradient_norm": float(torch.linalg.vector_norm(base)),
            "extended_mean_gradient_norm": float(torch.linalg.vector_norm(corr)),
            "cosine_similarity": float(torch.nn.functional.cosine_similarity(base, corr, dim=0)),
            "relative_gradient_change": float(torch.linalg.vector_norm(corr - base) / (torch.linalg.vector_norm(base) + 1e-12)),
            "branch_split_cosine": float(torch.nn.functional.cosine_similarity(split_a, split_b, dim=0)),
            "branch_split_a_vs_mean_cosine": float(torch.nn.functional.cosine_similarity(split_a, corr, dim=0)),
            "branch_split_b_vs_mean_cosine": float(torch.nn.functional.cosine_similarity(split_b, corr, dim=0)),
            "surrogate_only": True, "entropy_included": False}
    return per_lambda


def analyze(args) -> None:
    analysis_started = perf_counter()
    torch.set_num_threads(1)
    folder = args.folder.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    prior_analysis = (json.loads((folder / "analysis.json").read_text())
                      if args.reuse_gradients and (folder / "analysis.json").exists() else None)
    expected = manifest.pop("protocol_sha256")
    if digest_json(manifest) != expected:
        raise RuntimeError("Manifest digest mismatch")
    manifest["protocol_sha256"] = expected
    if sha256(Path(manifest["checkpoint"])) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash no longer matches capture")
    data = json.loads((folder / "branches.json").read_text())
    for filename, file_hash in manifest.get("cached_data_sha256", {}).items():
        if sha256(folder / filename) != file_hash:
            raise RuntimeError(f"Cached input hash mismatch: {filename}")
    gamma = float(manifest["ppo_config"]["gamma"])
    _, model, _, ppo, _ = load_runtime(Path(manifest["checkpoint"]), manifest["seed"],
                                       bool(manifest["current_d4_order_acknowledged"]))
    model.eval()
    values = []
    for episode in data["episodes"]:
        prefix = episode["prefix"]
        rewards = [r["reward"] for r in prefix]
        vals = [r["value"] for r in prefix]
        terminals = [r["done"] for r in prefix]
        truncations = [r["truncated"] for r in prefix]
        trunc_bootstrap = [value(model, ppo, r["next_board"]) if r["truncated"] else 0.0 for r in prefix]
        # The sampled cutoff is nonterminal; standard PPO bootstraps its critic value.
        cutoff_value = value(model, ppo, episode["cutoff_board"])
        for lam in args.lambdas:
            ordinary_vector = gae_vector(rewards, vals, terminals, gamma, lam, cutoff_value,
                                         truncations, trunc_bootstrap)
            extended = []
            tail_advantages = []
            episode_differences = []
            for branch in episode["branches"]:
                suffix = branch["transitions"]
                all_rewards = rewards + [r["reward"] for r in suffix]
                all_values = vals + [r["value"] for r in suffix]
                all_terminals = terminals + [r["done"] for r in suffix]
                all_truncated = truncations + [False] * len(suffix)
                all_trunc_bootstrap = trunc_bootstrap + [0.0] * len(suffix)
                extended_vector = gae_vector(all_rewards, all_values, all_terminals, gamma, lam,
                                             float(branch["bootstrap_value"]), all_truncated,
                                             all_trunc_bootstrap)[:len(rewards)]
                tail_advantages.append(gae([r["reward"] for r in suffix],
                    [r["value"] for r in suffix], [r["done"] for r in suffix], gamma, lam,
                    float(branch["bootstrap_value"])))
                episode_differences.append(np.asarray(extended_vector) - np.asarray(ordinary_vector))
                extended.append(float(extended_vector[0]))
            mean_difference = np.mean(episode_differences, axis=0)
            bins = {}
            for label, distance in (("last20", 20), ("last100", 100), ("last200", 200), ("full_prefix", len(rewards))):
                indexes = list(range(max(0, len(rewards) - distance), len(rewards)))
                bins[label] = {"mean_correction": float(mean_difference[indexes].mean()),
                               "mean_absolute_correction": float(np.abs(mean_difference[indexes]).mean()),
                               "mean_correction_sign_flip_fraction": float(np.mean([
                                   np.sign(ordinary_vector[i]) != np.sign(ordinary_vector[i] + mean_difference[i])
                                   for i in indexes if abs(ordinary_vector[i]) > 1e-8]))
                                   if any(abs(ordinary_vector[i]) > 1e-8 for i in indexes) else None}
            values.append({"slot": episode["slot"], "lambda": lam,
                           "cutoff_value": cutoff_value,
                           "ordinary_prefix_gae_at_start": ordinary_vector[0],
                           "extended_prefix_gae_at_start_mean": float(np.mean(extended)),
                           "prefix_correction_at_start": float(np.mean(extended) - ordinary_vector[0]),
                           "mean_boundary_tail_advantage": float(np.mean(tail_advantages)),
                           "boundary_tail_advantage_se": float(np.std(tail_advantages, ddof=1) / np.sqrt(len(tail_advantages))) if len(tail_advantages) > 1 else None,
                           "bins": bins,
                           "censored_fraction": float(np.mean([b["censored"] for b in episode["branches"]]))})
    result = {"protocol_sha256": expected, "target_units": "reward-multiplied critic units",
              "lambdas": list(args.lambdas), "episode_results": values,
              "summary": {str(lam): {"episodes": sum(r["lambda"] == lam for r in values),
                    "mean_correction_at_start": float(np.mean([r["prefix_correction_at_start"] for r in values if r["lambda"] == lam])),
                    "mean_boundary_tail_advantage": float(np.mean([r["mean_boundary_tail_advantage"] for r in values if r["lambda"] == lam])),
                    "episode_cluster_se_tail_advantage": float(np.std([r["mean_boundary_tail_advantage"] for r in values if r["lambda"] == lam], ddof=1) / np.sqrt(sum(r["lambda"] == lam for r in values))) if sum(r["lambda"] == lam for r in values) > 1 else None,
                    "mean_cutoff_value": float(np.mean([r["cutoff_value"] for r in values if r["lambda"] == lam])),
                    "mean_extended_boundary_target": float(np.mean([r["cutoff_value"] + r["mean_boundary_tail_advantage"] for r in values if r["lambda"] == lam])),
                    "mean_conditional_branch_se": float(np.mean([r["boundary_tail_advantage_se"] for r in values if r["lambda"] == lam])),
                    "censored_branch_fraction": float(np.mean([r["censored_fraction"] for r in values if r["lambda"] == lam])),
                    "bins": {name: {"mean_correction": float(np.mean([r["bins"][name]["mean_correction"] for r in values if r["lambda"] == lam])),
                                    "episode_cluster_se": float(np.std([r["bins"][name]["mean_correction"] for r in values if r["lambda"] == lam], ddof=1) / np.sqrt(sum(r["lambda"] == lam for r in values))) if sum(r["lambda"] == lam for r in values) > 1 else None,
                                    "mean_absolute_correction": float(np.mean([r["bins"][name]["mean_absolute_correction"] for r in values if r["lambda"] == lam])),
                                    "mean_correction_sign_flip_fraction": float(np.mean([r["bins"][name]["mean_correction_sign_flip_fraction"] for r in values if r["lambda"] == lam if r["bins"][name]["mean_correction_sign_flip_fraction"] is not None])) if any(r["lambda"] == lam and r["bins"][name]["mean_correction_sign_flip_fraction"] is not None for r in values) else None}
                             for name in ("last20", "last100", "last200", "full_prefix")}}
                    for lam in args.lambdas}}
    write_json(folder / "analysis.json", result)
    if not args.skip_gradients:
        before = tensor_tree_sha256(model.state_dict())
        current_gradient_code = gradient_code_sha256()
        prior = prior_analysis
        if (prior is not None and prior.get("protocol_sha256") == expected
                and prior.get("lambdas") == list(args.lambdas)
                and prior.get("gradient_code_sha256") == current_gradient_code
                and prior.get("model_state_sha256_before_after", {}).get("before") == before
                and prior.get("model_state_sha256_before_after", {}).get("unchanged") is True):
            result["pooled_selected_prefix_policy_gradients"] = prior.get(
                "pooled_selected_prefix_policy_gradients", prior.get("full_batch_policy_gradients"))
            result["model_state_sha256_before_after"] = prior["model_state_sha256_before_after"]
        else:
            rollout_data = json.loads((folder / "rollouts.json").read_text())
            result["pooled_selected_prefix_policy_gradients"] = batch_gradient_audit(
                ppo, manifest, data, rollout_data, args.lambdas)
            after = tensor_tree_sha256(model.state_dict())
            if before != after:
                raise RuntimeError("Frozen checkpoint weights changed during gradient analysis")
            result["model_state_sha256_before_after"] = {"before": before, "after": after, "unchanged": True}
        result["gradient_code_sha256"] = current_gradient_code
        write_json(folder / "analysis.json", result)
    lines = ["# Rollout boundary audit", "", f"Checkpoint iteration: {manifest['iteration']}",
             f"Cutoff episodes: {manifest['episodes']}; branches per episode: {manifest['branches_per_episode']}",
             "", f"Corrections compare ordinary prefix GAE with the mean target from cached continuations. Censored tails use the checkpoint critic at the {manifest['max_moves']}-move cap.", "",
             "| lambda | episodes | critic V(cutoff) | extended target | tail A | episode cluster SE | censored |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for lam, summary in result["summary"].items():
        lines.append(f"| {lam} | {summary['episodes']} | {summary['mean_cutoff_value']:.6g} | {summary['mean_extended_boundary_target']:.6g} | {summary['mean_boundary_tail_advantage']:.6g} | {summary['episode_cluster_se_tail_advantage']:.6g} | {summary['censored_branch_fraction']:.3f} |")
    lines += ["", f"Mean correction by distance before the rollout cutoff (cluster SE uses {manifest['episodes']} episode-level means):", "", "| lambda | distance | mean correction | mean absolute correction | mean-advantage sign-flip fraction |", "|---:|---|---:|---:|---:|"]
    for lam, summary in result["summary"].items():
        for distance, stats in summary["bins"].items():
            flip = "n/a" if stats["mean_correction_sign_flip_fraction"] is None else f"{stats['mean_correction_sign_flip_fraction']:.3f}"
            lines.append(f"| {lam} | {distance} | {stats['mean_correction']:.6g} ± {stats['episode_cluster_se']:.3g} | {stats['mean_absolute_correction']:.6g} | {flip} |")
    lines += ["", "The report uses reward-multiplied value units. This is a paired descriptive audit; branch seeds are continuations from the same cutoff board.", ""]
    if "pooled_selected_prefix_policy_gradients" in result:
        prefix_count = len(data["episodes"])
        steps_per_game = int(manifest["ppo_config"]["steps_per_game_per_rollout"])
        pooled_count = prefix_count * steps_per_game
        lines += [f"The surrogate policy-gradient comparison pools one complete {steps_per_game}-step prefix from each of the {prefix_count} independent cutoff slots. It uses PPO masks, saved behavior log probabilities, the scheduled action mixture, and one mean/std normalization across all {pooled_count:,} transitions. It is an assembled diagnostic batch, not a single collected PPO batch. Entropy is excluded because advantage replacement does not change it.", "",
                  f"| lambda | transitions | ordinary/corrected cosine | relative gradient change | {len(data['episodes'][0]['branches']) // 2}/{len(data['episodes'][0]['branches']) - len(data['episodes'][0]['branches']) // 2} split cosine |", "|---:|---:|---:|---:|---:|"]
        for lam, stats in result["pooled_selected_prefix_policy_gradients"].items():
            lines.append(f"| {lam} | {stats['pooled_transitions']} | {stats['cosine_similarity']:.6g} | {stats['relative_gradient_change']:.6g} | {stats['branch_split_cosine']:.6g} |")
        lines += ["", f"Checkpoint model weights remained unchanged: {result['model_state_sha256_before_after']['unchanged']}.", ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    result["analysis_seconds"] = perf_counter() - analysis_started
    write_json(folder / "analysis.json", result)
    print(f"Wrote {folder / 'analysis.json'} and report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("checkpoint", type=Path)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--episodes", type=int, default=32)
    cap.add_argument("--branches", type=int, default=8)
    cap.add_argument("--batches", type=int, default=2)
    cap.add_argument("--seed", type=int, default=20482027)
    cap.add_argument("--max-moves", type=int, default=512)
    cap.add_argument("--current-d4-order", action="store_true")
    ana = commands.add_parser("analyze")
    ana.add_argument("folder", type=Path)
    ana.add_argument("--lambdas", type=float, nargs="+", default=[0.95, 0.99])
    ana.add_argument("--skip-gradients", action="store_true", help="Skip full-rollout policy-gradient comparison")
    ana.add_argument("--reuse-gradients", action="store_true", help="Reuse matching cached gradient vectors without recomputing autograd")
    args = parser.parse_args()
    if args.command == "capture":
        if min(args.episodes, args.batches, args.max_moves) <= 0 or args.branches < 2:
            parser.error("episodes, batches, and max-moves must be positive; branches must be at least 2")
        capture(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
