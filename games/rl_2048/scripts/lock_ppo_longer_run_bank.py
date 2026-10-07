"""Select and lock a fresh 96-state bank from the intended post-refactor capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.scripts.audit_ppo_update import (
    counterfactual_gae_horizon, deserialize_step, load_runtime, mixture_probabilities,
    sha256, tensor_tree_sha256, write_json,
)

ROOT = Path(__file__).resolve().parents[3]


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lock(args):
    out = args.audit.resolve()
    if (out / "locked_plan.json").exists():
        raise FileExistsError("refuse to replace an existing lock")
    native = out / "native"
    selection = json.loads((native / "selection.json").read_text())
    capture_path = native / "capture.pt"
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    checkpoint = Path(selection["source_checkpoint"])
    if sha256(checkpoint) != selection["source_sha256"]:
        raise ValueError("checkpoint hash changed since native capture")
    if capture.get("source_sha256") != selection["source_sha256"]:
        raise ValueError("capture source checkpoint mismatch")
    torch.set_num_threads(int(selection["threads"]))
    _, model, _, ppo, epsilon = load_runtime(
        checkpoint, int(selection["seed"]), bool(selection["current_d4_order_acknowledged"])
    )
    model.load_state_dict(capture["model_state_dict"])
    model.eval().requires_grad_(False)
    steps = [deserialize_step(row) for row in capture["rollout"]]
    per_game = int(ppo.config.steps_per_game_per_rollout)
    eligible = [i for i, step in enumerate(steps) if int((~step.legal_actions_mask).sum()) >= 2]
    positions = np.unique(np.linspace(0, len(eligible) - 1, min(args.candidate_limit, len(eligible)), dtype=int))
    selected_rows = [(eligible[int(pos)], steps[eligible[int(pos)]]) for pos in positions]
    states = [step.state for _, step in selected_rows]
    with torch.no_grad():
        logits = torch.cat([
            model(ppo._state_batch_tensor(states[start:start + 256])).action_logits.cpu()
            for start in range(0, len(states), 256)
        ])
    rows, successors, action_meta = [], [], []
    for local, (rollout_index, step) in enumerate(selected_rows):
        game = Game2048(seed=0)
        game.replace_board(step.state)
        legal = game.legal_actions()
        z = logits[local].clone()
        mask = torch.ones_like(z, dtype=torch.bool)
        mask[legal] = False
        actor_probs = torch.softmax(z.masked_fill(mask, -torch.inf), -1).numpy()
        row_q = {}
        rows.append((rollout_index, step, legal, actor_probs, row_q))
        for action in legal:
            moved, reward, applied = game._simulate_move(step.state.grid, action)
            if not applied:
                raise RuntimeError("legal action did not move")
            empty = np.argwhere(moved == 0)
            terms = []
            if len(empty):
                for rr, cc in empty:
                    for tile, probability in ((2, .9), (4, .1)):
                        spawned = moved.copy()
                        spawned[rr, cc] = tile
                        nxt = Game2048(seed=0)
                        nxt.replace_board(spawned)
                        if nxt.legal_actions():
                            terms.append((len(successors), probability / len(empty)))
                            successors.append(nxt.state)
                        else:
                            terms.append((None, probability / len(empty)))
            action_meta.append((local, int(action), float(reward), terms))
    values = []
    with torch.no_grad():
        for start in range(0, len(successors), 256):
            output = model(ppo.rollout_collector.state_batch_tensor(successors[start:start + 256])).value
            values.extend((ppo._untransform_predicted_value(output) / ppo.config.reward_multiplier).cpu().tolist())
    for local, action, reward, terms in action_meta:
        rows[local][4][action] = reward + ppo.config.gamma * sum(
            (0. if j is None else float(values[j])) * weight for j, weight in terms
        )
    candidates = []
    for rollout_index, step, legal, probs, qs in rows:
        actor = int(np.argmax(probs))
        critic = max(qs, key=qs.get)
        gap = float(qs[critic] - qs[actor])
        if actor == critic or gap < args.minimum_gap:
            continue
        slot, offset = divmod(rollout_index, per_game)
        episode_start = slot * per_game
        for prior in range(rollout_index - 1, slot * per_game - 1, -1):
            if steps[prior].done or steps[prior].truncated:
                episode_start = prior + 1
                break
        episode_index = sum(1 for s in steps[slot * per_game:episode_start] if s.done or s.truncated)
        episode_id = f"slot{slot:02d}_episode{episode_index:03d}"
        remaining = 0
        boundary = "rollout_segment"
        for following in steps[rollout_index:rollout_index + per_game - offset]:
            remaining += 1
            if following.done:
                boundary = "termination"
                break
            if following.truncated:
                boundary = "truncation"
                break
        age = rollout_index - episode_start
        candidates.append({
            "rollout_index": int(rollout_index), "source_episode_id": episode_id,
            "rollout_slot": int(slot), "episode_index_within_slot": int(episode_index),
            "captured_episode_age": int(age),
            "remaining_rollout_horizon": int(remaining),
            "counterfactual_gae_horizon_limit": int(counterfactual_gae_horizon(per_game - offset, age, ppo.config.max_episode_steps)),
            "boundary_after": boundary,
            "board": step.state.grid.tolist(), "legal_actions": [int(a) for a in legal],
            "actor_action": actor, "critic_action": critic,
            "actor_probabilities": [float(x) for x in probs],
            "behavior_probabilities": [float(x) for x in mixture_probabilities(probs, legal, epsilon)],
            "exact_q": {str(a): float(v) for a, v in qs.items()}, "delta_q": gap,
            "target_actor_probability": float(probs[critic]),
            "raw_advantage": float(step.advantage), "sampled_action": int(step.action),
            "old_log_prob": float(step.log_prob), "largest_tile": int(step.state.grid.max()),
            "empty_cells": int((step.state.grid == 0).sum()),
        })
    cells = {(g, p): [] for g in range(3) for p in range(3)}
    for row in sorted(candidates, key=lambda x: (-x["delta_q"], x["rollout_index"])):
        row["gap_stratum"] = int(np.digitize(row["delta_q"], [750., 1250.]))
        row["probability_stratum"] = int(np.digitize(row["target_actor_probability"], [.01, .10]))
        cells[(row["gap_stratum"], row["probability_stratum"])].append(row)
    selected, episode_counts, seen = [], {}, set()
    while len(selected) < args.states:
        progressed = False
        for cell in sorted(cells):
            choices = cells[cell]
            eligible_choices = [
                row for row in choices
                if tuple(map(tuple, row["board"])) not in seen
                and episode_counts.get(row["source_episode_id"], 0) < args.max_per_episode
            ]
            if eligible_choices:
                # Within each cell, take the highest-gap candidate from an
                # as-yet-unrepresented source episode when one exists. This
                # improves source-episode coverage without changing strata,
                # quotas, within-cell gap priority, or the per-episode cap.
                row = next((r for r in eligible_choices if episode_counts.get(r["source_episode_id"], 0) == 0), eligible_choices[0])
                choices.remove(row)
                row["id"] = len(selected)
                row["selection_order"] = len(selected)
                selected.append(row)
                episode_counts[row["source_episode_id"]] = episode_counts.get(row["source_episode_id"], 0) + 1
                seen.add(tuple(map(tuple, row["board"])))
                progressed = True
            if len(selected) >= args.states:
                break
        if not progressed:
            raise RuntimeError(f"Only selected {len(selected)} states; expand candidate screen before relaxing rules")
    if len(episode_counts) < args.minimum_episodes:
        raise RuntimeError(f"Bank covered only {len(episode_counts)} actual source episodes")
    plan = {
        "title": "Fresh locked disagreement bank for longer_run_post_refactor iteration 1000",
        "status": "selected mechanically from the pre-update rollout capture using fixed rules; native update endpoints excluded",
        "checkpoint_sha256": selection["source_sha256"], "checkpoint_iteration": 1000,
        "capture_sha256": sha256(capture_path), "native_selection_sha256": sha256(native / "selection.json"),
        "rollout_size": len(steps), "rollout_seed": int(selection["seed"]), "epsilon": float(epsilon),
        "selection_inputs": "Only the intended source checkpoint, its pre-update native capture/rollout and boundary markers, and the fixed screen/strata/diversity rules. The native capture's separate 8-board/512-candidate exploratory selection and the native update endpoint probabilities/GAEs are excluded.",
        "native_exploratory_capture_selection": {"board_count": len(selection["boards"]), "candidate_limit": selection.get("candidate_limit", 512), "used_for_bank": False},
        "screen": {"eligible": len(eligible), "candidate_limit": args.candidate_limit, "screened": len(positions),
                   "qualifying_candidates": len(candidates), "minimum_gap_raw_points": args.minimum_gap},
        "bank": {"n": len(selected), "source_episode_count": len(episode_counts), "max_states_per_source_episode": args.max_per_episode,
                 "all_new_states": True, "selection": "deduplicate exact boards; 3x3 tested-pair gap/probability strata round-robin; within each cell prefer highest-gap row from unrepresented source episode, otherwise next highest-gap available row; cap per actual source episode", "selection_rule_version": 2},
        "strata": {"gap_bins_raw_points": [args.minimum_gap, 750, 1250], "actor_target_probability_bins": [0, .01, .10, 1],
                   "cell_counts": {f"{g},{p}": sum(r["gap_stratum"] == g and r["probability_stratum"] == p for r in selected) for g in range(3) for p in range(3)}},
        "trajectory_budget": {"training_per_action": 32, "heldout_per_action": 32, "max_total": 24576,
                              "frozen_checkpoint_policy": True, "forced_first_action": True, "no_tree_expansion": True},
        "boards": selected,
    }
    plan["lock_sha256"] = canonical_hash({k: v for k, v in plan.items() if k != "lock_sha256"})
    write_json(out / "locked_plan.json", plan)
    manifest = {"checkpoint_sha256": plan["checkpoint_sha256"], "capture_sha256": plan["capture_sha256"],
                "selection_sha256": plan["native_selection_sha256"], "candidate_screen": plan["screen"],
                "selection": plan["bank"], "strata": plan["strata"], "screened_candidates": candidates,
                "selected_indices": [r["rollout_index"] for r in selected]}
    manifest["candidate_manifest_sha256"] = canonical_hash(manifest)
    write_json(out / "candidate_screen_manifest.json", manifest)
    print(f"Locked {len(selected)} states across {len(episode_counts)} episodes; {len(candidates)} qualifying candidates; plan={plan['lock_sha256']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--candidate-limit", type=int, default=2048)
    p.add_argument("--states", type=int, default=96)
    p.add_argument("--minimum-gap", type=float, default=250.0)
    p.add_argument("--max-per-episode", type=int, default=4)
    p.add_argument("--minimum-episodes", type=int, default=24)
    lock(p.parse_args())


if __name__ == "__main__":
    main()
