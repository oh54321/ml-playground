"""Recreate and verify the pre-outcome candidate screen for the iteration-1000 bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from games.rl_2048.environment.game_2048 import Game2048
from games.rl_2048.scripts.audit_ppo_update import (
    deserialize_step, exact_action_q, load_runtime, mixture_probabilities, sha256, write_json,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = ROOT / "artifacts/ppo_2048/ppo_iteration_1000_audit"


def run(audit: Path, candidate_limit: int = 2048):
    audit = audit.resolve()
    native = audit / "native"
    selection = json.loads((native / "selection.json").read_text())
    lock = json.loads((audit / "locked_plan.json").read_text())
    capture_path = native / "capture.pt"
    if sha256(capture_path) != lock["capture_sha256"]:
        raise ValueError("capture hash differs from locked bank")
    checkpoint = Path(selection["source_checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from locked bank")
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    torch.set_num_threads(int(selection["threads"]))
    _, model, _, ppo, epsilon = load_runtime(
        checkpoint, int(selection["seed"]), bool(selection["current_d4_order_acknowledged"])
    )
    model.load_state_dict(capture["model_state_dict"])
    model.eval().requires_grad_(False)
    steps = [deserialize_step(row) for row in capture["rollout"]]
    per_game = int(ppo.config.steps_per_game_per_rollout)
    eligible = [i for i, step in enumerate(steps) if int((~step.legal_actions_mask).sum()) >= 2]
    positions = np.unique(np.linspace(0, len(eligible) - 1, min(candidate_limit, len(eligible)), dtype=int))
    rows = [(eligible[int(pos)], steps[eligible[int(pos)]]) for pos in positions]
    states = [step.state for _, step in rows]
    with torch.no_grad():
        logits = torch.cat([
            model(ppo._state_batch_tensor(states[start:start + 256])).action_logits.cpu()
            for start in range(0, len(states), 256)
        ])

    candidates = []
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]
        q_inputs = []
        successors = []
        action_meta = []
        for local, (rollout_index, step) in enumerate(chunk):
            game = Game2048(seed=0)
            game.replace_board(step.state)
            legal = game.legal_actions()
            z = logits[start + local].clone()
            mask = torch.ones_like(z, dtype=torch.bool)
            mask[legal] = False
            actor_probs = torch.softmax(z.masked_fill(mask, -torch.inf), -1).numpy()
            q_inputs.append((rollout_index, step, legal, actor_probs))
            for action in legal:
                moved, reward, applied = game._simulate_move(step.state.grid, action)
                if not applied:
                    raise RuntimeError("candidate legal move failed")
                empty = np.argwhere(moved == 0)
                terms = []
                if len(empty):
                    for rr, cc in empty:
                        for tile, probability in ((2, .9), (4, .1)):
                            spawned = moved.copy()
                            spawned[rr, cc] = tile
                            next_state = Game2048(seed=0)
                            next_state.replace_board(spawned)
                            if next_state.legal_actions():
                                terms.append((len(successors), probability / len(empty)))
                                successors.append(next_state.state)
                            else:
                                terms.append((None, probability / len(empty)))
                action_meta.append((local, int(action), float(reward), terms))
        with torch.no_grad():
            values = []
            for value_start in range(0, len(successors), 256):
                output = model(ppo.rollout_collector.state_batch_tensor(successors[value_start:value_start + 256])).value
                values.extend((ppo._untransform_predicted_value(output) / ppo.config.reward_multiplier).cpu().tolist())
        qs = [{} for _ in chunk]
        for local, action, reward, terms in action_meta:
            qs[local][action] = reward + ppo.config.gamma * sum(
                (0. if index is None else float(values[index])) * weight for index, weight in terms
            )
        for local, (rollout_index, step, legal, actor_probs) in enumerate(q_inputs):
            q = qs[local]
            actor = int(np.argmax(actor_probs))
            critic = max(q, key=q.get)
            gap = float(q[critic] - q[actor])
            if actor == critic or gap < 250.0:
                continue
            slot, offset = divmod(rollout_index, per_game)
            episode_start = slot * per_game
            for prior_index in range(rollout_index - 1, slot * per_game - 1, -1):
                if steps[prior_index].done or steps[prior_index].truncated:
                    episode_start = prior_index + 1
                    break
            episode_index = sum(
                1 for item in steps[slot * per_game:episode_start] if item.done or item.truncated
            )
            episode_id = f"slot{slot:02d}_episode{episode_index:03d}"
            target_probability = float(actor_probs[critic])
            gap_stratum = 0 if gap < 750 else 1 if gap < 1250 else 2
            probability_stratum = 0 if target_probability < .01 else 1 if target_probability < .10 else 2
            candidates.append({
                "rollout_index": int(rollout_index), "source_episode_id": episode_id,
                "rollout_slot": int(slot), "episode_index_within_slot": int(episode_index),
                "episode_age": int(rollout_index - episode_start),
                "board": step.state.grid.tolist(), "legal_actions": [int(a) for a in legal],
                "actor_action": actor, "critic_action": int(critic),
                "actor_probabilities": [float(x) for x in actor_probs],
                "behavior_probabilities": [float(x) for x in mixture_probabilities(actor_probs, legal, epsilon)],
                "exact_q": {str(a): float(v) for a, v in q.items()}, "delta_q": gap,
                "target_actor_probability": target_probability,
                "gap_stratum": gap_stratum, "probability_stratum": probability_stratum,
            })

    cells = {(g, p): [] for g in range(3) for p in range(3)}
    for candidate in sorted(candidates, key=lambda row: (-row["delta_q"], row["rollout_index"])):
        cells[(candidate["gap_stratum"], candidate["probability_stratum"])].append(candidate)
    selected = []
    per_episode = {}
    represented_episodes = set()
    dedup = set()
    while len(selected) < 96:
        advanced = False
        for cell in sorted(cells):
            choices = cells[cell]
            eligible_choices = [
                row for row in choices
                if tuple(map(tuple, row["board"])) not in dedup
                and per_episode.get(row["source_episode_id"], 0) < 4
            ]
            if eligible_choices:
                candidate = next(
                    (row for row in eligible_choices if row["source_episode_id"] not in represented_episodes),
                    eligible_choices[0],
                )
                choices.remove(candidate)
                selected.append(candidate)
                per_episode[candidate["source_episode_id"]] = per_episode.get(candidate["source_episode_id"], 0) + 1
                represented_episodes.add(candidate["source_episode_id"])
                dedup.add(tuple(map(tuple, candidate["board"])))
                candidate["selection_order"] = len(selected) - 1
                advanced = True
            if len(selected) >= 96:
                break
        if not advanced:
            break

    lock_rows = lock["boards"]
    if len(selected) != 96 or len(lock_rows) != 96:
        raise ValueError(f"Candidate screen could not reproduce the 96-state lock: {len(selected)}")
    for position, (candidate, locked) in enumerate(zip(selected, lock_rows)):
        if candidate["rollout_index"] != locked["rollout_index"]:
            raise ValueError(f"Selection order mismatch at position {position}")
        if candidate["board"] != locked["board"] or candidate["critic_action"] != locked["critic_action"]:
            raise ValueError(f"Board/action mismatch at position {position}")

    payload = {
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "capture_sha256": lock["capture_sha256"],
        "locked_plan_sha256": lock["lock_sha256"],
        "screen": {"eligible": len(eligible), "evenly_spaced_limit": candidate_limit,
                   "screened_count": len(positions), "qualifying_count": len(candidates),
                   "minimum_gap_raw_points": 250.0},
        "selection": {"count": len(selected), "source_episode_count": len(per_episode),
                      "max_per_episode": max(per_episode.values()),
                      "cell_counts": {f"{g},{p}": sum(
                          1 for row in selected if row["gap_stratum"] == g and row["probability_stratum"] == p
                      ) for g in range(3) for p in range(3)},
                      "selected_rollout_indices": [row["rollout_index"] for row in selected]},
        "candidates": candidates,
    }
    payload["candidate_manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(audit / "candidate_screen_manifest.json", payload)
    print(f"Verified {len(candidates)} qualifying candidates and {len(selected)} locked rows; source episodes {len(per_episode)}; manifest SHA {payload['candidate_manifest_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--candidate-limit", type=int, default=2048)
    args = parser.parse_args()
    run(args.audit, args.candidate_limit)


if __name__ == "__main__":
    main()
