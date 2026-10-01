"""Collect locked 32-trial train/held-out action pools for the iteration-1000 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from games.rl_2048.scripts.audit_ppo_update import load_runtime, sha256, write_json
from games.rl_2048.scripts.balanced_sampling_ppo import _branches_for_action


ROOT = Path(__file__).resolve().parents[3]


def collect(args):
    audit = args.audit.resolve()
    lock_path = audit / "locked_plan.json"
    plan = json.loads(lock_path.read_text())
    lock_hash = plan.pop("lock_sha256")
    calculated = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan["lock_sha256"] = lock_hash
    if calculated != lock_hash:
        raise ValueError("locked plan digest does not match")
    if len(plan["boards"]) != 96:
        raise ValueError("expected exactly 96 locked boards")

    protocol_path = audit / "prebranch_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha256(protocol_path)
    branch = protocol["branch_rng_pairing"]
    training_namespace = int(branch["training_namespace"])
    heldout_namespace = int(branch["heldout_namespace"])
    if (args.seed != branch["seed"] or args.trials != branch["trials_per_legal_action_per_pool"]
            or args.max_moves != branch["trajectory_cap_moves"]):
        raise ValueError("pool CLI settings differ from the locked protocol")
    if not branch.get("forced_first_action_then_frozen_behavior_policy"):
        raise ValueError("collector only implements forced-first-action frozen-policy continuations")
    if args.pool_kind == "heldout":
        lr_path = audit / "learning_rate_lock.json"
        if not lr_path.exists():
            raise FileNotFoundError("held-out collection is gated on a training-only learning_rate_lock.json")
        lr_lock = json.loads(lr_path.read_text())
        if lr_lock.get("protocol_sha256") != protocol_sha or lr_lock.get("lock_sha256") != lock_hash:
            raise ValueError("held-out collection is gated on a matching protocol/bank LR lock")

    native = audit / "native"
    manifest = json.loads((native / "selection.json").read_text())
    capture_path = native / "capture.pt"
    if sha256(capture_path) != plan["capture_sha256"]:
        raise ValueError("native capture differs from lock")
    if sha256(Path(manifest["source_checkpoint"])) != plan["checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from lock")
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    torch.set_num_threads(int(manifest["threads"]))
    _, model, _, ppo, epsilon = load_runtime(
        Path(manifest["source_checkpoint"]), int(manifest["seed"]),
        bool(manifest["current_d4_order_acknowledged"]),
    )
    if abs(float(epsilon) - float(branch["captured_exploration_probability"])) > 1e-12:
        raise ValueError("loaded checkpoint exploration probability differs from protocol")
    model.load_state_dict(capture["model_state_dict"])
    model.eval().requires_grad_(False)
    settings = {
        "version": 1,
        "lock_sha256": lock_hash,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "capture_sha256": plan["capture_sha256"],
        "training_per_state_action": args.trials,
        "heldout_per_state_action": args.trials,
        "max_moves": args.max_moves,
        "rng_seed": args.seed,
        "training_namespace": training_namespace,
        "heldout_namespace": heldout_namespace,
        "protocol_sha256": protocol_sha,
        "candidate_screen_manifest_sha256": sha256(audit / "candidate_screen_manifest.json"),
        "frozen_policy": True,
        "forced_first_action": True,
    }
    settings_path = audit / "branch_pool_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("existing pools use different settings; use a fresh audit root")
    if not settings_path.exists():
        write_json(settings_path, settings)

    count = 0
    pool_hashes = {}
    for bi, board in enumerate(plan["boards"], 1):
        for action in board["legal_actions"]:
            phase = training_namespace if args.pool_kind == "training" else heldout_namespace
            for label, phase in ((args.pool_kind, phase),):
                path = audit / f"{label}_board_{board['id']:02d}_action_{action}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get("count") != args.trials or saved.get("board_id") != board["id"]
                            or saved.get("action") != action or saved.get("rng_namespace") != phase):
                        raise ValueError(f"existing pool metadata mismatch: {path}")
                else:
                    rows = _branches_for_action(
                        model, ppo, board, int(action), args.trials, phase,
                        args.seed, args.max_moves,
                    )
                    write_json(path, {
                        "board_id": int(board["id"]), "action": int(action),
                        "count": args.trials, "rng_namespace": phase,
                        "records": rows,
                    })
                pool_hashes[path.name] = sha256(path)
                count += 1
        if bi % 8 == 0:
            print(f"Completed branch pools for {bi}/96 boards ({count} files)", flush=True)

    write_json(audit / f"{args.pool_kind}_branch_pool_manifest.json", {
        "settings": settings,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "pool_file_count": count,
        "pool_sha256": pool_hashes,
        "trajectory_count": sum(int(args.trials) for b in plan["boards"] for _ in b["legal_actions"]),
        "pool_kind": args.pool_kind,
        "records_are_frozen_policy_forced_first_action": True,
        "protocol_sha256": settings["protocol_sha256"],
        "locked_plan_sha256": lock_hash,
        "candidate_screen_manifest_sha256": settings["candidate_screen_manifest_sha256"],
    })
    print(f"Saved {count} {args.pool_kind} pools under {audit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20482200)
    parser.add_argument("--pool-kind", choices=("training", "heldout"), default="training")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
