"""Corrected restore-only replay checks for PPO2048 saved fixed-state updates.

No trajectories are generated. The script reuses the locked 24-state training
pools, sample manifests, checkpoint, captured Adam state, and selected LR.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from games.rl_2048.scripts.audit_ppo_update import load_runtime
from games.rl_2048.scripts.balanced_sampling_ppo import (
    _evaluate_probabilities,
    _load_inputs,
    _make_replicate_batches,
    _restore_and_update,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "artifacts/ppo_2048/ppo_balanced_sampling_iter100"


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tree_sha(obj):
    digest = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            item = value.detach().cpu().contiguous()
            digest.update(str(item.dtype).encode()); digest.update(str(tuple(item.shape)).encode()); digest.update(item.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str): digest.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
        else: digest.update(repr(value).encode())
    visit(obj)
    return digest.hexdigest()


def _step_values(opt):
    return sorted({int(state["step"].item()) if torch.is_tensor(state["step"]) else int(state["step"])
                   for state in opt.state.values() if "step" in state})


def run(folder: Path):
    manifest, capture = _load_inputs(folder)
    prior = json.loads((folder / "selection/selection.json").read_text())
    old_replicates = json.loads((folder / "conservative_sensitivity_replicates.json").read_text())
    draw_manifest = json.loads((folder / "sensitivity_training_draw_manifest.json").read_text())
    selection = json.loads((folder / "sensitivity_lr_selection.json").read_text())
    divisor = int(selection["selected_divisor"])
    lr = float(selection["selected_learning_rate"])
    seed = int(draw_manifest["seed"])
    boards = manifest["boards"]
    torch.set_num_threads(int(manifest["threads"]))
    _, model, optimizer, ppo, _ = load_runtime(Path(manifest["source_checkpoint"]), manifest["seed"], manifest["current_d4_order_acknowledged"])
    model.load_state_dict(copy.deepcopy(capture["model_state_dict"]))
    source_probs, _ = _evaluate_probabilities(model, ppo, boards)
    old_probs = [{int(a): float(source_probs[i, int(a)]) for a in board["legal_actions"]} for i, board in enumerate(boards)]
    batches = _make_replicate_batches(folder, boards, old_probs, 32, 12, seed, capture)
    regenerated = []
    for rep in batches:
        arms = {}
        for arm in ("natural", "balanced"):
            arms[arm] = [[int(r["state_id"]), int(r["action"]), int(r["training_pool_trial_index"])] for r in rep[arm]]
        regenerated.append({"replicate": int(rep["replicate"]), "arms": arms})
    saved = [{"replicate": int(r["replicate"]), "arms": {a: r["arms"][a]["samples"] for a in ("natural", "balanced")}}
             for r in draw_manifest["replicates"]]
    if _json_sha(regenerated) != _json_sha(saved):
        raise ValueError("reconstructed corrected replay draws differ from preserved draw manifest")
    source_capture_hash = _tree_sha(capture["optimizer_state_dict"])
    start_steps = sorted({int(x["step"].item()) if torch.is_tensor(x["step"]) else int(x["step"])
                          for x in capture["optimizer_state_dict"]["state"].values() if "step" in x})
    if len(start_steps) != 1:
        raise ValueError(f"captured optimizer does not have one shared step counter: {start_steps}")
    initial_step = start_steps[0]
    outputs = []
    max_endpoint_errors = {"natural": [], "balanced": []}
    for rep, draw in enumerate(batches):
        for arm in ("natural", "balanced"):
            restored = optimizer.load_state_dict  # keep the target optimizer instance, restore helper deep-copies both states
            result = _restore_and_update(folder, manifest, capture, boards, ppo, optimizer, model,
                                         draw[arm], 8, learning_rate=lr)
            final_steps = _step_values(optimizer)
            if final_steps != sorted({initial_step, initial_step + 8}):
                raise ValueError(f"{arm} replicate {rep} ended with Adam counters {final_steps}; expected updated and untouched actor params only")
            if _tree_sha(capture["optimizer_state_dict"]) != source_capture_hash:
                raise ValueError(f"{arm} replicate {rep} mutated the captured optimizer state")
            old = old_replicates["replicates"][rep][arm]
            max_prob = float(np.max(np.abs(np.asarray(result["behavior_probabilities"]) - np.asarray(old["behavior_probabilities"]))))
            max_endpoint_errors[arm].append(max_prob)
            outputs.append({"replicate": rep, "arm": arm, "start_adam_step": initial_step,
                            "epoch_adam_steps": [int(x["adam_step_counter_sample"]) for x in result["epochs"]],
                            "end_adam_steps": final_steps, "max_behavior_probability_error_vs_prior_saved_result": max_prob,
                            "result": result})
        if rep % 3 == 2: print(f"completed corrected 24-state replay {rep + 1}/12", flush=True)
    action_changes={"natural":{"board_means_pp":{},"up_count":0,"down_count":0},"balanced":{"board_means_pp":{},"up_count":0,"down_count":0}}
    confirmed_ids=[0,1,2,3,5]
    for arm_index,arm in enumerate(("natural","balanced")):
        all_changes=[]
        for board_id in confirmed_ids:
            board=boards[board_id]; target=int(board["critic_action"]); old=float(board["behavior_probabilities"][target]); changes=[]
            for rep in range(12):
                result=outputs[2*rep+arm_index]["result"]
                changes.append((float(result["behavior_probabilities"][board_id][target])-old)*100.0)
            all_changes.extend(changes)
            action_changes[arm]["board_means_pp"][str(board_id)]={"mean":float(np.mean(changes)),"increase_count":sum(x>0 for x in changes),"decrease_count":sum(x<0 for x in changes)}
        action_changes[arm]["mean_target_probability_change_pp"]=float(np.mean(all_changes))
        action_changes[arm]["up_count"]=sum(x>0 for x in all_changes);action_changes[arm]["down_count"]=sum(x<0 for x in all_changes)
    payload = {
        "status": "corrected independent-restore replay; no new trajectories",
        "source_artifact_sha256": hashlib.sha256((folder / "conservative_sensitivity_replicates.json").read_bytes()).hexdigest(),
        "source_draw_manifest_sha256": hashlib.sha256((folder / "sensitivity_training_draw_manifest.json").read_bytes()).hexdigest(),
        "checkpoint_sha256": manifest["source_sha256"], "selected_learning_rate": lr, "selected_divisor": divisor,
        "training_draw_manifests_exact_match": True, "captured_adam_steps": start_steps,
        "all_independent_arms_start_from_same_capture": True, "capture_optimizer_hash_unchanged": True,
        "prior_saved_endpoint_max_abs_behavior_probability_error": {k: max(v) for k, v in max_endpoint_errors.items()},
        "confirmed_five_action_probability_changes": {"board_ids":confirmed_ids,"units":"percentage points","arms":action_changes},
        "arms": outputs,
    }
    (folder / "corrected_restore_replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Saved corrected 24-state restore replay; prior endpoint errors: " + json.dumps(payload["prior_saved_endpoint_max_abs_behavior_probability_error"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT)
    run(parser.parse_args().folder.resolve())
