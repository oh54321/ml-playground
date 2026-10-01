"""Frozen three-controller evaluation for the verified PPO2048 checkpoint.

The critic controller is greedy with respect to exact one-step spawn expectation
of the checkpoint's frozen value head. It is a controller diagnostic, not a
trained policy or a multi-step search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from games.rl_2048.environment.game_2048 import Game2048, State2048
from games.rl_2048.scripts.audit_ppo_update import load_runtime


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ppo_2048/longer_run_post_refactor_20260928_011841_456254/final.pt"
OUT = ROOT / "artifacts/ppo_2048/ppo_longer_run_post_refactor_policy_evaluation"
EXPECTED_SOURCE_SHA = "757832a35457094657b76433553e04b805443a0e0a00e00ee789baa9b1d40505"
ITERATION = 1000
GAMES = 256
MASTER_SEED = 20482991
PILOT_SEED = 20482992
MAX_MOVES = 1048576
INFERENCE_BATCH = 1024
ACTOR_RUNNER = OUT / "actor_baselines/evaluate_actor_baselines.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def lock_protocol(args) -> None:
    if not SOURCE.exists() or sha256(SOURCE) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Source checkpoint is missing or its verified SHA changed")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if int(checkpoint["outer_iteration"]) != ITERATION:
        raise RuntimeError("Source checkpoint is not outer_iteration 1000")
    cfg = dict(checkpoint["ppo_config"])
    if float(cfg["gamma"]) != 1.0 or float(cfg["reward_multiplier"]) != 0.01:
        raise RuntimeError("Unexpected gamma or reward multiplier")
    if not cfg.get("value_transform", "identity"):
        raise RuntimeError("Unexpected value transform configuration")
    seed_rng = np.random.default_rng(MASTER_SEED)
    env_seeds = seed_rng.integers(1, 2**32 - 1, size=GAMES, dtype=np.uint32).astype(np.uint64).tolist()
    action_seeds = seed_rng.integers(1, 2**32 - 1, size=GAMES, dtype=np.uint32).astype(np.uint64).tolist()
    pilot_rng = np.random.default_rng(PILOT_SEED)
    pilot_stream = pilot_rng.integers(1, 2**32 - 1, size=16, dtype=np.uint32).astype(np.uint64).tolist()
    pilot_seeds, pilot_action_seeds = pilot_stream[:8], pilot_stream[8:]
    implementation_files = [
        Path(__file__).resolve(),
        (ROOT / "games/rl_2048/environment/game_2048.py").resolve(),
        (ROOT / "games/rl_2048/training/evaluation.py").resolve(),
        ACTOR_RUNNER.resolve(),
    ]
    implementation_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in implementation_files
    }
    plan = {
        "task": "frozen three-controller full-game comparison",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": EXPECTED_SOURCE_SHA,
        "outer_iteration": ITERATION,
        "model_config": checkpoint["resnet_config"],
        "ppo_config": cfg,
        "controller_definitions": {
            "sampled_actor": "sample legal action from checkpoint actor pi at every decision using a dedicated per-game action RNG",
            "greedy_actor": "argmax legal action probability under checkpoint actor pi at every decision; lowest action index wins ties",
            "greedy_critic": "at every decision, choose legal action maximizing immediate raw score reward + gamma * exact expectation over uniform empty-cell and 0.9/0.1 tile-2/tile-4 spawn of frozen critic value; terminal successors have value 0; lowest action index wins ties",
        },
        "game_rules": {
            "environment": "current Game2048 implementation",
            "initial_board_and_spawns": "Game2048(seed=environment_seed); built-in uniform empty-cell spawn and tile-2 probability 0.9 / tile-4 probability 0.1",
            "termination": "Game2048 terminal condition (no legal moves); continue after reaching 2048, matching the production evaluate_games loop",
            "max_moves": MAX_MOVES,
            "max_moves_semantics": "matches the checkpoint's max_episode_steps safety cap; production evaluation itself runs until terminal. A game reaching the cap is an error, never treated as terminal or scored as complete",
            "critic_q": "gamma from checkpoint; raw immediate game reward plus discounted untransformed critic value converted from reward-multiplied units; exact one-ply spawn expectation, no sampled spawn for Q and no deeper search",
            "legal_actions": "current Game2048.legal_actions(); invalid actions are excluded before controller selection",
            "ties": "lowest legal action index for both greedy controllers",
        },
        "design": {
            "primary_comparison": "paired per-seed greedy_critic minus greedy_actor difference in reach-2048 indicator",
            "secondary_controllers": ["sampled_actor"],
            "games_per_controller": GAMES,
            "seed_master": MASTER_SEED,
            "seed_generation": "numpy.default_rng(master_seed).integers(1, 2**32-1, dtype=uint32), first 256 environment seeds then 256 independent action seeds",
            "environment_stream_pairing": "same environment seed for all three controllers; each controller begins with the same seeded initial spawn stream, but different trajectories consume it differently",
            "action_rng": "sampled actor uses its per-game action seed; environment spawn randomness is not consumed by actor sampling",
            "pilot": {"games": 8, "seed_master": PILOT_SEED, "seed_generation": "same generator rule; first eight values are environment seeds, next eight independent action seeds; all predeclared and excluded from final outcomes", "purpose": "throughput and runner validation only; retain aggregate decisions/time but suppress scores, reach, max-tile, and per-game outcome fields"},
            "statistics": "Primary greedy_critic minus greedy_actor reach-2048 difference: paired discordant counts, exact McNemar test, and paired-seed percentile bootstrap 95% interval. Secondary sampled_actor-vs-each-greedy reach differences: paired discordants and nominal exact McNemar tests; paired-seed percentile bootstrap 95% intervals for mean/median score and mean log2(max_tile) differences. Secondary contrasts are exploratory with no multiplicity-adjusted significance claims. Report marginal score and max-tile distributions and runtime; resample game seeds, never within-game states. Bootstrap uses 20,000 replicates, PCG64 seed 20482993, and NumPy's default linear percentile quantile.",
            "analysis_seed": 20482993,
            "inference_batch_size": INFERENCE_BATCH,
        },
        "exclusions": ["no model training or optimizer step", "no gradient or reward changes", "no selection of game seeds from model predictions or results", "no score-curve analysis"],
        "environment_code": "games/rl_2048/environment/game_2048.py",
        "evaluation_code": "games/rl_2048/training/evaluation.py",
        "implementation_sha256": implementation_hashes,
    }
    plan["protocol_sha256"] = canonical_sha(plan)
    manifest = {"protocol_sha256": plan["protocol_sha256"], "environment_seeds": env_seeds,
                "action_seeds": action_seeds, "pilot_seeds": pilot_seeds,
                "pilot_action_seeds": pilot_action_seeds}
    manifest["seed_manifest_sha256"] = canonical_sha(manifest)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "locked_protocol.json", plan)
    write_json(out / "seed_manifest.json", manifest)
    print(json.dumps({"protocol_sha256": plan["protocol_sha256"], "seed_manifest_sha256": manifest["seed_manifest_sha256"], "output": str(out)}))


def load_inputs(root: Path):
    plan = json.loads((root / "locked_protocol.json").read_text())
    manifest = json.loads((root / "seed_manifest.json").read_text())
    digest = plan.pop("protocol_sha256")
    if canonical_sha(plan) != digest:
        raise RuntimeError("Protocol digest mismatch")
    plan["protocol_sha256"] = digest
    m_digest = manifest.pop("seed_manifest_sha256")
    if canonical_sha(manifest) != m_digest:
        raise RuntimeError("Seed manifest digest mismatch")
    manifest["seed_manifest_sha256"] = m_digest
    if manifest.get("protocol_sha256") != digest:
        raise RuntimeError("Seed manifest is not cross-linked to the locked protocol")
    if plan.get("source_checkpoint") != str(SOURCE.relative_to(ROOT)) or plan.get("outer_iteration") != ITERATION:
        raise RuntimeError("Locked checkpoint path or iteration does not match this authorized run")
    for relative_path, expected_sha in plan["implementation_sha256"].items():
        path = ROOT / relative_path
        if not path.exists() or sha256(path) != expected_sha:
            raise RuntimeError(f"Controller/evaluation code hash mismatch: {relative_path}")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA or plan["source_sha256"] != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Source checkpoint identity mismatch")
    return plan, manifest


@torch.no_grad()
def _critic_actions(ppo, games: list[Game2048], inference_batch: int):
    """Return exact one-step critic-greedy actions for a batch of current games."""
    queries: list[State2048] = []
    references: list[tuple[int, int, float]] = []
    q_values: list[dict[int, float]] = []
    for game_index, game in enumerate(games):
        legal = game.legal_actions()
        if not legal:
            raise RuntimeError("Attempted critic decision in terminal state")
        q_values.append({})
        for action in legal:
            moved, reward, applied = game._simulate_move(game.state.grid, action)
            if not applied:
                raise AssertionError("legal action did not move")
            empties = np.argwhere(moved == 0)
            if len(empties) == 0:
                q_values[game_index][action] = float(reward)
                continue
            q_values[game_index][action] = float(reward)
            for row, col in empties:
                for tile, tile_probability in ((2, 0.9), (4, 0.1)):
                    board = moved.copy()
                    board[row, col] = tile
                    successor = Game2048(seed=0)
                    successor.state = State2048(grid=board)
                    weight = float(tile_probability / len(empties))
                    if not successor.legal_actions():
                        continue
                    else:
                        queries.append(successor.state)
                        references.append((game_index, action, weight))
    for start in range(0, len(queries), inference_batch):
        states = queries[start:start + inference_batch]
        output = ppo.model(ppo._state_batch_tensor(states))
        values = ppo._untransform_predicted_value(output.value).reshape(-1).cpu().numpy()
        for value, (game_index, action, weight) in zip(values, references[start:start + inference_batch], strict=True):
            q_values[game_index][action] += weight * float(ppo.config.gamma) * float(value) / float(ppo.config.reward_multiplier)
    actions, all_q = [], []
    for game, q in zip(games, q_values, strict=True):
        legal = game.legal_actions()
        if sorted(q) != legal:
            raise AssertionError("Q values do not cover exactly the legal actions")
        action = min(legal, key=lambda candidate: (-q[candidate], candidate))
        actions.append(action)
        all_q.append({str(k): q[k] for k in legal})
    return actions, all_q


@torch.no_grad()
def run_critic(args) -> None:
    root = args.audit
    plan, manifest = load_inputs(root)
    if args.pilot:
        seeds = manifest["pilot_seeds"]
        label = "critic_pilot"
    else:
        seeds = manifest["environment_seeds"]
        label = "greedy_critic"
    checkpoint, model, optimizer, ppo, epsilon = load_runtime(SOURCE, 27183, allow_current_d4=True)
    if epsilon != 0.0 or len(seeds) != (8 if args.pilot else GAMES):
        raise RuntimeError("Checkpoint epsilon or seed count does not match the lock")
    model.eval()
    model_sha_before = canonical_sha({k: v.detach().cpu().numpy().tobytes().hex() for k, v in model.state_dict().items()})
    games = [Game2048(seed=int(seed)) for seed in seeds]
    initial_boards = [game.state.grid.tolist() for game in games]
    active = list(range(len(games)))
    steps = [0] * len(games)
    max_tiles = [2] * len(games)
    reached = [False] * len(games)
    start = time.perf_counter()
    while active:
        next_active = []
        for group_start in range(0, len(active), args.game_batch):
            ids = active[group_start:group_start + args.game_batch]
            group = [games[i] for i in ids]
            actions, _ = _critic_actions(ppo, group, INFERENCE_BATCH)
            for i, game, action in zip(ids, group, actions, strict=True):
                if not game.step(action).action_applied:
                    raise AssertionError("critic selected an illegal action")
                steps[i] += 1
                max_tiles[i] = max(max_tiles[i], int(game.state.grid.max()))
                reached[i] = reached[i] or max_tiles[i] >= 2048
                if not game.done:
                    if steps[i] >= MAX_MOVES:
                        raise RuntimeError(f"game {i} reached the safety move cap; result is not scored")
                    next_active.append(i)
        active = next_active
        if steps and max(steps) % 1000 < args.game_batch:
            print(f"{label}: completed {sum(g.done for g in games)}/{len(games)} games; max moves={max(steps)}", flush=True)
    elapsed = time.perf_counter() - start
    model_sha_after = canonical_sha({k: v.detach().cpu().numpy().tobytes().hex() for k, v in model.state_dict().items()})
    if model_sha_before != model_sha_after:
        raise RuntimeError("Frozen model state changed during evaluation")
    rows = []
    for i, (game, seed) in enumerate(zip(games, seeds, strict=True)):
        rows.append({"game_index": i, "environment_seed": int(seed), "score": int(game.total_score),
                     "steps": steps[i], "max_tile": max_tiles[i], "reached_2048": reached[i],
                     "terminated": bool(game.done), "initial_board": initial_boards[i]})
    result = {"controller": label, "protocol_sha256": plan["protocol_sha256"],
              "seed_manifest_sha256": manifest["seed_manifest_sha256"], "checkpoint_sha256": EXPECTED_SOURCE_SHA,
              "epsilon": epsilon, "games": len(rows), "elapsed_seconds": elapsed,
              "games_per_second": len(rows) / elapsed, "model_state_immutable": model_sha_before == model_sha_after,
              "rows": rows if not args.pilot else None,
              "pilot_only_throughput": bool(args.pilot),
              "pilot_completed_games": len(rows) if args.pilot else None,
              "pilot_total_decisions": sum(steps) if args.pilot else None}
    write_json(root / f"{label}_results.json", result)
    print(json.dumps({k: result[k] for k in ("controller", "games", "elapsed_seconds", "games_per_second")}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--output", type=Path, default=OUT)
    critic = sub.add_parser("critic")
    critic.add_argument("--audit", type=Path, default=OUT)
    critic.add_argument("--pilot", action="store_true")
    critic.add_argument("--game-batch", type=int, default=8)
    args = parser.parse_args()
    if args.command == "lock":
        lock_protocol(args)
    else:
        run_critic(args)


if __name__ == "__main__":
    main()
