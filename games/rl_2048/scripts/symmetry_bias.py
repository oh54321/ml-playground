"""Test whether a trained 2048 policy respects board symmetries.

The script collects deterministic-policy states, evaluates each state under all
eight symmetries of the square, and compares the transformed policy with the
original policy after applying the corresponding action permutation.

Run from the repository root:

    .venv/bin/python -m games.rl_2048.scripts.symmetry_bias

An explicit checkpoint can be supplied with ``--checkpoint``.  By default the
latest ``artifacts/ppo_2048/run_*`` directory is used, preferring ``final.pt``
over ``best.pt``.  The test is read-only with respect to the checkpoint and
training runs; it writes only its JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make direct execution work as well as ``python -m ...``.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.models.actor_critic import ActorCriticModel  # noqa: E402
from core.symmetries.d4_transforms import transform_array  # noqa: E402
from games.rl_2048.legacy_agent import model_from_legacy_config  # noqa: E402
from games.rl_2048.environment.game_2048 import Game2048, State2048  # noqa: E402
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS  # noqa: E402


ACTION_NAMES = ("left", "up", "right", "down")
ActionMap = tuple[int, int, int, int]
SYMMETRIES = SQUARE_SYMMETRY_SPECS
SYMMETRY_LABELS = tuple(
    symmetry.transform.name.lower() for symmetry in SYMMETRIES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to test; defaults to the latest training run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "ppo_2048" / "symmetry_bias_test.json",
        help="JSON report path.",
    )
    parser.add_argument("--games", type=int, default=12, help="Rollout games used to collect states.")
    parser.add_argument(
        "--states-per-game",
        type=int,
        default=24,
        help="Maximum states sampled from each rollout game.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=20,
        help="Sample one state every N deterministic-policy moves.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20482026,
        help="First game seed; subsequent games use consecutive seeds.",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=2000,
        help="Safety cap for each state-collection game.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of transformed boards per inference batch.",
    )
    return parser.parse_args()


def find_latest_checkpoint() -> Path:
    run_root = REPO_ROOT / "artifacts" / "ppo_2048"
    runs = sorted(
        (path for path in run_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for run in runs:
        for filename in ("final.pt", "best.pt"):
            checkpoint = run / filename
            if checkpoint.exists():
                return checkpoint
    raise FileNotFoundError(f"No final.pt or best.pt found under {run_root}")


def load_model(checkpoint_path: Path) -> tuple[ActorCriticModel, dict, torch.device]:
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_from_legacy_config(checkpoint["resnet_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def legal_actions(board: np.ndarray) -> list[int]:
    """Return legal directions without adding a random tile."""
    game = Game2048(seed=0)
    game.state = State2048(grid=np.asarray(board, dtype=np.int64).copy())
    game.done = False
    return game.legal_actions()


def policy_batch(
    model: ActorCriticModel,
    boards: list[np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return legal-masked probabilities, greedy actions, and legal masks."""
    tensors = torch.stack([State2048(grid=board).to_tensor() for board in boards]).to(device)
    with torch.no_grad():
        logits = model(tensors).action_logits

    legal_masks = np.zeros((len(boards), 4), dtype=bool)
    for row, board in enumerate(boards):
        legal_masks[row, legal_actions(board)] = True
    mask = torch.as_tensor(legal_masks, device=device)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=-1).cpu().numpy()
    actions = masked_logits.argmax(dim=-1).cpu().numpy()
    return probabilities, actions, legal_masks


def collect_states(
    model: ActorCriticModel,
    device: torch.device,
    games: int,
    states_per_game: int,
    sample_every: int,
    seed_base: int,
    max_moves: int,
) -> list[dict]:
    states: list[dict] = []
    for game_index in range(games):
        game = Game2048(seed=seed_base + game_index)
        sampled = 0
        step = 0
        while not game.done and step < max_moves and sampled < states_per_game:
            if step % sample_every == 0:
                board = game.state.grid.copy()
                probabilities, actions, legal_masks = policy_batch(model, [board], device)
                states.append(
                    {
                        "game": game_index,
                        "seed": seed_base + game_index,
                        "step": step,
                        "board": board.tolist(),
                        "base_action": int(actions[0]),
                        "base_probabilities": probabilities[0].tolist(),
                        "legal_actions": np.flatnonzero(legal_masks[0]).tolist(),
                    }
                )
                sampled += 1
            board = game.state.grid.copy()
            probabilities, actions, legal_masks = policy_batch(model, [board], device)
            action = int(actions[0])
            if not legal_masks[0, action]:
                raise RuntimeError("Greedy policy selected an illegal action")
            result = game.step(action)
            if not result.action_applied:
                raise RuntimeError("Greedy policy selected a non-moving action")
            step += 1
    return states


def inverse_action_map(action_map: ActionMap) -> ActionMap:
    inverse = [0] * 4
    for original_action, transformed_action in enumerate(action_map):
        inverse[transformed_action] = original_action
    return tuple(inverse)  # type: ignore[return-value]


def evaluate_symmetries(
    model: ActorCriticModel,
    device: torch.device,
    states: list[dict],
    batch_size: int,
) -> tuple[dict, list[dict]]:
    transformed_jobs: list[tuple[int, int, np.ndarray]] = []
    for state_index, state in enumerate(states):
        board = np.asarray(state["board"], dtype=np.int64)
        for symmetry_index, symmetry in enumerate(SYMMETRIES):
            transformed = transform_array(board, symmetry.transform)
            transformed_jobs.append(
                (state_index, symmetry_index, transformed)
            )

    outputs: dict[tuple[int, int], tuple[np.ndarray, int, np.ndarray]] = {}
    for start in range(0, len(transformed_jobs), batch_size):
        batch = transformed_jobs[start : start + batch_size]
        boards = [job[2] for job in batch]
        probabilities, actions, legal_masks = policy_batch(model, boards, device)
        for row, (state_index, symmetry_index, _board) in enumerate(batch):
            outputs[(state_index, symmetry_index)] = (
                probabilities[row],
                int(actions[row]),
                legal_masks[row],
            )

    per_state: list[dict] = []
    metric_rows: dict[str, list[dict]] = {
        label: [] for label in SYMMETRY_LABELS
    }
    for state_index, state in enumerate(states):
        base_probs = np.asarray(state["base_probabilities"], dtype=float)
        base_action = int(state["base_action"])
        state_result = {
            "game": state["game"],
            "seed": state["seed"],
            "step": state["step"],
            "board": state["board"],
            "base_action": ACTION_NAMES[base_action],
            "symmetries": {},
        }
        for symmetry_index, symmetry in enumerate(SYMMETRIES):
            probabilities, action, legal_mask = outputs[(state_index, symmetry_index)]
            expected_action = symmetry.action_permutation[base_action]
            expected_probs = np.zeros(4, dtype=float)
            for original_action, transformed_action in enumerate(symmetry.action_permutation):
                expected_probs[transformed_action] = base_probs[original_action]
            max_probability_error = float(np.max(np.abs(probabilities - expected_probs)))
            l1_probability_error = float(np.sum(np.abs(probabilities - expected_probs)))
            row = {
                "actual_action": ACTION_NAMES[action],
                "expected_action": ACTION_NAMES[expected_action],
                "raw_same_direction": action == base_action,
                "mapped_action_match": action == expected_action,
                "max_probability_error": max_probability_error,
                "l1_probability_error": l1_probability_error,
                "legal_actions": np.flatnonzero(legal_mask).tolist(),
                "probabilities": probabilities.tolist(),
            }
            label = SYMMETRY_LABELS[symmetry_index]
            state_result["symmetries"][label] = row
            metric_rows[label].append(row)
        per_state.append(state_result)

    summary: dict[str, dict] = {}
    for symmetry_index, symmetry in enumerate(SYMMETRIES):
        label = SYMMETRY_LABELS[symmetry_index]
        rows = metric_rows[label]
        inverse = inverse_action_map(symmetry.action_permutation)
        raw_counts = {name: 0 for name in ACTION_NAMES}
        mapped_back_counts = {name: 0 for name in ACTION_NAMES}
        for row in rows:
            raw_counts[row["actual_action"]] += 1
        # Count transformed predictions in the original board's frame too.
        for row in rows:
            transformed_action = ACTION_NAMES.index(row["actual_action"])
            mapped_back_counts[ACTION_NAMES[inverse[transformed_action]]] += 1
        summary[label] = {
            "states": len(rows),
            "mapped_action_match_rate": float(np.mean([row["mapped_action_match"] for row in rows])),
            "raw_same_direction_rate": float(np.mean([row["raw_same_direction"] for row in rows])),
            "mean_max_probability_error": float(np.mean([row["max_probability_error"] for row in rows])),
            "median_max_probability_error": float(np.median([row["max_probability_error"] for row in rows])),
            "mean_l1_probability_error": float(np.mean([row["l1_probability_error"] for row in rows])),
            "transformed_action_counts": raw_counts,
            "mapped_back_action_counts": mapped_back_counts,
        }
    return summary, per_state


def print_report(
    checkpoint_path: Path,
    checkpoint: dict,
    states: list[dict],
    summary: dict,
) -> None:
    evaluation = checkpoint.get("evaluation", {})
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training iteration: {checkpoint.get('outer_iteration', 'unknown')}")
    if evaluation:
        print(
            "Saved evaluation: "
            f"mean score {evaluation.get('mean_score', 'unknown')}, "
            f"mean moves {evaluation.get('mean_moves', 'unknown')}"
        )
    print(f"Collected states: {len(states)}")
    base_counts = {name: 0 for name in ACTION_NAMES}
    for state in states:
        base_counts[ACTION_NAMES[state["base_action"]]] += 1
    print("Original greedy actions: " + ", ".join(f"{name}={count}" for name, count in base_counts.items()))
    print()
    print("Symmetry results (mapped match should be high; raw same direction is bias):")
    for symmetry_index, _symmetry in enumerate(SYMMETRIES):
        label = SYMMETRY_LABELS[symmetry_index]
        row = summary[label]
        print(
            f"  {label:18} "
            f"mapped={row['mapped_action_match_rate']:.1%}  "
            f"raw_same={row['raw_same_direction_rate']:.1%}  "
            f"mean_max_prob_error={row['mean_max_probability_error']:.4f}  "
            f"median={row['median_max_probability_error']:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.states_per_game <= 0 or args.sample_every <= 0:
        raise ValueError("--games, --states-per-game, and --sample-every must be positive")
    if args.max_moves <= 0 or args.batch_size <= 0:
        raise ValueError("--max-moves and --batch-size must be positive")

    checkpoint_path = (args.checkpoint or find_latest_checkpoint()).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model, checkpoint, device = load_model(checkpoint_path)
    print(f"Collecting states from {args.games} deterministic games...")
    states = collect_states(
        model,
        device,
        args.games,
        args.states_per_game,
        args.sample_every,
        args.seed_base,
        args.max_moves,
    )
    if not states:
        raise RuntimeError("No states were collected")
    summary, per_state = evaluate_symmetries(model, device, states, args.batch_size)
    report = {
        "version": 1,
        "checkpoint": str(checkpoint_path),
        "training_iteration": checkpoint.get("outer_iteration"),
        "model_config": checkpoint.get("resnet_config"),
        "collection": {
            "games": args.games,
            "states_per_game": args.states_per_game,
            "sample_every": args.sample_every,
            "seed_base": args.seed_base,
            "max_moves": args.max_moves,
        },
        "action_names": list(ACTION_NAMES),
        "summary": summary,
        "states": per_state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(checkpoint_path, checkpoint, states, summary)
    print(f"\nJSON report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
