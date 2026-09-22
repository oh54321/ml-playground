# Does PPO's advantage signal favor the better moves?

`inspect_advantages.py` tests a **frozen trained actor and critic**. For each
selected board it forces every legal first move, samples the remaining actions
from the same actor until game over, and calculates both eventual return and the
GAE learning signal from those same trajectories. No optimizer is created and no
training code, source checkpoint, or saved training run is modified.

## 1. Freeze a checkpoint and select boards

Run from the repository root, using a trusted local checkpoint:

```bash
.venv/bin/python -m games.rl_2048.inspect_advantages select \
  artifacts/ppo_2048/run_20260921_003303_661733_separate_trunks_c32_b4_bigger_model_lr_3e-4_minibatch_4096_parallel_32_adaptive_no_shared_trunk/final.pt \
  --output artifacts/ppo_2048/advantage_audit_final
```

Defaults: 96 fresh sampled games; 12 representative boards; 6 crowded boards
(at most two empty cells); 6 high-tile boards (maximum tile at least 1024).
All selected boards have at least two legal moves. Selection is random over
eligible games, then eligible time steps within a game—not uniform over the pool
of all transitions, which would overweight long games. No source game is reused
across groups, and duplicate boards are rejected. Representative games are
selected first; stress-test groups use the remaining games.

Selection does not consult branch returns, advantage values, or eventual game
scores. `boards.json` locks the selected boards, actor probabilities, critic
values, provenance, seed, and settings **before discovery begins**. The copied
checkpoint is saved as `checkpoint.pt` and verified by SHA-256.

Options: `--games`, `--representative`, `--crowded`, `--high-tile`,
`--crowded-cells`, `--high-tile-threshold`, and `--seed`. If there are too few
qualifying games for a stress group, selection stops rather than silently changing
the sample. Use a new output directory with more games or fewer requested boards.
The output directory must not already exist, protecting previous investigations.

## 2. Measure all first actions

```bash
.venv/bin/python -m games.rl_2048.inspect_advantages evaluate \
  artifacts/ppo_2048/advantage_audit_final --trials 64
```

This runs 64 complete games per legal action per board: 3,072–6,144 continuation
games for the default 24 boards. It is a substantive CPU experiment, not a quick
unit test. Progress is printed by board and during long continuations.

Outputs under `discovery/`:

- `settings.json`: the fixed action-comparison plan and manifest hash.
- `board_000.json`, etc.: per-action summaries and every trial's measurements.
- `results.json`: the combined complete result.
- `report.md`: readable boards, policy probabilities, returns, GAE, and pairwise
  differences, with representative and stress groups kept separate.

All values are expressed in **game-point units**. The critic's configured value
transform is inverted and its reward multiplier removed. For gamma=1, Monte
Carlo return equals remaining game score. For gamma<1, action comparisons use
discounted return (the PPO objective), and undiscounted score is also retained.

### Which GAE is being measured?

- **Primary `gae`:** computed with the checkpoint's gamma/lambda, ending at the
  board's next synthetic rollout boundary or episode time limit. A nonterminal
  boundary bootstraps from the frozen critic, just as the trainer does.
- **`full_episode_gae`:** the same calculation continued to natural game over.
- **Monte Carlo return:** the complete observed return, without critic bootstrap.

Each source game starts at synthetic rollout offset zero. For source move `t`,
the primary horizon is `min(rollout_steps - t % rollout_steps,
episode_limit - t)`. This is a documented training-like boundary, not a claim to
reconstruct that board's position in an actual past training buffer. Natural
game termination always uses next value zero. Complete continuation games are
still played even if the primary GAE horizon ends earlier.

With lambda=1 and no intervening boundary, full-episode GAE must equal discounted
return minus the initial critic value. Tests verify this identity and truncated
bootstrapping separately. The primary signal is not rollout-normalized: common
mean subtraction and positive standard-deviation scaling preserve action rankings.

For a pair “left minus right,” a positive return difference but a negative GAE
difference is a potential misleading learning signal. Uncertainty matters: one
noisy trajectory or a point-estimate ranking reversal is not enough.

## 3. Independently confirm discrepancies

```bash
.venv/bin/python -m games.rl_2048.inspect_advantages confirm \
  artifacts/ppo_2048/advantage_audit_final --top 3 --trials 256
```

This selects up to three discovery boards with opposing return/GAE point-estimate
rankings, prioritizing standardized gaps. It records one fixed action pair per
board **before** running new trials. Confirmation uses a separate RNG namespace,
plays all legal first actions, but tests only those preselected pairs. It does not
pick new “best actions” for its confirmation comparisons.

To check particular boards instead, use `--boards 0 7 12` rather than `--top`.
If discovery has no opposing point estimates, automatic confirmation reports
that fact rather than manufacturing a discrepancy; explicit board IDs can still
be used to confirm agreement or investigate an inconclusive case.

Confirmation outputs live under `confirmation/`, leaving discovery unchanged.
Statuses are `opposite`, `aligned`, or `inconclusive`. They use paired differences
and approximate normal intervals adjusted across the predeclared action pairs
and both primary metrics (return and GAE). Nominal 95% intervals are also saved.
Full-episode GAE is a secondary diagnostic, not part of that multiple-comparison
decision. These finite-sample intervals are approximate; heavy tails, very small
samples, and zero observed sample variance warrant caution.

All trials share spawn/action random streams across first actions for pairing.
Different actions can map the same random draw to different board cells; this is
common-random-number pairing, not identical future boards. Selection, discovery,
and confirmation RNG namespaces are distinct. Board IDs remain stable when only
a subset is confirmed.

## Interruptions, safeguards, and tests

Repeat an interrupted `evaluate` or `confirm` command with **the same options**
and `--resume`. Completed board files are reused. An incomplete board is rerun
from its deterministic seeds; its partial trials are never treated as complete.
Changing trial counts/settings during resume is refused, and existing experiment
folders cannot be silently overwritten. `--max-moves` defaults to 10,000; hitting
the safety cap fails rather than reporting a truncated return as a complete game.
`--threads 1` and `--batch-size 256` are the CPU defaults. Shared, separate, and
older shared-trunk checkpoint configurations are supported.

```bash
.venv/bin/python -m unittest games.rl_2048.test_inspect_advantages -v
```

This is the **learning-signal audit only**. It does not perform a PPO update,
choose a new learning rate, or modify the model. Testing whether an actual update
follows a correct signal is a subsequent experiment. Rotation/reflection and
action-direction-bias tests are also separate follow-ups.
