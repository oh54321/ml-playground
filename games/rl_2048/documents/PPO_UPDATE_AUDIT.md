# One-iteration PPO update audit

This audit captures one ordinary full rollout from a checkpoint, then locks a
small set of rollout states where the actor's preferred legal move differs
from the exact one-step critic-Q preferred move. The diagnostic continuations
and the optimizer replay are separate commands. Continuation data never enters
the PPO rollout.

## Run

From the repository root:

```bash
.venv/bin/python -m games.rl_2048.scripts.audit_ppo_update capture \
  artifacts/ppo_2048/<run>/final.pt --output artifacts/ppo_2048/update_audit \
  --boards 8 --current-d4-order
.venv/bin/python -m games.rl_2048.scripts.audit_ppo_update branch \
  artifacts/ppo_2048/update_audit --trials 64
.venv/bin/python -m games.rl_2048.scripts.audit_ppo_update update \
  artifacts/ppo_2048/update_audit
```

`--current-d4-order` is required for D4 checkpoints. Current training writes an
older `resnet_config` field name, and old D4 experiments used a different action
orientation order. The flag means the checkpoint came from the current
`SQUARE_SYMMETRY_SPECS` implementation; this script does not infer that from the
serialized config. Non-D4 current Conv checkpoints need no flag. Use `--seed`,
`--candidate-limit`, and `--threads` to control capture cost. Use fewer
`--trials` for a quick branch smoke. An interrupted branch phase can resume with
the same trial count and `branch ... --resume`; completed board results are
retained. Capture refuses an existing output directory. Update refuses to
overwrite an existing result.

The default capture inspects up to 512 uniformly spaced eligible states from
the captured rollout, computes Q only for those candidates, and fixes up to
eight disagreement boards before any continuation is run. This candidate cap
keeps exact spawn enumeration bounded. Selection is a diagnostic sample, not a
population estimate. It records the flattened game-major rollout index, sampled
action, old behavior log probability, raw and normalized rollout advantage,
and the true number of transitions to the captured termination, truncation, or
rollout boundary. It does not reconstruct a historical position beyond the
captured rollout.

The exact one-step target is
`Q(s,a) = reward + gamma * E_spawn[V(next_state)]`, reported in raw game points.
The critic transform is inverted, and its reward-multiplied values are divided
by the reward multiplier. Spawn cells are uniform, with probabilities 90% for
tile 2 and 10% for tile 4; terminal successor value is zero. PPO GAE remains in
training-scale reward units. The current D4 model allows captured symmetry-view
states/actions to be replayed directly; other legacy symmetry adapters are not
claimed as supported.

## Outputs and interpretation

- `selection.json` fixes actor-vs-critic action pairs before branch results.
- `capture.pt` stores the original rollout, model and optimizer state, original
  normalized-advantage inputs, RNG states after rollout collection, and source
  checkpoint hash.
- `branches.json` stores raw paired trials, remaining score, discounted return,
  training-scale GAE, and paired approximate intervals.
- `update.json` stores selected legal-action probabilities and sampled-action
  log odds after every replay epoch, each selected sample's actual clipped
  surrogate gradient activity, and exact full-rollout KL from the frozen
  pre-update behavior policy.
- `update_state.pt` stores the replayed model and optimizer state for comparison
  or further inspection; it is separate from the source checkpoint.
- `report.md` summarizes the paired continuation results and per-epoch update.

PPO replay uses the complete captured rollout, original normalized GAE, restored
optimizer state, production PPO objective, minibatch shuffle RNG, and the
configured epoch/approximate-KL stopping rule. Exact KL is diagnostic only and
does not change the production stop rule. Action-probability changes are
measured outcomes; clipping activity is per-sample. This implementation does
not calculate a per-board local policy-gradient dot product with the epoch
parameter displacement, so it does not attribute the full update direction to
an individual board. The checkpoint has no historical RNG snapshot, so capture
uses a recorded fresh seed. No cross-iteration forgetting is measured.

Branch trials force only the preselected first move, then sample the frozen
behavior policy including the configured legal-action exploration mixture.
The paired actor-minus-critic estimates use common random streams across the
two forced actions. Their normal intervals are approximate, especially for
heavy-tailed game scores; a noisy point ranking is not evidence of correctness.
The PPO update is independent of branch execution and uses no branch data.
Capture, continuation evaluation, and replay run on CPU. Checkpoints do not
store historical device/RNG state, so this audit makes no bitwise parity claim
about the original GPU update.

The current 2048 project training run uses D4 convolutions, `step` symmetry
augmentation, and a uniform legal-action exploration schedule. A tiny
refactored checkpoint is used only for the implementation smoke; the
substantive audit uses the completed 100-iteration checkpoint named in the
selection manifest.

## Balanced-versus-natural fixed-state diagnostic

The separate [balanced-sampling report](../../../artifacts/ppo_2048/ppo_balanced_sampling_iter100/report.md)
uses 24 fixed high-gap states from distinct rollout games. Its standalone
implementation is `games/rl_2048/scripts/balanced_sampling_ppo.py`; the locked
design and raw training/evaluation pools are stored beside the report. The
balanced arm gives each legal first action equal or near-equal sample counts,
then weights each action's clipped PPO term by its frozen behavior probability.
This estimates the same old-policy objective as natural action sampling, with
different finite-sample coverage. Both arms use the same frozen behavior
mixture for continuation actions, and held-out continuations never enter the
updates.

This diagnostic trains only the actor on the fixed first-action states and
excludes critic loss. Its 12 paired optimizer replicates reuse the same finite
training pools; the nested bootstrap interval is conditional on the fixed
state bank and held-out pool. It does not estimate full-game training effects,
state prevalence, or whether the saved ordinary PPO update raises the
critic-preferred action probability. The 24-state report found no detectable
balanced-sampling advantage under its return-weighted primary endpoint.
The report also contains one post-hoc lower-learning-rate sensitivity chosen
only from first-replicate training KL. Both small updates had positive
return-weighted changes on the selected bank, but their difference remained
inconclusive; see its explicit selection rule and conditional intervals before
interpreting that follow-up.
