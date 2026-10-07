# GPU utilization profiling

Use `games/rl_2048/scripts/profile_gpu_usage.py` to collect one-second GPU
telemetry and align it with timestamped training phases. This keeps future
profiling runs comparable and avoids estimating phase time from iteration
averages.

## Capture one hour on a Runpod Pod

Run from the repository root on the Pod. Save the CSV inside the active run's
directory so it persists on the mounted network volume:

```sh
python games/rl_2048/scripts/profile_gpu_usage.py collect \
  --seconds 3600 \
  --output artifacts/ppo_2048/RUN_NAME/gpu_telemetry_1h_YYYYMMDDTHHMMZ.csv
```

The collector samples `nvidia-smi` once per second and records UTC timestamp,
GPU utilization, memory-controller utilization, VRAM use, and power draw. It
flushes each sample immediately. `Ctrl+C` stops collection and preserves
samples already written. Keep training running during the capture. Save the
training output in a file with timestamps enabled; the Runpod launcher uses
`artifacts/runpod-training-launch.log` for live output. TensorBoard does not
provide the phase timings needed for this analysis.

If the script is only on the local computer, copy it to the Pod first or run it
from a synced repository checkout. Keep the CSV and the training log together
with the run artifacts.

## Analyze and compare

After capture, run from the repository root:

```sh
python games/rl_2048/scripts/profile_gpu_usage.py analyze \
  --csv artifacts/ppo_2048/RUN_NAME/gpu_telemetry_1h_YYYYMMDDTHHMMZ.csv \
  --log artifacts/runpod-training-launch.log \
  --output artifacts/ppo_2048/RUN_NAME/gpu_telemetry_report.json
```

The analyzer accepts both the current collector CSV and the earlier
`timestamp,gpu_util_pct,...` CSV format. It uses timestamped lines emitted by
the trainer to divide intervals into:

- **Rollout:** rollout-start marker through the rollout-collected marker.
- **PPO update:** rollout-collected marker through outer-iteration-complete.
- **Evaluation:** evaluation-start marker through evaluation-complete.
- **Other:** time after iteration completion, outside evaluation, until the
  next rollout starts (checkpointing and reporting can fall here).

It reports each phase's elapsed share, mean and median GPU utilization, share
of samples at or above 50% GPU utilization, memory-controller utilization,
VRAM, and power. It also reports rollout starts per hour as a throughput check.
Overall metrics summarize the entire capture. Inspect
`phase_markers_in_capture` and sample counts before drawing conclusions; incomplete
log coverage, timestamps disabled in the run, or a capture beginning midway
through a phase can leave time unclassified.

## How to interpret it

Elapsed-time share measures wall-clock time. GPU utilization and power samples
show whether the GPU was busy during that time. A long rollout with low GPU
utilization points toward CPU-side environment/game work, synchronization, or
data movement as a likely optimization target; it does not mean that GPU time
is separately billed by utilization. Runpod Pod charges are based on provisioned
time. Higher utilization can reduce cost per training iteration when it makes
the run finish sooner.

Compare the **rollout elapsed share** and **rollout GPU utilization** across
runs with the same sampling cadence, GPU, training settings, and phase markers.
Also compare total iteration throughput and evaluation cadence; a lower rollout
share alone can result from a longer PPO update or evaluation phase rather than
faster training. Record the commit, GPU model, run settings, capture start/end,
and Pod hourly rate alongside the JSON report.

## Previous baseline

The October 6, 2026 capture used approximately one sample per second for one
hour. Its source CSV is
`artifacts/ppo_2048/one_hot_lambda_0.95_20261006_054411_637781/gpu_telemetry_1h_20261006T0600Z.csv`
on the shared Runpod network volume. Re-run `analyze` with that CSV and its
timestamped `artifacts/runpod-training-launch.log` to regenerate exact baseline
figures with this script before comparing future captures.
