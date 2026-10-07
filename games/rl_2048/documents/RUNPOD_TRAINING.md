# Run PPO training on RunPod

This workflow lets a future agent select an available GPU, launch the current
requested PPO run, save the complete artifact directory to persistent storage,
and close the Pod without keeping a Codex session or the user's computer online.

## Storage rule

`/workspace` is only a mount path. Before launching, verify in the live RunPod
Pod details that `mounts.network` contains the Network Volume ID. A Pod's own
volume disk can also appear at `/workspace`, but it is deleted with the Pod.
Only the separate Network Volume persists when its Pod is terminated.

The trainer creates `artifacts/ppo_2048/<run-name>_<timestamp>` relative to the
repository root. Therefore the repository must be at
`/workspace/ml-playground`, and the training command must run from that repo.
This stores checkpoints, raw and formatted summaries, diagnostic plots, and
TensorBoard event files together on the Network Volume. The launcher also adds
`runpod-training-launch.log`, `runpod-training-status.txt`, and a `SHA256SUMS`
manifest to the run directory.

Record the selected resources before each launch:

- Network Volume ID: `<volume-id>`
- Datacenter: `<datacenter-id>`
- Pod ID: `<pod-id>`

Do not assume this volume or Pod is still available or idle. Read the live Pod
and Network Volume state first. Do not run two trainers that write to the same
repository/artifact tree at once. If the volume is unavailable or attached to
another active training run, create a separate Network Volume in the chosen
GPU's datacenter and attach it when creating the Pod.

## Future-agent launch procedure

1. Read the RunPod plugin's `runpod` router skill and follow its current
   RunPod MCP or `runpodctl` instructions. Use live GPU catalog and capacity
   data to choose a GPU; do not assume that a GPU or region from an old run is
   still available.
2. Read the current Pod list and Network Volume details. Choose an unused
   Network Volume in the selected GPU datacenter. Attach it at Pod creation,
   mounted at `/workspace`, and confirm the resulting Pod has the expected
   `mounts.network[].volumeId`.
3. Put the requested repository revision at `/workspace/ml-playground`. Use
   Git if the requested commit is available from a remote; otherwise transfer
   the requested working tree over SSH. Record the exact commit and whether
   the remote tree is dirty. Do not silently include unrelated local changes.
4. Install the project's runtime dependencies in the Pod. TensorBoard is an
   optional dependency, but this workflow requires it so every run has event
   files. Install it with `python -m pip install tensorboard` if it is missing.
5. Set `RUNPOD_POD_ID` to the new Pod's ID if RunPod has not already injected
   it. Launch from the Pod:

   ```sh
   cd /workspace/ml-playground
   games/rl_2048/scripts/runpod_train_and_close.sh \
     --outer-iterations 4000
   ```

   Pass the requested PPO overrides after the script name, for example
   `--num-parallel-games 128` or warm-start/exploration options. The wrapper
   always enables TensorBoard. The trainer's existing command-line options are
   documented by `python -m games.rl_2048.train_2048 --help`.
6. At Pod creation, set a generous `--stop-after` limit as a cost guard. It is
   a time limit, not a completion detector; choose enough time for the run,
   final evaluation, and artifact manifest. Prefer the current RunPod CLI or
   MCP capability that supports the field.
7. Observe progress through Pod logs or TensorBoard while the Pod is running.
   Do not rely on the observing session to perform final cleanup. The wrapper
   writes status and checksums after the training process exits, verifies the
   manifest, then asks RunPod to delete the Pod. If deletion fails, it tries
   the legacy CLI spelling and then tries stopping the Pod. The `--stop-after`
   limit is the fallback if all cleanup calls fail.

The wrapper only auto-deletes after it has identified the new run directory
and successfully written and verified `SHA256SUMS`. If startup fails before a
run directory is created, it preserves the launch log under
`artifacts/ppo_2048/.runpod-launch-failures/` and leaves the Pod for recovery;
the configured time limit remains the cost guard. Set
`PPO_RUNPOD_AUTO_DELETE=0` for a deliberate short smoke run where the Pod must
remain available for inspection.

The RunPod CLI is documented to be available in Pods with a Pod-scoped API key.
Self-deletion behavior can vary by CLI version, so verify this cleanup path on
a short run before depending on it for an unattended long run. Do not put a
personal RunPod API key in the repository or training command.

## Retrieve a run later

The Network Volume remains after Pod deletion. RunPod exposes its files through
an S3-compatible API, so download can happen later from the user's Mac without
starting another GPU Pod. Configure `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` locally using a RunPod S3 API key; never commit the
secret. The RunPod S3 key is separate from `RUNPOD_API_KEY`.

From the local repository root, run:

```sh
RUNPOD_NETWORK_VOLUME_ID='<volume-id>' \
RUNPOD_VOLUME_DATACENTER_ID='<datacenter-id>' \
games/rl_2048/scripts/fetch_runpod_artifacts.sh \
  <run-directory-name>
```

The helper syncs the entire run directory into local
`artifacts/ppo_2048/<run-directory-name>` and checks every file against
`SHA256SUMS`. It does not delete remote files. To keep the run in an alternate
local directory, pass that path as the second argument.

Network Volume storage remains billable while retained. Keep enough RunPod
credit available and make an off-platform backup for any results that cannot
be recreated. Do not delete the Network Volume until all needed runs have been
downloaded and verified.

## References

- [RunPod storage guide](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod-usage/reference/storage.md)
- [RunPod CLI Pod reference](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods)
- [PPO TensorBoard instructions](TENSORBOARD.md)
