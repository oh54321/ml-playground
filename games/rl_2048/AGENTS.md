# Agent instructions for 2048 training

When asked to launch PPO 2048 training on RunPod, first read
[`documents/RUNPOD_TRAINING.md`](documents/RUNPOD_TRAINING.md) and the current
RunPod plugin skill. Verify the Pod has the intended Network Volume mounted at
`/workspace`, record the requested code revision, and use
[`scripts/runpod_train_and_close.sh`](scripts/runpod_train_and_close.sh) so the
complete run artifacts are checksummed before Pod cleanup. Never delete the
Network Volume as part of Pod cleanup. Use a time-based Pod stop limit as a
fallback, and recheck current GPU, Pod, volume, and CLI state before each launch.
