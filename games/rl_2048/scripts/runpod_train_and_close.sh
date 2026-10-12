#!/usr/bin/env bash
# Run PPO training from a RunPod Network Volume and close the Pod after saving
# a checksum manifest for the complete run directory.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/ppo_2048"
AUTO_DELETE="${PPO_RUNPOD_AUTO_DELETE:-1}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

case "$REPO_ROOT" in
  /workspace/*) ;;
  *) fail "Expected this repository beneath /workspace on the RunPod Network Volume; found $REPO_ROOT" ;;
esac

command -v git >/dev/null 2>&1 || fail "git is not available"
git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || fail "No Git repository found at $REPO_ROOT"
[[ -n "${RUNPOD_POD_ID:-}" ]] || fail "RUNPOD_POD_ID is unset; pass the live Pod ID before starting training"
command -v python >/dev/null 2>&1 || fail "python is not available"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required to verify run artifacts"

case "$AUTO_DELETE" in
  0|1) ;;
  *) fail "PPO_RUNPOD_AUTO_DELETE must be 0 or 1" ;;
esac

if [[ "$AUTO_DELETE" == "1" ]]; then
  command -v runpodctl >/dev/null 2>&1 || fail "runpodctl is unavailable; refusing an unattended run without automatic Pod cleanup"
fi

if ! python -c 'import tensorboard' >/dev/null 2>&1; then
  fail "TensorBoard is required so the run includes event files. Install it with: python -m pip install tensorboard"
fi

mkdir -p "$ARTIFACT_ROOT/.runpod-launch-failures"
before_dirs="$(mktemp)"
after_dirs="$(mktemp)"
launch_log="$(mktemp)"
temporary_launch_log="$launch_log"
cleanup_temps() {
  rm -f "$before_dirs" "$after_dirs" "$temporary_launch_log"
}
trap cleanup_temps EXIT

find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.runpod-launch-failures' -print \
  | LC_ALL=C sort > "$before_dirs"

cd "$REPO_ROOT"
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
printf 'Output root: %s\n' "$ARTIFACT_ROOT"
printf 'Pod: %s\n' "$RUNPOD_POD_ID"
printf 'Starting PPO training with TensorBoard enabled.\n'

python -u -m games.rl_2048.train_2048 --tensorboard "$@" 2>&1 | tee "$launch_log"
training_exit_code=${PIPESTATUS[0]}

find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.runpod-launch-failures' -print \
  | LC_ALL=C sort > "$after_dirs"
mapfile -t new_run_dirs < <(comm -13 "$before_dirs" "$after_dirs")

run_dir=""
identified_run_dir=1
if [[ ${#new_run_dirs[@]} -eq 1 ]]; then
  run_dir="${new_run_dirs[0]}"
else
  identified_run_dir=0
  failure_dir="$ARTIFACT_ROOT/.runpod-launch-failures/$(date -u +%Y%m%dT%H%M%SZ)_${RUNPOD_POD_ID}"
  mkdir -p "$failure_dir"
  run_dir="$failure_dir"
  printf 'Could not identify exactly one new training run directory; found %s.\n' \
    "${#new_run_dirs[@]}" > "$run_dir/runpod-training-status.txt"
fi

if [[ "$launch_log" != "$run_dir/runpod-training-launch.log" ]]; then
  mv "$launch_log" "$run_dir/runpod-training-launch.log"
  launch_log="$run_dir/runpod-training-launch.log"
fi

completion_state="incomplete"
if [[ "$training_exit_code" -eq 0 && -f "$run_dir/final.pt" ]]; then
  completion_state="completed"
fi

{
  printf 'state=%s\n' "$completion_state"
  printf 'training_exit_code=%s\n' "$training_exit_code"
  printf 'pod_id=%s\n' "$RUNPOD_POD_ID"
  printf 'repo_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'repo_dirty=%s\n' "$(if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then printf yes; else printf no; fi)"
  printf 'command=python -u -m games.rl_2048.train_2048 --tensorboard'
  printf ' %q' "$@"
  printf '\n'
} > "$run_dir/runpod-training-status.txt"

(
  cd "$run_dir"
  find . -type f ! -name 'SHA256SUMS' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
manifest_exit_code=$?

if [[ "$manifest_exit_code" -ne 0 ]]; then
  printf 'Artifact manifest verification failed; leaving the Pod running for recovery.\n' >&2
  exit 3
fi

sync
printf 'Run artifacts saved at %s\n' "$run_dir"
printf 'Training state: %s (exit code %s)\n' "$completion_state" "$training_exit_code"

if [[ "$identified_run_dir" -ne 1 ]]; then
  printf 'Could not identify exactly one run directory; leaving the Pod running for recovery.\n' >&2
  exit 5
fi

if [[ "$AUTO_DELETE" == "0" ]]; then
  printf 'Automatic Pod deletion disabled by PPO_RUNPOD_AUTO_DELETE=0.\n'
  exit "$training_exit_code"
fi

printf 'Deleting Pod %s after artifact verification.\n' "$RUNPOD_POD_ID"
# The modern CLI spelling is attempted first. Current RunPod Pod-scoped CLI
# credentials have also required the legacy spelling for self-removal, so keep
# that fallback until self-deletion is confirmed on the account's CLI version.
if runpodctl pod delete "$RUNPOD_POD_ID"; then
  exit "$training_exit_code"
fi
if runpodctl remove pod "$RUNPOD_POD_ID"; then
  exit "$training_exit_code"
fi
if runpodctl stop pod "$RUNPOD_POD_ID"; then
  printf 'Pod deletion failed; the Pod was stopped instead.\n' >&2
  exit "$training_exit_code"
fi

printf 'Automatic Pod cleanup failed. The verified artifacts remain on the Network Volume; stop or delete Pod %s in RunPod.\n' \
  "$RUNPOD_POD_ID" >&2
exit 4
