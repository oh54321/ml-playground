#!/usr/bin/env bash
# Download one completed PPO run from a RunPod Network Volume and verify it.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  RUNPOD_NETWORK_VOLUME_ID='<volume-id>' \
  RUNPOD_VOLUME_DATACENTER_ID='<datacenter-id>' \
  games/rl_2048/scripts/fetch_runpod_artifacts.sh <run-directory-name> [local-destination]

Example:
  RUNPOD_NETWORK_VOLUME_ID='<volume-id>' \
  RUNPOD_VOLUME_DATACENTER_ID='<datacenter-id>' \
  games/rl_2048/scripts/fetch_runpod_artifacts.sh one_hot_minLR_3e-5_20261007_120000_123456

AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be configured locally for
RunPod's S3-compatible API. The secret is shown only once when created; do not
put it in the repository.
EOF
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
[[ -n "${RUNPOD_NETWORK_VOLUME_ID:-}" ]] || usage
[[ -n "${RUNPOD_VOLUME_DATACENTER_ID:-}" ]] || usage
command -v aws >/dev/null 2>&1 || { echo "ERROR: install and configure the AWS CLI first" >&2; exit 2; }

run_name="$1"
case "$run_name" in
  */*|.*) echo "ERROR: pass only the run directory name, not a path" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
local_destination="${2:-$repo_root/artifacts/ppo_2048/$run_name}"
remote_path="ml-playground/artifacts/ppo_2048/$run_name"
region="$RUNPOD_VOLUME_DATACENTER_ID"
endpoint="https://s3api-$(printf '%s' "$region" | tr '[:upper:]' '[:lower:]').runpod.io/"

mkdir -p "$local_destination"
aws s3 sync \
  --region "$region" \
  --endpoint-url "$endpoint" \
  --only-show-errors \
  "s3://${RUNPOD_NETWORK_VOLUME_ID}/${remote_path}/" \
  "$local_destination/"

if [[ ! -f "$local_destination/SHA256SUMS" ]]; then
  echo "ERROR: downloaded run has no SHA256SUMS manifest; refusing to report it verified" >&2
  exit 3
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$local_destination" && sha256sum -c SHA256SUMS)
elif command -v shasum >/dev/null 2>&1; then
  (cd "$local_destination" && shasum -a 256 -c SHA256SUMS)
else
  echo "ERROR: sha256sum or shasum is required to verify the download" >&2
  exit 2
fi

echo "Verified artifacts at $local_destination"
