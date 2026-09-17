#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/workspace/cloud_provisioning.json}"
json_value() {
  python -c 'import json,sys; value=json.load(open(sys.argv[1]))[sys.argv[2]]; assert value is not None, sys.argv[2]; print(value)' "$CONFIG" "$1"
}

REMOTE_ROOT="$(json_value remote_root)"
REMOTE_BUNDLE="$(json_value remote_bundle)"
BUNDLE_SHA256="$(json_value bundle_sha256)"
REMOTE_DATASET_BUNDLE="$(json_value remote_dataset_bundle)"
DATASET_BUNDLE_SHA256="$(json_value dataset_bundle_sha256)"
DATASET_MANIFEST_SHA256="$(json_value dataset_manifest_sha256)"
VENV_ROOT="$(json_value venv_root)"
DEADLINE="$(json_value deadline_unix)"
TORCH_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_runtime"]["torch"])' "$CONFIG")"
NUMPY_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_runtime"]["numpy"])' "$CONFIG")"
SCIPY_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_runtime"]["scipy"])' "$CONFIG")"
PILLOW_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_runtime"]["pillow"])' "$CONFIG")"

printf '%s  %s\n' "$BUNDLE_SHA256" "$REMOTE_BUNDLE" | sha256sum -c -
printf '%s  %s\n' "$DATASET_BUNDLE_SHA256" "$REMOTE_DATASET_BUNDLE" | sha256sum -c -
if [ -e "$REMOTE_ROOT" ]; then
  test -f "$REMOTE_ROOT/portable/manifest.json"
else
  mkdir -p "$REMOTE_ROOT"
  tar -xzf "$REMOTE_BUNDLE" -C "$REMOTE_ROOT"
fi
tar -xzf "$REMOTE_DATASET_BUNDLE" -C "$REMOTE_ROOT"
printf '%s  %s\n' \
  "$DATASET_MANIFEST_SHA256" \
  "$REMOTE_ROOT/PreAttentiveVision/data/bsds500/manifest.json" |
  sha256sum -c -
test "$(find "$REMOTE_ROOT/PreAttentiveVision/data/bsds500/images" -type f | wc -l)" = 500

python -c 'import json,sys; c=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert float(c["deadline_unix"])==float(m["deadline_unix"]); assert m["fixed_exposure"]=={"updates":4000,"episodes":160000,"batch_size":8}' "$CONFIG" "$REMOTE_ROOT/portable/manifest.json"

if [ ! -x "$VENV_ROOT/bin/python" ]; then
  python -m venv "$VENV_ROOT"
  "$VENV_ROOT/bin/pip" install --disable-pip-version-check \
    "torch==$TORCH_VERSION" --extra-index-url https://download.pytorch.org/whl/cu117
  "$VENV_ROOT/bin/pip" install --disable-pip-version-check \
    "numpy==$NUMPY_VERSION" "scipy==$SCIPY_VERSION" "Pillow==$PILLOW_VERSION"
fi

if [ -f "$REMOTE_ROOT/remote_supervisor.pid" ]; then
  PRIOR_PID="$(cat "$REMOTE_ROOT/remote_supervisor.pid")"
  if kill -0 "$PRIOR_PID" 2>/dev/null; then
    printf 'Refusing duplicate supervisor; PID %s is active\n' "$PRIOR_PID" >&2
    exit 1
  fi
fi

test "$(python -c 'import time,sys; print(int(float(sys.argv[1])-time.time()>300))' "$DEADLINE")" = 1
cd "$REMOTE_ROOT"
nohup "$VENV_ROOT/bin/python" -B WorkingMemory/ProspectiveQuery/sweep.py \
  portable/manifest.json > remote_supervisor.log 2>&1 < /dev/null &
SUPERVISOR_PID="$!"
printf '%s\n' "$SUPERVISOR_PID" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$SUPERVISOR_PID"
