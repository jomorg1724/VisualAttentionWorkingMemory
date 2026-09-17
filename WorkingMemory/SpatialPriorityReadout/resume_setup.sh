#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/workspace/spatial_priority_resume_cloud.json}"
value() {
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$CONFIG" "$1"
}
ROOT="$(value remote_root)"
BUNDLE="$(value remote_bundle)"
BUNDLE_SHA="$(value bundle_sha256)"
DATASET="$(value remote_dataset_bundle)"
DATASET_SHA="$(value dataset_bundle_sha256)"
VENV="$(value venv_root)"
DEADLINE="$(value deadline_unix)"

printf '%s  %s\n' "$BUNDLE_SHA" "$BUNDLE" | sha256sum -c -
test ! -e "$ROOT"
mkdir -p "$ROOT"
tar -xzf "$BUNDLE" -C "$ROOT"
python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["authorized_arms"]==["spatial_priority_readout_scratch"]; assert m["resume_step"]==1600; assert m["fixed_exposure"]=={"updates":4000,"episodes":160000,"batch_size":8,"microbatches_per_update":5}' "$ROOT/portable/resume_manifest.json"

python -m venv "$VENV"
"$VENV/bin/pip" install --disable-pip-version-check \
  torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
"$VENV/bin/pip" install --disable-pip-version-check \
  numpy==1.23.1 scipy==1.8.1 Pillow==9.1.1
printf '%s  %s\n' "$DATASET_SHA" "$DATASET" | sha256sum -c -
tar -xzf "$DATASET" -C "$ROOT"
test "$(python -c 'import time,sys; print(int(float(sys.argv[1])-time.time()>600))' "$DEADLINE")" = 1
cd "$ROOT"
nohup "$VENV/bin/python" -B WorkingMemory/SpatialPriorityReadout/resume_sweep.py \
  portable/resume_manifest.json > remote_supervisor.log 2>&1 < /dev/null &
PID="$!"
printf '%s\n' "$PID" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$PID"
