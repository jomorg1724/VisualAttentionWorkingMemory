#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/workspace/attention_context_comparator_resume_cloud.json}"
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

python -c 'import json,sys; c=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert c["authorized_arms"]==m["authorized_arms"]==["attention_context_comparator_scratch"]; assert m["initialization_kind"]=="exact_durable_resume"; assert m["resume_step"]==1024; assert m["fixed_exposure"]=={"updates":4000,"episodes":160000,"batch_size":8,"microbatches_per_update":5}; assert float(c["deadline_unix"])==float(m["deadline_unix"])' "$CONFIG" "$ROOT/portable/manifest.json"

if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install --disable-pip-version-check \
    torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  "$VENV/bin/pip" install --disable-pip-version-check \
    numpy==1.23.1 scipy==1.8.1 Pillow==9.1.1
fi

printf '%s  %s\n' "$DATASET_SHA" "$DATASET" | sha256sum -c -
tar -xzf "$DATASET" -C "$ROOT"
test "$(python -c 'import time,sys; print(int(float(sys.argv[1])-time.time()>600))' "$DEADLINE")" = 1
cd "$ROOT"
nohup "$VENV/bin/python" -B WorkingMemory/AttentionContextComparator/resume_sweep.py \
  portable/manifest.json > remote_supervisor.log 2>&1 < /dev/null &
PID="$!"
printf '%s\n' "$PID" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$PID"
