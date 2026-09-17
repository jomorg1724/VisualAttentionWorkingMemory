#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/workspace/dual_attention_cloud.json}"
value() {
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$CONFIG" "$1"
}

ROOT="$(value remote_root)"
BUNDLE="$(value remote_bundle)"
BUNDLE_SHA="$(value bundle_sha256)"
DATASET="$(value remote_dataset_bundle)"
DATASET_SHA="$(value dataset_bundle_sha256)"
DATASET_MANIFEST_SHA="$(value dataset_manifest_sha256)"
VENV="$(value venv_root)"
DEADLINE="$(value deadline_unix)"

printf '%s  %s\n' "$BUNDLE_SHA" "$BUNDLE" | sha256sum -c -
test ! -e "$ROOT"
mkdir -p "$ROOT"
tar -xzf "$BUNDLE" -C "$ROOT"

python -c 'import json,sys; c=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); assert c["authorized_arms"]==m["authorized_arms"]==["dual_attention_spatial_priority_scratch"]; assert m["initialization_kind"]=="full_model_from_scratch"; assert m["loaded_parent"] is False and m["loaded_optimizer_state"] is False; assert "parent_sha256" not in m; assert m["fixed_exposure"]=={"updates":4000,"episodes":160000,"batch_size":8,"microbatches_per_update":5}; assert float(c["deadline_unix"])==float(m["deadline_unix"])' "$CONFIG" "$ROOT/portable/manifest.json"

if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install --disable-pip-version-check \
    torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  "$VENV/bin/pip" install --disable-pip-version-check \
    numpy==1.23.1 scipy==1.8.1 Pillow==9.1.1
fi

if [ -f "$DATASET" ]; then
  printf '%s  %s\n' "$DATASET_SHA" "$DATASET" | sha256sum -c -
  tar -xzf "$DATASET" -C "$ROOT"
else
  cd "$ROOT"
  "$VENV/bin/python" -B -m PreAttentiveVision.natural_stimuli --prepare
fi
GENERATED_MANIFEST="$ROOT/PreAttentiveVision/data/bsds500/manifest.json" \
  "$VENV/bin/python" - <<'PY'
import json,os,pathlib
generated=json.load(open(os.environ["GENERATED_MANIFEST"]))
assert generated["dataset"]=="BSDS500"
assert generated["archive_bytes"]==70763455
assert generated["archive_sha256"]=="97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e"
assert generated["split_counts"]=={"train":200,"val":100,"test":200}
assert generated["image_count"]==500
root=pathlib.Path(os.environ["GENERATED_MANIFEST"]).parent
assert len(list((root/"images").rglob("*.jpg")))==500
PY

test "$(python -c 'import time,sys; print(int(float(sys.argv[1])-time.time()>600))' "$DEADLINE")" = 1
cd "$ROOT"
nohup "$VENV/bin/python" -B WorkingMemory/SpatialPriorityReadout/DualAttention/sweep.py \
  portable/manifest.json > remote_supervisor.log 2>&1 < /dev/null &
PID="$!"
printf '%s\n' "$PID" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$PID"
