#!/usr/bin/env bash
# Remote setup for the accumulator program: verify bundles, build the pinned venv, unpack the dataset, start two lanes.
set -euo pipefail
CONFIG="${1:-/workspace/accum_cloud.json}"
value() { python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$CONFIG" "$1"; }
ROOT="$(value remote_root)"; BUNDLE="$(value remote_bundle)"; BUNDLE_SHA="$(value bundle_sha256)"
DATASET="$(value remote_dataset_bundle)"; DATASET_SHA="$(value dataset_bundle_sha256)"; VENV="$(value venv_root)"
DEADLINE="$(value deadline_unix)"; SEEDS="$(value seeds)"; WORKERS="$(value workers)"; RESULTS="$(value remote_results)"

printf '%s  %s\n' "$BUNDLE_SHA" "$BUNDLE" | sha256sum -c -
test ! -e "$ROOT"; mkdir -p "$ROOT"; tar -xzf "$BUNDLE" -C "$ROOT"
if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install --disable-pip-version-check torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  "$VENV/bin/pip" install --disable-pip-version-check numpy==1.23.1 scipy==1.8.1 Pillow==9.1.1
fi
printf '%s  %s\n' "$DATASET_SHA" "$DATASET" | sha256sum -c -
tar -xzf "$DATASET" -C "$ROOT"
test "$(ls "$ROOT/PreAttentiveVision/data/bsds500/images" | wc -l)" -ge 1
cd "$ROOT"
"$VENV/bin/python" -B -c 'import torch;assert torch.cuda.is_available();print(torch.__version__,torch.cuda.get_device_name(0))'
"$VENV/bin/python" -B -m WorkingMemory.PlainBaseline.baseline --task orientation_ring --out /tmp/accum_profile --profile 3 --workers 2 --stack 3 --center --encoder accum --accumulator convgru 2>&1 | tail -1
mkdir -p "$RESULTS"
LANE1="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["lanes"]["lane1"])' "$CONFIG")"
LANE2="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["lanes"]["lane2"])' "$CONFIG")"
nohup "$VENV/bin/python" -B -m WorkingMemory.PlainBaseline.program --out "$RESULTS/lane1" --arms "$LANE1" --seeds "$SEEDS" --deadline "$DEADLINE" --workers "$WORKERS" --chunk 32 > "$RESULTS/lane1.log" 2>&1 < /dev/null &
P1=$!
nohup "$VENV/bin/python" -B -m WorkingMemory.PlainBaseline.program --out "$RESULTS/lane2" --arms "$LANE2" --seeds "$SEEDS" --deadline "$DEADLINE" --workers "$WORKERS" --chunk 32 > "$RESULTS/lane2.log" 2>&1 < /dev/null &
P2=$!
printf '%s %s\n' "$P1" "$P2" > "$RESULTS/program.pids"
printf 'REMOTE_PIDS=%s,%s\n' "$P1" "$P2"
