#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/vawm_spatial_priority_control_recovery_3200
BUNDLE=/workspace/spatial_priority_control_recovery_bundle.tar.gz
DATA=/workspace/bsds500_runtime.tar.gz
VENV=/workspace/spatial-priority-control-recovery-venv
echo "63c64313d29eccb05b9dd8e9de1c2e4c42a690e2419a6c19fffb38dac46d3391  $BUNDLE" | sha256sum -c -
echo "990e2bd405f6bcc4c940299f437685fc08765dcb56923ab0ab639f5f37c1883e  $DATA" | sha256sum -c -
test ! -e "$ROOT"
mkdir -p "$ROOT"
tar -xzf "$BUNDLE" -C "$ROOT"
tar -xzf "$DATA" -C "$ROOT"
python -m venv "$VENV"
"$VENV/bin/pip" install --no-cache-dir torch==1.13.1+cu117 numpy==1.23.1 scipy==1.8.1 Pillow==9.1.1 --extra-index-url https://download.pytorch.org/whl/cu117
cd "$ROOT"
nohup "$VENV/bin/python" -B WorkingMemory/SpatialPriorityReadout/recovery_sweep.py portable/recovery_manifest.json > remote_recovery_supervisor.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > remote_supervisor.pid
printf 'RECOVERY_SUPERVISOR_PID=%s
' "$pid"
