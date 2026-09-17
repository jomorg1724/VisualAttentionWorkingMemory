#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/vawm_spatial_priority_control_recovery_3200
cd "$ROOT"
test ! -e spatial_priority_recovery_results
nohup /workspace/spatial-priority-control-recovery-venv/bin/python -B WorkingMemory/SpatialPriorityReadout/recovery_sweep.py portable/recovery_manifest.json > remote_recovery_supervisor.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > remote_supervisor.pid
printf 'RECOVERY_SUPERVISOR_PID=%s
' "$pid"
