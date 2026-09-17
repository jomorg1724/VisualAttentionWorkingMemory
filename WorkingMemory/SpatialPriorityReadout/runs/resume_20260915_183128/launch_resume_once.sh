#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/vawm_spatial_priority_scratch_resume_1600
cd "$ROOT"
test ! -e spatial_priority_results
nohup /workspace/spatial-priority-scratch-resume-venv/bin/python -B WorkingMemory/SpatialPriorityReadout/resume_sweep.py portable/resume_manifest.json > remote_supervisor.log 2>&1 < /dev/null &
pid=$!
printf '%s
' "$pid" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s
' "$pid"
