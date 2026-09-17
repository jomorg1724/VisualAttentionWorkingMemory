#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/vawm_spatial_priority_scratch_resume_1600
cd "$ROOT"
/workspace/spatial-priority-scratch-resume-venv/bin/python -m py_compile \
  WorkingMemory/SpatialPriorityReadout/resume_sweep.py continue_after_active.py
echo "49cee4eb1440fcc7239ba938bce8a83f9d91c7367bc45fdc5954d20f9982392c  WorkingMemory/SpatialPriorityReadout/resume_sweep.py" | sha256sum -c -
grep -q '"extended_target_updates": 12000' portable/resume_manifest.json
kill -0 1716
kill -0 3052
nohup /workspace/spatial-priority-scratch-resume-venv/bin/python -B \
  continue_after_active.py > extension_handoff.log 2>&1 < /dev/null &
handoff=$!
echo "$handoff" > extension_handoff.pid
kill -TERM 1716
sleep 3
if kill -0 1716 2>/dev/null; then
  echo "Old supervisor 1716 did not exit" >&2
  exit 2
fi
kill -0 3052
kill -0 "$handoff"
printf 'HANDOFF_PID=%s\nWORKER_PID=3052\n' "$handoff"
tail -n 1 spatial_priority_results/spatial_priority_readout_scratch/training/metrics.csv
