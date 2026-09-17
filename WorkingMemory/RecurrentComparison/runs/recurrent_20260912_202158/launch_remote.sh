#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '50ca98694c547b42860f012aaee864cd620de4ddb553a05ea1f4a9218861491a  /workspace/ei_portable_bundle.tar.gz' | sha256sum -c -
mkdir -p /workspace/vawm_recurrent
tar -xzf /workspace/ei_portable_bundle.tar.gz -C /workspace/vawm_recurrent
cd /workspace/vawm_recurrent
test ! -e remote_results/budget.json
nohup /workspace/wm-venv/bin/python -B WorkingMemory/RecurrentComparison/remote_sweep.py > remote_supervisor.log 2>&1 < /dev/null &
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$!"
