#!/usr/bin/env bash
set -euo pipefail
cd /workspace
test "$#" -eq 1
printf '%s  exposure_bundle.tar.gz\n' "$1" | sha256sum -c -
mkdir -p /workspace/vawm_exposure
tar -xzf exposure_bundle.tar.gz -C /workspace/vawm_exposure
python -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
python -m venv /workspace/exposure-venv
timeout 600 /workspace/exposure-venv/bin/pip install --disable-pip-version-check 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
timeout 300 /workspace/exposure-venv/bin/pip install --disable-pip-version-check 'numpy==1.23.1' 'scipy==1.8.1' 'Pillow==9.1.1'
cd /workspace/vawm_exposure
test ! -e remote_results/budget.json
nohup /workspace/exposure-venv/bin/python -B WorkingMemory/TrainingExposure/Cloud/remote_sweep.py > remote_supervisor.log 2>&1 < /dev/null &
printf '%s\n' "$!" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$!"
