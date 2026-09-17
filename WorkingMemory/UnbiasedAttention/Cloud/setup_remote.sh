#!/usr/bin/env bash
set -euo pipefail
cd /workspace
printf '%s  unbiased_bundle.tar.gz\n' "$1" | sha256sum -c -
mkdir -p /workspace/vawm_unbiased
tar -xzf unbiased_bundle.tar.gz -C /workspace/vawm_unbiased
python -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
python -m venv /workspace/unbiased-venv
timeout 600 /workspace/unbiased-venv/bin/pip install --disable-pip-version-check 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
timeout 300 /workspace/unbiased-venv/bin/pip install --disable-pip-version-check 'numpy==1.23.1' 'scipy==1.8.1' 'Pillow==9.1.1'
cd /workspace/vawm_unbiased
nohup /workspace/unbiased-venv/bin/python -B WorkingMemory/UnbiasedAttention/sweep.py portable/manifest.json > remote_supervisor.log 2>&1 < /dev/null &
printf '%s\n' "$!" > remote_supervisor.pid
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$!"
