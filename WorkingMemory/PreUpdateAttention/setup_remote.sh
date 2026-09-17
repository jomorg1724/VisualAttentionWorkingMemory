#!/usr/bin/env bash
set -euo pipefail
cd /workspace
printf '%s\n' '1e15d822218cfcb9a3e4fb39fb6a3f0691436bb39e3126b9f4a344680df29947  attention_bundle.tar.gz' | sha256sum -c -
mkdir -p /workspace/vawm_attention
tar -xzf attention_bundle.tar.gz -C /workspace/vawm_attention
python -m venv /workspace/attention-venv
/workspace/attention-venv/bin/pip install --disable-pip-version-check 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
/workspace/attention-venv/bin/pip install --disable-pip-version-check 'numpy==1.23.1' 'scipy==1.8.1' 'Pillow==9.1.1'
cd /workspace/vawm_attention
test ! -e remote_results/budget.json
nohup /workspace/attention-venv/bin/python -B WorkingMemory/PreUpdateAttention/remote_sweep.py > remote_supervisor.log 2>&1 < /dev/null &
printf 'REMOTE_SUPERVISOR_PID=%s\n' "$!"
