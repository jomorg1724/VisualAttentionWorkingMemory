#!/usr/bin/env bash
set -euo pipefail
python -m venv /workspace/unbiased-venv
timeout 600 /workspace/unbiased-venv/bin/pip install --disable-pip-version-check 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
timeout 300 /workspace/unbiased-venv/bin/pip install --disable-pip-version-check 'numpy==1.23.1' 'scipy==1.8.1' 'Pillow==9.1.1'
touch /workspace/runtime_ready
