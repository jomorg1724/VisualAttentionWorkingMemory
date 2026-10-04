#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python -u ./deploy.py deploy --use-existing-guard-credential --storage-reserve-usd 1 --retention-hours 24 "$@"
