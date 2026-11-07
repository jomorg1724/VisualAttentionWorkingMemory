#!/bin/bash
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu-n72
#SBATCH --open-mode=append
#SBATCH --output=/ix/jpherman/hermanj/rvit_zk/logs/%x_%A_%a.out
# RVIT_EXISTENCE_PROOF_PLAN.md rev 2 Part C (2026-11-04): position-gated LSTM edits on the 20 rscaled10 start models
# (= the warm-up's transformer and actor), at the SENS-C low-value location. Array task i: 5 models. Task 0 also runs
# the CPU tests of zk/ep_train.py first. Resumable (ep_edit skips finished cells).
set -uo pipefail
module load python/pytorch_251_311_cu124 2>/dev/null || true
export PYTHONUNBUFFERED=1
cd /ix/jpherman/hermanj/rvit_zk/tree_zk
R=/ix/jpherman/hermanj/rvit_zk/runs; O=$R/existence/partC2; mkdir -p $O   # v2 direction (position coordinates); v1 outputs in partC/
if [[ ${SLURM_ARRAY_TASK_ID:-0} == 0 ]]; then
  CUDA_VISIBLE_DEVICES= python -m pytest -q zk/tests/test_ep_train.py 2>&1 | tail -5
fi
M=(); for s in $(seq 42 51); do for L in 0 3; do M+=("$L $s"); done; done
for ((j = SLURM_ARRAY_TASK_ID * 5; j < SLURM_ARRAY_TASK_ID * 5 + 5 && j < ${#M[@]}; j++)); do
  read -r L s <<< "${M[$j]}"
  python -u -m zk.ep_edit --ckpt $R/whichloss/rscaled10/loc${L}_s$s/ckpt_final.pt --hl $L --out $O/edit_loc${L}_s$s.json || true
done
echo "[ep_edit] task $SLURM_ARRAY_TASK_ID done $(date)"
