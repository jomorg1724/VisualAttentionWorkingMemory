#!/bin/bash
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --exclude=gpu-n72
#SBATCH --open-mode=append
#SBATCH --output=/ix/jpherman/hermanj/rvit_zk/logs/%x_%A_%a.out
# RVIT_EXISTENCE_PROOF_PLAN.md (2026-11-04). Array task i runs PACK consecutive TABLE lines ("ARM SEED HL") concurrently
# on one GPU via run_ep.sh. Idempotent, so a requeue resumes.
set -uo pipefail
: "${TABLE:?}"; : "${PACK:=3}"
BASE=/ix/jpherman/hermanj/rvit_zk
module load python/pytorch_251_311_cu124 2>/dev/null || true
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); a=torch.randn(2048,2048,device='cuda'); (a@a).sum(); print('GPU OK', torch.cuda.get_device_name(0))" \
  || { echo "FATAL: GPU health check failed on $(hostname)"; exit 1; }
mapfile -t LINES < <(tr -d '\r' < "$TABLE" | grep -v '^\s*#' | grep -v '^\s*$')
START=$((SLURM_ARRAY_TASK_ID * PACK)); pids=()
for ((j = START; j < START + PACK && j < ${#LINES[@]}; j++)); do
  read -r ARM SEED HL <<< "${LINES[$j]}"; RUN=$BASE/runs/existence/$ARM/loc${HL}_s${SEED}; mkdir -p "$RUN"
  echo "[job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}] line $j: ${LINES[$j]}"
  bash "$BASE/slurm/run_ep.sh" "$ARM" "$SEED" "$HL" >> "$RUN/cell.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "[job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}] finished rc=$rc $(date)"; exit $rc
