#!/usr/bin/env bash
set -euo pipefail

cd /workspace/vawm_dual_attention_resume_800
EXPECTED=12377dabe2bf3ce92e022bb5084e51189873af343dbfa32a9f45812a720a5206
test "$(sha256sum WorkingMemory/SpatialPriorityReadout/DualAttention/extend_live.py | cut -d' ' -f1)" = "$EXPECTED"
test -d /proc/465
test -d /proc/3108

kill -STOP 465
sleep 2
test "$(awk '{print $3}' /proc/3108/stat)" != Z
kill -KILL 465
sleep 1

nohup /workspace/dual-attention-resume-venv/bin/python -B \
  WorkingMemory/SpatialPriorityReadout/DualAttention/extend_live.py \
  extension_config.json > extension_supervisor.log 2>&1 < /dev/null &
new_pid="$!"
printf '%s\n' "$new_pid" > extension_supervisor.pid
sleep 3

printf 'NEW_SUPERVISOR=%s\n' "$new_pid"
ps -o pid,ppid,etime,stat,cmd -p "$new_pid",3108
cat dual_attention_results/extension_receipt.json
echo "---LEDGER---"
cat dual_attention_results/budget.json
