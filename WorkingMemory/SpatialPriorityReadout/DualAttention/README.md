# Dual-attention spatial-priority experiment

This independent arm trains `dual_attention_spatial_priority_scratch_v1` from
scratch on the unchanged five-task spatial battery. It does not share a pod,
run directory, process, checkpoint, optimizer state, watcher, or lifecycle
with either the scratch control or attention-context comparator.

## Architecture

At each frame, one width-64 pre-update head uses `R_(t-1)` queries and
`[H_t; R_(t-1)]` keys/values. Its raw merged value context is
`C_t[B,64,13,13]`; `W_O_pre(C_t)` is the sole external drive to the unchanged
spatial E/I update. One width-64 post-update head then uses `R_t` queries and
`[C_t; H_t; R_t]` keys/values. Both attentions retain learned source biases and
softplus-locality penalties, without masks or privileged coordinates.

Only terminal `P_T` from the post output reaches the decoder:
`1x1 64→96`, SiLU, `3x3 96→64`, then task-specific local selection/evidence
maps and a 169-way spatial-softmax weighted logit sum. There is no old
comparator, gamma, raw-field concatenation, global pooling, or classifier
bypass.

## Verified live execution

Run `runs/dual_20260914_213518` uses independent Palladio RunPod
`f7y0zw02f4fzum`, a community RTX 3090 (24 GB) listed at $0.22/hour. The fixed
target is 4,000 updates / 160,000 episodes (five task microbatches of eight)
with validation at 800-update intervals. The hard deadline is
`2026-09-15T12:35:18Z`.

Local and pinned-runtime construction checks passed all requested tensor
shapes, finite forward/backward, nonzero gradients through both Q/K/V paths,
normalized priority weights, P-only decoder wiring, empty optimizer state,
zero loaded tensors, and scratch-control equality for 119 shared tensors.
Remote initial model SHA256 is
`e2ac2db5a4175e783bbcd1acaa54381d7a4c57b01c4afbf9234c472ef94167a1`.

Three profile updates averaged 2.5437 seconds/update, with 991,733,760 bytes
peak allocated. Production was verified with supervisor PID 186 and worker
PID 265. The first three training rows were finite; by the captured snapshot
step 20 had loss 0.77457, aggregate batch accuracy 0.55, and pre-clip gradient
norm 0.63206. GPU utilization was 67% in the direct snapshot and 79% in the
first 45-second mirror. These are live training observations, not validation.

Local lifecycle PID 15536 and watcher PID 20144 mirror metrics and immutable
evaluation artifacts every 45 seconds, verify final hashes, then stop/delete
only pod `f7y0zw02f4fzum`.

## Evidence

- `construction_checks.json`: local essential checks.
- `runs/dual_20260914_213518/remote_construction_stdout.log`: required remote
  equality and gradient checks.
- `runs/dual_20260914_213518/remote_profile_result.json`: measured profile.
- `runs/dual_20260914_213518/remote_initialization.json`: zero-inheritance
  lineage and parameter count.
- `runs/dual_20260914_213518/live_status.json`: 45-second live mirror.
- `runs/dual_20260914_213518/verified_live_snapshot.txt`: process, GPU, and
  initial metric evidence.
- `runs/dual_20260914_213518/cloud_provisioning.json`: isolated provider,
  endpoint, paths, price, and lifecycle ownership.
