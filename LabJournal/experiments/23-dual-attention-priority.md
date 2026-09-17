# 23 — Dual attention with an exclusive priority-map decoder

[Current status](../CURRENT_STATUS.md) ·
[implementation and live evidence](../../WorkingMemory/SpatialPriorityReadout/DualAttention/README.md) ·
[launch receipt](../../WorkingMemory/SpatialPriorityReadout/DualAttention/launch_receipt.json)

## Question

Can separate pre-update integration and post-update selection support the
unchanged five-task battery when the decision path is forced through a spatial
priority field? This is a multi-component architecture experiment, not a
single-variable ablation.

## Architecture and lineage

A one-head width-64 pre attention lets previous E/I rates query current sensory
and previous-memory tokens. Its raw value context `C_t` is retained while its
projected context drives the unchanged spatial E/I update. A separate one-head
width-64 post attention lets updated rates query 507 `C/H/R` tokens. Only its
terminal 64×13×13 output reaches the convolutional spatial-softmax decoder.
Both stages use learned source identity/bias and locality terms; no gamma,
mask, old comparator, privileged coordinates, global pooling, or bypass exists.

All 549,344 trainable parameters and Adam state start fresh. Remote tests
proved zero loaded tensors/states and exact name/value equality to scratch
control checkpoint 0 for 119 unchanged tensors. The architecture-specific
parameters use deterministic documented seeds.

## Verified live execution

Independent community RTX 3090 pod `f7y0zw02f4fzum` is listed at $0.22/hour.
It has separate source, paths, run ID, supervisor (PID 186), worker (PID 265),
45-second watcher (PID 20144), lifecycle (PID 15536), and deletion ownership.
Existing control pod `ce00y2ooosl7wc` and comparator pod `h1ygacz4zcsfj3`
were not modified.

The three-update profile averaged 2.5437 seconds with 0.992 GB peak allocation.
Production began under a finite `2026-09-15T12:35:18Z` deadline and fixed
4,000-update / 160,000-episode exposure. First rows had finite losses and
gradients; a direct snapshot at step 20 showed loss 0.77457, aggregate batch
accuracy 0.55 and pre-clip gradient norm 0.63206, with 67% GPU utilization.
These are optimization-health observations, not validation evidence.

## Dated closure — 2026-09-15T22:35-07:00

The resumed dual-attention pod `629g1utqkk8non` (post-hoc extension to
12,000) was stopped and deleted at the user's request at logged step ≈5,464.
Retrieval before deletion failed, so no checkpoint of this arm survives.
Mirrored validations 800–4800 remain under
`WorkingMemory/SpatialPriorityReadout/DualAttention/runs/dual_resume_20260915_183345/incremental/`;
validation-selected step so far was 4800 with every task at chance-normalised
BA ≈ 0. No held-out test exists. This arm is closed without an architecture
verdict.

## Next decision

Allow the independent lifecycle to complete scheduled validation, held-out
evaluation, hash-verified retrieval, and deletion. Compare all five tasks and
conditions against the separately trained scratch control and comparator only
after their final held-out artifacts exist.
