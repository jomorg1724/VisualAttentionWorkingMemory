# 22 — Scratch spatial priority-map readout

[Current status](../CURRENT_STATUS.md) ·
[implementation](../../WorkingMemory/SpatialPriorityReadout/README.md) ·
[live report](../../WorkingMemory/SpatialPriorityReadout/report.md) ·
[launch receipt](../../WorkingMemory/SpatialPriorityReadout/launch_receipt.json) ·
[correction receipt](../../WorkingMemory/SpatialPriorityReadout/lineage_correction_receipt.json)

## Question and architecture

The five-task model's terminal classifier globally pooled sensory, memory and
comparison fields. The new architecture instead concatenates final
`H_T/R_T/C_T` at 13×13, applies 1×1 and padded 3×3 convolutional fusion, and
learns task-specific spatial selection and local evidence maps. A 169-location
softmax weighted sum gives logits. The map is saved for diagnostics.

Opponent processing, original `JointAttention` with source/locality terms,
spatial E/I memory and comparator classes remain. There is no global decision
shortcut, prospective gamma, coordinate oracle, cue-label input or task
change.

## Lineage correction

The initial launch inherited attention8400 tensors and Adam state. The user
superseded that lineage with full scratch training. Monitor PIDs 30480/38436
were stopped, and RTX 4090 pod `mbi39b005jp884` was stopped, deleted and
confirmed absent. It had reached 320 updates / 12,800 fresh episodes with no
validation. It is a cancelled implementation attempt, not evidence about the
architecture.

Version `spatial_priority_readout_scratch_v2` loads no parent and starts at
step 0 with every one of 579,423 parameters and the optimizer freshly
initialized from recorded seeds. Local and remote tests independently
recreated identical state within their runtimes; no cross-platform bitwise
identity is claimed. The cloud bundle contains no checkpoint.

## Live execution

Replacement Palladio pod `ce00y2ooosl7wc` uses one community RTX 3090
(24,576 MiB) at $0.22/hour. Three profile updates averaged 3.4266 seconds with
1.014 GB allocated. The projected training/evaluation/retrieval total is
5.51 hours / $1.21; the eight-hour backstop ends
`2026-09-15T11:47:20.740518Z`.

**Launch snapshot, not validation:** step 46 / 1,840 episodes, loss 0.8250,
accuracy 0.55, gradient norm 0.5971. No validation had completed. A detached
45-second watcher and lifecycle process mirror evidence and own verified
retrieval and pod deletion.

## Decision boundary

The unchanged suite receives 4,000 updates / 160,000 episodes with five
microbatches of eight and validation every 800 updates. Historical trained
models differ in initialization and prior exposure; comparisons are
descriptive architecture comparisons, not initialization-controlled
ablations.

## Complementary frozen-core readout result

A separate authorized [local diagnostic](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
completed without touching the active cloud worker. It used the terminal
motion-only step12200 checkpoint, which had received30,400 cumulative motion
episodes but remained near chance. All computations through final aligned
`H_T/R_T/C_T` stayed frozen and in evaluation mode.

Independent train/validation/test base streams used seeds731091/731092/731093
and 512/128/256 base movies per delay. Each base movie was rendered at
D0/4/12/24; delay repeats were grouped rather than counted as independent.
Both probes selected epoch20 (640 updates /40,960 repeated presentations) on
validation. The capacity-matched pooled probe had74,303 parameters and the
single-task spatial-priority probe74,373.

On the once-only held-out test, pooled BA/AUC was23.83%/0.4805 and spatial
BA/AUC24.32%/0.4832. The paired spatial-minus-pooled BA difference was+0.49
percentage points (95% grouped bootstrap CI -1.76 to+2.64). Per-delay BA was
22.66%,24.22%,25.39%,25.00% for spatial versus
22.27%,25.00%,23.05%,25.00% for pooled; every paired interval included zero.
Thus this post-hoc function class did not decode the four-way duration winner
from these final frozen fields.

The spatial map assigned13.13% mean probability mass to a 3×3 region around
the held-out target location versus5.33% for a uniform map, with20.80% peak
hits. Target coordinates entered only this post-test metric. Spatial
alignment without successful label decoding is not evidence of causal
attention or a positive forecast for the independently trained cloud model.
Checkpoint and frozen-state hashes were unchanged. Smoke plus production and
finalization used738.5 seconds of the1,800-second local cap.
