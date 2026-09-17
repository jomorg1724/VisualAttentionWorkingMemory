# Scratch spatial priority-map readout

**Status (live, 2026-09-15 UTC): production training verified.** No validation
has completed; all metrics below are launch-health evidence.

## Correction and construction

The inherited attention8400 launch was invalidated by the user's lineage
correction. Pod `mbi39b005jp884` was stopped/deleted and confirmed absent at
last-known step 8720 (12,800 fresh episodes); no validation completed.

The corrected `spatial_priority_readout_scratch_v2` run loads no parent,
model, Adam, stream or RNG state. Its bundle contains no checkpoint. All
579,423 parameters and all 137 optimizer parameter states begin fresh from
documented seeds. Remote construction checks passed deterministic same-seed
reconstruction, different-seed sensitivity, empty Adam state, original
`JointAttention`/`SpatialEI`/comparator classes, source/locality bias
trainability, gamma absence, no terminal pooled bypass, priority
normalization, task-specific heads and finite full forward/backward.

The remote initial model-state digest is `9e2a507a…9a45`; the atomic step-0
checkpoint is `97b7e6ce…3f76`. The local state digest differs under its newer
runtime, so no cross-platform bitwise-equivalence claim is made.

## Architecture and fixed protocol

Final `H_T`, `R_T` and `C_T` are concatenated as `[B,192,13,13]`, processed by
shared 1×1 `192→96` and padded 3×3 `96→64` convolutions, then task-specific
selection and class-evidence maps. A spatial softmax weighted sum yields class
logits. No global mean/max decision branch, privileged coordinates, task
change, attention modification or prospective gamma exists.

Training retains the five historical tasks and exact stimuli, cues, labels,
losses, seeds, conditions and evaluation method. Each of 4,000 updates uses
five microbatches of eight and one clipped Adam step: 160,000 episodes.
Validation occurs every 800 updates from scratch.

## Verified cloud launch

Palladio pod `ce00y2ooosl7wc` is one community RTX 3090 (24,576 MiB), listed
at $0.22/hour; it is not an A100. The pinned runtime is Python 3.10.12,
PyTorch 1.13.1+cu117, NumPy 1.23.1, SciPy 1.8.1 and Pillow 9.1.1. Three
profile updates averaged 3.4266 seconds/update, allocated 1.014 GB and reserved
1.122 GB. The conservative run/evaluation/retrieval projection is 5.51 hours
and $1.21; the eight-hour deadline is
`2026-09-15T11:47:20.740518Z` ($1.76 listed ceiling).

The immutable launch receipt snapshot reached step 46 / 1,840 episodes with
finite loss 0.8250, accuracy 0.55 and pre-clip gradient norm 0.5971.
Supervisor PID 254 and worker PID 332 were active. These aggregate
microbatches are not performance estimates.

Detached lifecycle PID 36128 and 45-second watcher PID 36812 mirror metrics,
GPU state, checkpoint hashes and immutable validations. They own complete
retrieval and provider stop/delete at terminal state or deadline. See
`runs/scratch_20260915_034720/` and `launch_receipt.json`.

## Interpretation boundary

This experiment asks whether the complete architecture can acquire the five
tasks when trained from scratch with spatially preserved terminal evidence.
It is not a one-variable continuation and cannot support an
initialization-controlled claim against previously trained checkpoints.

## Completed complementary local diagnostic

An analysis-only [frozen-core diagnostic](LocalDiagnostic/report.md) used the
motion-only step-12200 checkpoint, not this live scratch cloud model. Its
encoder, opponent traces, original attention, spatial E/I memory and
comparator stayed frozen in evaluation mode. Two fresh probes received
identical cached final `H_T/R_T/C_T` fields.

The 74,303-parameter pooled probe scored 23.83% held-out balanced accuracy
(95% grouped CI 21.48–26.27%) and macro OVR AUC0.4805. The
74,373-parameter spatial-priority probe scored 24.32% (22.07–26.66%) and
AUC0.4832. Their paired BA difference was +0.49 percentage points
(95% CI -1.76 to +2.64), with no reliable per-delay advantage at D0/4/12/24.
Both are near the 25% motion chance level, so this diagnostic did not expose a
motion-duration solution in these frozen terminal fields.

The spatial probe's map assigned 13.13% mean mass to an evaluation-only 3×3
target region versus uniform expectation5.33%, but alignment did not produce
correct direction decoding. This post-hoc finding does not predict the
outcome of end-to-end scratch cloud training. The local job made no cloud
calls and did not modify the source checkpoint.
