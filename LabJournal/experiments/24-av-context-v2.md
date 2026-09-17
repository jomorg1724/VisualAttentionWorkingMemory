# 24 — AV-context v2: five combined changes, local scratch run

[Journal index](../README.md) ·
[implementation and pre-flight](../../WorkingMemory/AttentionContextComparator/V2/README.md) ·
[live status](../../WorkingMemory/AttentionContextComparator/V2/runs/v2_local_20260915_220124/live_status.json)

Status: closed as a negative result; both arms user-stopped. Last updated: 2026-09-16T08:30-07:00. The local arm was killed at logged step 491 (durable checkpoints 0 and 256 only; see `runs/v2_local_20260915_220124/user_stop_receipt.json`) before any validation, so it contributes no result.

## Question and reason for this test

The v1 scratch attention-context arm (experiment 22/23 lineage, cloud pod
`6mopzgoioemdp2`) is class-collapsed on four of five tasks at validation 4800,
with priority entropy at 99.1–99.9% of uniform on the cued tasks. Four
mechanisms were argued to keep it there, and one configuration fault:

1. The opponent direction sign and magnitude computed by the accumulators are
   discarded before the readout (`energy,_=core.energy_channels(...)`).
2. Average pooling to 13×13 dilutes sparse local motion evidence.
3. Both attention heads start at λ = 4 with a +2 visual bias, so a query cell
   places ~1e-16 of its mass three cells away; cue and patch cannot interact.
4. Zero-initialised priority selection maps: under uniform priority the loss
   drives evidence towards spatial constancy, which zeroes the selection
   gradient. Uniform priority is a self-reinforcing fixed point.
5. The `parent_lr`/`new_lr` split is a warm-start convention; on a scratch arm
   it trains 73% of the parameters (encoder, readout, accumulators,
   projections) ten times slower for no reason.

This is deliberately a combined arm: a single-factor result at chance teaches
nothing, so the combination runs first and ablations follow only a positive
result.

## Design and ancestry

No parent. Version `attention_context_comparator_v2` subclasses the pinned
v1 code; all v1 source hashes are unchanged. Changes A–E and their
initialisation are listed in the
[package README](../../WorkingMemory/AttentionContextComparator/V2/README.md).
With C, D and E disabled the v2 forward is bit-identical to v1 on CPU
([construction_checks.json](../../WorkingMemory/AttentionContextComparator/V2/construction_checks.json)).
Parameters 590,015 (v1 534,239), every one at learning rate 3e-4. Stimuli,
cues, losses, five-task recipe, seeds, batch 8 × 5 microbatches, gradient clip
1 and Adam ε are the v1 values.

## Pre-flight (forward-only, no training)

- Identity gate: passed, max |Δlogit| 0 on CPU (3.1e-5 on CUDA from kernel
  selection).
- Attention mass versus distance on the hash-verified local scratch
  motion-only v1 step-3200 checkpoint: both heads λ ≈ 4.02; mass beyond 3
  cells 1.8e-17 per head; 83% on the query's own cell. v2 head 1 at init:
  71.6% beyond 3 cells.
  [preflight_attention.json](../../WorkingMemory/AttentionContextComparator/V2/preflight/preflight_attention.json)
- Linear probes for the cued patch's longest-duration direction on the same
  checkpoint (768 fit / 256 test, D0): H_T, R_T, C_T and last-evidence H_9 as
  full fields, mean+max pools and cued 3×3 regions are all at chance (best
  28.1%, Wilson 23.0–33.9%; permuted-label control 25.8%). By the
  pre-stated rule the field is not decodable, so A and B were kept and the
  full A–E arm launched.
  [preflight_probe.json](../../WorkingMemory/AttentionContextComparator/V2/preflight/preflight_probe.json)

## Exposure, selection and resources

Local RTX 3070 Laptop GPU, run
`WorkingMemory/AttentionContextComparator/V2/runs/v2_local_20260915_220124`,
supervisor PID 34688, 40-hour runtime cap. Planned 12,000 updates / 480,000
episodes, validation every 800 (64 per cell, 100 per Krauzlis cell), selection
by maximum [minimum chance-normalised task BA, mean task AUC] then earlier
step, held-out test 256 per cell of selected and terminal.

Profile: 6.51 s/update, 1.05 GB peak; projected 4.3 h of training to the gate
and 21.7 h to 12,000 before validations.

Pre-registered stopping rule at validation 2,400: priority entropy below 98%
of ln 169 on at least one of orientation/motion/Krauzlis, and head-1 attention
mass beyond 3 cells above 1%, both measured on the validation trials. Either
failing stops the arm with no held-out test and no extension.

Live instrumentation every 32 updates
(`training/live_diagnostics.jsonl`): per-task priority entropy fraction,
attention mass beyond 3 cells per head, spatial variance of evidence and of
H_T, softplus(raw_locality) per head, source bias, selection weight norm,
priority and attention gradient norms.

## Cloud overnight arm (same model, long exposure)

At the user's instruction (2026-09-15T22:28-07:00) both earlier cloud pods
were deleted and one replacement pod was provisioned for the identical v2
model: RunPod community RTX 3090 `vqpgk21cpi53b6`, $0.22/hour, run directory
`WorkingMemory/AttentionContextComparator/V2/runs/cloud_20260915_222845`,
hard deadline 20 hours after launch (maximum spend $4.40). Target 30,000
updates / 1.2 M episodes, validations at 800, 1600, 2400 and then every 2,000
from 4,000; training stops gracefully 45 minutes before the deadline and the
latest checkpoint is validated and tested, so the reachable exposure is set by
measured throughput. The stopping rule at 2,400 is evaluated and recorded but
does not stop this arm. Supervisor: `V2/cloud_sweep.py`; the same worker,
model, protocol and construction gate as the local run; a detached local
lifecycle monitor mirrors artifacts every 45 s and retrieves, verifies and
deletes the pod at completion, failure or deadline.

## Dated progress

- 2026-09-15T22:01-07:00 (local): gate passed, profile completed, `train_800`
  started. No validation exists.
- 2026-09-15T22:31-07:00 (cloud): pod `vqpgk21cpi53b6` set up; remote
  identity gate passed bit-exactly; profile 3.48 s/update, 1.04 GB peak
  (projected ≈19,900 reachable updates at profile speed, more at production
  speed); `train_800` running at 60% GPU utilisation. Deadline
  2026-09-16T12:28-07:00. Local lifecycle monitor relaunched after fixing a
  watcher snippet that MSYS `ssh` mangled; the monitor now restarts a crashed
  watcher instead of treating it as a terminal state.

- 2026-09-16T07:55-07:00: overnight every local process (local supervisor,
  worker, cloud lifecycle monitor and watcher) had died at ≈22:40 the
  previous evening; the local run stalled at step 321 with checkpoint 256
  durable, and nothing was mirroring the pod. The pod itself trained on
  unattended: step ≈10,280 in `train_12000`, 2.6 s/update, 48 indexed
  checkpoints, validations 800/1600/2400/4000/6000/8000/10000 complete. The
  lifecycle monitor was relaunched (PID 38220), the pod's results and all
  checkpoints were pulled to
  `runs/cloud_20260915_222845/pulled/` while training continued, and the local
  run was resumed in place from checkpoint 256 (65 uncheckpointed metric rows
  moved aside; `resume_local.py`).

## Results

Cloud validations (64 per cell, 100 per Krauzlis cell), provisional:

| step | min normalised BA | mean AUC | priority entropy / ln169: orient, motion, Krauzlis | head-1 mass > 3 cells | λ head 0 |
|---|---:|---:|---|---:|---:|
| 800 | 0.000 | 0.521 | 0.995, 0.999, 0.996 | 0.714 | 4.004 |
| 2400 (gate) | 0.000 | 0.497 | 0.991, 0.997, 0.994 | 0.712 | 4.005 |
| 4000 | 0.000 | 0.508 | 0.991, 0.996, 0.995 | 0.712 | 4.006 |
| 6000 | 0.000 | 0.526 | 0.989, 0.993, 0.995 | 0.719 | 3.989 |
| 8000 | 0.000 | 0.511 | 0.979, 0.990, 0.996 | 0.721 | 3.974 |
| 10000 | -0.016 | 0.531 | 0.985, 0.979, 0.992 | 0.719 | 3.973 |

Every task is at chance-normalised BA ≈ 0 through 400,000 episodes. The
pre-registered gate at 2,400 would have failed on the entropy criterion
(minimum cued entropy 0.991 > 0.98) and passed on head-1 locality; by 10,000
motion entropy has reached 0.979, the first cued task below the 98% line, but
without any accuracy movement. Head-1 stays wide (λ ≈ 0.0195) and head-0
locality is slowly loosening (4.0 → 3.97).

Training cross-entropy over the cloud run (500-update window means; chance
0.693 for the binary tasks, 1.386 for motion):

| updates | orientation | motion | Krauzlis | binding | recognition | grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 1–500 | 0.696 | 1.390 | 0.689 | 0.694 | 0.609 | 0.63 |
| 2000–2500 | 0.694 | 1.387 | 0.685 | 0.693 | 0.523 | 0.33 |
| 9500–10000 | 0.693 | 1.387 | 0.684 | 0.693 | 0.520 | 0.26 |

## Dated closure — 2026-09-16T08:15-07:00

The user judged the arm a failure and stopped the pod at logged step 10,621
(424,840 episodes). `cloud_shutdown.py` retrieved and hash-verified the
complete results directory and all 49 indexed checkpoints (0–10,496) to
`runs/cloud_20260915_222845/retrieved_user_stop/` before deletion; stop 200,
delete 204, lookup 404, account pod list empty
([receipt](../../WorkingMemory/AttentionContextComparator/V2/runs/cloud_20260915_222845/user_stop_receipt.json)).
No held-out test was run. Approximate spend: 9.8 pod-hours at $0.22.

## What this shows, what it does not show, and the next decision

Measured: with all five changes, from-scratch five-task training stayed at
chance on orientation, motion, Krauzlis and binding for 424,840 episodes, with
decaying gradient norm and constant outputs. The wide, source-neutral head 1
was never recruited (72% of its mass beyond 3 cells throughout, λ unchanged);
the non-zero priority selection init was flattened back toward uniform, and
evidence spatial variance collapsed within 300 updates, which is the
uniform-priority fixed point acting against the initialization. Unifying the
learning rate did not change this. The pre-flight probe had already found no
decodable direction signal in the frozen fields, so changes A and B widened a
representation that was not carrying the signal.

Interpretation: the failure is not specific to attention, priority readout or
motion. Every from-scratch arm on this battery (v1 scratch control, dual
attention, v2 cloud, v2 local) failed to learn even spatial binding and image
recognition, tasks the warm-started attention-8400 lineage solved to 100% and
97% within ≈140,000 new episodes. From-scratch optimisation of the whole
sensory-to-memory chain at batch 8 does not leave the constant-output regime
on this battery, so the cue-routing question was never actually posed to the
network. Attribution among A–E is therefore moot for this arm.

Not shown: that C, D or E are useless under a working representation; that
the battery is unlearnable; anything about seeds (one seed per arm).

Next decision (design only): apply C, D and E as a warm start from attention
8400 or cloud 10000, where binding and recognition already work, and test
whether the wide head and non-uniform priority get used when there is a
representation to route. That separates the attention/readout question from
the scratch-optimisation question this design confounded.

## Evidence and closure

[model](../../WorkingMemory/AttentionContextComparator/V2/model.py) ·
[protocol](../../WorkingMemory/AttentionContextComparator/V2/protocol.py) ·
[check](../../WorkingMemory/AttentionContextComparator/V2/check.py) ·
[worker](../../WorkingMemory/AttentionContextComparator/V2/worker.py) ·
[supervisor](../../WorkingMemory/AttentionContextComparator/V2/run.py) ·
[aggregate](../../WorkingMemory/AttentionContextComparator/V2/runs/v2_local_20260915_220124/aggregate.json)
