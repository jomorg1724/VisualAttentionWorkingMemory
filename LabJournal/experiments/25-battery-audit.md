# 25 — Battery audit: ideal observers, streams and one-step training diagnostics

[Journal index](../README.md)

Status: completed (audit only; no training). Last updated: 2026-09-16T21:45-07:00.

## Question and reason for this test

After experiment 24 closed the from-scratch lineage, the handoff (section 4) asked whether the five tasks are learnable at all at this resolution, whether the streams and resume path are correct, and whether the training logic itself (five-way loss average, clip at 1, batch 8, activation checkpointing, E/I time constants) is a plausible cause of every scratch arm sitting at chance. None of these had been measured. Without them, a chance-level result on the battery cannot be attributed to an architecture.

## Design and ancestry

No model was trained. Three CPU-side measurements and one forward/backward pass:

1. Hand-coded ideal observers per task, reading only the rendered frames and decoding the cue from pixels: quadrature Gabor bank for orientation and binding, block matching for motion duration, Lucas-Kanade flow for Krauzlis, raster hashing for recognition. Validation stream seed 63973001. Sample sizes 768 (orientation, motion), 256 per baseline (Krauzlis), 256 + 128 (binding D0, D24), 128 per condition (recognition).
2. Stream checks on the train, validation and test seeds: label and target balance over 10,000 sampler draws per task, JSON round trip of the stream state, cross-process determinism, rendered length versus `frame_count` for all 54 conditions, recognition split isolation.
3. One-step gradient diagnostics on the AV-context v2 model under its exact recipe (loss/5 per task, batch 8), at the scratch initialisation (model seed 41973001) and at the retrieved cloud checkpoint 10240 (SHA256 in the JSON). No optimizer step.

Code and outputs: [WorkingMemory/BatteryAudit](../../WorkingMemory/BatteryAudit/README.md) (`report.md`, `results.json`, `stream_checks.json`, `training_diagnostics_scratch.json`, `training_diagnostics_step10240.json`).

## Results

Ideal observers at D0 (or per baseline):

| Task | BA | Other |
|---|---:|---|
| Orientation | 1.000 | AUC 1.000 at 15, 30 and 45 deg; angle error median 0.06 deg |
| Motion duration | 0.954 | per-transition direction agreement 0.982; label margin is one frame in 61% of trials |
| Krauzlis | 0.973-0.987 at 13 deg | AUC 0.999 at B12/B20/B28; d' 4.4-4.7; dots are rendered with bilinear sub-pixel splatting, not integer rounding |
| Binding | 1.000 (D0 and D24) | swapped pair recovered 1.000 |
| Recognition | 1.000 | probe frames identical across the hold; study rasters unique; manifest splits disjoint |

Cue necessity: uncued motion patches predict the cued label at 0.258; the sign-normalised orientation rotation multiset over all four locations has TV distance 0.065 between labels; P(label=1 | swapped pair) in binding is 0.43-0.64 across pairs. Cue decoding from pixels is 1.000 for every task.

Streams: labels and targets exactly balanced per task and cell; Krauzlis events 5700/2900/1400 per 10,000; state round trip byte-identical through JSON; two subprocesses and the in-process stream agree on the validation and test seeds; no frame-count mismatch in 54 conditions; recognition source ids disjoint across splits.

Training logic, v2 recipe at scratch init: summed gradient norm 2.79, so the clip at 1 scales every update by 0.36 from the first step. Orientation and Krauzlis carry 36% and 32% of the summed gradient, binding 20%, recognition 8%, motion 5%. Logit spread across the eight trials of a batch is 0.002-0.04 at initialisation. At checkpoint 10240 every task's scaled gradient norm is below 0.003 except Krauzlis (0.07) and recognition (0.17), the summed norm is 0.18, and logit spread is at most 0.0009: outputs are constant across inputs. Activation checkpointing reproduces logits exactly and gradients to 3e-8 absolute. The E/I rate norm grows across 24 blank frames (ratio 1.64 at scratch, 2.14 trained) rather than decaying; the fixed sensory traces decay to 1e-3 and 4e-15.

## What this shows, what it does not show, and the next decision

Measured: every task's label is recoverable from the rendered frames by a simple fixed computation at 0.95-1.00 balanced accuracy, so the environments are not mis-specified for this resolution. The streams are correct. The v2 recipe's failure has a mechanical signature independent of architecture: heavy clipping from step one, an uneven five-way gradient split, and a network that is nearly input-invariant at initialisation and exactly so after 400k episodes.

Not shown: that any learned model can reach the observer numbers, or which of the recipe's factors (batch 8, clip, loss averaging, the fixed priors, initialisation) is responsible. The observers use knowledge of the stimulus geometry (centre positions, wavelength, cardinal directions) that a model must discover.

Next decision: train the plain baseline of the handoff's ladder rung 1 (per-frame CNN + GRU, everything trainable, one task at D0, batch 64, lr 1e-3, no clip) on the laptop, one task per run, and record its D0 balanced accuracy per task as the floor.

## Evidence and closure

- [Audit README with all tables](../../WorkingMemory/BatteryAudit/README.md)
- [Observer report](../../WorkingMemory/BatteryAudit/report.md), [results.json](../../WorkingMemory/BatteryAudit/results.json)
- [stream_checks.json](../../WorkingMemory/BatteryAudit/stream_checks.json)
- [training_diagnostics_scratch.json](../../WorkingMemory/BatteryAudit/training_diagnostics_scratch.json), [training_diagnostics_step10240.json](../../WorkingMemory/BatteryAudit/training_diagnostics_step10240.json)
- `stimuli.py` SHA256 `16b7efd4575eba27...` (full hash in `results.json`). No checkpoints were produced; the cloud checkpoint 10240 used is the retrieved v2 arm's, unchanged.
