# Battery audit: environments, streams and training logic

Handoff section 4, executed 2026-09-16. CPU-only ideal observers and stream checks, plus one-step gradient diagnostics on the AV-context v2 model. No optimizer step was taken and nothing was trained.

Files:

- `observers.py`: hand-coded observers (ring and glyph decoders, quadrature Gabor bank, block-matching motion, Lucas-Kanade flow) and metrics.
- `run_audit.py` -> `results.json`, `report.md`: observer accuracy per task and condition, cue-necessity checks, frame-role table.
- `stream_checks.py` -> `stream_checks.json`: label/target balance, state round trip, cross-process determinism, frame counts, split isolation.
- `training_diagnostics.py` -> `training_diagnostics_scratch.json`, `training_diagnostics_step10240.json`: per-task gradient norms before the shared clip, checkpoint-recompute identity, E/I state across a 24-frame blank.

All numbers below are from those files, validation stream seed 63973001, `stimuli.py` SHA256 `16b7efd4575eba27...`.

## 1. Every task carries its label at 100x100

| Task | Observer | n | Result | Cue decoded from pixels |
|---|---|---:|---|---:|
| Orientation D0 | Gabor bank, sign x rotation at cued location > 7.5 deg | 768 | BA 1.000, AUC 1.000, at 15/30/45 deg all 1.000; angle error median 0.06 deg, p95 0.17 deg | 1.000 |
| Motion duration D0 | block matching per transition, majority vote | 768 | BA 0.954 hard vote (0.879 soft sum); per-transition direction agreement 0.982 | 1.000 |
| Krauzlis B12/B20/B28 | Lucas-Kanade mean flow, pre vs post event | 256 each | AUC 0.999 at every baseline; BA 0.973/0.979/0.987 at a fixed 13 deg threshold; d' 4.4-4.7 | 1.000 |
| Binding D0 / D24 | Gabor bank, pair with largest change, retrocue location | 256 / 128 | BA 1.000 / 1.000; swapped pair recovered 1.000 | 1.000 |
| Recognition N0/N4/N24 | raster hash | 128 each | accuracy 1.000; probe hold frames identical; study rasters unique; metadata hashes match | n/a |

Motion detail. Step size matters only for the soft observer (0.96 / 0.95 / 0.73 at 0.8 / 1.2 / 1.6 px; the hard vote is 1.00 / 1.00 / 0.87). The label margin is one frame in 61% of trials (472 of 768), and the soft observer drops to 0.82 there while margin >= 2 is >= 0.97. The task is hard at the label level but the per-frame motion signal is nearly perfect despite integer rasterisation and 50% dot replacement.

Krauzlis detail. The handoff's suspect, sub-pixel steps plus integer rasterisation, does not apply: `_krauzlis` renders dots with bilinear sub-pixel splatting. A linear flow estimator recovers the pre-event direction to a median 2 deg and the signed 26-28 deg change with SD about 6 deg. Estimated |change| on an unchanged cued patch is about 5 deg (SD 4). The 57/29/14 event split is exact per 200 draws (stream check).

Cue necessity. Motion: the uncued patches predict their own winner at 0.944 but the cued label at 0.258, so the cue carries the answer. Orientation: the sign-normalised rotation multiset over all four locations has total-variation distance 0.065 between labels at n=768 (sampling noise for 3 keys), so a no-cue observer cannot read the label from what changed. Binding: P(label=1 | swapped pair) is 0.43-0.64 over six pairs at about 60 trials each, consistent with 0.5. The uncued locations, given the cue, do leak a little in orientation (under label 0 the target is 0 or -1, so the set of uncued rotations differs), but that requires the cue first.

Frame roles. For every task the metadata frame indices are inside the tensor and the last referenced frame is the last frame; `frame_count` matches the rendered length for all 54 train and eval conditions. Blank frames carry only the 108-120 corner-glyph pixels. The orientation glyph for top-row targets lands at rows 4-14 as the handoff notes, and decodes at 1.000 there (BA top row 1.000, bottom row 1.000).

## 2. Streams are balanced, resumable and deterministic

From `stream_checks.json`, 10,000 `_case` draws per task on the train seed:

- Orientation, binding, recognition: labels 5000/5000, targets 2500 x 4, every (label, target) cell 1250, cue signs 5000/5000.
- Motion: 2500 per class, 2500 per target, every cell 625.
- Krauzlis: label 1 = 5700, label 0 = 4300; events target/foil/catch = 5700/2900/1400; targets 5000/5000.
- State round trip: saving the stream state, drawing six batches, restoring from the JSON-serialised state and redrawing gives byte-identical tensors. Same for the direct dict. A fresh stream differs, so the check is not vacuous.
- Cross-process: two subprocesses and the in-process stream produce identical batch hashes for both the validation seed and the test seed.
- Recognition photographs: train/val/test source ids disjoint; manifest sha256 and base ids disjoint across splits (200/100/200 images).

## 3. Training logic: the clip, not the tasks, set the effective learning rate; the trained model is dead

`training_diagnostics.py` runs the v2 recipe's exact loss (`cross_entropy / 5` per task, batch 8, one D0 microbatch per task) and inspects gradients before `clip_grad_norm_(1.0)`.

At scratch initialisation (model seed 41973001):

| Task | Loss (chance) | Scaled grad norm | Share of summed gradient | Predictions on 8 trials |
|---|---|---:|---:|---|
| orientation | 0.774 (0.693) | 1.714 | 0.356 | all class 1 |
| motion | 1.405 (1.386) | 0.491 | 0.048 | all class 0 |
| krauzlis | 0.770 (0.693) | 1.551 | 0.321 | all class 0 |
| binding | 0.706 (0.693) | 0.882 | 0.196 | all class 1 |
| recognition | 0.693 (0.693) | 0.557 | 0.079 | all class 1 |

Summed norm 2.79, clip factor 0.359. Orientation and Krauzlis take two thirds of every update; motion gets 5%. Per module, the readout and priority readout carry most of the gradient (readout 0.4-1.5 per task, priority readout 0.2-0.8), the encoder 0.12-0.41, the E/I memory 0.02-0.08, and memory_input 0.002-0.006. Pairwise cosines between task gradients are between -0.16 and 0.24. The logit spread across trials in a batch is 0.002-0.04, so the network is already nearly input-invariant at initialisation.

At cloud checkpoint 10240 (409,600 episodes):

| Task | Loss | Scaled grad norm | Share | Logit spread across trials |
|---|---|---:|---:|---:|
| orientation | 0.693 | 0.0023 | 0.000 | 0.0009 |
| motion | 1.386 | 0.0027 | 0.000 | 0.0001 |
| krauzlis | 0.669 | 0.0714 | 0.147 | 0.0001 |
| binding | 0.693 | 0.0016 | 0.000 | 0.0000 |
| recognition | 0.727 | 0.1695 | 0.853 | 0.0003 |

Summed norm 0.18, nothing clipped. Every module's gradient is below 0.004 except the priority readout on Krauzlis and recognition. Outputs are constant across inputs to four decimals. This is a dead network, not a slow one: the gradient has nothing to push against because the output no longer depends on the input.

Other checks:

- Activation checkpointing: logits identical to the direct forward; gradients differ by at most 3e-8 absolute (relative 3e-8 at scratch, 7e-7 at step 10240). Not bit-identical, but at float32 accumulation-order noise. Not a cause.
- E/I state across 24 blanks (orientation D24, eval mode): the rate norm does not decay. At scratch it rises from 34.7 at the last sample frame to 57.0 at the last blank (ratio 1.64); at step 10240 from 77.3 to 165.3 (ratio 2.14). Initial tau_r spans 2.0-8.0 (median 4.0) so passive decay over 24 frames would be 0.001 at the median, but the recurrent E/I weights sustain and grow activity. The inactive fraction settles near 0.27 (scratch) and 0.45 (trained). The memory does not annihilate by construction; whether it carries stimulus-specific information is a separate question the linear probes in experiment 24 answered negatively.
- The fixed sensory traces still decay to 0.25^24 = 3.6e-15 and 0.75^24 = 1.0e-3 over 24 blanks; they cannot carry anything across D24.

## What this settles

1. The five environments are not mis-specified. Hand-coded observers reach 0.95-1.00 on every task at D0, including motion duration and Krauzlis, using only the rendered frames and the pixel-decoded cue. Any model that stays at chance on these tasks at D0 is failing to learn, not failing at an impossible task.
2. The streams and resume path are correct.
3. The v2 recipe's failure has a mechanical signature: clipping to a third of the step from the first update, a five-way loss average with two tasks taking two thirds of the gradient, batch 8, and a network whose outputs are almost input-invariant from initialisation and become exactly so by 400k episodes. None of the audited parameters (glyph placement, motion step and replacement, Krauzlis rendering, event proportions, frame indexing) needs to change before a plain baseline is trained.

Next rung: the plain baseline (per-frame CNN + GRU, all parameters trainable, one task at D0, batch 64, lr 1e-3, no clip), one task per run, sequentially on the laptop. Its D0 result per task is the floor every later model must beat.
