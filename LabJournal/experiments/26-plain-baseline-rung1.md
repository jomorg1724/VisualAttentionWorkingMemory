# 26 — Plain baseline, ladder rung 1: can a standard model learn the battery from scratch?

[Journal index](../README.md)

Status: completed (orientation family; other families have record runs only). Last updated: 2026-09-17T02:20-07:00.

## Question and reason for this test

Experiment 25 showed every task's label is recoverable from pixels by a fixed observer. The handoff's rung 1 asks whether a standard model with everything trainable learns each task at D0 within 100k episodes, to give a floor for every later component and to separate task difficulty from optimisation failure.

## Design and ancestry

No parent. Model: per-frame CNN (four stride-2 conv blocks with GroupNorm and ReLU to 128x7x7), flatten, Linear to 256, GRU(256), per-task linear heads; about 2.2M parameters. Adam, batch 64 (two chunks of 32), no gradient clipping, no weight decay, no fixed priors. One task per run, 100,000 episodes (1,563 updates) unless stated. Train stream seed 61973001 (single thread) or 61973001 + 7919*(worker+1) per producer process; validation seed 63973001, 256 trials per cell every 200 updates; test seed 64973001, 512 trials per cell. Code: [WorkingMemory/PlainBaseline](../../WorkingMemory/PlainBaseline/README.md). Options added during the experiment, each described there: `--stack K` (K consecutive frames as input channels), `--feature-norm layernorm`, `--center` (input minus 0.5), `--zero-head`, and a within-family difficulty ladder (`variants.py`).

## Results

### Record runs, the recipe as specified (lr 1e-3, per-frame, uncentred)

| Task | Episodes | Test BA (terminal) | Chance | Note |
|---|---:|---:|---:|---|
| orientation_cued D0 | 100,032 | 0.488 | 0.50 | loss exactly log 2 from update 100; gradient norm 0.000 |
| spatial_binding D0 | 20,032 | 0.514 | 0.50 | same collapse by update 143 |
| image_recognition N4 (H3/H4/H5) | 20,032 | 0.525 / 0.500 / 0.504 | 0.50 | collapsed |
| motion_duration_cued D0 | 20,032 | 0.250 | 0.25 | predicts one class for every trial |
| krauzlis B12/B20/B28 | 20,032 | 0.500 / 0.500 / 0.500 | 0.50 | predicts class 1 for every trial |

Runs after orientation were shortened to 20k episodes because the collapse is complete within 150 updates and the remaining exposure adds nothing.

### Mechanism of the collapse (orientation, lr 1e-3)

Update 1 loss 0.696; update 2 loss 1.086 with gradient norm 7.1 and accuracy exactly 0.5000 from then on; the loss then rings with a six-update period down to log 2 by update 40. Probing the terminal checkpoint: CNN feature variation across trials fell from 0.052 at initialisation to 0.0015, 75% of projection units dead, recurrent state identical across trials to 1e-4, logit spread 0.0001. The first Adam step makes the output input-independent; afterwards only the head bias is tuned. The v2 lineage's checkpoint 10240 has the same signature (experiment 25).

A six-step probe at initialisation (orientation batch of 64): centring the input raises relative feature variation across trials from 0.18 to 0.47; at lr 1e-3 feature variation collapses within six steps with or without centring unless the head is zero-initialised; at lr 1e-4 it survives.

### Full task, recipe sweep (orientation D0, 100k episodes each, test BA on 512)

| stack | feature norm | centred | zero head | lr | Test BA | Test AUC | Training loss at end | Outcome |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | none | no | no | 1e-3 | 0.488 | 0.491 | 0.6932 | first-step collapse |
| 3 | layernorm | no | no | 1e-3 | 0.498 | 0.499 | 0.6932 | first-step collapse |
| 3 | none | no | no | 1e-3 | 0.498 | 0.516 | 0.6931 | first-step collapse |
| 1 | layernorm | no | no | 1e-3 | (killed at 1400) | | 0.6932 | first-step collapse |
| 3 | layernorm | no | no | 1e-4 | 0.477 | 0.497 | 0.6932 | loss spike at update 2, then plateau |
| 1 | layernorm | no | no | 1e-4 | 0.520 | 0.518 | 0.6931 | plateau |
| 3 | layernorm | no | no | 3e-5 | 0.518 | 0.500 | 0.6934 | plateau |
| 3 | none | yes | yes | 1e-3 | 0.500 | 0.518 | 0.6932 | no spike; plateau, gradient 0.001 |
| 3 | none | yes | no | 1e-4 | 0.535 | 0.506 | 0.6932 | no spike; plateau |
| 3 | none | yes | yes | 3e-4 | (killed at 921) | | 0.6932 | plateau |

No recipe learns the full task within 100k episodes. Two distinct failure modes: first-step collapse (uncentred, lr >= 1e-4) and a flat plateau with no consistent gradient (centred or very low lr).

### Difficulty ladder inside the orientation family (centred, stack 3)

| Rung | Label interaction | lr | Val BA by update 200 / 400 / 600 | Test BA (512) |
|---|---|---:|---|---:|
| orientation_single: one Gabor, no cue, no glyph | rotation only | 1e-4 | 0.938 / 0.992 / 1.000 | 1.000 |
| orientation_single | rotation only | 3e-4, zero head | 0.988 / 0.996 / 1.000 | 1.000 |
| orientation_ring: four Gabors, ring cue, no glyph | location x rotation | 1e-4 | 0.477 / 0.535 / 0.992 | 1.000 |
| orientation_ring | location x rotation | 3e-4, zero head | 0.484 / 0.996 / 1.000 | 1.000 |
| orientation_sign: one Gabor, sign glyph | sign x rotation | 1e-4 | 0.484 / 0.500 / 0.500, then 0.988 at 1000 | 1.000 |
| orientation_cued: the real task | location x sign x rotation | 1e-4 | 0.531 / 0.496 / 0.473 | 0.535 (AUC 0.506) |
| orientation_cued, 1M episodes (scratch) | same | 1e-4 | chance at all 17 looks through update 8500 (544k episodes); gradient norm 0.005; stopped by me at 8500 to free the GPU | stopped |

The same model and recipe learn the first-order rung in under 40k episodes, the location-conditioned two-way rung by about 38k, and the sign-conditioned two-way rung by about 64k (slower, consistent with its XOR structure: neither the glyph sign nor the rotation sign correlates with the label alone). The three-way real task is not learned in 100k episodes, nor in the first 513k of the 1M-episode run.

### Curriculum: the real task from a two-way parent (rule: the parent learned from scratch)

| Parent (terminal checkpoint, test BA 1.000) | Child task | Val BA at update 200 / 400 / 1000 | Test BA (512) at 100k episodes |
|---|---|---|---:|
| orientation_sign, lr 1e-4 | orientation_cued D0 | 0.977 / 0.988 / 1.000 | 0.998 (AUC 1.000) |
| orientation_ring, lr 1e-4 | orientation_cued D0 | 0.984 / 0.988 / 0.996 | 1.000 |

From either two-way parent the real task is learned within 200 updates (12,800 episodes) and reaches test BA 0.998-1.000. The same model from scratch had not left chance after 544,000 episodes. Zero-shot at other delays, the sign-curriculum model gives D0 0.998, D4 0.545, D12 0.512, D24 0.520 (`zero_shot_delays_test.json`): the three-frame stack solves D0 by direct comparison and nothing is carried across a blank. Rung 2, continuation of the sign-curriculum model on mixed delays D0/4/12/24 (200k episodes, 3,125 updates cycling the four cells): test BA D0 0.996, D4 0.498, D12 0.512, D24 0.490. Per-cell training loss over the last 300 updates: D0 0.021, D4/D12/D24 0.693. D0 is retained and no delay is learned; the model finds no gradient toward carrying the cued angle through even four blank frames. The delay curve of this model is therefore a step: 1.0 at D0, chance at every delay. Rung 2 with a delay ladder (D1, D2, ...) is the obvious next test and was not run.

### Gradient signal-to-noise at initialisation (negative diagnostic)

`gradient_snr.py` measured the consistency of the loss gradient across 16 independent batches of 64 at initialisation, with a label-shuffle control isolating the label-informative part. The label-informative part has SNR 0.96-1.27 and mean pairwise cosine within noise of zero for every task, including the two rungs that were then learned within 600 updates. The label-independent part (predict the class prior) dominates the raw gradient (cosine 0.5-0.6 for motion, binding, Krauzlis). The probe therefore does not predict learnability and is recorded only to prevent re-deriving it.

## What this shows, what it does not show, and the next decision

Measured: the battery as specified is not learned from scratch by a standard CNN+GRU at batch 64 in 100k episodes under any of ten recipes, and the failure is optimisation, not information (experiment 25). The standard recipe fails by a first-step collapse that is independent of architecture: the v2 lineage's model shows the identical signature. Within the orientation family, learnability breaks between the two-way cue-conditioned rung (learned) and the three-way real task (not learned at this budget).

Not shown: whether the other families break at the same interaction order, whether the curriculum result holds across seeds, and whether larger batches would rescue the scratch run (untested).

Decision taken: a two-way rung (ring or sign) is the parent for the real task at D0, and the curriculum works from either (0.998-1.000 in 12.8k episodes). Delays are a second, separate break: nothing tested here learns to hold the cued angle across a blank.

Recommended next steps, in order: (1) a delay ladder from the D0 model (D1, D2, D4, ...), the same curriculum logic applied to retention; (2) the same three-rung ladder for motion, binding, recognition and Krauzlis, so every task has a from-scratch floor and a curriculum path; (3) a second seed of the orientation ladder and curriculum; (4) only then any architectural component, compared against this baseline on the same seeds.

## Evidence and closure

- Record runs: `WorkingMemory/PlainBaseline/runs/rung1_20260916_212434/` (`summary.md`, per-task `receipt.json`, `metrics.csv`, `validation.json`, `terminal.pt`, `best.pt`).
- Sweeps: `runs/sweep_orientation_20260916/`, `runs/sweep_orientation_lowlr_20260916/`, `runs/sweep_orientation_center_20260916/` (each with `summary.md`; killed runs have `metrics.csv` and `validation.json` only).
- Ladder: `runs/ladder_orientation_20260916/`. Long run: `runs/long_orientation_20260916/`. Curriculum: `runs/curriculum_orientation_20260916/` (receipts record the parent checkpoint hash). Delay curve: `runs/delay_orientation_20260916/`.
- Probe: `WorkingMemory/PlainBaseline/gradient_snr_stack3_center.json`.
- Source hashes are in each `receipt.json` (`stimuli.py` and `baseline.py` at run time).
