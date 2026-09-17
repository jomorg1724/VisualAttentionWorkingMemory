# Local frozen-core spatial-readout diagnostic

This post-hoc analysis asks whether a spatial output can decode cued
motion-duration evidence that is not exposed by a similarly sized global
pooling probe. It does **not** update or replace the deployed model and is not
a substitute for the independently active cloud training.

## Frozen source and common examples

- Frozen source: terminal motion-only checkpoint step 12200, after 30,400
  cumulative motion episodes. Its deployed output remains near 25% chance.
- Checkpoint SHA256:
  `7cec4c48c3da81b4d4935c65098b58974a38a8ff3a825dfe12e4d8d16e9a0e26`.
- Encoder, opponent traces, original biased pre-update attention, spatial E/I
  memory and comparator are loaded in evaluation mode with gradients disabled.
- A single extraction computes and caches the exact same float32 terminal
  `concat[H_T,R_T,C_T]` tensor for both probes.
- Existing `motion_duration_cued` examples cover D0/4/12/24. Separate
  train/validation/test streams use seeds 731091/731092/731093 and their
  corresponding repository split names. Re-instantiating the stream per delay
  pairs the underlying motion movie across delays. Base episodes, not repeated
  delay presentations, are the grouping unit.

## Probes and fitting

Both probes have four outputs and receive only train-derived per-channel
standardization:

1. **Pooled:** global mean and max over all 192 channels, linear 384→191,
   SiLU, linear 191→4 (74,303 parameters).
2. **Spatial:** the cloud candidate's single-task form: 1×1 192→96,
   GroupNorm/SiLU, padded 3×3 96→64, GroupNorm/SiLU, then independent 1×1
   selection and four-class evidence maps (74,373 parameters).

The 70-parameter (0.094%) difference is reported rather than hidden. Adam uses
LR 3e-4, weight decay 1e-4 and clipping at 1. Both see identical deterministic
minibatch orders. Validation looks at epochs 5/10/20/40 select mean per-delay
balanced accuracy, then mean AUC, then the earlier epoch. Test remains unread
until both selections are fixed.

The smoke extracted 64 delay-presentations in 14.925 seconds. The resulting
linear projection was comfortably below the remaining wall allowance, so the
production target was pinned without reduction at 512/128/256
independent base episodes per delay for train/validation/test. Thus each split
has four paired delay presentations per base episode. A class-stratified
paired bootstrap resamples base episodes and reports model-by-probe
differences. Chance accuracy/BA is 25%.

## Priority-map diagnostic

Test maps are saved for every presentation. Alignment is the priority mass and
peak-hit rate inside a 3×3 feature-grid neighborhood centered on the target
patch after linear 100→13 coordinate projection. Target coordinates are used
only after predictions are fixed; they never enter fitting. The 3×3 region has
uniform-map expectation 9/169=5.33%. This approximate receptive-field metric
is not pixel attribution and cannot establish causal selection.
