# Probe adequacy diagnostic — selected2297 and fresh random encoder

Fresh train/validation/test; one original native movie per independent group. Frozen analysis-only models. No deployed change.

Train600 / validation150 / test300; test events258, catches42. All72 readouts frozen before test generation. No prior movies reused.

## Decision: this probe family is not validated as an information-loss assay

**Measured probe limitation, not neural erasure.** On the same fresh movies, endpoint image-only optical flow localized the changed side at BA **0.740 [0.684,0.792]**, while matched ridge readouts on the **raw endpoint pixels themselves** achieved only **0.500–0.512**. Endpoint flow exceeded unprojected pixel ridge by **+0.240 [0.184,0.292]**. A negative result from this low-effective-capacity ridge family is therefore not diagnostic of absent motion-change information—even before a CNN is involved. This supports an adequacy concern about interpreting the earlier neural negatives, but does not prove that their exact older probe basis would fail: the new df-controlled polynomial kernel is not an exact replication.

**Removing projection did not rescue these fits.** Trained-CNN event-side BA was0.499/0.508 with projected linear/quadratic readouts and0.500/0.500 unprojected. Unprojected-minus-projected gains were **+0.001 [−0.053,+0.060]** linear and **−0.008 [−0.043,+0.021]** quadratic. Thus projection is not established as the sole problem; underfitting, insufficient sample support, unsuitable motion/comparison inductive bias, ROI sampling and threshold calibration remain unresolved. This finite matrix does not identify which one is responsible.

**Trained versus random did not separate reliably.** Projected trained-minus-random side-BA gains were+0.022 [−0.060,+0.103] linear and+0.036 [−0.031,+0.106] quadratic; both unprojected classifiers were constant decisions. This is failure to distinguish one trained encoder from one freshly initialized same-architecture control under these probes—not evidence of equivalent encoding or trained destruction. Trained unprojected linear side AUC was0.558 [0.494,0.633], a weak unresolved trend, not a rescue.

**Direct signed-change readouts largely shrank toward zero.** Across real-label ridge arms, changed-patch MAE was26.84–27.10°, versus27.02° for always-zero. Endpoint optical flow gave21.04° and full-history flow7.05°. Catch error and output variability are separately tabulated in [DETAILS.md](DETAILS.md); small all-patch error is not successful event recovery. Any-event BA remained0.476–0.526 for real-label ridge arms. Full/endpoint flow event BA was0.871/0.632, with210/258 and154/258 event hits, and3/42 and14/42 catch false positives respectively. Shuffled controls show no consistent real-label rescue; no post-test winner was selected.

**Temporal comparison remains unlocalized.** Both endpoints were explicitly supplied to every matched readout, so these failures cannot be attributed only to lack of endpoint access. But known signal in endpoint pixels plus failed raw-pixel and neural ridge fits prevents inferring an encoding lesion or a specific neural comparison failure. Full-history flow side BA0.946 [0.917,0.971] shows additional usable temporal evidence for that privileged observer, not a deployed-head remedy. No optical-flow deployment, architecture fix or change to teaching is recommended or authorized here.

Intervals are descriptive and unadjusted across the fixed exploratory matrix. BA intervals[0.500,0.500] and gain intervals[0,0] arise from constant decisions and empirical bootstrap degeneracy, **not certainty or proof of equivalence**; AUC intervals and raw confusion counts retain the uncertainty. The original native target-report task/stimuli were unchanged, but these analyses score physical event/side/change targets, not successful cue-conditioned reporting. No cueing, inhibition, microstimulation or biological circuit claim is made.

## Matched readouts
BA [95% grouped bootstrap CI]; signed-change MAE is on actual changed patches (degrees).
|Representation|Projection|Degree|Event BA|Side BA|Side AUC|Changed MAE|Side shuffled BA|
|---|---|---:|---:|---:|---:|---:|---:|
|pixels|projected|1|0.506 [0.431,0.575]|0.512 [0.483,0.546]|0.536|27.10|0.496 [0.472,0.519]|
|pixels|projected|2|0.520 [0.455,0.588]|0.500 [0.500,0.500]|0.541|27.04|0.500 [0.500,0.500]|
|pixels|unprojected|1|0.503 [0.468,0.548]|0.500 [0.500,0.500]|0.477|27.01|0.500 [0.500,0.500]|
|pixels|unprojected|2|0.500 [0.500,0.500]|0.500 [0.500,0.500]|0.475|27.04|0.500 [0.500,0.500]|
|trained|projected|1|0.493 [0.450,0.533]|0.499 [0.440,0.553]|0.527|26.84|0.493 [0.465,0.527]|
|trained|projected|2|0.476 [0.424,0.517]|0.508 [0.479,0.543]|0.500|26.93|0.478 [0.434,0.518]|
|trained|unprojected|1|0.499 [0.451,0.561]|0.500 [0.500,0.500]|0.558|26.96|0.503 [0.470,0.541]|
|trained|unprojected|2|0.499 [0.451,0.561]|0.500 [0.500,0.500]|0.555|26.97|0.500 [0.500,0.500]|
|random|projected|1|0.486 [0.412,0.561]|0.477 [0.413,0.541]|0.481|27.05|0.503 [0.444,0.561]|
|random|projected|2|0.512 [0.448,0.560]|0.473 [0.410,0.537]|0.472|27.03|0.529 [0.477,0.581]|
|random|unprojected|1|0.515 [0.485,0.539]|0.500 [0.500,0.500]|0.516|26.96|0.485 [0.455,0.512]|
|random|unprojected|2|0.526 [0.456,0.591]|0.500 [0.500,0.500]|0.519|26.98|0.500 [0.500,0.500]|

## Positive controls and zero-change baseline
|Image observer|Event BA|Side BA|Changed MAE|Catch MAE|
|---|---:|---:|---:|---:|
|full|0.871 [0.823,0.911]|0.946 [0.917,0.971]|7.05|6.18|
|endpoint|0.632 [0.551,0.704]|0.740 [0.684,0.792]|21.04|20.66|
|Always zero|—|—|27.02|2.54|

## Capacity and protocol
- Twelve feature/readout arms × three targets × real/shuffled training targets =72 fitted predictors. Frozen selected2297 and independently constructed random architecture share identical input movies, preprocessing, eval state, fixed CNN ROIs and projection matrix. Random was never loaded from trained weights and had no optimizer updates. The raw-pixel arm has no weight state.
- Raw grayscale windows:6348 coordinates; CNN:2304. RGB planes are exactly duplicates. Projection:32 per endpoint,64 total. Linear ridge and degree-two polynomial ridge are separately matched. The unprojected quadratic maps have millions of implicit coefficients, NOT modest nominal capacity. Their sample-effective ridge df is capped/selected from16/32/48, just like every other arm; intercept adds one. Selection.json records exact dimensions, coefficients-equivalent, alpha and df for every fit. Matching df does not equate inductive bias or representation geometry.
- Train-only coordinate scaling and train-centered kernel. K=dot/d (linear) or dot/d+(dot/d)^2 (quadratic), so off-diagonal monomials receive the standard sqrt2 weighting. This is not numerically identical to the earlier explicit unweighted triangle feature basis. Projected results are a controlled new diagnostic, not exact replication of the old probes.
- Validation-only df/decision-threshold selection. Event/side threshold grid0.05..0.95; signed-change target cos/sin of each patch actual circular change, validation minimum all-patch circular error. Side fit/evaluation is restricted to event groups; no event/side/cue truth is passed to feature extraction or predictions. No true cue chooses any ROI. All-patch selection can favor zero shrinkage; changed-patch and catch errors must be read separately.
- CNN early layer is before first KDA, so native early endpoint extraction can omit downstream recurrent compute; exact forward-hook and native-logit parity checked for trained AND random at B12/B20/B28. Endpoint inputs are the same native centered three-frame stacks ending B+7/B+15. External storage of both endpoints is an analyst privilege, not native final-state access.
- Raw windows cover both fixed23×23 dot ROIs; CNN samples3×3 neighborhoods at stride4. CNN GroupNorm additionally depends on the full image. This is endpoint-time matched, not an exact spatial-information/nominal-dimension match. Neither whole-map decoding nor CNN ROI ablation is tested.
- Pixel centroid-flow positive control is image-only, with known geometry/timing, nonlinear isolated-dot matching, and full-history or last-two-transition access. Full-history is privileged, not a matched ridge arm. No deployment is proposed.
- Independent fresh streams, native cycle/renderer unchanged,600/150/300 independent movies. Same held-out groups and500 resamples across arms. Intervals conditional on fixed fitted probes, one trained checkpoint and one random seed; exploratory hypothesis follow-up, no multiple-comparison correction. Grouped uncertainty does not measure training-seed variability.
- Saved raw endpoint windows, both CNN features, dot-angle schedules, flow sums, metadata/hash identities, random state, projections, fitted dual coefficients/scalers, predictions and fit audit support replay. Original files unchanged. The absolute1800s cap includes profile through report/replay, threads≤2, one worker, no cloud.

## Verification
{
  "exact_prediction_replay_max_abs": 0.0,
  "normal_equation_max_relative_residual": 3.648969395474919e-15,
  "train_only_scalers_exact": true,
  "all_splits_and_prior_movies_disjoint": true,
  "independent_groups_total": 1050,
  "checkpoint_and_both_states_unchanged": true,
  "renderer_and_wrapper_parity": "profile.json, both encoders, all three durations, bit exact",
  "pixel_flow_replay_exact": true,
  "example_raster_replay_exact": true,
  "all_choices_frozen_before_fresh_test": true,
  "torch_threads": 2,
  "interop_threads": 1,
  "workers": 1,
  "no_optimizers": true,
  "no_cloud": true,
  "extract_seconds": {
    "test": 5.517329216003418
  },
  "elapsed_seconds": 367.81493616104126,
  "within_cap": true
}

## Reproduction and closeout

`replay.py` performs no fitting and reproduces saved predictions/metrics under the same original deadline; it refuses execution after expiry. `test_contract.py` checks kernel algebra and actual feature/projection shape. Re-extraction/refitting is not automatically authorized. Exact dimensions, all classifier confusions/AUC intervals, all shuffled fits, signed-change errors/gains, and failure evidence: [DETAILS.md](DETAILS.md). `independent_replay.json` records the independent process audit. `final_receipt.json` records final elapsed time and artifact hashes after report/journal completion.

**Final closeout:** report, journal, source archive, all fits and independent no-fit replay completed within the original1800-second cap; elapsed at closeout1272.3s. No automatic extension or further fitting. Exact final elapsed and hashes are in `final_receipt.json`.
