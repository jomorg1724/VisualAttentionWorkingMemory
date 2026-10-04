# Matched final-only versus time-separated access

**Executed frozen terminal6760 diagnostic, native D0 only.** Primary comparison: whole spatial ConvGRU3136. Secondary: whole post-ReLU256 readout. Early visual ROIs are an explanatory reference.

## Held-out decisions
512 fresh episodes per task, shared by all readouts and deployed head. BA/AUC intervals are paired label-stratified whole-episode bootstrap (2000 resamples); they condition on these fixed fits and one checkpoint, not training variability.

|Task|Layer/access|BA [95%]|AUC [95%]|
|---|---|---|---|
|orientation_cued|deployed|0.4922 [0.4492, 0.5371]|0.4912 [0.4397, 0.5407]|
|orientation_cued|gru_final_only|0.6309 [0.5859, 0.6699]|0.6589 [0.6086, 0.7059]|
|orientation_cued|gru_time_separated|0.6133 [0.5703, 0.6523]|0.6473 [0.5989, 0.6932]|
|orientation_cued|readout_final_only|0.4980 [0.4570, 0.5391]|0.4992 [0.4499, 0.5469]|
|orientation_cued|readout_time_separated|0.5117 [0.4688, 0.5547]|0.5086 [0.4588, 0.5569]|
|orientation_cued|early_final_only|0.9082 [0.8828, 0.9317]|0.9691 [0.9551, 0.9806]|
|orientation_cued|early_time_separated|0.9336 [0.9121, 0.9531]|0.9763 [0.9646, 0.9858]|
|spatial_binding|deployed|0.5000 [0.4551, 0.5410]|0.4955 [0.4449, 0.5457]|
|spatial_binding|gru_final_only|0.8574 [0.8281, 0.8848]|0.9289 [0.9063, 0.9495]|
|spatial_binding|gru_time_separated|0.8770 [0.8477, 0.9024]|0.9402 [0.9190, 0.9583]|
|spatial_binding|readout_final_only|0.6250 [0.5801, 0.6660]|0.6801 [0.6329, 0.7271]|
|spatial_binding|readout_time_separated|0.7324 [0.6934, 0.7695]|0.7763 [0.7337, 0.8172]|
|spatial_binding|early_final_only|1.0000 [1.0000, 1.0000]|1.0000 [1.0000, 1.0000]|
|spatial_binding|early_time_separated|1.0000 [1.0000, 1.0000]|1.0000 [1.0000, 1.0000]|

### Matched temporal-access effect
Positive differences favor external sample/cue-time access. All feature dimensions, decoder/calibrator parameter counts, example counts, supervision and four-alpha selection opportunities match within each pair.

|Task|Layer|BA gain [95%]|AUC gain [95%]|
|---|---|---|---|
|orientation_cued|gru|-0.0176 [-0.0527, +0.0156]|-0.0115 [-0.0426, +0.0185]|
|orientation_cued|readout|+0.0137 [-0.0450, +0.0742]|+0.0094 [-0.0579, +0.0751]|
|orientation_cued|early|+0.0254 [+0.0078, +0.0449]|+0.0072 [-0.0001, +0.0146]|
|spatial_binding|gru|+0.0195 [-0.0020, +0.0430]|+0.0114 [+0.0012, +0.0218]|
|spatial_binding|readout|+0.1074 [+0.0566, +0.1563]|+0.0962 [+0.0423, +0.1487]|
|spatial_binding|early|+0.0000 [+0.0000, +0.0000]|+0.0000 [+0.0000, +0.0000]|

## Measured interpretation
- **orientation_cued, primary whole-field:** final-only BA 0.6309; time-separated BA 0.6133; paired difference -0.0176 with 95% interval [-0.052734375, 0.015625]. Final-only versus deployed BA gain is +0.1387, interval [0.078125, 0.1953125].
- **spatial_binding, primary whole-field:** final-only BA 0.8574; time-separated BA 0.8770; paired difference +0.0195 with 95% interval [-0.001953125, 0.04296875]. Final-only versus deployed BA gain is +0.3574, interval [0.3046875, 0.412109375].
- An above-deployed final-only result establishes accessibility to this auxiliary-supervised circular-comparison diagnostic, not acquisition from native task labels or a trained deployed replacement. The time-separated contrast changes access privilege only within a layer. Failure of one finite linear component-decoding scheme does not prove erasure.
- Across-layer capacities and ROI geometries are unmatched: layerwise differences cannot establish causal compression or irreversible information loss. No new label-only arm or shuffled-label arm was authorized; the earlier control is not treated as a control on these new fits.

## Instruction and orientation precision
All four locations decoded independently; true target is used only for scoring, never input or ROI selection. Errors are axial degrees. Location/sign columns are balanced accuracy.

|Task|Layer/access|Sample error° (target / other)|Probe error° (target / other)|Cue location BA|Cue sign BA|
|---|---|---|---|---|---|
|orientation_cued|gru_final_only|5.49 (5.68 / 5.43)|21.66 (23.16 / 21.16)|1.0000|1.0000|
|orientation_cued|gru_time_separated|4.56 (4.73 / 4.50)|21.66 (23.16 / 21.16)|1.0000|1.0000|
|orientation_cued|readout_final_only|25.26 (25.07 / 25.32)|32.57 (33.22 / 32.35)|1.0000|0.9941|
|orientation_cued|readout_time_separated|14.34 (14.77 / 14.20)|32.57 (33.22 / 32.35)|1.0000|1.0000|
|orientation_cued|early_final_only|3.83 (3.62 / 3.91)|7.50 (7.28 / 7.57)|1.0000|1.0000|
|orientation_cued|early_time_separated|1.61 (1.56 / 1.63)|7.50 (7.28 / 7.57)|1.0000|1.0000|
|spatial_binding|gru_final_only|5.61 (5.70 / 5.58)|16.33 (16.76 / 16.19)|1.0000|not defined|
|spatial_binding|gru_time_separated|3.70 (3.65 / 3.71)|16.33 (16.76 / 16.19)|1.0000|not defined|
|spatial_binding|readout_final_only|19.42 (22.04 / 18.55)|25.40 (26.25 / 25.12)|1.0000|not defined|
|spatial_binding|readout_time_separated|9.88 (9.73 / 9.93)|25.40 (26.25 / 25.12)|1.0000|not defined|
|spatial_binding|early_final_only|2.43 (2.47 / 2.42)|3.32 (3.37 / 3.30)|1.0000|not defined|
|spatial_binding|early_time_separated|1.67 (1.72 / 1.66)|3.32 (3.37 / 3.30)|1.0000|not defined|

Orientation changes are 0/±15/±30/±45°; 7.5° is half the smallest nonzero separation. Binding target changes are 0/45/90° axially. `*_results.json` includes actual-change-stratified sample/probe errors, fractions below7.5°, per-location cued/uncued errors, instruction label strata, and decision target-change/sign/location/label strata. Predictions retain truth for scoring separately from the input-only prediction API.

## Matched capacity and fitting
|Task|Layer (each access condition)|Weights+biases|Scaler values|Selected fits|Alpha candidates|
|---|---|---:|---:|---:|---:|
|orientation_cued|gru|69050|62790|11|44|
|orientation_cued|readout|5690|5190|11|44|
|orientation_cued|early|45306|34502|11|44|
|spatial_binding|gru|62776|56518|10|40|
|spatial_binding|readout|5176|4678|10|40|
|spatial_binding|early|34488|23686|10|40|

- Whole ConvGRU3136 and readout256 enter every component decoder. Early angle inputs are the identical prior fixed32×5×5 pooled local ROI (800 values); early cue inputs concatenate four raw32×6×6 glyph ROIs and32×5×5 global pooled values (5408). No target-selected crop. The relational calibrator receives35 values in every arm.
- Each location/sample/probe predicts cos(2θ), sin(2θ). Cue predicts one-hot location and, for orientation, sign. Binding sign is the task constant +1, not a fitted or metadata-supplied instruction. Hard predicted cue/sign enter the exact prior circular relation: normalized axial dot/cross products, absolute/signed half-angle, sign-aligned angle and absolute cross; selected-target relationships, all-location relationships, predicted cue and sign feed a ridge task-label calibrator.
- Reused immutable train1024 and validation256 per task. TRAIN first512 fit components; TRAIN last512 fit calibration using out-of-component-fit predictions. The same validation256 choose α∈{0.1,1,10,100} independently for each component and calibrator, first grid entry wins ties. Train-only column standardization, feature-width-normalized exact dual ridge; no PCA, no train/validation refit. Float16 features promoted to float32; ridge algebra float64, saved weights/scalers float32.
- Both tasks and all12 structured models were fit, serialized and hashed before generating any fresh test episode. Test seeds and counts are pinned in protocol.json. No old test predictions are loaded; old metadata/hashes are read only for exclusion. Fresh1024 episode IDs and full-raster hashes are disjoint from all3584 previous FeatureDiagnostic train/validation/test episodes and each other.
- final_only uses report features for every decoder. time_separated uses sample2, report probe, orientation cue0 or binding query3. Stored earlier activations are external diagnostic memory. Six actual final-only test predictions and every intermediate component were bitwise invariant after earlier timesteps were removed and separately filled with NaNs; the input bundle contained no labels, logits, metadata or other layers.

## Verification and limits
- Main checkpoint, saved source dependencies, and all prior diagnostic files unchanged: True. Main-model state tensors bitwise unchanged: True. Native direct/extraction wrapper maximum error: 0.0. Zero main-model optimizer updates; eval/no-grad only.
- Independent artifact replay reopens all12 fit states and test features, recomputes component outputs, decisions, BA/AUC and paired bootstrap intervals, rechecks capacities, fit separation, invariance and hashes. See independent_verification.json; completion.json is written only after replay and this report finish.
- Single local worker; CPU≤2, interop1; MPS used only for fresh frozen test extraction. One immutable1200s wall cap begins before the first numerical fit and includes extraction, scoring, replay and reporting. Exact elapsed time is in completion.json. No restart, cloud use, model update or architecture experiment.
- **D0 stack3 is not clean retention.** Orientation report3 contains sample1/sample2/probe; binding report4 contains sample2/query3/probe. Removing early *diagnostic feature access* is not removing those samples from the native input stack or recurrent computation. No long-delay memory, biology, attention allocation, cue-validity benefit, inhibition or microstimulation conclusion is measured here.
- Bootstrap intervals are descriptive, uncorrected for multiple secondary comparisons and do not model native balancing-queue dependence. At a perfect score a nonparametric bootstrap degenerates; saved accuracy Wilson intervals retain finite-sample uncertainty. Ridge outputs are decision scores (threshold0.5), not calibrated probabilities.

## Artifacts and reproduction
`run.py`, `test_access.py`, `verify.py`, `protocol.json`, `prior_artifacts.json`, `models/*.pt`, `features/*`, `predictions/*`, `*_results.json`, `results.json`, `capacity.json`, `selection_frozen.json`, `verification.json`, `independent_verification.json`, `budget.json`, `completion.json`, `summary.csv`, `progress.jsonl`.

Run from repo root: `/tmp/vawm-task-suite-venv/bin/python -B -m SecondPass.SpatialReadout.TemporalAccessDiagnostic.run --run`. Existing budget deliberately prevents rerunning; a separately authorized new output directory and cap would be required. `--precheck` and `test_access` do CPU setup/routing checks only. Replay is executable via `verify` only while the same cap remains active; no renewal. Prior artifacts and LabJournal were not edited.
