# Matched final-only versus time-separated feature access

Authorization: user said "continue with the experiment" after proposal to hold diagnostic supervision and readout capacity constant and compare final-state-only access with separate sample/probe activations.

## Question and scope

Does the prior successful auxiliary-supervised structured diagnostic require external storage of sample-time activations, or can the same diagnostic supervision/operations recover useful comparisons from a final representation alone?

Actual frozen terminal6760: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt`; SHA256 `1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841`. Native D0 orientation_cued and spatial_binding only. No main-model updates, task-law edits, curriculum, cloud or new deployed architecture.

## Matched contrasts

Primary: final spatial ConvGRU field. Secondary: post-ReLU256 readout. Early visual features are an explanatory reference reproducing prior early-access advantage. This is three layer-specific paired contrasts, not three production model arms. No other layer/architecture sweep.

For EACH layer, fit identical structured diagnostic components and calibration under two information-access conditions:
- final_only: sample-angle, probe-angle, cue-location and sign decoders all receive that layer's REPORT-time feature(s); no stored earlier activations can enter any prediction.
- time_separated: same dimension/crop/feature family, component supervision, scalers, fit split, regularization grid and relational calibration; sample decoder receives sample-time feature, probe decoder report-time feature, cue decoder native cue/query-time feature. This explicitly has external diagnostic memory.

Match parameter counts and selection opportunities exactly WITHIN each layer comparison, including cue decoder feature dimension. Prefer whole3136 ConvGRU field and whole256 readout for the primary/secondary component decoders so a restrictive crop does not masquerade as full-state inadequacy; early may use prior fixed local/glyph geometry, identically in each access condition. Document exact dimensions/counts. Across-layer capacity is NOT matched and cannot support a clean causal compression claim.

Use same auxiliary targets (actual axial cos2theta/sin2theta and cue location/sign) and same circular relational functions as FeatureDiagnostic. Disjoint first512 TRAIN examples for component fits and second512 for task-label calibration; validation256 selects regularization only. No actual target, sign, angle or label may enter operational prediction or select a feature ROI at evaluation. All four location decoders fitted, predicted cue does selection. Final_only implementation must be tested with early timesteps removed or corrupted: its predictions unchanged.

Reuse immutable prior train/validation feature artifacts and splits if valid to avoid redundant extraction. Do NOT select using prior test predictions. Freeze all fits/selections for all comparisons before extracting/scoring fresh512 test episodes PER TASK from independent unused seed namespaces. Same fresh episodes across all readouts and deployed head. Verify no base raster/ID overlap with previous train/validation/test. If a split artifact shape prevents reuse, document exact reason before additional extraction.

## Results and verification

Report paired held-out BA/AUC plus final_only versus time_separated paired gains and95% intervals at each layer, deployed baseline, instruction decoding, sample/probe axial errors relative to actual changes, and target/sign/location strata. Keep D0 stack3 caveat: samples remain in raw input stack, so this is not clean retention. A final-only success with auxiliary supervision is NOT native-label acquisition or a trained deployment. A failure of these probes is not proof of erasure. Reproduce prior structured design's inductive biases explicitly. A difference between within-layer access conditions tests external temporal access, not a biological memory mechanism.

Save source, pinned protocol, feature hashes, fit states, frozen selection receipt, per-trial predictions, summary/report and verification of unchanged main checkpoint/state/source hashes and exact native forward/extraction parity. Verify final_only lacks early-frame access and matched fit counts/dimensions. Don't overwrite earlier diagnostics.

## Finite execution

One researcher owns execution without repeated review gates. One local MPS extraction worker maximum, CPU threads≤2. A new nonrenewable1200-second wall cap starts before FIRST numerical diagnostic fit or accelerator profile/extraction (whichever earlier), covering CPU fitting, fresh-test extraction/scoring and reporting. Pin counts/method before scoring; reserve reporting time. No restart or automatic extension. Setup/syntax checks can precede numerical fitting; do not wait on process tracker alone (prior null exits were EOF while worker lived). Researcher returns actual executed results; parent verifies and journals.
