# Frozen suite analysis

This analysis preserves the selected suite checkpoint at update 54,808. It uses the shared visual/memory backbone and its 13 task-selected output heads, without an optimizer, parameter updates, threshold fitting, or checkpoint reselection.

The user requested psychometrics across the suite, Krauzlis and cued-orientation attention interventions, and projected attention maps at every timestep. The clarification explicitly includes both target stimulation and distractor inhibition as candidate benefits, as well as the reverse costs.

## Scope

- 33,568 psychometric presentations, including native memory load/delay conditions and extended evidence strengths.
- 83,168 matched intervention presentations: 39 Krauzlis arms and 41 cued-orientation arms, with dose, site, phase and normalization controls.
- 8,960 native atlas presentations: 256 per each of 35 conditions, grouped by relevant stimulus strength, cue location/sign, event, or recognition status.
- One finite 14,400-second local inference allocation; no training or cloud use.

The actual checkpoint contains one CLS query over 100 patches, not a full patch-query self-attention matrix. The requested column average over queries therefore equals the recorded attention row. The report uses its actual normalized weights, projected to the 10-by-10 spatial grid with nearest-neighbor expansion. Color normalization is fixed across tasks and timesteps.

## Entry points

`experiment.py` writes raw outcomes, attention mass by location, and atlas arrays. `supervise.py` launches the worker and an independent owned-process deadline guard. `report.py` produces PNG/PDF figures, raw tables, GIFs, every-timestep contact sheets and an offline HTML gallery. `audit.py` independently checks counts, pairing, control invariance, checkpoint hashes and every exported sprite pixel.

The generated report is `run01/gallery/index.html`. Runtime receipts and the prospective protocol are under `run01`. Model source and original checkpoints are never edited. Rendering adapters have bitwise native-equivalence checks. Strength zero is an explicitly ambiguous scheduled-label control; native ring orientation rejects zero and it is omitted there.

## Interpretation

These are priority-logit interventions in a computational model, not literal electrical stimulation or biological inactivation. The four primary benefit comparisons per task use paired scene-level differences and one-sided sign-flip tests, with Holm correction across the eight primary tests. All other dose/timing/location results are exploratory, with pointwise paired bootstrap intervals. One trained checkpoint does not establish reproducibility across training seeds.

Photographs differ across recognition trials. Individual images are paired only with their own maps; the condition mean map is separately labeled. A mean-image mode is available but explicitly shows the blurred mean of different photographs. Every original model frame is retained in floating point in the atlas arrays; browser images are 8-bit displays.

## Completed findings

All 125,696 inference presentations completed and passed the independent audit. The report contains 77 attention groups, 1,376 group timesteps, 179 contact sheets, 13 GIFs and 12 PNG/PDF scientific figures. Both requested benefit directions were tested; none of the eight primary tests established an accuracy gain after Holm correction. Strong impairment effects occurred in both tasks. See the report opening findings and raw paired tables for magnitudes and uncertainty. The orientation-conflict subgroup uses class-stratified scene resampling because finer post-hoc cells are sparse. No model weights changed and no training is queued.
