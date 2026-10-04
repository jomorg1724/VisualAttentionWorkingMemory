# KDA results graphics

Generated exclusively from committed metrics, test receipts and analysis JSON. The architecture and trial timelines are labeled implementation schematics, not observations. No training or inference was run.

## Open
- `index.html`: offline searchable gallery, full-resolution SVGs, PNG/PDF downloads, source links.
- `KDA_results_atlas.pdf`: one figure per page.
- `01_...` through `29_...`: each figure in PNG, SVG and PDF.
- `data/`: normalized CSV tables for every plotted numerical result.
- `manifest.json`: inventory, source paths and SHA256 hashes.
- `build_figures.py`: reproducible plotting script; needs matplotlib, pandas, numpy, scipy.

## What cannot be plotted honestly from this checkout
The source analysis script saves gates_D*_scale*.npz arrays containing per-trial spatial maps and signed implicit coefficients. They are not in the checkout, and neither are trained model weights. A spatial movie or per-trial heatmap therefore cannot be recovered here. The temporal heatmaps in this gallery display the stored region averages only.

## Interpretation limits
- 512 trials per psychometric point; saved 95% intervals are trial-bootstrap intervals. The bootstrap is degenerate at perfect accuracy; it does not quantify uncertainty about all future trials.
- One local seed for the probes and psychometrics; cloud comparison shows both original model seeds separately.
- Gaussian curve parameters are replayed as saved, not refitted. The saved slope is inverse spread, not a derivative. Fit-parameter uncertainty was not saved. Some fitted thresholds lie below the smallest tested magnitude.
- The all-ceiling delay curve cannot identify a forgetting time constant. The degenerate saved tau is not interpreted.
- KDA coefficients are signed and unnormalized before aggregation. Stored attention summaries average their absolute magnitudes across trials, regions and heads; they do not give attention probabilities or final-decision attribution.
- R2 probes are separate, same-time, unregularized least-squares fits with 64 fit / 64 held-out trials. Post-probe values may reflect sample/probe correlations; pre-probe values are identified explicitly. No cross-time decoding or angular error is inferred.
- Those probes have 257 features including the intercept and are underdetermined. No repeated-split or shuffle controls are present. The first two nominal blank frames still contain sample frames in the rolling three-frame input; they are not pure recurrent retention measurements.
- Reset interventions affect all KDA states, not the downstream GRU. Clamping does not prove blank writes are universally irrelevant.
- Beta=0 disables both writes and delta correction; alpha=1 removes diagonal decay, not all forgetting. Same-magnitude distractor mode makes all uncued rotations nonzero; the original pattern also uses a shared nonzero magnitude but permits unchanged Gabors.
- Cue jitter is a uniform integer offset per axis, sampled per cue-bearing frame, bounded by the stated value. Distraction varies uncued Gabors, not delay-inserted distractors.
- No significant improvement with delay, lossless memory, causal readout locus, or general memory capacity is claimed from these plots.
