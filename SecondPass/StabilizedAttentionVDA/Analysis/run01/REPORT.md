# Frozen VDA analysis: checkpoint 16000

Top-left detection rose from 48.2% to 68.0% when bottom-right attention was inhibited during the change. Bottom-right detection fell from 87.0% to 6.5%.

The primary VDA-4 test uses 384 matched scenes per event/arm, a 100% bottom-right
cue, 18-degree changes, and manipulation only in frames 5–6. With bottom-right
inhibition −4, top-left detection improves by 19.79pp
(paired 95% interval 15.62 to24.22).
False alarms fall from 3.65% to1.82%.
Top-left/catch balanced accuracy improves from 72.27% to83.07%.
The reciprocal loss of bottom-right detection is -80.47pp.

This deliberately violates the perfectly predictive cue; such uncued changes
are absent from native 100%-validity training. Uncued physical changes are still
positive VDA labels. False alarms refer only to no-change catches.

131,616 inference presentations completed, including
8,064  primary, 4,608  native, 18,144  psychometric, 100,800 exploratory intervention
presentations. 90 attention groups contain every one of 7 model frames, three
individual examples each, and a 48-scene condition mean. Original checkpoint
and learned tensors unchanged; no training or threshold tuning occurred.

Eight predeclared directional comparisons use exact paired binary tests with
Holm correction. Full results: primary_hypothesis_tests.csv. All other effects
are exploratory; cluster-bootstrap intervals resample scenes, treating mirrored
geometries as repeated observations. Rendering adapters passed native equality;
zero and global-shift attention controls passed. Maps use the actual single
CLS-query row, projected without inventing an NxN attention matrix.

The model intervention is additive priority-logit bias, not biological current.
This is one trained checkpoint, not a training-seed replication. The 100×100 input
adaptation and different stimulus-cell sizes preclude a pure capacity comparison.

Open gallery/index.html for figures, interactive attention frames, GIFs and data.
