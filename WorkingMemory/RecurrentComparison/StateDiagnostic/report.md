# Information remains accessible; fit the readout first

The frozen E/I model's final firing-rate state retained substantial information about early motion. A newly fitted **linear** diagnostic readout improved held-out duration decisions by **9.47 percentage points**. The nonlinear probe provided no observed advantage. These findings favor a targeted readout-fitting follow-up before altering recurrent retention or adding a slow excitatory state.

| Output on the same1024 held-out movies | Balanced accuracy | Change from existing output, paired95% CI | Raw-score macro OVR-AUC |
|---|---:|---:|---:|
| Existing E/I output |68.65%|—|0.8959|
| Linear ridge, sensory+r |78.13%|+9.47pp [+6.84,+12.11]|0.9293|
| MLP64, sensory+r |77.64%|+8.98pp [+6.15,+11.82]|0.9141|

The diagnostic output receives existing sensory128 features and final firing rates r256. It never reads adaptation state a. Thus the rescue is available through an ordinary output route, rather than requiring a decoder to read intrinsic adaptation variables. The original output factors a linear function through r→128, adds sensory128, then applies the task head; the probe concatenates the two fields. Improvement may reflect better fitting of this learned projection/fusion, not just a class-bias correction. This experiment does not establish that changing the architecture is necessary.

Early evidence was measured as the first four directional counts, using three independent count coordinates. Within-pair prediction differences remove identical suffix cues. Paired-difference R² was:

| Frozen feature | Linear count R² (clustered95% CI) |
|---|---:|
| r after transition4 |0.875 [0.860,0.890]|
| r at final report |0.800 [0.782,0.818]|
| [r,a] at final report |0.827 [0.810,0.843]|
| Existing sensory pathway at final report |0.002 [−0.002,0.005]|

The modest nonlinear count probe yielded0.873 at the prefix and0.800 at the final report using[r,a], offering no improvement over the corresponding linear probe. Final r therefore carries useful accessible evidence; adaptation is not its sole repository. The decline from prefix to final linear r decoding was−0.075 R² (CI[−0.089,−0.061]), but this is not a causal estimate of memory leak: transitions5–6 vary and may correlate with the first four counts under these selected schedules. The final sensory pathway includes opponent traces and must not be described as a current-frame-only representation.

The frozen checkpoint was selected step5000 after40,000 prior training episodes. We collected3072 fresh probe-training,768 validation and1024 held-out movies. Entire canonical direction templates and all cyclic rotations were disjoint between splits. Each pair shared its last two directions, identical last two visual evidence rasters, instruction and report; different early histories were regenerated as continuous dot movies with restored nuisance RNG. The held-out set contained128 canonical pair-template clusters, each with both partners and four rotations. Bootstrap intervals resampled these128 clusters, keeping all eight movies together; they do not treat1024 movies or multiple probe methods as independent templates.

Train-only standardization and validation-only selection preceded locked held-out collection/scoring. Ridge selected among three predeclared mean-MSE regularizers; the fixed MLP64 trained150 epochs with validation selection every10. All retained selections/traces are in `fit_selection.json`. The MLP duration fit selected epoch20; later validation loss worsened, so no test-based epoch choice or extra architecture was added. AUC here uses raw class scores/logits and should not be numerically equated with earlier reports that used softmax scores.

This is a controlled suffix-matched schedule distribution, not a fresh confirmation of performance on the original random six-cell task stream. It tests one trained checkpoint and a limited family of probes. Successful readout rescue demonstrates usable accessibility on these held-out templates; weak probes would not prove information erasure. No unique leak/adaptation mechanism or model-wide failure is claimed.

**Recommended single follow-up:** freeze sensory and recurrent dynamics; refit the existing memory-output projection and appropriate task heads on fresh standard-task examples, then compare against the unchanged parent on fresh standard held-out cells. Retain the original architecture and protect orientation/anchor performance when fitting the shared projection. A cheaper frozen-feature/output fit is appropriate; this result does not require a slow-E state, a nonlinear adapter, or40,000 full-BPTT updates. The next training budget and exact stream must be pinned separately before launch.

Execution finished in247.40seconds through the successful supervisor receipt, inside the same1200-second allowance. A CPU duration-loss attempt initially rejected Windows int32 labels; explicit int64 targets fixed it. All first-attempt files are preserved in `attempt_1`, and saved training/validation features were reused without repeating GPU collection. Deterministic CPU probe fits were replayed because their full selection traces had not yet been committed; retained artifacts record the completed protocol, while the completion receipt distinguishes actual repeated fitting. The original E/I parameters received zero optimizer steps. Frozen neural extraction and MLP probes used fp32; offline ridge fits used NumPy/SciPy float64. Both used two CPU threads; GPU extraction was sequential on the local RTX3070 Laptop. No cloud action occurred.

See `summary.json`, `count_uncertainty.json`, `heldout_predictions.npz`, split feature/metadata files, source/checkpoint hashes in `config.json`, and `completion_receipt.json`. The independent reviewer agreed with readout fitting and requested no further diagnostic.
