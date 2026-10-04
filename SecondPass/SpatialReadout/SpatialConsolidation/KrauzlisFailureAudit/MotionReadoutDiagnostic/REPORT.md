# Motion-specific readout diagnostic — frozen selected2297

## Decision and scope

**Analysis-decoder training only.** The original native task, stimuli, curriculum and deployed model were not modified. These readouts estimate physical motion events and changed side; this is not successful acquisition of the complete cue-conditioned task. No cloud or deployed-model optimizer was used.

Raw-pixel adequacy gate: **PASS**, fixed before fitting and before independent test generation. Train600 and validation150 are exact hash-verified rerenders of the prior adequacy train/validation movies; test300 is newly generated, disjoint from every prior saved movie.

Validation pixels: side BA **0.984 [0.961, 1.000]**, event BA **0.988 [0.976, 1.000]**, event AUC **0.996 [0.987, 1.000]**. All seven prespecified criteria were evaluated without consulting test results.

Fresh test pixels: side BA **0.988 [0.974, 1.000]**, event BA **0.893 [0.830, 0.946]**. Frozen early-CNN maps with the same displacement/comparison family: side BA **0.531 [0.469, 0.592]**, event BA **0.500 [0.500, 0.500]**.

**The pixel control succeeded; the identical motion-comparison family did not transfer to frozen early-CNN maps.** The neural event classifier reports an event on every test movie, including every catch, and side decoding is not reliably above chance. This narrows the failure to the tested substrate/readout pairing rather than repeating the old raw-pixel probe failure. It does **not** prove that the CNN erased motion information: brightness/feature constancy, stride, nonlinear feature transformations, channel weighting and spatial support are not equivalent across substrates. No random encoder was included, so trained-weight effects cannot be separated from architectural effects.

**Next decision:** retain the frozen deployed observer and unchanged teaching. Close this bounded comparison; do not install this analysis decoder or infer an encoder lesion. If the neural readout underperforms despite the pixel control, the unresolved question is whether the frozen representation preserves usable motion under a representation-appropriate, independently calibrated comparison—not whether an external-history decoder already remedies the deployed task. Any further diagnostic requires a separately justified design and authorization.

## Fresh paired test

Independent movies: 300; events 258, catches 42; event-left 130, event-right 128. Native26/28-degree magnitudes, baseline12/20/28 transitions; original target/foil/catch sampling and renderer.

|Arm|Event BA [95% CI]|Event AUC [95% CI]|Event-side BA [95% CI]|Event hits|Catch false positives|
|---|---|---|---|---|---|
|pixels|0.893 [0.830, 0.946]|0.975 [0.955, 0.990]|0.988 [0.974, 1.000]|246/258|7/42|
|pixels/shuffled|0.502 [0.500, 0.506]|0.314 [0.235, 0.397]|0.201 [0.153, 0.254]|1/258|0/42|
|trained|0.500 [0.500, 0.500]|0.523 [0.418, 0.623]|0.531 [0.469, 0.592]|258/258|42/42|
|trained/shuffled|0.502 [0.441, 0.558]|0.471 [0.373, 0.563]|0.481 [0.416, 0.538]|44/258|7/42|
|flow/full|0.863 [0.804, 0.914]|0.930 [0.896, 0.957]|0.957 [0.929, 0.981]|218/258|5/42|
|flow/endpoint3|0.643 [0.560, 0.721]|0.652 [0.548, 0.747]|0.740 [0.684, 0.790]|166/258|15/42|
|flow/window5|0.681 [0.604, 0.760]|0.762 [0.682, 0.834]|0.806 [0.754, 0.854]|198/258|17/42|

## Paired differences
|Comparison|Event BA gain [95% CI]|Event AUC gain [95% CI]|Side BA gain [95% CI]|
|---|---|---|---|
|pixels MINUS pixels/shuffled|0.391 [0.327, 0.444]|0.661 [0.578, 0.740]|0.787 [0.730, 0.838]|
|flow/window5 MINUS pixels|-0.212 [-0.306, -0.115]|-0.213 [-0.289, -0.142]|-0.182 [-0.232, -0.132]|
|trained MINUS trained/shuffled|-0.002 [-0.058, 0.059]|0.052 [-0.081, 0.191]|0.051 [-0.041, 0.146]|
|pixels MINUS trained|0.393 [0.330, 0.446]|0.452 [0.353, 0.556]|0.457 [0.394, 0.517]|

## Native event-type breakdown
Counts are predicted **any physical event**, not cue-conditioned positive reports.
|Arm|Target-event detections|Foil-event detections|Catch false positives|
|---|---|---|---|
|pixels|165/171|81/87|7/42|
|pixels/shuffled|1/171|0/87|0/42|
|trained|171/171|87/87|42/42|
|trained/shuffled|33/171|11/87|7/42|
|flow/full|145/171|73/87|5/42|
|flow/endpoint3|108/171|58/87|15/42|
|flow/window5|130/171|68/87|17/42|

## Unchanged native-head and optical-flow target-report reference

Frozen deployed head target-report BA: **0.500 [0.500, 0.500]**, evaluated on these same unchanged movies. Physical event/side readout scores above are different targets and cannot be directly called a deployed-head improvement.

|Image observer + image-decoded cue|Target-report BA [95% CI]|
|---|---|
|full|0.883 [0.849, 0.916]|
|endpoint3|0.642 [0.589, 0.694]|
|window5|0.745 [0.694, 0.795]|

Flow uses the previously implemented sparse isolated-dot centroid matching, with known geometry, native dot mass/speed and external time windows. Target-report flow uses the **image-decoded cue**, with the already frozen any-event threshold; it is not a separately optimized target-report comparator. Full history is temporally privileged. Endpoint3 preserves the old reference; window5 matches this diagnostic’s total raw-frame access. No flow parameters or rendering were changed.

## Prespecified adequacy and minimal learned family

ALL required on rawpixel validation: sideBA>=0.65 with bootstrap95%lower>0.50; eventBA>=0.60; eventAUC>=0.65 with bootstrap95%lower>0.50; paired real-minus-shuffled sideBA AND eventAUC bootstrap95%lower>0. No neural-feature extraction/fit unless gate passes. If fail, stop localization; fresh pixel-only test may characterize failure, never override gate.

- Three iterations of robust multichannel first-order displacement least-squares (small-displacement SSD matching/Lucas–Kanade), fitting dx/dy from image/map spatial derivatives and temporal differences. Not the sparse optical-flow reference, and not a raw-pixel ridge baseline. Robust weights reduce replacement/boundary residuals; no dot IDs, angle truth, cue truth, event truth or changed-side truth enters descriptors or prediction APIs.
- Each fixed patch independently yields eight features:1−direction cosine, absolute direction cross product, normalized-direction distance, relative speed difference, summed speeds, two normalized fit residuals and their absolute difference. A patch-shared balanced logistic decoder learns physical per-patch-change supervision from train only:8 standardized coefficients+intercept, fixedL2=1, fresh zero initialization and a single convex fit. Neural fit is separately fresh, not inherited from pixel or deployed weights. This is real **analysis-decoder training**, not model-free scoring.
- One matched group-shuffled-supervision fit per representation, same features, fit budget and validation event-threshold rule. Complete two-patch label vectors are shuffled between independent movies. The side decision is right-probability minus left-probability>0; no true-event mask is used by predictions. Event mask is used only to score side on actual events. Event threshold0.05..0.95 is validation-only; no hyperparameter sweep or validation refit.
- Raw input:three maps per phase, each a centered native three-frame grayscale stack, fixed29×29 patches. Corresponding CNN input:three64-channel7×7 maps per phase, from native full25×25 maps before first KDA, stride4; neural displacement multiplied by4 to express pixel units. Same eight descriptor dimensions and nine learned coefficients; same600 independent train groups/1200 patch rows. CNN receptive fields/global GroupNorm and spatial resolution/channel count remain unavoidable mismatches, not matched information content.
- **Temporal access differs from the earlier adequacy report:** native stacks ending B+5/B+6/B+7 and B+13/B+14/B+15 expose five raw frames per phase. Both current arms receive the same six native stack times; external storage of both phases is privileged. The old saved3×3 CNN ROIs were not reused. This is not final-state-only decoding and cannot establish a deployed output-head remedy.
- Both fixed spatial sites are always decoded. True cue is never used to choose a patch or comparator input. Native means are90° apart; patch-shared fitting does not use the other patch’s angle as an input, although event/side aggregation compares the two predicted patch probabilities.
-1000 independent-movie bootstrap resamples, paired across arms,95% percentile intervals, conditional on one fitted decoder per condition and one frozen checkpoint. These intervals do not include training-sample/seed variation, representation-selection uncertainty or multiplicity correction. The validation gate is a pragmatic bounded go/no-go check, not a formal population test.
- The single raw-pixel shuffled-supervision fit is strongly **below chance** for test side (BA0.201), not a clean chance-level null. A random training-label alignment can reverse the sign of a compact informative descriptor; this seeded control is not a many-permutation significance test. We retain it rather than flipping its scores or choosing a different shuffle. Raw-pixel adequacy also required absolute side/event performance and intervals above chance, independently of the gain over this shuffle.
- A near-ceiling empirical interval ending1 is not certainty. Constant shuffled decisions can yield BA0.5 with a degenerate bootstrap interval; this is not evidence of equivalence. AUCs, confusion counts, native event/catch denominators and paired gains are retained in metrics.json.

## Validation results

|Arm|Event BA [95% CI]|Event AUC [95% CI]|Event-side BA [95% CI]|Event hits|Catch false positives|
|---|---|---|---|---|---|
|pixels|0.988 [0.976, 1.000]|0.996 [0.987, 1.000]|0.984 [0.961, 1.000]|126/129|0/21|
|pixels/shuffled|0.508 [0.500, 0.519]|0.399 [0.278, 0.522]|0.248 [0.175, 0.316]|2/129|0/21|
|trained|0.500 [0.500, 0.500]|0.415 [0.277, 0.553]|0.490 [0.406, 0.572]|129/129|21/21|
|trained/shuffled|0.526 [0.444, 0.588]|0.475 [0.344, 0.613]|0.534 [0.444, 0.624]|19/129|2/21|

## Verification, recovery and artifacts

Checkpoint: `/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run/validation_checkpoint_002297.pt`; SHA256 `cda0058e75232befdd7e0d96814d182e8a9b9fe211357c6bad5b00b155eba02e`. State tensors and original model/renderer source hashes were verified unchanged after inference. Native renderer/capture parity includes rasters, labels, metadata and RNG state for all three durations.
- Conditional neural extraction passed bit-exact native full-sequence-hook versus direct-block ROI parity at all six selected updates and unchanged hooked/native logits, for B12/B20/B28 (neural_hook_parity.json).
- Independent separate-process no-fit replay recalculated every train/validation/test descriptor from saved maps, every serialized real/shuffled prediction and test metric, train-only scaling, validation predictions, source/checkpoint/freeze hashes, all movie disjointness, and first three native test rasters/flow references. Prediction maximum absolute difference:0.
- One execution encountered a strict-JSON serialization error on a NumPy boolean **after the two pixel fits but before gate receipt, neural extraction or test generation**. The failed source/log and recovery reason are preserved. A serialization-only fix reran the exact same deterministic extraction/two fits once under the **same original deadline**; no additional architecture, seed, criterion or candidate was tried. This repeated computation is included in the cap.
- Background tracker null-exit notifications were not used as completion evidence: OS PID checks showed the recovery worker still live. Completion relies on completed numerical artifacts, hash verification and independent replay, not the tracker notification.
- CPU only, torch2 threads/interop1, numerical libraries≤2threads, one extraction/fitting worker at a time. No MPS, cloud, main-model training, task alteration or existing90checkpoint restart. Replay/report complete inside the nonrenewable1800second budget; final_receipt.json gives final inclusive elapsed time.
- Saved:protocol.json, budget.json, adequacy_gate.json, original/recovery logs, all train/validation/test raw or neural maps and descriptors, native metadata/movie hashes, exact fits.npz, shuffle permutation, selections/thresholds, pre-test freeze hashes, trial predictions, native head logits, full/endpoint/window5 flow angles, bootstrap indices, metrics, identity/parity/verification/replay receipts and source archive. Large artifacts remain local in this directory; prior analysis outputs are preserved.

## Limits relative to the neuroscience SOP

This is a bounded mechanism-supporting failure diagnostic, not a new attention experiment. Cued-versus-uncued psychometric curves across validity, neutral/invalid cue manipulations, retention/inhibition/microstimulation and biological-circuit claims are **not measured here**. Native target/foil/catch counts and unchanged head scores are reported without relabeling foil events as physical catches. No timing/action policy was introduced; no reaction-time claim.

Reproduction:replay.py is no-fit and refuses execution beyond this run’s original deadline. Any later re-execution requires a newly authorized audit budget; do not edit/reset budget.json.
