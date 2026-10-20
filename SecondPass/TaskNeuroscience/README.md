# Computational neurophysiology of the previous 19-task model

Requested 18 October 2026: a scientific paper for every task, including behavioral functions, threshold-defined timing, spatial attention, causal priority and memory perturbations, decoding and mnemonic population geometry.

## Fixed subject

- Main: `SecondPass/MemoryGuidedSearch/LocalRuntime/static_memory4x01/best.pt`, update **204289**, SHA-256 `596bc351783b3ddbfa8d37a9cf8d65146d1333dcc0097d40f0f75665b6451bcf`.
- Robustness: the same directory's `latest.pt`, update **220637**, SHA-256 `c13338ff6958c5dbb617914da538bca139944b89122b848b72885ed3241995d4`.
- Weights frozen. No optimizer, new agent training or checkpoint modification. Local GPU analysis is separate from the new spatial-query model training on RunPod.

## Experiments

1. **Native behavior:** 128 new trials for every one of 437 conditions (55,936 total), plus matched 32-trial final-checkpoint comparisons. Record class probabilities, all-frame attention and unchanged-head prefix probabilities.
2. **Psychometrics:** 128 trials per audited sensory/motion/orientation strength and VDA event/angle condition. Native full factorial supplies memory load, delay, similarity and exposure curves. Retain target, foil and catch labels separately.
3. **Chronometry:** first post-evidence crossings of 0.6/0.7/0.8/0.9 confidence thresholds, crossing coverage, conditional accuracy/latency, premature crossings and censored survival traces. Units are model frames.
4. **Causal priority:** 25 sham/spatial arms per selected condition, including +/-2/4/8 evidence logit doses, +/-4 encoding/delay perturbations, off-site and global controls. Exact same pixels across arms. Missing epochs are marked inactive. In sample-search tasks, a spatial `target_encoding` arm denotes the future array-target coordinate during encoding; sample-centered manipulations are a distinct experiment.
5. **Memory causality:** reset or amplify spatial, CLS or comparison memory at a specified retention boundary, six arms, while preserving subsequent input.
6. **VDA dual effect:** 256 nuisance draws per event and seven arms, in VDA4/9/16. A 100% bottom-right cue is followed by bottom-right change, top-left change, or catch. Manipulation is restricted to change frames. Uncued changes remain positive-response events.
7. **Neural recording:** 192 fitting and 128 held-out trials per selected condition. Six recording regions, all time points, capacity-matched external ridge probes, response/location/content/continuous-feature targets, temporal generalization, dimensionality and state trajectories.
8. **Full-state controls:** uncompressed ridge probes with inner-training penalty selection, unit selectivity and cross-validated centroid distances.
9. **Reporting:** one LaTeX/PDF paper per task, task-specific figures and native-condition tables, all-frame representative atlases, an integrated volume, source data and executable analysis code.

Additional analyses developed during the experiment are explicitly exploratory:

- Matched attention-map manipulation checks on 32 replayed trials per selected condition, including central-sample encoding controls for static and motion search.
- Identical-array query switches: 64 fresh scenes per selected memory-search condition, four sample identities/directions per scene, and one blank-sample control. Search movie identity is asserted before inference.
- Weak-change VDA follow-up: 128 matched trials per 2/4/8-degree cued or uncued event, plus catches, seven interventions in every VDA size. This follows the high uncued detection seen in the initial 18-degree VDA4 counterfactual; tests receive their own within-task Holm correction and a joint dual-effect assessment corrected across all three VDA tasks.
- Descriptive Weibull fits to native-label sensory accuracy sweeps, with matched-nuisance bootstrap thresholds; event-conditioned detection curves remain empirical.
- All-64-unit held-out population profiles and four example units chosen using fitting data only.
- Full 28-by-28 RGB sample reconstruction from independently recorded states, followed by external pixel-distance candidate matching on the same held-out scenes. Full-coordinate ridge penalties are selected within fitting data; paired differences from agent behavior receive Holm correction across all 72 comparisons.
- A subsequent joint readout decodes each current candidate from its local appearance features and compares that reconstruction with the independently decoded delay template. Inner candidate-decoder validation splits complete scenes. Six spatial-memory joint-readout comparisons form a separate exploratory Holm family.
- Matched-scene timing contrasts, all-location orientation-content decoding, recognition hit/false-alarm and serial-position functions, and condition-resolved attention maps.
- Native VDA probability/value stratification, including prechange cue allocation and the fraction of offered value earned.
- Commitment checks compare each first-threshold choice with the terminal choice. Retrospective settling times separately identify the final choice revision, with correct and incorrect terminal outcomes retained.

The shared-condition timing generators deliberately reuse scenes. Task-level intervals cluster these scenes together across sample/delay/exposure conditions. Probe fitting and testing use separate procedural namespaces and natural-image splits. Primary +/-4 evidence comparisons receive a cross-task Holm correction; other sweeps remain exploratory.

## Audit and operation

`recording.py` transforms the inspected authoritative forward in a small explicit set of places. `Results/audited_forward.py` records the exact function. Mandatory all-task checks establish exact original logits, attention and final prefix-readout parity, prefix causality, zero-bias identity and global-softmax shift invariance. Renderer parity checks separately verify the difficulty adapters against native pixels.

```powershell
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.collect --check
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.collect --hours 16
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.statistics --watch
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.deep_analysis --watch
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.papers --watch --rebuild
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.attention_controls --wait-for-collection
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.sample_reconstruction --wait
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.joint_readout
C:/Python310/python.exe -m SecondPass.TaskNeuroscience.finalize --wait
```

Completed data units are atomically saved and skipped on explicit continuation. A `Results/stop_requested.json` file stops collection at the next batch/arm boundary. The collector uses the existing local GPU lock and a finite wall-clock allocation. Statistical/reporting workers are CPU processes. Runtime arrays, weights, PDFs and tool binaries are ignored by Git under `Results/` and `Tools/`.

Final verification must establish all 19 completion receipts, all 437 native cells, all expected experiment units, unchanged checkpoint hashes, finite arrays, correct label/probability shape, corrected statistical reporting, successful LaTeX compilation, rendered page review and a portable paper package.

For publication-only continuation after completed experiments, run
`C:/Python310/python.exe -m SecondPass.TaskNeuroscience.finish_publication`.
This refreshes the methods clarification, descriptive factor tables, compilation
and audits without additional model inference. Inspect the rendered contact
sheets and full-size pages, record the result in `Results/pdf_audit.json`, then
call `finalize.package()`. Packaging requires a passed visual review and unchanged
PDF hashes. `finalize` itself stops for that review rather than packaging
unreviewed output.

## Condition-by-time attention atlas

`attention_dynamics` selects fixed conditions and physical target/cue sites from
the saved native trials, generates 63 condition groups and 888 condition-frame
maps, and exports regional time courses with 2,000 cluster-bootstrap resamples.
It replays stimulus pixels for the viewer without running the network again.
`attention_report.gallery()` builds `Results/attention/index.html`; the paper
compiler adds the corresponding main-text figures and all-frame appendices.
The layout follows the condition rows, time columns and regional trajectories
in Morgan, Albanna and Herman (2025), Figure 4.

`verify_attention` independently reconciles displayed averages and region masses
against the saved arrays. `node SecondPass/TaskNeuroscience/test_attention_viewer.cjs`
tests all selector/frame controls and asset paths in a simulated DOM. This code
test is separate from browser rendering. The delivery includes the portable
viewer, its figure assets, condition definitions, numerical maps and CSV traces.

## Neuroscience integrated into each task manuscript

Each of the 19 manuscripts now introduces the relevant human and animal
experiments before defining the model task. Its Discussion revisits those same
studies alongside the measured behavioral, attention, intervention and decoding
results, followed by the computational account and a specific follow-up test.
The temporal-shortcut and recognition signal-detection reanalyses appear in
the Results of their respective papers. Citations resolve to each paper's own
bibliography. The focused neuroscience companion remains available separately.

`C:/Python310/python.exe -m SecondPass.TaskNeuroscience.integrate_literature`
rebuilds these existing manuscripts without rerunning model inference.
`verify_literature` checks section placement, study coverage, citation resolution
and repeatable integration, and renders the affected pages for visual review.

## Reproduction from the delivery archive

The archive contains the LaTeX manuscripts with their relative figure paths, summary JSON/CSV tables, analysis modules, and imported repository Python source files with a SHA-256 manifest and package versions. It is a report/source package; model weights, natural-image datasets, and raw trial arrays remain external. Use the pinned checkpoints listed above and the repository's BSDS500 setup, restore source files at their recorded repository paths, and run the commands from the repository root. The neural collector stores fresh raw arrays locally. Tectonic compiles the manuscripts in the `papers` directory. Figures and summaries can be inspected without installing the training environment.
