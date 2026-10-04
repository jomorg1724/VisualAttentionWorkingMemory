# Frozen feature and comparator diagnostic

**Native D0 only; terminal6760 unchanged.** Independent 1024/256/512 train/validation/test base episodes per task unless counts below state otherwise.

|Task|Readout/control|Test BA|Test AUC|Paired accuracy gain vs deployed (95% interval)|
|---|---|---:|---:|---|
|orientation_cued|deployed|0.4863|0.4852|+0.0000 [0.0, 0.0]|
|orientation_cued|label_prior|0.5000|0.5000|+0.0137 [-0.044921875, 0.072265625]|
|orientation_cued|label_only_relation|0.4922|0.5028|+0.0059 [-0.056640625, 0.068359375]|
|orientation_cued|shuffled_label_relation|0.5312|0.5507|+0.0449 [-0.015625, 0.10546875]|
|orientation_cued|aux_supervised_structured|0.9160|0.9773|+0.4297 [0.37890625, 0.478515625]|
|spatial_binding|deployed|0.5078|0.4964|+0.0000 [0.0, 0.0]|
|spatial_binding|label_prior|0.5000|0.5000|-0.0078 [-0.068408203125, 0.0546875]|
|spatial_binding|label_only_relation|0.5625|0.5804|+0.0547 [-0.00390625, 0.11328125]|
|spatial_binding|shuffled_label_relation|0.5410|0.5436|+0.0332 [-0.029296875, 0.09375]|
|spatial_binding|aux_supervised_structured|1.0000|1.0000|+0.4922 [0.447265625, 0.53515625]|

## Measured conclusion

**The trained network contains the cue and usable local orientation information, but its deployed decision does not combine them successfully.** This is strongest for early activations: auxiliary-trained component decoders plus a feature-predicted relational readout solve91.6% of signed-orientation and100% of binding held-out trials. This is **not** a successful label-only rescue, nor proof that the final256-unit representation can solve either task.

- **Cue blindness is not supported.** At report, location decoding is100% across all four layers for both tasks. Signed-cue decoding is100% in early/final-visual/ConvGRU features and99.8% in the256-unit readout. Native label×location×sign balancing and label-stratified cue results prevent task-label association from explaining these cue scores.
- **Sensory precision changes along the processing path.** Mean early sample-angle error is1.33° for orientation and1.39° for binding. At report, early probe-angle errors are5.89° and2.66°; final-visual errors18.20° and11.46°; ConvGRU errors18.39° and13.81°; readout errors30.06° and22.14°. Orientation's15° minimum change is therefore small relative to late-stage probe errors. Sample-angle information remains more accessible than new probe-angle information. These are different finite-capacity linear decoders and ROI summaries, so this pattern does not prove irreversible information loss or a causal bottleneck.
- **Label-only learning remains unresolved.** The small relation-aware comparator scores49.2% BA on orientation and56.3% on binding versus deployed48.6%/50.8%. Both paired gain intervals include zero. Shuffled-label controls score53.1%/54.1%, reinforcing caution rather than establishing a reliable label-only improvement.
- **Parts can be combined with explicit auxiliary supervision and a supplied circular relation.** Using only predicted cue/sign/angles at evaluation, the structured diagnostic scores91.6%/100%, with paired gains42.97/49.22 percentage points (95% intervals37.89–47.85/44.73–53.52). Auxiliary decoders used only512 TRAIN episodes; the task-label calibrator used the other512. The orientation decoder's small errors still matter at15° and at unchanged targets. This comparator has geometry/timing, auxiliary-supervision and relational-inductive-bias privileges; it is neither the deployed head nor equivalent to the label-only arm.
- **No clean-retention conclusion.** All D0 report stacks still contain a sample image. The successful structured arm also accesses stored sample-time activations, an external diagnostic memory privilege. It demonstrates feature accessibility and composability across known times, not a successful single-final-state readout or recurrent memory mechanism.

**Next decision:** prioritize the acquisition/use of the cue-conditioned circular comparison, while preserving the evidence that late probe precision is weaker. These results do not support complete cue erasure or a retention-only explanation. They do not authorize more model training or establish an architectural remedy. A future comparison of equivalent-capacity readouts restricted to one final state versus time-separated activations would distinguish the diagnostic memory privilege from final-state accessibility; not run here.

## Feature accessibility
Mean axial error in degrees (uniform-angle uninformed reference 45°, not an empirical control). Each local probe predicts cos(2θ), sin(2θ); all four locations fit separately. Sample-at-report is not clean retention: raw stack3 still contains a sample.

|Task|Layer|Time|Content|Cued error°|Uncued error°|All error°|
|---|---|---|---|---:|---:|---:|
|orientation_cued|early|sample|sample|1.35|1.32|1.33|
|orientation_cued|early|report|sample|3.16|3.15|3.16|
|orientation_cued|early|report|probe|6.15|5.81|5.89|
|orientation_cued|late|sample|sample|7.76|6.92|7.13|
|orientation_cued|late|report|sample|11.85|10.36|10.73|
|orientation_cued|late|report|probe|19.59|17.74|18.20|
|orientation_cued|gru|sample|sample|3.77|3.84|3.82|
|orientation_cued|gru|report|sample|4.77|4.61|4.65|
|orientation_cued|gru|report|probe|19.95|17.87|18.39|
|orientation_cued|readout|sample|sample|12.24|11.32|11.55|
|orientation_cued|readout|report|sample|22.36|21.69|21.86|
|orientation_cued|readout|report|probe|31.66|29.52|30.06|
|spatial_binding|early|sample|sample|1.36|1.40|1.39|
|spatial_binding|early|preprobe|sample|1.34|1.30|1.31|
|spatial_binding|early|report|sample|2.16|2.01|2.05|
|spatial_binding|early|report|probe|2.61|2.67|2.66|
|spatial_binding|late|sample|sample|6.06|6.32|6.25|
|spatial_binding|late|preprobe|sample|9.86|8.58|8.90|
|spatial_binding|late|report|sample|10.64|8.88|9.32|
|spatial_binding|late|report|probe|14.20|10.55|11.46|
|spatial_binding|gru|sample|sample|2.82|3.06|3.00|
|spatial_binding|gru|preprobe|sample|3.48|3.85|3.76|
|spatial_binding|gru|report|sample|5.18|4.53|4.69|
|spatial_binding|gru|report|probe|15.39|13.28|13.81|
|spatial_binding|readout|sample|sample|7.79|8.23|8.12|
|spatial_binding|readout|preprobe|sample|12.49|12.65|12.61|
|spatial_binding|readout|report|sample|18.13|15.58|16.22|
|spatial_binding|readout|report|probe|23.63|21.64|22.14|

|Task|Layer|Time|Cue content|Balanced accuracy|
|---|---|---|---|---:|
|orientation_cued|early|sample|location|1.0000|
|orientation_cued|early|sample|sign|0.9961|
|orientation_cued|early|report|location|1.0000|
|orientation_cued|early|report|sign|1.0000|
|orientation_cued|late|sample|location|1.0000|
|orientation_cued|late|sample|sign|1.0000|
|orientation_cued|late|report|location|1.0000|
|orientation_cued|late|report|sign|1.0000|
|orientation_cued|gru|sample|location|1.0000|
|orientation_cued|gru|sample|sign|1.0000|
|orientation_cued|gru|report|location|1.0000|
|orientation_cued|gru|report|sign|1.0000|
|orientation_cued|readout|sample|location|0.9980|
|orientation_cued|readout|sample|sign|0.9219|
|orientation_cued|readout|report|location|1.0000|
|orientation_cued|readout|report|sign|0.9980|
|spatial_binding|early|preprobe|location|1.0000|
|spatial_binding|early|report|location|1.0000|
|spatial_binding|late|preprobe|location|1.0000|
|spatial_binding|late|report|location|1.0000|
|spatial_binding|gru|preprobe|location|1.0000|
|spatial_binding|gru|report|location|1.0000|
|spatial_binding|readout|preprobe|location|1.0000|
|spatial_binding|readout|report|location|1.0000|

## Scientific scope and fit discipline
- Signed orientation reports positive iff the selected rotation agrees with the instruction sign. Actual rotations are 0/±15/±30/±45°. Error relative to the smallest nonzero change is important: ≥7.5° is already half that separation. Binding swaps exactly one pair; nonzero axial differences are 45° or 90°. Per-location cued/uncued errors and fraction within7.5° are in probe_results.json; no scalar latent-level regression substitutes for angles.
- Cue location/sign are balanced independently of task label by the native queues. Cue tables include per-task-label accuracy and full class counts/confusions, preventing a label-prior decoder being misrepresented as cue decoding. Binding sign is not defined and is not probed; target location before the retrocue is not probed.
- Early visual means first learned Conv/GN/ReLU output32×50×50, stored as local2×2 means32×25×25 plus four unpooled32×6×6 tiny-glyph ROIs. The final visual field is160×7×7 (CNN+KDA output), ConvGRU64×7×7, and post-ReLU readout256. Large maps preserve spatial positions; all local orientation probes use fixed5×5 early or3×3 late/GRU crops, without target-based selection. Readout probes receive all256 units. Cue early probes use all four unpooled glyph ROIs plus fixed5×5 global pooling. Other cue probes flatten the full field.
- Fixed geometry, known task identity and known sample/query/report times are diagnostic privileges. All locations are decoded; none is selected with true target metadata as comparator input. Features are float16 on disk, promoted to float32, while exact dual ridge algebra is float64. A negative probe does not prove information was erased; pooling, linearity, small sample size and independent channel scaling may limit detection.
- Ridge fits standardize using TRAIN only; regularization α∈{0.1,1,10,100} after kernel/feature-width normalization is selected only on validation angular error or cue balanced accuracy. No PCA. Same base episode groups all locations/times. Seeds, rasters hashes, native metadata, fitted scaling/weights and predictions are saved.
- Primary label-only comparator: shared local sample/probe encoder16/32 units, ordered values/difference/product relation32 units, learned16-unit cue representation and softmax routing across all four locations. Receives only early learned activations at fixed times/ROIs; no true cue/sign/angle auxiliary loss. AdamW lr.002, batch128,120 epochs; hidden/weight-decay and 10-epoch checkpoints selected by validation BA. Shuffled-label control uses same feature pipeline and one32-unit configuration; its train labels are independently permuted while validation uses real labels. Thus its tuning opportunity is smaller than the primary grid.
- Auxiliary-supervised structured diagnostic is DISTINCT from the label-only comparator. First half of TRAIN fits early-feature angle/cue decoders with auxiliary metadata targets; second half fits a ridge task-label readout from predicted cue and circular sample/probe relationships. Validation selects α. Calibration examples were not used to fit those decoders. Operational TEST inputs are exclusively predicted quantities. True target/sign/angles never enter the operational score. This diagnoses accessible parts plus an explicitly supplied circular-comparison inductive bias, not spontaneous native task learning.
- TEST is loaded for scoring only after all model/regularization/checkpoint selections for BOTH tasks are serialized and hashed. Test metadata is generated during extraction, but no test outcome is used for selection. No retraining on validation; no test-driven retry. Label-prior rule predicts positive at0.5. Deployed native head predictions use the exact same episodes. Wilson accuracy intervals and paired whole-episode bootstrap gain intervals condition on these fitted probes/checkpoint, not model-training variability; balanced-label queue dependence and multiple descriptive probes are not corrected.
- D0 has no inserted blanks. For signed orientation, sample frame2 is also preprobe; report3 stack contains sample1/sample2/probe. Binding sample2 precedes query3; report4 stack contains sample2/query3/probe. Preprobe3 is not a clean memory assay either. No long-delay, clean-retention, cue-validity benefit, attention allocation, lesion, microstimulation, motion or recognition conclusion is supported.

## Verification and coverage
- Counts: {"train": 1024, "val": 256, "test": 512} per task. Completed tasks: 2/2.
- Exact wrapper parity maximum: 0.0. Main state tensors bitwise unchanged: True; main-model optimizer updates:0.
- Checkpoint/source hashes unchanged: True. CPU cap2, interop1, one MPS worker. Elapsed through verification/report: see completion.json and budget.json; immutable1800-second budget.
- Artifacts: run.py; features/*.npz and *_metadata.json; models/*.pt; predictions/*_test.npz; *_probe_selection.json; *_probe_results.json; *_comparator_results.json; selection_frozen.json; verification.json; budget.json; completion.json; progress.jsonl. All are local to FeatureDiagnostic.

## Interpretation boundary / next decision
Positive cue/orientation decoding establishes accessibility to these independent readouts, not use by the deployed network or a causal locus. The measured structured advantage supports usable early component information and successful explicitly supervised combination, not a final-state-only or label-only rescue. Poor label-only comparator performance is inconclusive at this small fitting budget. Do not infer a proven architecture fix, generic learning impossibility or long-delay retention failure. Compare the measured layer/time error progression and the two explicitly different comparator supervision regimes before choosing a next experiment.

## Independent readback and descriptive tables
- Main worker completed extraction, fitting, frozen held-out scoring and initial report in213.16s. Independent CPU readback finished at227.57s from first MPS work, below the1800s cap. All136 saved probe models and6 learned comparator models replay their saved test predictions; all feature/model hashes and metric calculations match. No new fits or test-driven selection occurred during readback.
- `angular_error_strata.csv` adds all4 locations × cued/uncued/all × actual absolute-change strata from saved predictions. `target_change_strata.json` reports task-decision accuracy by actual selected-target change, including unchanged cases. `comparator_summary.csv` gives compact metrics and uncertainty. These are descriptive post-fit summaries of the same held-out predictions, not additional experiments.
- Native/direct renderer equality and direct/extraction-wrapper equality were checked on each split's first8 episodes per task, plus the separate16-episode profile; native stream generation remains unmodified for every episode. Raster hashes/IDs are unique across all3584 production base episodes; profile episodes are separate. Exactly two completed tasks, four measured layers,136 selected angle/cue probes.
- Reproduction: from the repository root use `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 /tmp/vawm-task-suite-venv/bin/python -B -m SecondPass.SpatialReadout.FeatureDiagnostic.run --run`. The original directory deliberately refuses a second run because its immutable budget exists. Reproduction requires separately authorized fresh output/budget, not deleting or renewing this receipt. Readback used `-m SecondPass.SpatialReadout.FeatureDiagnostic.verify`.
