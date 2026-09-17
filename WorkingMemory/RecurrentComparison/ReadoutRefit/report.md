# Existing readout refit: held-out results

**Existing output fitting improved eight-transition motion from70.31% to79.30% (+8.98 percentage points, paired95% CI[+5.86,+12.11]).** No clear orientation cost was detected, which is not an equivalence guarantee. Sensory/recurrent weights and architecture remained unchanged. Trained38,720 new standard-task episodes. Validation selected step9840 (38,720 new episodes at that checkpoint).

| Standard task cell | Parent BA | Refit BA | Change, percentage points (paired95% CI) | Parent/refit AUC |
|---|---:|---:|---:|---:|
|motion_anchor|99.80%|100.00%|+0.20 [+0.00,+0.59]|1.0000/1.0000|
|orientation_anchor|99.61%|99.41%|-0.20 [-0.98,+0.39]|1.0000/1.0000|
|motion_L2|98.24%|99.22%|+0.98 [+0.20,+1.95]|0.9998/1.0000|
|motion_L8|70.31%|79.30%|+8.98 [+5.86,+12.11]|0.9304/0.9425|
|orientation_recall_minimal|92.77%|93.55%|+0.78 [-0.20,+1.95]|0.9744/0.9791|
|orientation_recall_D4|83.98%|84.57%|+0.59 [-0.78,+1.95]|0.9097/0.9175|

The mean six-cell AUC changed by+0.0041, paired95% CI[0.002701420254177505, 0.005552164713541702]. Different binary/four-way chance accuracies are not pooled into one raw headline BA.

Both model views used exactly the same3072 fresh held-out standard-task movies; complete metadata matched. No validation or test example was used for optimizer fitting. Confidence intervals resample complete paired episodes, stratified by class within each cell; they quantify this selected-checkpoint comparison, not training-seed variation or the uncertainty of an architecture search.

Only existing memory_output plus motion/orientation linear heads trained (33,670 parameters). All sensory and recurrent parameters remained exactly unchanged at every training-block check. Parent Adam moments, sampler and RNG resumed in a versioned readout-only protocol. The separate profiling updates were excluded from production initialization/exposure. The selected result can precede the endpoint; both counters are recorded.

Training clipped37.91% of updates; median preclip norm0.6886. Measured production collection/inference/output-fit time was662.17seconds. Full GPU worker/supervisor/evaluation costs are in the run budget/exit receipts. No full sequence BPTT was performed because upstream computation was fixed.

This is a targeted readout-fitting intervention compared with the unchanged EI parent. It does not establish superiority over equally exposed end-to-end continuation, a biological memory mechanism, or an adequate horizon for every task. The preceding controlled StateDiagnostic motivated this branch; its suffix-matched task distribution and raw-score AUC differ from this standard-task softmax-score evaluation.

See analysis.json for paired effects, intervals, error overlap and training exposure; results.json for validation curves, class confusions, per-class recall and binary hit/miss/false-alarm/correct-rejection statistics. Immutable checkpoints, original source/config snapshots, metrics and raw predictions remain in the run directory. No new experiment is launched automatically.

The independent reviewer confirmed that this supports a useful readout-training limitation in the parent, without requiring new memory dynamics. The old LSTM score came from different held-out draws; no claim that this refit matches or exceeds LSTM is made. All11 GPU jobs exited successfully in828.01seconds total supervisor time, including the profile, training, validation and both final tests. Reporting remained inside the original3600-second cap.
