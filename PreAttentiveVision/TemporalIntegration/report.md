# Causal temporal integration: measured comparison

All three accumulators completed the same fresh training exposure with the selected spatial encoder frozen. The per-task 95% point-estimate screen was met by opponent.

This tests learned two-step sensory integration with these fitted systems. It does not establish long-duration memory, a population architecture ranking, or a biological mechanism. A spending cap is not a sufficient acquisition horizon.

The opponent model scored100% on motion and spatial frequency, and98.44–99.78% on the other five tasks. KDA scored31.70% on motion while scoring96.88–99.33% on the other tasks; ConvGRU scored75.67% on motion and94.64–99.11% elsewhere. These are outcomes after the stated exposure, not claims that the other architectures cannot learn motion.

## Held-out task performance

Balanced accuracy, fixed argmax decisions. Each column uses exactly the same 448 fresh generated pairs per task. The preserved reference retains the original direct pair decoder; the three new models receive only the current field and their causal state.

| Task | Preserved pair reference | spatial_kda | convgru | opponent |
|---|---:|---:|---:|---:|
| motion_direction | 100.00% | 31.70% | 75.67% | 100.00% |
| orientation | 100.00% | 98.88% | 98.44% | 99.78% |
| contrast | 99.55% | 97.99% | 97.99% | 99.33% |
| spatial_frequency | 99.78% | 98.66% | 95.98% | 100.00% |
| chromatic_increment | 99.55% | 96.88% | 94.64% | 99.11% |
| contour | 95.54% | 98.21% | 99.11% | 98.88% |
| natural_spectrum | 99.55% | 99.33% | 98.21% | 98.44% |

Task AUC and uncertainty are retained in results_temporal.json and the interactive HTML report. There is no pooled raw accuracy across binary and four-class tasks.

## Paired changes from the preserved reference

Differences are percentage points in balanced accuracy, with paired 95% percentile bootstrap intervals. Intervals resample procedural pairs or natural-image source photos, preserving each model/reference pairing. The two-percentage-point retention tolerance is descriptive; this is not a formal noninferiority analysis.

For the opponent model, contour BA increased3.35percentage points versus the preserved reference (paired95% interval+1.58 to+5.05). Natural-image spectral BA decreased1.12points (source-photo-cluster interval−2.40 to−0.22). All point differences meet the descriptive two-point retention tolerance, but the natural-image interval extends beyond it. The contour increase is a system-level comparison that also includes new modules/readout and more training; it does not isolate a benefit of the fixed energy channels.

| Accumulator | Task | Delta BA, pp | 95% interval, pp | Delta AUC |
|---|---|---:|---:|---:|
| spatial_kda | motion_direction | -68.30 | [-71.61, -65.06] | -0.3141 |
| spatial_kda | orientation | -1.12 | [-2.27, -0.22] | +0.0000 |
| spatial_kda | contrast | -1.56 | [-2.96, -0.40] | -0.0010 |
| spatial_kda | spatial_frequency | -1.12 | [-2.37, -0.03] | +0.0000 |
| spatial_kda | chromatic_increment | -2.68 | [-4.36, -1.22] | -0.0001 |
| spatial_kda | contour | +2.68 | [1.02, 4.33] | +0.0056 |
| spatial_kda | natural_spectrum | -0.22 | [-0.98, 0.46] | -0.0008 |
| convgru | motion_direction | -24.33 | [-27.35, -21.54] | -0.0713 |
| convgru | orientation | -1.56 | [-2.89, -0.62] | -0.0043 |
| convgru | contrast | -1.56 | [-2.79, -0.47] | -0.0005 |
| convgru | spatial_frequency | -3.79 | [-5.78, -2.03] | -0.0028 |
| convgru | chromatic_increment | -4.91 | [-7.19, -2.75] | -0.0062 |
| convgru | contour | +3.57 | [1.98, 5.28] | +0.0057 |
| convgru | natural_spectrum | -1.34 | [-2.77, -0.24] | -0.0006 |
| opponent | motion_direction | +0.00 | [0.00, 0.00] | +0.0000 |
| opponent | orientation | -0.22 | [-0.74, 0.00] | +0.0000 |
| opponent | contrast | -0.22 | [-1.22, 0.72] | -0.0001 |
| opponent | spatial_frequency | +0.22 | [0.00, 0.73] | +0.0000 |
| opponent | chromatic_increment | -0.45 | [-1.57, 0.50] | -0.0001 |
| opponent | contour | +3.35 | [1.58, 5.05] | +0.0058 |
| opponent | natural_spectrum | -1.12 | [-2.40, -0.22] | -0.0001 |

## Acquisition and checkpoint selection

Checkpoints were selected on validation only: highest minimum task BA, then mean task OVR-AUC, with exact ties resolved in favor of the earlier checkpoint. The fixed training exposure was completed regardless of early scores. Final test results did not choose the checkpoint.

| Accumulator | Selected update | Terminal update | Validation minimum BA at successive looks | Validation mean AUC at successive looks |
|---|---:|---:|---|---|
| spatial_kda | 4032 | 4032 | 1008: 25.89%, 2016: 26.79%, 3024: 33.48%, 4032: 33.93% | 1008: 0.8441, 2016: 0.9132, 3024: 0.9386, 4032: 0.9483 |
| convgru | 4032 | 4032 | 1008: 30.36%, 2016: 25.45%, 3024: 61.61%, 4032: 72.77% | 1008: 0.8699, 2016: 0.9206, 3024: 0.9735, 4032: 0.9873 |
| opponent | 4032 | 4032 | 1008: 63.39%, 2016: 56.70%, 3024: 86.16%, 4032: 98.66% | 1008: 0.9388, 2016: 0.9936, 3024: 0.9976, 4032: 0.9992 |

## Temporal diagnostics

Each diagnostic repeats the same test pairs. Reset-before-second deletes prior state and may create an out-of-distribution state. A decrease supports dependence on prior state without identifying a neural mechanism. Frame swap reverses direction labels and flips binary interval/signed-orientation labels. Some task marginals may retain single-frame information; no fixed chance threshold was imposed.

For the opponent model, motion BA fell from100% to25.89% after resetting state before frame2, and was99.78% after reversing the frames and transforming labels. This supports use of temporal history and the tested order sensitivity; the reset removes the whole state and does not establish that the energy branch is necessary.

| Accumulator | Task | Normal BA | Reset-before-second BA | Reversed-order, transformed-label BA |
|---|---|---:|---:|---:|
| spatial_kda | motion_direction | 31.70% | 24.11% | 33.48% |
| spatial_kda | orientation | 98.88% | 50.22% | 99.11% |
| spatial_kda | contrast | 97.99% | 50.45% | 98.66% |
| spatial_kda | spatial_frequency | 98.66% | 51.56% | 99.33% |
| spatial_kda | chromatic_increment | 96.88% | 51.34% | 98.21% |
| spatial_kda | contour | 98.21% | 73.66% | 98.44% |
| spatial_kda | natural_spectrum | 99.33% | 47.32% | 98.66% |
| convgru | motion_direction | 75.67% | 25.00% | 75.00% |
| convgru | orientation | 98.44% | 48.44% | 96.43% |
| convgru | contrast | 97.99% | 54.02% | 97.77% |
| convgru | spatial_frequency | 95.98% | 60.04% | 95.31% |
| convgru | chromatic_increment | 94.64% | 54.02% | 94.20% |
| convgru | contour | 99.11% | 64.29% | 98.88% |
| convgru | natural_spectrum | 98.21% | 64.51% | 98.88% |
| opponent | motion_direction | 100.00% | 25.89% | 99.78% |
| opponent | orientation | 99.78% | 51.12% | 99.78% |
| opponent | contrast | 99.33% | 49.55% | 99.33% |
| opponent | spatial_frequency | 100.00% | 48.66% | 99.78% |
| opponent | chromatic_increment | 99.11% | 53.79% | 99.55% |
| opponent | contour | 98.88% | 58.04% | 98.66% |
| opponent | natural_spectrum | 98.44% | 51.12% | 99.78% |

## Exposure and measured cost

Each model trained for 4,032 updates at batch32 (129,024 fresh generated pairs), with identical per-task stream prefixes and the interleaved contour-focused twelve-update schedule. These are new projection/core/readout updates; the inherited encoder had zero parameter updates. The encoder retains prior training exposure, and all three depend on the same trained parent.

| Task | Terminal pairs per model | spatial_kda selected pairs | convgru selected pairs | opponent selected pairs |
|---|---:|---:|---:|---:|
| motion_direction | 10,752 | 10,752 | 10,752 | 10,752 |
| orientation | 10,752 | 10,752 | 10,752 | 10,752 |
| contrast | 10,752 | 10,752 | 10,752 | 10,752 |
| spatial_frequency | 10,752 | 10,752 | 10,752 | 10,752 |
| chromatic_increment | 10,752 | 10,752 | 10,752 | 10,752 |
| contour | 64,512 | 64,512 | 64,512 | 64,512 |
| natural_spectrum | 10,752 | 10,752 | 10,752 | 10,752 |

| Accumulator | Trainable parameters | Core parameters | State MiB/example | State MiB/batch32 | Profile peak allocated MiB | Profile mean training ms/update | Production worker wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_kda | 209,222 | 74,262 | 3.217 | 102.94 | 861.3 | 200.5 | 734.96 |
| convgru | 301,136 | 166,176 | 0.402 | 12.87 | 382.0 | 184.5 | 646.20 |
| opponent | 141,968 | 7,008 | 0.804 | 25.73 | 1030.5 | 213.5 | 838.81 |

Supervisor receipt: completed in 2505.47 seconds of the new 7,200-second allowance, including profiles, validation, final inference and paired analysis. The previous unused allowance was not consumed or renewed. One GPU worker ran at a time, fp32, no AMP/TF32, two CPU PyTorch threads. Persistent state bytes exclude activations and gradients; peak allocated VRAM excludes desktop/driver allocations. Profile48-update fits were separate, preserved diagnostics and were not used as production initialization. Their shared initial stream prefixes are repeated presentations, not extra unique production data. Production worker wall time includes process startup and checkpointing; profile update time measures a different scope.

## Reproduction and limits

Run directory: `C:\Users\jomor\Documents\VisualAttentionWorkingMemory\PreAttentiveVision\TemporalIntegration\runs\temporal_20260912_165510`. The immutable source snapshot, fixed_config.json, dataset_manifest.json, budget.json, worker logs, metrics.csv, checkpoint indices, full-state checkpoints, raw predictions and exit.json are retained there. results_temporal.json is the readable aggregate; paired_analysis.json contains the saved-score comparison.

Frozen parent: `C:\Users\jomor\Documents\VisualAttentionWorkingMemory\PreAttentiveVision\runs\allocation_20260912_160414\contour_focus_seed20271\checkpoint_002268.pt`; SHA256 `31dc0f4b6a2ef7706e5f724837281a80787cb53462a57011be5ac48279b69ba2`. Its recorded training step is2268; the reference evaluator does not train it.

Training base seed310001, validation910001, test1010001, with task seeds base+100003*(registry index+1). Model seed30301, common projection/readout initialization30312, temporal-core initialization30313. Fresh Adam lr.001, weight decay.0001, gradient norm clip5; full two-step BPTT through the new modules. Source manifests and versioned full state define strict resumption; do not resume a prior optimizer for new temporal weights.

Normal task intervals use1,000 bootstrap replicates. Natural-image sources are clustered because fresh generated crops/pairs reuse the held-out BSDS photo pool; procedural pairs are the unit for other tasks. Repeated diagnostics/checkpoints are not independent data or training seeds. Point-perfect empirical bootstrap intervals can collapse at100% and do not imply population certainty. Per-difficulty summaries and confusion matrices remain in the aggregate. No held-out tuning or additional architecture sweep was performed.

A two-step recurrent state can retain enough information to implement an online pair comparator. These results therefore concern replacing direct two-encoding access under the tested objectives; persistent working memory, longer accumulation, speed/coherence tuning and MT pattern-motion correspondence require separately authorized tasks.
