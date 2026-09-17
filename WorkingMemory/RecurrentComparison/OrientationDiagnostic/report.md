# Frozen orientation diagnostic

**The model still retains useful orientation information after 24 blank frames. Its deployed comparison does not use that information effectively on these episodes.** Immediately before the probe, a linear readout of firing rates recovers the sample's actual orientation with 4.25° mean absolute error (sample-time error 5.32°). A small label-trained comparator using those rates and the separately encoded probe reaches 73.44% balanced accuracy, versus 50.00% for the deployed model: a paired gain of 23.44 percentage points,95% CI 17.77–28.91.

A diagnostic comparator using independently decoded angles reaches 87.30% (gain 37.30pp,95% CI 32.23–42.19). It receives predicted angles, not the ground-truth angles; its angle decoders do receive extra orientation supervision during fitting. This establishes a useful diagnostic route, not a newly trained deployed capability. The label-only comparator's gain independently shows that the available features can support comparison without angle-supervised training.

The sample-trained linear decoder performs poorly after the delay (44.96°), whereas a delay-specific linear decoder works on those exact same held-out states (4.25°). The paired error reduction is 40.71°,95% CI 38.46–42.94. The sample-time decoding map is therefore not temporally stable; shifts in offsets, scale or representation can contribute. This does not require information erasure. Adding adaptation access changes the D24 error to 3.82°; recovery already succeeds through the ordinary firing-rate route.

The conditional linear readout of the **post-probe** state reaches 52.73%, with no clear gain. That narrows the useful next development question to comparison/probe processing and output use, but does not distinguish a probe-update problem from nonlinear information remaining after the probe. Preserve the trained memory core while investigating that interface; no further experiment was launched. Short-delay comparators underperform the deployed model, so this is not a universal improvement. The hardest 7.5° changes also remain unresolved: the angle-distance comparator detects 22/53, despite correctly rejecting 229/256 unchanged trials.

Selected Retention E/I checkpoint 14800; 512 independent held-out episodes, each shown at four delays. No parent-model updates.

## Pre-probe orientation recovery

| State/stage | D0 MAE° | D4 | D12 | D24 |
|---|---:|---:|---:|---:|
| r before query | 5.32 | 4.81 | 4.05 | 4.15 |
| r after query / before probe | 4.94 | 4.69 | 3.98 | 4.25 |
| [r,a] before probe | 4.37 | 4.30 | 3.67 | 3.82 |
| sensory path before probe | 7.23 | 5.56 | 7.26 | 22.19 |
| sample-trained r decoder transferred | 20.46 | 43.46 | 44.96 | 44.96 |

Sample-stage r ridge MAE: 5.32°. Isolated-probe ridge MAE: 1.88°. Smallest actual change is 7.5°; a 3.75° half-change benchmark is descriptive, not a sufficient comparator guarantee.

## Operational label comparison

| Readout | BA% [paired95% CI] | AUC | Change vs deployed model, pp [95% CI] |
|---|---:|---:|---:|
| comparator_D0 | 72.85 [69.34, 76.76] | 0.768 | -23.44 [-27.54, -19.33] |
| comparator_D24 | 73.44 [69.73, 76.95] | 0.793 | 23.44 [17.77, 28.91] |
| post_D24 | 52.73 [48.44, 56.84] | 0.518 | 2.73 [-2.54, 7.81] |
| existing_D0 | 96.29 [94.34, 97.85] | 0.993 | 0.00 [0.00, 0.00] |
| existing_D4 | 91.80 [89.45, 93.95] | 0.974 | 0.00 [0.00, 0.00] |
| existing_D12 | 61.91 [58.20, 65.43] | 0.650 | 0.00 [0.00, 0.00] |
| existing_D24 | 50.00 [45.90, 54.30] | 0.484 | 0.00 [0.00, 0.00] |
| circular_D0 | 83.59 [80.47, 86.72] | 0.909 | -12.70 [-16.21, -9.38] |
| circular_D24 | 87.30 [84.38, 89.84] | 0.930 | 37.30 [32.23, 42.19] |

## Fixed nonlinear angle fits

| Probe | Angular MAE° | 95% CI | Within3.75° |
|---|---:|---:|---:|
| sample_r_mlp | 4.92 | [4.50, 5.32] | 52.15% |
| preprobe_r_D0_mlp | 4.56 | [4.18, 4.96] | 57.03% |
| preprobe_r_D24_mlp | 4.13 | [3.79, 4.47] | 60.55% |

## Actual-change detail at D24

| Actual change° | Trials | Deployed correct% | MLP comparator correct% | Circular comparator correct% |
|---|---:|---:|---:|---:|
| 0.0 | 256 | 42.19 | 86.33 | 89.45 |
| 7.5 | 53 | 56.60 | 20.75 | 41.51 |
| 15.0 | 57 | 66.67 | 49.12 | 89.47 |
| 22.5 | 45 | 53.33 | 75.56 | 100.00 |
| 30.0 | 38 | 63.16 | 78.95 | 97.37 |
| 37.5 | 23 | 43.48 | 82.61 | 100.00 |
| 45.0 | 24 | 54.17 | 87.50 | 100.00 |
| 52.5 | 10 | 50.00 | 60.00 | 100.00 |
| 60.0 | 6 | 66.67 | 100.00 | 100.00 |

[Figure](orientation_diagnostic.png) · [Complete numeric results](summary.json) · [Paired transfer and change-size detail](paired_analysis.json) · [Frozen fit/threshold choices](fit_selection.json)

## Limits and accounting

Targets are actual rendered axial angles, not the latent level. All angular predictions use only pre-probe states; probe-angle decoding uses an independently initialized sensory path. The ordinary sensory feature includes opponent traces. Thus a comparator restricted to r plus an independently encoded probe omits that history route, and a D0 comparator failure can reflect that omission as well as limited probe fitting. Adaptation access is diagnostic-only. Weak probe performance never establishes erased information. An early decoder transfer penalty is evidence about decoder stability, not by itself a demonstration of a changed code.

Operational classifiers receive no angles or labels as inputs. Circular comparisons use inferred angles and a validation-only threshold. Postprobe label fits, if present, are operational rescue tests and make no sample-reconstruction claim. AUC uses continuous score differences. Change-size rows in summary.json contain changed-trial sensitivity (and zero-change specificity), not balanced accuracy within an all-positive stratum. Intervals condition on this one frozen trained model and use independent base episodes; repeated delays are paired presentations.

Training 2048 / validation 512 / test 512 independent groups produce 12288 model-delay presentations and 184320 sequence-frame encoder calls. Independent probe encoding adds 3072 image calls. The first profile batch is retained within those counts; the ordinary-renderer equivalence check is CPU-only. All choices were frozen before held-out collection. Neural extraction and MLP fits use fp32; NumPy/SciPy ridge uses float64. No original optimizer was constructed and no parent weights were updated. Exact checkpoint and runtime source identities are in config.json and dependency_manifest.json; source copies, all candidate traces and selected fitted weights are retained. Worker computation completed in 399.42 seconds; supervisor exited successfully at 400.03 seconds, within the 1800-second allowance. Final reporting time is recorded in completion_receipt.json.
