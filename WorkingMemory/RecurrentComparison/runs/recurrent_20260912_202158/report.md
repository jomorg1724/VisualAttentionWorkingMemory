# Two recurrent memories after the opponent visual system

Both models completed 40,000 fresh episodes (5,000 updates each). The shared WM parent was update 6,860. The experiment completed after 47.28 elapsed minutes. The user subsequently moved the E/I arm to the Palladio RunPod account while local LSTM training continued. Each location retained its declared absolute deadline.

| Held-out cell | LSTM BA | LSTM AUC | E/I BA | E/I AUC |
|---|---:|---:|---:|---:|
| motion_direction/motion_L2 | 99.80% | 1.0000 | 98.24% | 1.0000 |
| motion_direction/motion_L8 | 79.69% | 0.9460 | 68.95% | 0.9195 |
| motion_direction/motion_anchor | 99.62% | 1.0000 | 100.00% | 1.0000 |
| orientation/orientation_anchor | 100.00% | 1.0000 | 99.82% | 1.0000 |
| orientation/orientation_recall_D4 | 83.98% | 0.9079 | 85.74% | 0.9158 |
| orientation/orientation_recall_minimal | 95.31% | 0.9881 | 93.36% | 0.9758 |

Both models acquired the focused new tasks. LSTM scored higher on the longer motion-duration cell; the small E/I advantage on delayed orientation recall has a paired interval that includes zero. These results concern the tested tasks and fitted models, rather than a general memory-capacity or biological ranking.

Motion chance BA is 25%; orientation chance BA is 50%. AUC chance is 0.5 for both. Fixed argmax decisions and score ranking are both reported; test scores never select or calibrate the model. This is one initialization per arm, not replication across training seeds. The six cells contain 3,072 unique held-out episodes shared across both models and both state conditions: 12,288 scored presentations, not 12,288 independent episodes. Likewise, each model trains on the same 40,000-episode procedural stream; 80,000 model-episode presentations are not 80,000 independent source episodes.

## Paired effects and the recurrent-history intervention

Positive EI-minus-LSTM deltas favor EI. Positive normal-minus-reset deltas mean that preserving the new recurrent history helped the trained model. Resetting the added state before every frame leaves opponent traces and the current-frame memory computation intact. It is not a separately trained memory-free or parameter-matched control.

| Comparison / cell | BA difference (percentage points) |95% interval| AUC difference |95% interval|
|---|---:|---:|---:|---:|
| EI minus LSTM / motion_L2 | -1.56 | [-2.73,-0.39] | -0.0000 | [-0.0001,+0.0000] |
| EI minus LSTM / motion_L8 | -10.74 | [-14.26,-7.03] | -0.0264 | [-0.0384,-0.0146] |
| EI minus LSTM / motion_anchor | +0.38 | [+0.00,+0.97] | +0.0000 | [+0.0000,+0.0001] |
| EI minus LSTM / orientation_anchor | -0.18 | [-0.55,+0.00] | +0.0000 | [+0.0000,+0.0000] |
| EI minus LSTM / orientation_recall_D4 | +1.76 | [-0.59,+3.91] | +0.0079 | [-0.0154,+0.0316] |
| EI minus LSTM / orientation_recall_minimal | -1.95 | [-3.91,-0.20] | -0.0123 | [-0.0225,-0.0026] |
| lstm: normal minus reset / motion_L2 | +74.80 | [+74.41,+75.00] | +0.2172 | [+0.2073,+0.2279] |
| lstm: normal minus reset / motion_L8 | +54.69 | [+51.37,+58.40] | +0.4328 | [+0.4009,+0.4655] |
| lstm: normal minus reset / motion_anchor | +0.00 | [-0.56,+0.56] | -0.0000 | [-0.0001,+0.0000] |
| lstm: normal minus reset / orientation_anchor | +0.00 | [+0.00,+0.00] | +0.0000 | [+0.0000,+0.0000] |
| lstm: normal minus reset / orientation_recall_D4 | +7.81 | [+5.47,+10.35] | -0.0013 | [-0.0187,+0.0142] |
| lstm: normal minus reset / orientation_recall_minimal | +1.95 | [+0.39,+3.52] | -0.0008 | [-0.0058,+0.0029] |
| ei_adaptive: normal minus reset / motion_L2 | +73.24 | [+72.07,+74.41] | +0.0683 | [+0.0638,+0.0727] |
| ei_adaptive: normal minus reset / motion_L8 | +43.95 | [+40.42,+47.85] | +0.3940 | [+0.3578,+0.4269] |
| ei_adaptive: normal minus reset / motion_anchor | +0.20 | [+0.00,+0.59] | +0.0000 | [+0.0000,+0.0000] |
| ei_adaptive: normal minus reset / orientation_anchor | +0.00 | [+0.00,+0.00] | +0.0000 | [+0.0000,+0.0000] |
| ei_adaptive: normal minus reset / orientation_recall_D4 | +10.35 | [+7.42,+13.28] | -0.0012 | [-0.0049,+0.0026] |
| ei_adaptive: normal minus reset / orientation_recall_minimal | +0.20 | [-1.76,+2.15] | -0.0017 | [-0.0050,+0.0013] |

Resetting the new recurrent history reduces both motion-integration decisions to chance and substantially lowers their AUC. For delayed orientation recall, it reduces BA by 7.81 points in LSTM and 10.35 in E/I, while OVR-AUC changes by less than 0.002 in either model. The orientation effect therefore does not establish additional score-ranking information; decision bias and use of retained sensory traces remain relevant alternatives.

Intervals use 1,000 paired class-stratified episode resamples. Episode labels, identities and complete metadata match across each comparison. These intervals condition on the trained models; they neither measure seed variability nor correct for multiple exploratory cells. Empirical perfect scores can produce collapsed percentile intervals, which are not guarantees of population certainty.

## Selection, exposure and acquisition

The arithmetic mean of all six cell OVR-AUCs selects a checkpoint at four planned validation looks, with exact ties favoring the earlier checkpoint. Training still reaches the same final endpoint for both models.

| Arm | Selected update | Selected episodes | Terminal episodes | Clipped updates |
|---|---:|---:|---:|---:|
| lstm | 5,000 | 40,000 | 40,000 | 81.12% |
| ei_adaptive | 5,000 | 40,000 | 40,000 | 86.16% |

| Arm / cell | Selected-checkpoint episodes | Terminal episodes |
|---|---:|---:|
| lstm / motion_anchor | 2,000 | 2,000 |
| lstm / motion_L2 | 9,000 | 9,000 |
| lstm / orientation_recall_minimal | 9,000 | 9,000 |
| lstm / motion_L8 | 9,000 | 9,000 |
| lstm / orientation_recall_D4 | 9,000 | 9,000 |
| lstm / orientation_anchor | 2,000 | 2,000 |
| ei_adaptive / motion_anchor | 2,000 | 2,000 |
| ei_adaptive / motion_L2 | 9,000 | 9,000 |
| ei_adaptive / orientation_recall_minimal | 9,000 | 9,000 |
| ei_adaptive / motion_L8 | 9,000 | 9,000 |
| ei_adaptive / orientation_recall_D4 | 9,000 | 9,000 |
| ei_adaptive / orientation_anchor | 2,000 | 2,000 |

| Arm | Validation update | Mean six-cell AUC |
|---|---:|---:|
| lstm | 1,250 | 0.8801 |
| lstm | 2,500 | 0.9335 |
| lstm | 3,750 | 0.9664 |
| lstm | 5,000 | 0.9809 |
| ei_adaptive | 1,250 | 0.8345 |
| ei_adaptive | 2,500 | 0.9089 |
| ei_adaptive | 3,750 | 0.9403 |
| ei_adaptive | 5,000 | 0.9728 |

Clipping was frequent in both runs and is a material optimization qualification, despite finite computation and observed acquisition. No clipping-based restart or hidden tuning was performed. The following state maxima are from the planned diagnostic batches, not a scan of every activation.

| Arm | Median pre-clip norm | Maximum pre-clip norm | Largest logged state magnitude |
|---|---:|---:|---:|
| lstm | 21.139 | 545.256 | 8.013 |
| ei_adaptive | 21.056 | 412.306 | 2.392 |

## Numerical opportunity and limits

LSTM has 1,011,872 learned parameters; E/I has 714,912. Each adds 512 recurrent state values. All 408,728 transferred parameters remain trainable; unused task heads receive no task loss. Shared interface and residual tensors begin identically, and fresh task-local streams match across arms. The two cores have different initialization and parameterization, so equal exposure does not equate optimization difficulty.

LSTM forget biases initialize nominal retention time constants 4–128; its independent gate normalization precedes biases and never normalizes the stored cell. E/I starts with balanced incoming excitation/inhibition, rate time constants 2–8 and weak 0.05 adaptation. Dale signs remain constrained by positive magnitudes times presynaptic column signs. Rate/adaptation dynamics use old states synchronously; neither state nor recurrent current is normalized. Bounded leak timescales do not guarantee stability of the full recurrent loop.

Adam uses shared epsilon 1e-10, new-parameter LR 3e-4 and transferred LR 3e-5; clipping threshold 1. Raw sign-constrained parameters, biases, normalization and timescale/adaptation parameters have no generic decay. Profiles and planned training diagnostics report effective recurrent-weight changes, raw-gradient scales, clipping, state RMS/max, gate saturation or E/I activity, and early/late representation gradients. A nonzero gradient is not proof of a useful learned recurrent computation; neither a small gradient nor clipping alone proves lost information.

Minimal and longer cells must be read together. Weak minimal-rule acquisition prevents a clean memory-capacity interpretation. Improvement over the earlier broad battery could reflect focused exposure or the new shared interface as well as the recurrent core. A reset effect establishes sensitivity to this trained recurrent history under the tested intervention; it does not establish biological correspondence, universal memory capacity, or the necessity of a separately named WM module. No attention, extra models, independent sweep or automatic budget extension was added.

## Platform migration

LSTM trained locally on Windows/RTX3070 Laptop; E/I trained remotely on Linux/RTX3090 at the user’s explicit request. Torch 1.13.1+cu117, NumPy 1.23.1, SciPy 1.8.1 and Pillow 9.1.1 matched. Python patch versions differed (3.10.6/3.10.12). The remote initializer contained exact unfitted local E/I tensors, and its common input/output tensors were checked against local LSTM update 0. It was not an after-profile model. Full task metadata matching is checked for paired analysis; cross-platform rendering/training is not claimed bitwise identical. Platform variation is an additional confound for differences between models. The original local supervisor was replaced without interrupting its active LSTM worker, and local E/I production was excluded. Production wall times on different GPUs are not an architecture-throughput comparison. Cloud artifacts were retrieved separately; pod cleanup and actual billing duration require the saved cloud cleanup receipt.

## Reproduction

Run: `C:\Users\jomor\Documents\VisualAttentionWorkingMemory\WorkingMemory\RecurrentComparison\runs\recurrent_20260912_202158`. Read README.md for exact cells, frame counts, equations and initialization. The run stores pinned source/config/parent identities, full optimizer/sampler/RNG checkpoints, metric/state logs and raw paired predictions. results.json contains every cell confusion matrix, class recall, binary hit/miss/false-alarm/correct-rejection counts, sensitivity/criterion, uncertainty and strata. analysis.json contains paired intervals and selected/terminal exposure. Profile fits remain separate from production.
