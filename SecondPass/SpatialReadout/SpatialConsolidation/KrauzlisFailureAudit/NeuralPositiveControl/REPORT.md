# Conventional neural positive control — completed bounded experiment

**inconclusive: no established held-out task acquisition in budget.** One717,371-parameter learned spatiotemporal/axial-attention architecture; no KDA or GRU.

## Findings

- Native32-trial fixed-set fit: CE0.728147 → 0.112391; 32/32 correct; 245updates and 7840presentations. This is optimization/memorization evidence, not generalization.
- The first recipe failed with a saturated dense GELU readout:93.75% exactly-zero hidden activations; initial loss0.728147 → chance-level0.693147. All initial gradient paths and parameter updates existed, so startup movement alone did not validate sustained optimization.
- One targeted correction reused the same architecture and same initial seed, reduced AdamW lr0.001→0.0001 and accumulated all32fixed scenes instead of8; the initial2-example update was removed. This succeeded at the fixed-set gate. The combined correction does not identify which changed optimizer detail was causal. No architecture search.
- Original failed-fit/permutation evidence is preserved in initial_attempt_result.json; localization is in failed_fit_localization.json.
- Fresh shuffled-label diagnostic under the original failed recipe: CE0.708161 → 0.693233, BA0.5000, 516updates. This is a capacity/optimization check, not a many-permutation generalization null.

## Independent fresh-stream generalization

Fresh model72002, fresh empty AdamW, and native920001/2/3 streams; no fixed-set weights transferred. 2404updates ×8 = **19232fresh training trials**. Native B12/B20/B28 exposures: {'12': 6416, '20': 6408, '28': 6408}; event counts: {'target': 10961, 'catch': 2690, 'foil': 5581}. AdamW lr0.0001,wd0.0001,clip5,effective8/micro2.

Validation300native trials; checkpoint chosen only by validationCE, including step0. Initial validationCE0.693948; terminalCE0.683315. Choice frozen before test. Selected step2404; fixed terminal step2404. Test600fresh scenes (200perB), never used to tune model/optimizer. Both endpoint roles are reported, not test-selected.

| Endpoint | BA (95% bootstrap CI) | AUC (95% bootstrap CI) | CE | TP / TN / FP / FN |
|---|---|---|---|---|
| selected | 0.5000 [0.5000,0.5000] | 0.5000 [0.4762,0.5246] | 0.683315 | 342 / 0 / 258 / 0 |
| terminal | 0.5000 [0.5000,0.5000] | 0.5000 [0.4762,0.5246] | 0.683315 | 342 / 0 / 258 / 0 |

### Cued-target, foil and catch behavior

| Endpoint | Target hit rate | Foil false-report rate | Catch false-report rate |
|---|---|---|---|
| selected | 1.0000 (n=342) | 1.0000 (n=174) | 1.0000 (n=84) |
| terminal | 1.0000 (n=342) | 1.0000 (n=174) | 1.0000 (n=84) |

### Native baseline conditions

| Endpoint | B | BA | AUC | n |
|---|---|---|---|---|
| selected | 12 | 0.5000 | 0.4678 | 200 |
| selected | 20 | 0.5000 | 0.5523 | 200 |
| selected | 28 | 0.5000 | 0.4865 | 200 |
| terminal | 12 | 0.5000 | 0.4678 | 200 |
| terminal | 20 | 0.5000 | 0.5523 | 200 |
| terminal | 28 | 0.5000 | 0.4865 | 200 |

Uncertainty uses500bootstrap replicates of independent physical scenes, stratified byB×label. An all-positive classifier produces a degenerate BA interval[0.5,0.5]; this does **not** mean certainty about underlying representation or future performance. AUC intervals remain relevant. Single-seed, same-geometry, short-budget results do not establish reproducibility or absence of learnability.

### Endpoint computation integrity

Independent saved-prediction BA/AUC replay, native label/length checks, no scene-hash overlap across splits, all Adam counters equal checkpoint update count, and CPU-vs-saved-MPS prediction parity passed. Full numerical receipt: saved_artifact_verification.json.
The fresh terminal endpoint had only7.8125% exactly-zero dense GELU activations on the2-trial audit and retained nonzero gradients in all module groups. Therefore the original tiny-set saturation diagnosis does **not** establish the cause of fresh-stream generalization failure, and says nothing causal about prior KDA/GRU runs.

## Inputs and architecture

Native ±90renderer untouched:100×100RGB, every29/37/45frame, B12/20/28, exact cue/gap/dot/report timing and target/foil/catch law. Model receives images only. There is no cue crop, event boundary, privileged side, hand-coded optical flow or supplied relational rule. Length-homogeneous batches contain no sequence padding; no hidden label-bearing mask exists.

Learned stride-one3×3×3-equivalent convolution3→8 precedes spatial reduction. Learned5×5patch8→24 and2×2patch24→48 produce100tokens/frame. Two axial temporal-then-spatial self-attention blocks (width48,4heads) expose all retained time tokens; learned temporal attention pooling followed by4800→128→2MLP outputs finalCE. Details/equations in PROTOCOL.md. Spatial aggregation is lossy despite native-resolution input; no information-preservation or biological-plausibility claim.

## Budget, checks and artifacts

Original nonrenewable budget1800s; worker result completed at1669.71s, saved-artifact verification at1686.40s. One MPS worker at a time; Torch/numerical CPUthreads1. Corrected worker reused original absolute deadline. No cloud, prior model loads or external modifications. All new checkpoints are generated solely by this experiment.

CPU test was observed failing before implementation, then1passed in2.01s: direct-renderer equality, labels, batch/single inference, first/lastframe gradients, wholebatch/microbatch gradient parity(max error1.07e−6), everyparameter gradient and optimizer update. MPS profile independently confirmed allparameter updates and input gradients across everyframe. These are pathway checks, not learning claims.

Files: control.py; run_control.py; optimizer_correction.py; test_control.py; localize_failure.py; verify_saved.py; finalize_report.py; PROTOCOL.md; VERIFICATION.md; budget/progress/scene-manifest/check/selection/result JSON; fixed-set and held-out predictions; original/corrected fixed-set checkpoints and selected/terminal fresh checkpoints. See result.json for complete machine-readable metrics.

## Interpretation

The fixed-set diagnostic demonstrates that this conventional network can use its gradient path to fit native movies after an optimizer correction. It does not establish learned native cue-motion comparison on unseen trials. The held-out endpoint metrics above determine that question; negative results are limited to this architecture, optimizer recipe, one seed and finite exposure. No claim that all neural networks fail, or that an unchanged task is unsolvable, is supported.
