# Neural positive-control journal

**inconclusive: no established held-out task acquisition in budget.** One717,371-parameter learned spatiotemporal/axial-attention architecture; no KDA or GRU.

## Findings

- Native32-trial fixed-set fit: CE0.728147 → 0.112391; 32/32 correct; 245updates and 7840presentations. This is optimization/memorization evidence, not generalization.
- The first recipe failed with a saturated dense GELU readout:93.75% exactly-zero hidden activations; initial loss0.728147 → chance-level0.693147. All initial gradient paths and parameter updates existed, so startup movement alone did not validate sustained optimization.
- One targeted correction reused the same architecture and same initial seed, reduced AdamW lr0.001→0.0001 and accumulated all32fixed scenes instead of8; the initial2-example update was removed. This succeeded at the fixed-set gate. The combined correction does not identify which changed optimizer detail was causal. No architecture search.
- Original failed-fit/permutation evidence is preserved in initial_attempt_result.json; localization is in failed_fit_localization.json.
- Fresh shuffled-label diagnostic under the original failed recipe: CE0.708161 → 0.693233, BA0.5000, 516updates. This is a capacity/optimization check, not a many-permutation generalization null.

## Independent fresh-stream generalization

Fresh model72002, fresh empty AdamW, and native920001/2/3 streams; no fixed-set weights transferred. 2404updates ×8 = **19232fresh training trials**. Native B12/B20/B28 exposures: {'12': 6416, '20': 6408, '28': 6408}; event counts: {'target': 10961, 'catch': 2690, 'foil': 5581}. AdamW lr0.0001,wd0.0001,clip5,effective8/micro2.

Original deadline retained. Read REPORT.md and saved_artifact_verification.json for verified final results. No training continues.
