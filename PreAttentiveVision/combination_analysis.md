# Combining the measured strengths

**A targeted combination experiment is worth doing; a blanket equal-weight ensemble is not the right default.** The saved predictions contain useful contour complementarity: averaging ConvNeXt-GRN and MobileNet-SE probabilities improves contour accuracy over MobileNet alone. The same averaging harms ConvNeXt's motion accuracy. These results justify a task-aware combination reference before designing a compact fused encoder. They do not identify which architectural blocks caused the complementarity.

This analysis used only saved probability vectors, in **4.86 seconds on CPU**, with CUDA visibility disabled and numerical-library threads set to one. It performed zero model inferences, optimizer steps or GPU work. The original trained models remain unchanged.

## Fixed design and selection

Before computing ensemble metrics, `combination_protocol.json` fixed two pairs: ConvNeXt-GRN + MobileNet-SE and ConvNeXt-GRN + VOne-ResNet. Each uses the arithmetic mean of its probability vectors with weights exactly **0.5 / 0.5**, followed by argmax. There was no calibration, weight search or learned gate.

The pair was selected using the equal-task mean one-versus-rest AUC on **motion and contour validation predictions only**, from the existing final checkpoints. Selection was saved before the script loaded test predictions. ConvNeXt + MobileNet scored **0.9094**, versus **0.8920** for ConvNeXt + VOne, so MobileNet was selected. On validation, the selected pair already showed the tradeoff: contour BA increased from MobileNet's 75.4% to 75.9%, but motion BA fell from ConvNeXt's 81.3% to 71.4%.

The original test set was already inspected during the encoder comparison. The following is therefore **posthoc analysis of reused test predictions, not fresh confirmatory evidence**. Both predefined pairs are reported transparently. No subsequent ensemble was fitted or computed.

## Actual equal-weight ensemble results

Each row uses the same 448 held-out pair identities for both constituents. “Difference” compares with the stronger constituent in that task, as identified descriptively from the held-out results. Paired intervals compare against that named, fixed constituent and do not provide post-selection or multiple-comparison coverage guarantees.

| Pair and task | Stronger constituent BA | Ensemble BA | Difference, percentage points (95% paired bootstrap CI) | Constituent → ensemble AUC |
|---|---:|---:|---:|---:|
| ConvNeXt + MobileNet: motion | ConvNeXt 80.6% | 70.8% | −9.82 (−12.42, −7.37) | .936 → .929 |
| ConvNeXt + MobileNet: contour | MobileNet 71.7% | 74.8% | +3.13 (+1.12, +5.32) | .830 → .852 |
| ConvNeXt + VOne: motion | ConvNeXt 80.6% | 77.7% | −2.90 (−6.10, +0.36) | .936 → .922 |
| ConvNeXt + VOne: contour | VOne 72.1% | 74.8% | +2.68 (+0.66, +4.72) | .804 → .834 |

The selected contour ensemble fixes **19 MobileNet errors while breaking 5 correct decisions**: 14 net extra correct trials. Its AUC gain is +0.0214 (paired CI +0.0105 to +0.0325), so the improvement is not solely a fixed-argmax threshold effect. The VOne contour ensemble fixes 17 errors and breaks 5, with AUC gain +0.0304 (+0.0172 to +0.0433).

For motion, the selected MobileNet ensemble fixes only 4 ConvNeXt errors but breaks 48 correct decisions. Averaging a weak motion head with a stronger one changes margins and class decisions unfavorably. Neither constituent's probability calibration was established, so this is evidence about the specified arithmetic average, not a general impossibility of combining representations.

## How much overlap exists?

“First” is ConvNeXt-GRN in every row. Counts are paired trial outcomes, not independent model samples.

| Second encoder / task | Both correct | First only correct | Second only correct | Both wrong | Label-informed oracle correct |
|---|---:|---:|---:|---:|---:|
| MobileNet / motion | 110 | 251 | 14 | 73 | 375/448 = 83.7% |
| MobileNet / contour | 179 | 91 | 142 | 36 | 412/448 = 92.0% |
| VOne / motion | 87 | 274 | 20 | 67 | 381/448 = 85.0% |
| VOne / contour | 178 | 92 | 145 | 33 | 415/448 = 92.6% |

The oracle credits a trial whenever either constituent is correct. It requires the true label to choose the answer: **an unattainable deployable oracle construction and an upper bound only for choosing between these two saved answers**, not a measured ensemble, a promise of learnable gating, or an upper bound on a newly trained network. The contour error disagreement is substantial, but the realized fixed ensemble captures only part of that potential.

## Development implication and competing explanations

The next narrowly justified reference is **ConvNeXt for motion, with the validation-selected ConvNeXt + MobileNet combination for contour**. Task identity already selects the supervised output head; this does not require labels or a correctness oracle. This routing was not evaluated as a new seven-task system here and is a proposed next experiment. Confirming it should use fresh held-out pairs. It would also provide a concrete reference for a later compact shared encoder that preserves local feature and grouping computations, instead of assuming that copying favored blocks will reproduce ensemble behavior.

The current evidence does **not** distinguish encoder representation complementarity from separately learned decoder weights, calibration differences, initialization or unequal acquisition maturity. All models received equal exposure but only **108 updates / 3,456 training pairs per task and one seed**. ConvNeXt motion validation BA rose 25.9% → 51.8% → 81.3% across the three looks; MobileNet contour rose 52.7% → 55.8% → 75.4%. VOne contour AUC rose .550 → .721 → .843, and its intermediate BA lagged ranking (.540 at the second look). These trajectories leave both ongoing learning and biased decisions as live explanations. They do not establish permanent architectural specialization or biological mechanisms.

Keeping two encoders also costs more computation than keeping one. This cheap prediction analysis demonstrates a reason to test selective combination; it does not establish that a larger fused network is the best speed/accuracy tradeoff. No new training or GPU work was launched.

## Artifacts and uncertainty

- `analyze_combinations.py`: reproducible CPU-only analysis.
- `combination_protocol.json`: fixed weights, selection rule and interpretation scope.
- `combination_validation_selection.json`: validation results and selected pair, saved before test loading.
- `combination_analysis.json`: complete counts, metrics, paired differences, trajectories, runtime and input/script SHA256 identities.
- `combination_analysis.log`: retained execution output.

Intervals use **2,000 paired generated-pair bootstrap resamples per task**, with the same sampled trial indices applied to all compared probability vectors. They condition on the saved models and this task distribution; they do not measure training-seed variation or remove the limitations of reused-test exploration. Motion and contour pairs have no repeated natural source-image clustering issue. All original checkpoints and run records are preserved.
