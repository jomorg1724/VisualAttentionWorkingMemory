# Can complementary encoder errors be combined cheaply?

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Do the existing probability vectors contain useful complementary errors before we spend GPU time combining components?

**Design.** A CPU-only saved-score analysis fixed equal-weight ConvNeXt+MobileNet and ConvNeXt+VOne ensembles. Validation on motion and contour selected MobileNet as the partner before loading test predictions. No encoder inference, optimizer update or new training examples were used. Runtime was 4.86 seconds.

**Result.** ConvNeXt+MobileNet increased contour from the stronger constituent's 71.7% to 74.8%, but lowered motion from 80.6% to 70.8%. On contour it fixed 19 MobileNet errors and broke five correct answers; on motion it broke 48 correct ConvNeXt decisions while fixing only four.

**Interpretation.** Error complementarity is useful evidence for a combination experiment, but indiscriminate probability averaging destroys part of the strong model's performance. An oracle that selects a correct constituent using the true label is not a deployable model. Neither oracle accuracy nor ensemble improvement identifies which feature computation caused complementarity.

**Decision.** Test small learned feature additions against ordinary continuation. This reused already inspected test scores, so it is exploratory development evidence rather than a fresh confirmation.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.

| Pair and task | Stronger constituent BA | Ensemble BA | Difference, percentage points (95% paired bootstrap CI) | Constituent → ensemble AUC |
|---|---:|---:|---:|---:|
| ConvNeXt + MobileNet: motion | ConvNeXt 80.6% | 70.8% | −9.82 (−12.42, −7.37) | .936 → .929 |
| ConvNeXt + MobileNet: contour | MobileNet 71.7% | 74.8% | +3.13 (+1.12, +5.32) | .830 → .852 |
| ConvNeXt + VOne: motion | ConvNeXt 80.6% | 77.7% | −2.90 (−6.10, +0.36) | .936 → .922 |
| ConvNeXt + VOne: contour | VOne 72.1% | 74.8% | +2.68 (+0.66, +4.72) | .804 → .834 |

| Second encoder / task | Both correct | First only correct | Second only correct | Both wrong | Label-informed oracle correct |
|---|---:|---:|---:|---:|---:|
| MobileNet / motion | 110 | 251 | 14 | 73 | 375/448 = 83.7% |
| MobileNet / contour | 179 | 91 | 142 | 36 | 412/448 = 92.0% |
| VOne / motion | 87 | 274 | 20 | 67 | 381/448 = 85.0% |
| VOne / contour | 178 | 92 | 145 | 33 | 415/448 = 92.6% |

## Evidence

- [PreAttentiveVision/combination_analysis.md](../../PreAttentiveVision/combination_analysis.md)
- [PreAttentiveVision/combination_analysis.json](../../PreAttentiveVision/combination_analysis.json)
- [PreAttentiveVision/component_combination_research.md](../../PreAttentiveVision/component_combination_research.md)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
