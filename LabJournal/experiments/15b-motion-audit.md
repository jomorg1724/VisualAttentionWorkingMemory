# Motion errors include decision bias and predate attention

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** How much of weak motion accuracy reflects class-score bias, and what did the actual training schedule prioritize?

**Design.** CPU-only reuse of saved predictions, without Torch/model inference. Audit all four trained models on the same 512 motion bases. Fit three effective zero-sum class offsets using only 128 existing D0 validation episodes, fixed regularization, then evaluate saved D0/D24 scores. This diagnostic fit is not installed in the trained checkpoint.

**Result.** Attention predicts right/up/left/down 13/348/151/0 times at D0: it never chooses down despite down-versus-rest AUC .738. Calibration improves 39.65% → 52.54%, +12.89 pp [8.59,16.99], but D24 transfer is small and uncertain. Continuation also benefits from calibration and still has stronger discrimination.

**Historical finding.** The earlier separately paired spatial experiment already showed a major motion loss. Attention cannot explain that earlier decline. The current 90/10 runs gave each motion delay only 1,600 episodes and excluded motion from checkpoint selection. Logged gradients are finite; clipping declines in all three arms, with no unique evidence of attention-only numerical collapse.

**Decision.** Run a controlled equal-total-update allocation comparison: continued 10% versus 50% motion, preserving architecture and protecting every task in selection. This is the current Stage1. The audit used previously inspected tests and fit offsets conditional on a small validation set; it is exploratory, not a fresh benchmark proving calibration solves the system.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.

| Frozen model | D0 BA% / AUC | D24 BA% / AUC | D0 predicted counts | D24 predicted counts |
|---|---:|---:|---|---|
| Spatial parent4400 | 46.09 / .798 | 25.00 / .756 | 231 / 12 / 104 / 165 | 0 / 512 / 0 / 0 |
| Continuation8400 | 50.78 / .817 | 31.64 / .730 | 37 / 215 / 218 / 42 | 0 / 435 / 55 / 22 |
| Additive feedback8400 | 45.31 / .808 | 25.00 / .734 | 19 / 153 / 307 / 33 | 0 / 512 / 0 / 0 |
| Pre-update attention8400 | 39.65 / .774 | 35.35 / .701 | 13 / 348 / 151 / 0 | 123 / 279 / 110 / 0 |

| Model and test | Original BA% | Corrected BA% | Paired change pp [95% CI] |
|---|---:|---:|---:|
| Attention D0 | 39.65 | 52.54 | +12.89 [8.59, 16.99] |
| Continuation D0 | 50.78 | 56.64 | +5.86 [2.15, 9.38] |
| Attention D24 | 35.35 | 37.70 | +2.34 [−1.76, 7.04] |
| Continuation D24 | 31.64 | 31.05 | −0.59 [−3.32, 2.15] |

| Arm | Validation D0 BA% at5400 /6400 /7400 /8400 | Validation D24 BA% |
|---|---|---|
| Attention | 48.44 /55.47 /27.34 /37.50 | 25.00 /22.66 /26.56 /36.72 |
| Continuation | 54.69 /39.06 /50.78 /46.09 | 25.00 /25.00 /30.47 /28.13 |
| Feedback | 50.78 /31.25 /40.63 /44.53 | 25.00 /25.00 /25.00 /25.00 |

## Evidence

- [WorkingMemory/PreUpdateAttention/MotionAudit/report.md](../../WorkingMemory/PreUpdateAttention/MotionAudit/report.md)
- [WorkingMemory/PreUpdateAttention/MotionAudit/findings.json](../../WorkingMemory/PreUpdateAttention/MotionAudit/findings.json)
- [WorkingMemory/PreUpdateAttention/MotionAudit/completion_receipt.json](../../WorkingMemory/PreUpdateAttention/MotionAudit/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
