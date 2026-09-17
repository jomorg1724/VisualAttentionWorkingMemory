# Orientation survives the delay but the ordinary comparison fails

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Does the long-delay orientation failure reflect absent sample information, an unstable decoding map, or ineffective probe comparison?

**Design.** Freeze Retention 14800. Use 2,048 probe-training, 512 validation and 512 independent test base episodes, each paired across four delays. Decode the actual rendered axial angle, not the latent stimulus level: random context angle makes that shortcut invalid. Fit train-only scaling and validation-selected linear/MLP probes. Compare pre-probe firing rates with separately encoded probe images.

**Result.** At D24 a delay-specific linear readout recovers orientation from firing rates with 4.25° mean absolute error. The sample-trained decoder transferred to those states instead has 44.96° error. A label-trained comparator without angle inputs reaches 73.44% against deployed 50.00%; a separate angle-supervised diagnostic route reaches 87.30%. A post-probe label readout reaches only 52.73%.

**Interpretation.** The firing-rate state contains useful orientation information before the probe. Its useful decoding map is not stable across time; scale/offset changes can contribute. The operational comparator rescue shows that extra angle supervision is not necessary to demonstrate accessible comparison information. Post-probe probe failure does not prove erasure or rule out nonlinear information.

**Decision.** Investigate the memory/comparison interface and the user's spatial-memory competitor. Preserve the original model. The smallest 7.5° changes remain difficult even for the angle-based diagnostic comparator; this is not a universal replacement readout.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/RecurrentComparison/OrientationDiagnostic/report.md](../../WorkingMemory/RecurrentComparison/OrientationDiagnostic/report.md)
- [WorkingMemory/RecurrentComparison/OrientationDiagnostic/summary.json](../../WorkingMemory/RecurrentComparison/OrientationDiagnostic/summary.json)
- [WorkingMemory/RecurrentComparison/OrientationDiagnostic/paired_analysis.json](../../WorkingMemory/RecurrentComparison/OrientationDiagnostic/paired_analysis.json)
- [WorkingMemory/RecurrentComparison/OrientationDiagnostic/completion_receipt.json](../../WorkingMemory/RecurrentComparison/OrientationDiagnostic/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
