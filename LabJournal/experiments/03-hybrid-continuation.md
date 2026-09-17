# Gabor and late-SE additions did not solve contour

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Can a compact oriented-feature branch or channel gate reproduce the useful properties suggested by the ensemble audit?

**Design and ancestry.** Three siblings start from the exact trained ConvNeXt step-756 parent: unchanged continuation, a fixed-Gabor residual branch, and a late-SE residual gate. Compatible weights, Adam states and sampler progress are preserved. The two additions initialize to the parent function. Architecture additions and their subsequent learning are the interventions.

**Exposure.** Every arm receives 756 new updates / 24,192 new pairs; all select global step 1512. New paired final examples also evaluate the frozen parent. The predeclared useful-change screen required at least +3 pp contour BA against continuation, with motion regression no worse than −2 pp.

**Result.** Neither addition passed. Gabor contour was 58.7% against control 63.8%; late-SE was 63.4%. Continued training alone improved the frozen parent's contour by 12.50 pp. Both additions reached almost perfect motion decisions, but control motion ranking was already nearly perfect.

**Decision.** Do not claim that copying plausible components solved grouping. Keep late-SE as a useful low-cost candidate and test whether the remaining weak task simply needs more of the existing training budget. The negative result concerns these initialized additions and exposure, not every Gabor/SE design.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [PreAttentiveVision/runs/hybrids_20260912_152427/report.md](../../PreAttentiveVision/runs/hybrids_20260912_152427/report.md)
- [PreAttentiveVision/results_hybrids.json](../../PreAttentiveVision/results_hybrids.json)
- [PreAttentiveVision/results_hybrids_analysis.json](../../PreAttentiveVision/results_hybrids_analysis.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
