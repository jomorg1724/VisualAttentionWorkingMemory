# Contour succeeds when it receives enough training allocation

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Is contour weakness caused partly by how the common training budget is distributed?

**Design and ancestry.** Both arms start from late-SE step 1512 with identical architecture, weights and Adam state. Uniform training gives each of seven tasks 1/7 of updates; focused training gives contour 1/2 and each other task 1/12. Explicit new task-local RNG streams provide matching per-task prefixes, rather than pretending to replay the old single-stream sampler.

**Exposure and selection.** Profiled costs pin 756 additional updates / 24,192 pairs per arm. Uniform gets 3,456 contour pairs; focus gets 12,096, with 2,016 per other task. Both select terminal step 2268 by validation minimum task BA, then mean task AUC. A fresh test has 448 paired examples per task.

**Result.** Contour rises 67.9% → 96.2% with a paired +28.35 pp interval [23.91, 32.59]. AUC rises .771 → .994, showing an improvement in discrimination rather than just an argmax shift. All seven task point estimates exceed 95%. Contrast falls 1.56 pp, and its interval extends beyond the descriptive two-point preservation tolerance.

**Decision.** Adopt focused step 2268 as the PAV reference. No new architecture was necessary to close this particular contour gap. The task schedule is the experimentally changed package; gradient conflict or a specific optimization mechanism was not identified. Early focused motion temporarily fell before recovering, so a single bad early checkpoint would have given the wrong conclusion.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [PreAttentiveVision/runs/allocation_20260912_160414/report.md](../../PreAttentiveVision/runs/allocation_20260912_160414/report.md)
- [PreAttentiveVision/results_allocation.json](../../PreAttentiveVision/results_allocation.json)
- [PreAttentiveVision/results_allocation_analysis.json](../../PreAttentiveVision/results_allocation_analysis.json)
- [PreAttentiveVision/next_experiment_task_allocation.md](../../PreAttentiveVision/next_experiment_task_allocation.md)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
