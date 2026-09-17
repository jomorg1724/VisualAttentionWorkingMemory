# Ordering sensitivity: the E/I model benefits from later winning evidence

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Is the E/I duration deficit associated with the order of evidence even when the correct answer and final visual suffix are fixed?

**Design.** Freeze both selected step-5000 models. Construct 512 early/late matched pairs of continuous eight-transition dot movies. Match the complete direction count vector, winner, count margin, first direction, final two directions, total switch count and exact final two evidence rasters. Regenerate movies with restored nuisance RNG; do not shuffle images into physically discontinuous motion.

**Result.** E/I improves 72.27% → 78.52% when winning evidence occurs later: +6.25 pp [2.73, 9.58]. LSTM changes 85.74% → 84.18%, with an interval spanning zero. The model-by-order interaction is +7.81 pp [3.90, 11.53].

**What it identifies.** E/I is more sensitive to this controlled reordering. Moving the winner also changes distractor placement and local run structure; this does not uniquely identify leak, adaptation or saturation. The test is a selected schedule bank, not the earlier random six-cell distribution. Direction rotations are not independent schedule templates.

**Decision.** Test recoverability of early evidence from frozen E/I states. This diagnostic took 71.95 seconds with zero training and no cloud restart. It justified a question, not an automatic slow-synapse modification.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.

| Selected step5000 model | Early BA | Late BA | Late − early, percentage points (paired95% CI) |
|---|---:|---:|---:|
| LSTM |85.74% (439/512)|84.18% (431/512)|−1.56 [−4.88,+1.56]|
| E/I adaptive |72.27% (370/512)|78.52% (402/512)|+6.25 [+2.73,+9.58]|

## Evidence

- [WorkingMemory/RecurrentComparison/RecencyDiagnostic/report.md](../../WorkingMemory/RecurrentComparison/RecencyDiagnostic/report.md)
- [WorkingMemory/RecurrentComparison/RecencyDiagnostic/README.md](../../WorkingMemory/RecurrentComparison/RecencyDiagnostic/README.md)
- [WorkingMemory/RecurrentComparison/RecencyDiagnostic/summary.json](../../WorkingMemory/RecurrentComparison/RecencyDiagnostic/summary.json)
- [WorkingMemory/RecurrentComparison/RecencyDiagnostic/completion_receipt.json](../../WorkingMemory/RecurrentComparison/RecencyDiagnostic/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
