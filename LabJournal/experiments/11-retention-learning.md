# Retaining a motion judgment improves; delayed orientation comparison remains weak

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** What happens when the existing model must process 0, 4, 12 or 24 blank frames after the relevant evidence?

**Design and ancestry.** Start refit 9840. Motion asks which of four directions occupied the greatest total duration in eight transitions, followed by D blank frames and report. Orientation shows a sample, D blanks, an identity query and a comparison probe. The renderer's post_delay is query-to-probe and is fixed to zero; it is not the manipulated sample retention interval. Match underlying evidence/probes across delays.

**Training.** 39,680 fresh episodes; select terminal global 14800. Train the existing recurrent core and outputs, with sensory/opponent/memory_input fixed. The primary delays are trained, not extrapolated. No new architecture or attention is added.

**Result.** Motion becomes about 79–80% across all four delays. At D24 the parent moves from 25% to 80.08%. Orientation reaches 95.70% at D0, 65.82% at D12, and 49.02% at D24.

**Clear reading.** On about one in five motion trials the model chooses the wrong duration winner even without inserted blanks. Waiting through 24 blanks adds little observed error. Orientation differs: the answer depends on a later probe, so a previously computed category alone is insufficient.

**Decision.** Diagnose whether orientation is inaccessible, transformed, or badly compared. Parent motion D24 has AUC .789 despite 25% choices, so chance decisions cannot be called erased information. This trained near-flat motion curve is not a guarantee of perfect retention or proof of distinct biological stores.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/RecurrentComparison/Retention/report.md](../../WorkingMemory/RecurrentComparison/Retention/report.md)
- [WorkingMemory/RecurrentComparison/Retention/results.json](../../WorkingMemory/RecurrentComparison/Retention/results.json)
- [WorkingMemory/RecurrentComparison/Retention/analysis.json](../../WorkingMemory/RecurrentComparison/Retention/analysis.json)
- [WorkingMemory/RecurrentComparison/Retention/completion_receipt.json](../../WorkingMemory/RecurrentComparison/Retention/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
