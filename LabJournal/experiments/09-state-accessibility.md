# Earlier motion evidence remains in the firing-rate population

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Is early motion information missing from the E/I state, or is the deployed output failing to use it?

**Design.** Freeze EI 5000. Fit analysis-only ridge and small MLP readouts using 3,072 training movies, 768 validation and 1,024 held-out movies. Canonical templates and rotations are disjoint between splits. The duration probe sees sensory 128 plus firing rates 256; adaptation is not needed for this rescue. Count probes target early directional evidence through independent count coordinates.

**Result.** A linear duration readout improves deployed 68.65% to 78.13%, +9.47 pp [6.84, 12.11]. The MLP yields 77.64%, no observed advantage. Early-count paired-difference R² is .875 at transition four and .800 at final firing rates, while final sensory features alone give .002.

**Interpretation.** Useful early evidence remains accessible through the ordinary firing-rate route. The deployment problem includes fitting or use of the representation; changing recurrence is not yet necessary. A successful probe is constructive evidence of accessibility. A failed probe would not establish erasure.

**Decision.** Defer slow excitatory synapses and refit the existing output only. These controlled suffix-matched movies and raw-score AUC are distinct from the normal task stream and earlier softmax AUC. A Windows int 32 target issue was corrected to int 64; saved features and failed-attempt evidence were preserved. Total successful supervisor time was 247.40 seconds within the 1,200-second allowance.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/RecurrentComparison/StateDiagnostic/report.md](../../WorkingMemory/RecurrentComparison/StateDiagnostic/report.md)
- [WorkingMemory/RecurrentComparison/StateDiagnostic/summary.json](../../WorkingMemory/RecurrentComparison/StateDiagnostic/summary.json)
- [WorkingMemory/RecurrentComparison/StateDiagnostic/completion_receipt.json](../../WorkingMemory/RecurrentComparison/StateDiagnostic/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
