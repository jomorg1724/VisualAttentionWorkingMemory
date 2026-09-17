# An existing output-only refit recovers much of motion performance

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Does the diagnostic readout finding transfer into an improvement of the existing deployed computation without changing its architecture?

**Design and ancestry.** Continue EI 5000 but train only its existing memory_output and motion/orientation heads: 33,670 parameters. Keep encoder, sensory traces, memory input and recurrent computation fixed in evaluation mode. Preserve compatible weights, Adam, sampler and RNG through a recorded training-policy migration. Frozen upstream computation avoids unnecessary sequence BPTT.

**Exposure.** 38,720 fresh standard-task episodes; validation selects terminal global step 9840. Final tests pair 512 examples in each of the six existing cells against the untouched parent.

**Result.** Eight-transition motion improves 70.31% → 79.30%, a paired +8.98 pp [5.86, 12.11]. Orientation and anchors remain high without a clearly detected large cost. All upstream parameters remain exactly unchanged.

**Decision.** Retain the refitted model and test blank-period retention. This demonstrates a useful output-fitting limitation in the parent; it does not establish superiority over equally exposed end-to-end continuation. It also does not establish equality to the old LSTM score, which used a different held-out draw. The probe from the preceding study was not silently installed as a new architecture.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/RecurrentComparison/ReadoutRefit/report.md](../../WorkingMemory/RecurrentComparison/ReadoutRefit/report.md)
- [WorkingMemory/RecurrentComparison/ReadoutRefit/results.json](../../WorkingMemory/RecurrentComparison/ReadoutRefit/results.json)
- [WorkingMemory/RecurrentComparison/ReadoutRefit/analysis.json](../../WorkingMemory/RecurrentComparison/ReadoutRefit/analysis.json)
- [WorkingMemory/RecurrentComparison/ReadoutRefit/completion_receipt.json](../../WorkingMemory/RecurrentComparison/ReadoutRefit/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
