# The orientation improvement depends on memory-source routing during blanks

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** When is memory-source attention used, and does changing evidence-period routing recover motion?

**Design.** Freeze attention 8400 and continuation 8400. Reuse 512 paired orientation and 512 paired motion episodes; D0 orientation uses 128. An analysis wrapper excludes memory keys/values separately during sample, inserted blanks, identity query or probe. Previous-memory queries and ordinary E/I recurrence remain. Moving-frame motion also tests replacing attention drive with raw sensory H. No main-model training occurs.

**Result.** Excluding memory source only during D24 blanks lowers orientation 79.30% → 50.00%, −29.30 pp [−33.01,−25.78]. Sample, query and probe exclusions show no comparable cost. For motion, memory exclusion improves D0 by 3.12 pp; raw sensory bypass improves D0 by 3.91 pp and D24 by 5.86 pp.

**Interpretation.** The learned blank-period routing matters. Exclusion both removes recirculated values and renormalizes onto blank visual input, so active refresh and rejection of irrelevant drive remain entangled. Acute bypass is out of distribution and cannot prove that training with it would retain orientation gains.

**Decision.** Examine allocation and class decisions before adding new recurrence. These small routing rescues and the calibration rescue on page 15 b cannot be arithmetically added: they are different interventions on overlapping errors. The model's sensory branch already contains opponent history, not only current-frame information.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md](../../WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md)
- [WorkingMemory/PreUpdateAttention/MechanismDiagnostic/results.json](../../WorkingMemory/PreUpdateAttention/MechanismDiagnostic/results.json)
- [WorkingMemory/PreUpdateAttention/MechanismDiagnostic/branch_statistics.json](../../WorkingMemory/PreUpdateAttention/MechanismDiagnostic/branch_statistics.json)
- [WorkingMemory/PreUpdateAttention/MechanismDiagnostic/completion_receipt.json](../../WorkingMemory/PreUpdateAttention/MechanismDiagnostic/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
