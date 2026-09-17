# Additive controller feedback gives a narrow gain, not a maintenance explanation

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Can a learned recurrent controller preserve relevant memory through blanks while teaching stays exactly the same?

**User correction.** A proposed delayed orientation-report objective with explicit angle supervision was rejected before implementation. The experiment therefore changes architecture only, retaining existing tasks, cues, labels, losses, heads and the 90/10 schedule.

**Design and ancestry.** Both arms continue spatial 4400. Control is unchanged; competitor adds a 32-unit E/I controller with rate/adaptation states observing full sensory/memory summaries. It supplies additive previous-state currents to memory, with no cue crop, blank oracle, hold/forget gate or classifier shortcut. Added package: 45,537 parameters and 64 state scalars.

**Exposure and result.** Both complete 32,000 additional episodes and select global 8400. Single D24 rises 58.40% control → 64.26% feedback, but single D12 and motion worsen. Silencing only controller-to-memory currents during the 24 blanks changes single D24 by −0.20 pp [−1.17, .78] and no binding/motion choices.

**Interpretation and decision.** The gain does not require ongoing additive feedback into spatial memory during blanks under this intervention. Controller state continues updating and can influence later frames, so this does not remove every controller contribution. Test pre-update joint sensory/memory attention as the user's next alternative.

**Execution note.** A Windows status-file sharing violation interrupted the original supervisor, not completed training. A replacement adopted the independently completed validation and resumed fixed remaining jobs without repeating training or extending the deadline. Recovery evidence remains preserved.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/SelectiveMaintenance/report.md](../../WorkingMemory/SelectiveMaintenance/report.md)
- [WorkingMemory/SelectiveMaintenance/README.md](../../WorkingMemory/SelectiveMaintenance/README.md)
- [WorkingMemory/SelectiveMaintenance/analysis.json](../../WorkingMemory/SelectiveMaintenance/analysis.json)
- [WorkingMemory/SelectiveMaintenance/completion_receipt.json](../../WorkingMemory/SelectiveMaintenance/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
