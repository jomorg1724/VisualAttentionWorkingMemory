# The broad sequence battery did not isolate memory capacity

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Does the existing sensory opponent model support integration, retaining a decision, and remembering an item until a later query/probe?

**Design.** Extend all seven sensory families with sequence rules, visible/flashed cues, integration lengths, delays, distractors and retrospective probes. Warm-start opponent 4032 and allow all 408,728 learned parameters to adapt; fixed trace coefficients and energy equations remain fixed. No separate working-memory module is added.

**Exposure and selection.** 39,200 fresh episodes / 9,800 updates were completed. Validation selected update 6,860 after 27,440 episodes; terminal exposure must not be attributed to that selected checkpoint. Each final cell has 128 held-out episodes.

**Result.** Sensory anchor ranking remained strong (mean AUC .9964), while integration/decision and retrospective cell means were .5154 and .5184. Several simplest new-rule conditions already stayed near chance. Thus “minimal-delay rules remained weak” means the model often could not perform the newly instructed task even when almost no waiting was required.

**Decision.** Focus the next comparison on learnable motion-duration and one-item orientation tasks with explicit recurrent memory. Longer-delay failure here cannot be cleanly called a retention limit because the baseline rule was not consistently acquired. This broad battery is retained as a recorded negative acquisition result, not silently discarded or rewritten.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/report.md](../../WorkingMemory/report.md)
- [WorkingMemory/TASK_BATTERY.md](../../WorkingMemory/TASK_BATTERY.md)
- [WorkingMemory/results.json](../../WorkingMemory/results.json)
- [WorkingMemory/completion_receipt.json](../../WorkingMemory/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
