# Joint attention substantially improves delayed orientation, with a motion cost

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Before the next recurrent update, can the model select between incoming sensory information and its own remembered information?

**Design and ancestry.** Start the same spatial 4400 parent as ordinary continuation and additive feedback. Previous spatial memory supplies 169 queries. Current sensory plus previous memory supplies 338 keys/values. Two heads of dimension 32 may attend to either source; they are not hardwired “visual” and “memory” heads. The attended field replaces raw sensory drive into the existing E/I memory input. Original sensory and old-memory/current-sensory comparison routes remain.

**Exposure.** 32,000 additional episodes / 4,000 updates, selecting global 8400. Tasks, teaching, 90/10 schedule and compatible Adam state stay unchanged. The added module has 27,590 parameters, total model 556,128, with no added persistent state. Cloud RTX 3090 versus local controls is a disclosed platform difference.

**Result.** Single orientation D24 rises from continuation 58.40% to attention 79.30%, a paired +20.90 pp [15.82,25.98]. Binding stays near ceiling. Immediate motion falls 50.78% → 39.65%, while motion D24 is 35.35% against 31.64%. It is a substantial specific gain, not an all-task winner.

**Decision.** Preserve the advance and investigate the regressions. Frozen phase interventions and a saved-score motion audit follow in pages 15 a/15 b. The old cloud pod was retrieved and deleted; it is separate from the new TrainingExposure pod. High attention mass by itself cannot demonstrate useful memory content or biological attention.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/PreUpdateAttention/report.md](../../WorkingMemory/PreUpdateAttention/report.md)
- [WorkingMemory/PreUpdateAttention/model.py](../../WorkingMemory/PreUpdateAttention/model.py)
- [WorkingMemory/PreUpdateAttention/analysis.json](../../WorkingMemory/PreUpdateAttention/analysis.json)
- [WorkingMemory/PreUpdateAttention/retrieval_receipt.json](../../WorkingMemory/PreUpdateAttention/retrieval_receipt.json)
- [WorkingMemory/PreUpdateAttention/cloud_cleanup_receipt.json](../../WorkingMemory/PreUpdateAttention/cloud_cleanup_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
