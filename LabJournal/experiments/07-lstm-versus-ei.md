# Focused recurrent memories learn the rules; LSTM leads on motion duration

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Does an added recurrent population improve focused sequence learning, and how does a conventional gated baseline compare with a rate/adaptation circuit?

**Design and ancestry.** Both inherit the selected sequence 6860 sensory model. A common 64 × 13 × 13 → 8 × 13 × 13 → 128 interface feeds either a 256-cell normalized LSTM or dense 256-unit E/I adaptive network. Each adds 512 state scalars. All learned sensory parameters remain trainable at lower learning rate. There is no attention or frame-growing cache.

**Exposure and optimization.** Both receive 40,000 new episodes / 5,000 updates on matched six-cell streams and select step 5000. LSTM trains locally; E/I is explicitly moved to a cloud RTX 3090. Gate normalization, initialization, sign constraints and clipping are documented in the source; frequent clipping in both arms is an optimization qualification. Cross-platform differences preclude a pure controlled hardware comparison.

**Result.** Both learn the new tasks. LSTM scores 79.69% on eight-transition motion versus E/I 68.95%, a paired 10.74 pp gap. Delayed orientation is 83.98% versus 85.74%; the small E/I advantage is uncertain. Resetting the added recurrent state before every frame collapses motion choices toward chance.

**Decision.** Investigate what the E/I state retained rather than invent a new circuit immediately. Neither an LSTM “counter” nor a “carousel” was explicitly implemented; learned gating can accumulate evidence, and behavior alone does not specify its internal algorithm. This is a comparison of fitted systems, not a biological ranking.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/RecurrentComparison/report.md](../../WorkingMemory/RecurrentComparison/report.md)
- [WorkingMemory/Research/recurrent_memory_without_attention.md](../../WorkingMemory/Research/recurrent_memory_without_attention.md)
- [WorkingMemory/RecurrentComparison/model.py](../../WorkingMemory/RecurrentComparison/model.py)
- [WorkingMemory/RecurrentComparison/analysis.json](../../WorkingMemory/RecurrentComparison/analysis.json)
- [WorkingMemory/RecurrentComparison/cloud_cleanup_receipt.json](../../WorkingMemory/RecurrentComparison/cloud_cleanup_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
