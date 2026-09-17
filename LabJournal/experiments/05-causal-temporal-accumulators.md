# One frame at a time: opponent traces beat the tested KDA and ConvGRU fits

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Can a causal fixed-size state replace direct access to two encoded images while retaining the sensory capabilities?

**Design and ancestry.** The selected contour-focused encoder at step 2268 is frozen. New common input projections/readout train around three competitors: spatial KDA, ConvGRU, and fast/slow opponent traces with fixed quadrature energies. Each processes the current frame and previous state, never both original encodings at once. The preserved original pair system remains a reference.

**Exposure.** Each new temporal model receives 4,032 updates × 32 = 129,024 fresh pairs under the contour-focused schedule. All select terminal 4032. Their encoder gets zero new updates; projection, temporal output and readout learning still has substantial representational freedom. The opponent has 141,968 trainable parameters in this specific frozen-encoder experiment, not in later fully trainable memory models.

**Result.** Opponent scores 98.44–100% across seven tasks, including 100% motion. KDA and ConvGRU motion are 31.70% and 75.67%. Resetting opponent history before frame two lowers motion to 25.89%; reversed frames with transformed labels yield 99.78%.

**Decision.** Continue with the opponent winner, holding other cores aside. History use and order sensitivity are established for these two-step tasks. They do not establish long-duration working memory, an MT-equivalent circuit, or inability of KDA/ConvGRU to improve with different training. Opponent contour gains over the pair reference also include new readout and extra training, so they do not isolate energy-channel necessity.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [PreAttentiveVision/TemporalIntegration/report.md](../../PreAttentiveVision/TemporalIntegration/report.md)
- [PreAttentiveVision/TemporalIntegration/README.md](../../PreAttentiveVision/TemporalIntegration/README.md)
- [PreAttentiveVision/TemporalIntegration/accumulators.py](../../PreAttentiveVision/TemporalIntegration/accumulators.py)
- [PreAttentiveVision/TemporalIntegration/results_temporal.json](../../PreAttentiveVision/TemporalIntegration/results_temporal.json)
- [PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/completion_receipt.json](../../PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
