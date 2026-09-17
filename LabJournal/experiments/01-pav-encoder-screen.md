# Five convolutional encoders: the first corrected sensory screen

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Which compact shared convolutional encoder learns a useful representation across seven two-frame sensory tasks? The user corrected the initial task to four cardinal directions of random-dot motion, grounded in Krauzlis-style stimuli. The original binary/static dot benchmark was stopped and has no locked final test. Its surviving artifacts are superseded diagnostics within this post-reset project, not a completed comparison.

**Design and ancestry.** Five new compact encoder adaptations—anti-aliased ResNet, ConvNeXt-GRN, InceptionNeXt, MobileNet-SE and VOne-ResNet—used a common ordered decoder. Each frame is encoded separately with shared weights. Signed differences and local displacement correlations preserve motion direction; a swap-symmetric change decoder cannot distinguish reversed motion. Encoder weights are not shared between candidate models.

**Exposure and selection.** One seed per model; 756 updates × 32 pairs = 24,192 pairs each, only 3,456 per task. All selected checkpoints were update 756 using validation mean task AUC. Final tests had 448 matched pairs per task; natural examples cluster by source photograph.

**What we learned.** ConvNeXt-GRN was the best broad fitted candidate here, with 80.6% motion but only 60.3% contour. MobileNet-SE and VOne-ResNet were stronger on contour. This was a practical short acquisition screen, not a reason to commit permanently to one architecture. Some low accuracies coexisted with useful AUC: poorly placed class decisions do not prove absent information.

**Decision.** Preserve all candidates, investigate complementary contour errors, and continue the useful learned ConvNeXt lineage. High-level neuroscience resemblance did not reliably predict performance: VOne's Gabor front end had 23.9% motion in this run. That is not evidence against biological oriented filters generally.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [PreAttentiveVision/runs/multitask_20260912_141316/report.md](../../PreAttentiveVision/runs/multitask_20260912_141316/report.md)
- [PreAttentiveVision/research.md](../../PreAttentiveVision/research.md)
- [PreAttentiveVision/krauzlis_stimulus.md](../../PreAttentiveVision/krauzlis_stimulus.md)
- [PreAttentiveVision/two_frame_task_research.md](../../PreAttentiveVision/two_frame_task_research.md)
- [PreAttentiveVision/results_multitask.json](../../PreAttentiveVision/results_multitask.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
