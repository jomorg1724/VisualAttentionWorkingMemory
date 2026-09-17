# Keeping a spatial field helps binding but does not solve every task

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Task/metric definitions](../TASKS_AND_METRICS.md)

Status: **completed**. Journal reconstruction: 2026-09-14T00:21:31.576956+00:00. Source experiment dates and costs are retained in the linked run records.

**Question.** Does retaining a 64-channel 13 × 13 memory field avoid the information bottleneck of compressing everything into a vector?

**Design and ancestry.** Start compatible sensory weights from Retention 14800. Compare a dense E/I memory and a convolutional E/I rate/adaptation field. Both receive an explicit learned old-memory/current-sensory comparator before classification. Dense inherits its trained core; spatial initializes a new core. All learned components may train, with lower sensory LR.

**Tasks and exposure.** Existing single-item orientation, new two-location orientation preserve/swap binding at D0/4/12/24, and a 10% motion-duration allocation. Both get 35,200 episodes / 4,400 updates and select terminal 4400. Motion is excluded from the eight-primary-cell AUC checkpoint selection. Four additional binding-center evaluations measure interpolation.

**Result.** Spatial binding D24 is 93.36% against dense 49.61%, and unseen-center binding is 94.53%. Single D24 remains near chance in both. Spatial motion D0/D24 drops to 46.68%/25.00%, versus the same-draw untouched parent's 79.10%/77.54%.

**Interpretation.** Spatial organization is a useful binding candidate, not a general fix. Binding is not a harder two-item version of native orientation: rendering, change sizes and query differ; a full swap can be detected by storing just one location. Spatial has 42.25 × the dense recurrent-state scalars, with different geometry and prior experience. Locality is not isolated.

**Decision.** Preserve the motion-competent parent and spatial candidate, investigate selective maintenance without changing teaching. The major motion regression begins here, before the later attention module.

## Selected measured tables

The following tables are transcribed directly from the original report. They retain that report's dataset, selection and uncertainty conventions; they are not a new pooled evaluation. The source report contains further strata, definitions and uncertainty.



## Evidence

- [WorkingMemory/SpatialComparison/report.md](../../WorkingMemory/SpatialComparison/report.md)
- [WorkingMemory/Research/spatial_ei_memory.md](../../WorkingMemory/Research/spatial_ei_memory.md)
- [WorkingMemory/SpatialComparison/model.py](../../WorkingMemory/SpatialComparison/model.py)
- [WorkingMemory/SpatialComparison/analysis.json](../../WorkingMemory/SpatialComparison/analysis.json)
- [WorkingMemory/SpatialComparison/completion_receipt.json](../../WorkingMemory/SpatialComparison/completion_receipt.json)

Journal interpretation does not supersede immutable run source/configuration, predictions or receipts. Corrections should be dated and preserve the previous conclusion's context; see [maintenance](../MAINTENANCE.md).
