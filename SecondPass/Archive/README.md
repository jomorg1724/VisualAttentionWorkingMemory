# Archived experiments

Archived 2026-11-07. Each folder here was last touched before 2026-10-10 and is not imported by any current experiment. They were moved unchanged from `SecondPass/<name>/`.

- Anything that was imported as `SecondPass.<name>` now lives at `SecondPass.Archive.<name>`. Code inside these folders that imports sibling archived modules by their old path will need that prefix to run again.
- Links in `LabJournal/` and older reports still point to the old `SecondPass/<name>/` paths.

Old folders that current work still imports were left in place: PreAttentiveVision, WorkingMemory, TaskSuite, StabilizedAttentionKrauzlis, StabilizedAttentionVDA, PixelContrastiveVAE, PredictiveConsistencyVAE, JointTraining, VariationalMotionPredictor, OutcomeContrastiveRViT, ThreeFrameConvVAE, TwoFrameRViT, SpatialReadout, SpatialComparisonReadout, DelayedFrameGRU and SingleStimulusRViT. LocalComparisonKrauzlis also stays, because the manuscript reads its data. SecondPass/papers (the bibliography) also stays.
