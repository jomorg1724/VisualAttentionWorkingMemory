# Second pass

**October 4, 2026: angular contrastive motion learning completed successfully on the simplified task. No training is active.**

The [technical report](AngularContrastiveMotion/TECHNICAL_REPORT.md) documents the fresh CNN, direction-supervised spring loss and 100% held-out before/after change decisions. The original native cue/distractor task remains untested for this model. Use [current status](../LabJournal/CURRENT_STATUS.md), the [experiment catalog](../LabJournal/EXPERIMENT_CATALOG.md) and the [research history](../LabJournal/RESEARCH_HISTORY.md) for the distinction between final results and older launch snapshots.

## Latest motion-learning sequence

| Implementation | Question | Observed result |
|---|---|---|
| [WeightedMeanConvGRU](WeightedMeanConvGRU/README.md) | Does a decayed raw-frame mean plus CNN/GRU solve the cue-free single-stimulus task? | Cancelled at570updates for the VAE; no final test |
| [ThreeFrameConvVAE](ThreeFrameConvVAE/FINAL_REPORT.md) | Can a spatial variational representation reconstruct three motion frames? | 9,900 updates; reconstruction improved; no response decision |
| [VAERViT](VAERViT/README.md) | Does the explicitly transferred VAE encoder help recurrent change decisions? | Cancelled at 227 updates for the input×10 experiment |
| [VAERViTInput10](VAERViTInput10/README.md) | Does scaling visual tokens by 10 improve learning? | 330 updates; final BA 50%, mean AUC about 0.490 |
| [WeightedMeanRViT](WeightedMeanRViT/README.md) | Can a fresh CNN over the weighted mean feed RViT? | 10-epoch cloud run replaced at 1,304 updates; preserved cancellation evidence |
| [WeightedMeanRViTEpoch2](WeightedMeanRViTEpoch2/FINAL_REPORT.md) | Does refreshing after two epochs improve generalization? | 2,310 updates; selected final BA 50%, mean AUC 0.504 |
| [VariationalMotionPredictor](VariationalMotionPredictor/FINAL_REPORT.md) | Does prediction of frame 4 from frames 1–3 learn useful motion structure? | 50,240 updates; fresh prediction MSE 92.4% below copy-last |
| [PredictiveMotionChange](PredictiveMotionChange/FINAL_REPORT.md) | Can a frozen predictive representation plus a concatenation FFN detect a 26°/28° change? | 10,240 FFN updates; final BA 50%, AUC 0.495 |
| [AngularContrastiveMotion](AngularContrastiveMotion/TECHNICAL_REPORT.md) | Can a fresh CNN learn direction similarity using proportional angular supervision? | 25,024 updates; 100% BA/AUC 1.000 in all six simplified comparison cells |

These are different objectives and stimulus distributions. They are not an exposure-matched ablation proving one causal mechanism. Prediction loss, angular loss and final response accuracy are separate measures.

## Earlier second-pass implementations

- [JointTraining](JointTraining/README.md), [TaskSuite](TaskSuite/README.md) and the [task atlas](TaskSuite/Demo/README.md): unified 13-task/35-condition battery and learned KDA joint learner.
- [SpatialReadout](SpatialReadout/FreshRun/README.md), [spatial comparison](SpatialComparisonReadout/README.md), [recurrent transformer](SpatialRecurrentTransformer/README.md), [convolutional recurrent decoder](SpatialRecurrentConvDecoder/README.md): spatial readout/core changes with explicit fresh-versus-transferred lineages.
- [SequenceKDA](SequenceKDA/README.md), [SequenceKDA3](SequenceKDA3/README.md), [SequenceKDA16](SequenceKDA16/README.md): whole-sequence single-layer, deeper and multi-head alternatives on native Krauzlis movies.
- [DelayedFrameGRU](DelayedFrameGRU/README.md), [TwoFrameRViT](TwoFrameRViT/README.md), [TwoFrameRViTReplay](TwoFrameRViTReplay/FINAL_REPORT.md), [RandomFrameRViT](RandomFrameRViT/FINAL_REPORT.md): separate temporal encoders, visual/memory attention, replay schedules and selected-frame gradient policy.
- [StructuredMotionRViT](StructuredMotionRViT/FINAL_REPORT.md), [TrainingPathAudit](TrainingPathAudit/REPORT.md), [SingleStimulusRViT](SingleStimulusRViT/README.md): motion-energy inductive bias, objective/gradient audit and cue/distractor removal.
- [KDA paper](KDA_paper/KDA_paper.md), [bibliography](papers/BIBLIOGRAPHY.md), [pod runbook](PODS.md), [analysis protocol](../ANALYSIS_SOP.md): source rationale and historical operational records.

The [original active-baseline record](ACTIVE_BASELINE.md) is a historical architectural rollback. It is not a current training instruction or evidence that the latest contrastive model was initialized from that baseline. Superseded live-status entries are archived in the journal.
