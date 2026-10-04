# Predictive representation motion-change experiment


## October4 — predictive encoder + concatenated FFN, LOCAL TRAINING

User requested the simple before/after change decision and explicitly said train locally. Frozen validation-selected predictive encoder50,000; two deterministic512-vectors concatenated to a fresh281,026-parameter FFN256/64/2, trained by balanced binary CE. Six continuous persistent-dot frames, 3before/3after, no cue/distractor; same direction or +/−26/28° turn, speeds.375/1/2. No-change clips move and differ in pixels. Matched label pairs share nuisance variables and before clip; independent train/val/test splits. Batch64, Adam1e-3, two epochs/1000freshtrials then refresh, FP32 MPS/CPU2. Target10,240 updates/640,000presentations/320,000 unique; new8h ceiling through2026-10-04T15:42:47.446601-07:00, no extension. Only headbest/latest, preserve predictivebest/latest. Saved update768 independently verified all8Adamstates; guard armed. This is a new simple synthetic representation-readout test, not native fullKrauzlis evaluation. [Implementation](../SecondPass/PredictiveMotionChange/README.md).


## Completed October4, 8:05AM PDT — chance performance

All10,240 planned FFN updates completed in23m06s:640,000 presentations/320,000 unique trials. Frozen predictive encoder50,000 unchanged; all8 headAdam states in best5632/latest10240 independently reloaded at their saved steps. Final independent3072-trial test: balanced accuracy50.00%, meanAUC0.49519, cross-entropy0.693165. Every cell predicted no-change (sensitivity0%, specificity100%); all6speed/angle cells had50%BA. Last100 training batches meanCE0.693178. No process remains; guard normal_owner_exit. This frozen-encoder/concatenation FFN did not learn the comparison. Prediction success alone did not translate into this simple readout; the result does not establish that motion information is absent from the representation. No end-to-end tuning or additional diagnostic was run. Both classifier checkpoints and predictive source checkpoints retained, no autonomous new training or cloud job.
