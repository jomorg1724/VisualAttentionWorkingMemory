# Weighted-sum transformer change classifier

Each consecutive three-frame clip becomes one image:
`S/6 = (oldest + 2*middle + 3*latest)/6`.
The two clips continue the same six-frame persistent-dot movie. A shared
single-layer transformer encodes each weighted image into its CLS vector.
Ordered `[CLS_before, CLS_after]` vectors feed a nonlinear binary classifier.

The transformer uses 10×10 nonoverlapping RGB patches, dimension128, four heads,
one pre-normalized encoder layer with GELU feedforward512 and dropout0, and
learned position/CLS embeddings. All learned parameters start fresh and train
jointly on ordinary binary cross entropy. No VAE, frozen encoder, latent cache,
angular target, contrastive loss or prediction objective is included.

Default requested exposure: 25,024 updates, 391,000 unique fresh six-frame movies
and 782,000 pair presentations. Each 1,000-movie pool is reused for two shuffled
epochs, effective batch32/microbatch8; partial tails receive correct weighting.
Labels are balanced with fresh independent nuisance draws. Validation and test
use independent streams and paired unchanged/changed nuisance examples.

Optimization uses one Adam learning rate3e-4, epsilon1e-8, no weight decay or
clipping, FP32/TF32off, one local GPU worker and at most two CPU threads.
Validation at64/5000/10000/15000/20000/terminal selects minimum BCE. A positive
scalar logit threshold is fitted only on validation; final test reports both
this decision and the ordinary zero-logit decision, with AUC and all six
speed×angle cells. Final 3,072 presentations represent1,536 paired contexts.

A new finite four-hour deadline begins at the first local GPU profile and covers
profiling, training, evaluation and reporting. Measured update/render/evaluation
costs pin complete64-update pools before production with1.25 safety margin and
600 seconds plus evaluation reserve. Profile weights are discarded. One GPU
lock and independent supervisor deadline guard prevent concurrent workers and
automatic budget renewal. Only best/latest compatible optimizer/RNG checkpoints
are retained. No cloud job or second arm is included.

`python -m SecondPass.WeightedSumTransformer.launch --run ABSOLUTE_RUN_DIRECTORY`
profiles and then launches fresh production under its guard.
`python -m SecondPass.WeightedSumTransformer.runner_check` executes a disposable
CPU fixture through64 actual model updates, validation/checkpoint saving,
selected-weight loading and paired final evaluation. It does not activate the
scientific GPU experiment.

The stimuli remain simplified coherent dots; results do not establish native
Krauzlis replication, broad task generalization or biological validity.


Actual activation: October4,9:00PM PDT, `LocalRuntime/run01/`. Native profile
pinned25,024updates; worker16652/supervisor32432/guard39176. Actual GPU validation64
and checkpoint saving passed; all24 aligned finite Adam states independently
CPU-verified. Source remains frozen. New cap endsOctober5,1:00:19AM PDT.
No final trained outcome yet. Current journal/runtime receipts supersede plan text.


Final result: completed October4,9:45PM PDT, all25,024 updates; selected5,000.
Final3,072 presentations/1,536contexts: BA49.25%, AUC0.492292 (chance).
Best/latest independently verified with24 aligned finite Adam states. All workers
exited normally; no follow-up queued. [Final report](LocalRuntime/run01/FINAL_REPORT.md)
supersedes the launch/live statements above.
# Faster-dot rerun

The user subsequently requested increased dot speed and a rerun. `--speed-scale 4`
uses 1.5/4/8 pixels per frame, preserving the same 26/28-degree turns, renderer,
architecture, fresh initialization/shuffle seeds, nuisance streams, Adam settings
and 25,024-update target. The intervention is recorded in both profile and saved
production config; profile/production scale mismatch is rejected. Separate
`LocalRuntime/run02_speed4` preserves the completed original run. A new finite
four-hour local allowance begins at profiling, includes evaluation/reporting,
and does not extend automatically. No attention modification or extra arm.
