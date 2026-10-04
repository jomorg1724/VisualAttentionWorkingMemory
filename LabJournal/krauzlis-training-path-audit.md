# Actual objective and long-range gradient diagnosis

2026-10-03 UTC. User asked whether losses, decoding or disconnected gradients could be responsible for the failures.

[Targeted report](../SecondPass/TrainingPathAudit/REPORT.md) and [results](../SecondPass/TrainingPathAudit/results.json) cover the three active architecture paths. Two raw logits with ordinary cross-entropy, target/foil/catch label mapping, final readout and microbatch normalization are consistent. Seven comparisons of the executed update functions reproduce full-batch loss, gradients and Adam updates, including replay tails. All learned tensors receive finite nonzero gradients on the sampled native trial; saved optimizer evidence already shows all learned tensors updating.

The material new finding is severe temporal gradient attenuation in RViT. On the same full29-frameB12 target trial, cue-input/final-input gradient ratios were1.33e-23 for cloudRViTreplay1500 and3.07e-14 for motionRViT100; first-memory/final-memory ratios3.04e-23/4.43e-20. The16headKDA4100 cue-input ratio was0.8795, with key/value gradients at every frame. Gradients are connected but tiny in theRViTs. Norms were measured inFP64 afterFP32backprop to avoid norm-squaring underflow.

The current recurrence carries a residual fromX_t, while previousH enters only through learned memory attention. No identity carry preserves the previous-memory gradient. The state-gradient trajectory is consistent with weak long-range credit assignment. Earlier all-parameter-update and short-gradient checks did not rule this out. One sampled trial/checkpoint per architecture is a descriptive diagnostic, not proof of information erasure or a unique explanation for all failures.

No live state, training code, objective, task or cap changed. The finding motivates considering a residual/gated previous-memory carry on a subsequent architecture request; no extra run or modification was launched. Current runs remain under their existing allocations.
