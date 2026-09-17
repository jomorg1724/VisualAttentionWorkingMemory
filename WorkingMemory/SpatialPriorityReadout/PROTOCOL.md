# Protocol: scratch spatial-priority readout

## Initialization and architecture

- Version/arm: `spatial_priority_readout_scratch_v2` /
  `spatial_priority_readout_scratch`.
- Start: step 0, with no parent checkpoint, migrated model tensor, optimizer
  state, sampler state or inherited RNG state.
- All trainable computations are freshly initialized from the seed inventory
  recorded in `construction_checks.json` and every checkpoint.
- Existing classes/configuration are retained for opponent sensory emission,
  `JointAttention` with source/locality biases, `SpatialEI`, and the spatial
  comparator.
- Prospective-query `gamma` is absent.
- Final `H_T/R_T/C_T` remain aligned at 13×13 and enter only the convolutional
  priority readout. There is no terminal global-pooling or classifier bypass.

## Exposure and evaluation

- 4,000 optimizer updates from scratch.
- Five task microbatches/update × eight episodes = 40 episodes/update.
- 160,000 fresh episodes total; profile episodes are isolated and excluded.
- Unchanged five-task stimuli, cues, labels, conditions and losses.
- Full BPTT, fp32, one GPU worker.
- Existing learning-rate policy: sensory/base parameters `3e-5`;
  memory/attention/comparator/priority-readout parameters `3e-4`; Adam epsilon
  `1e-10`, weight decay and clip 1 unchanged.
- Validation: steps 800, 1600, 2400, 3200 and 4000 with unchanged seeds and
  sample counts.
- Selection: maximum lexicographic `[minimum chance-normalized task BA, mean
  task AUC]`, then earlier step.
- Held-out evaluation: selected and terminal checkpoints on the unchanged test
  stream.
- Diagnostics: normalized 13×13 priority map per evaluation trial.

## Guards and interpretation

The bundle must contain no `.pt` parent. The supervisor rejects a manifest
with `parent_sha256`; workers construct a fresh model/Adam pair and only load
their own hash-indexed scratch checkpoints when resuming or evaluating.
Construction tests repeat initialization under the same seeds, perturb the
model seed, verify empty Adam state, original component classes and bias terms,
gamma/global-pooling absence, shape/normalization invariants, and finite
gradients through all three terminal fields.

This is an architecture-training experiment. Previous trained models and the
cancelled inherited launch are historical context, not
initialization-controlled ablations.
