# Spatial recurrent transformer with convolutional readout (NO CLS)

**Fresh-from-scratch training worker; no inherited weights or training state.**
`worker.py` constructs every learned tensor anew, empty Adam, fresh native
training streams, fresh BalancedScheduler and RNG, zero counters and no winner.
No predecessor checkpoint is opened, loaded, inspected, hashed or packaged.
The unchanged model and native 13 tasks / 35 cells are reused as source only.
The parent owns RunPod deployment, billing guard and retrieval; this directory
does not provision cloud resources. A CPU optimizer persistence test is not a
claim that cloud production has started.

## Exact architecture

`SpatialRecurrentConvDecoder(task_classes())` accepts the unchanged suite's
`[B,T,3,100,100]` image sequence plus external task-head identity.

- Reuse `AccumulatorBaseline` input centering, causal stack-3 preprocessing,
  convolutional encoder and three spatial KDA accumulators unchanged. Remove its
  flatten-to-feature layer, global GRU and feature norm, just as in the CLS model.
- Project the deepest `160×7×7` encoder field through `Conv2d(160,64,1)` to Z.
- H is **49 spatial tokens × 64 channels**, initialized from a learned spatial
  parameter at each sequence boundary. There is no CLS token or CLS parameter.
- At each presented frame, apply two pre-norm transformer blocks. Each has two
  32-dimensional attention heads: 49 H queries attend jointly to 49 Z and 49 H
  keys/values (98 keys). Learned position `[1,49,64]` and two source embeddings
  enter normalized inputs. Q/K/V, output projection and `64→128→64` GELU FFN
  use 3×3 convolutions. Attention output and FFN each add to H residually; the
  second block's updated H recurs to the next frame. No attention-logit bias.
- **Only the final updated H**, reshaped to `64×7×7`, reaches the decoder:
  actual imported `PreAttentiveVision.decoder.ConvNormAct(64,32)` →
  `ConvNormAct(32,64)` → concatenate spatial mean and max (128 features) →
  `Linear(128,256)` → SiLU → freshly initialized 256-input task head.
  Both ConvNormAct stages are 3×3 bias-free conv, GroupNorm(8), SiLU.
- All parameters remain trainable. No raw sensory/earlier-frame readout bypass,
  pair decoder, oracle crop, phase metadata, extra recurrence, dropout or detach.
  Encoder KDA and H reset on every `forward`/`recurrent_states` call.

The decoder adapts the **convolution-before-pooling pattern** of
`PreAttentiveVision/TemporalIntegration/accumulators.py::StreamingReadout`.
It is not byte-identical reuse: one final selected field replaces three
current/emitted field pairs; fusion is `32→64`, not `96→64`; the trunk is
256-wide, not 128-wide, and has no dropout. It does not import the older frozen
encoder policy or pair-interaction decoder.

Use `SecondPass.TaskSuite.suite.task_classes()` for all 13 native heads.
Renderers, labels, cues, sequence lengths, objectives and teaching are unchanged.
Task identity selects a head externally; it is not a sensory input.

## Focused CPU verification

From the repository root, with the existing isolated environment:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python \
-m pytest SecondPass/SpatialRecurrentConvDecoder/test_model.py -q \
--junitxml=SecondPass/SpatialRecurrentConvDecoder/checks/pytest.xml
```

Executed result: **6 passed in 1.80s**; machine-readable receipt:
[`checks/pytest.xml`](checks/pytest.xml). The initial behavioral TDD test was
first run against the untouched CLS class and failed as intended (`[2,50,64]`
versus required `[2,49,64]`, 1 failed in 2.79s). After implementing the new class
and switching that test's import, it passed (1 passed in 1.54s).

Coverage: 49-token/no-CLS state; independent two-head 49×98 joint-attention
reference; early-frame gradients outside the last raw stack window; finite
nonzero parameter gradients; all new decoder tensors changing after **one
bounded disposable Adam update**; causal prefix/explicit carry/sequence reset;
final-map intervention excluding sensory/earlier-state readout bypass; actual
conv-before-pooling shapes; and all 13 native head output dimensions.
No native dataset rendering, performance evaluation or acquisition claim.
The untouched predecessor's safe model-only node
`SecondPass/SpatialRecurrentTransformer/test_transformer.py::test_recurrent_spatial_cls_is_causal_and_learns`
also passed (1 passed in 1.30s; its own bounded synthetic Adam update).
Do not run the original transformer's whole test file: its migration tests
access real checkpoints, outside this work's authorization.

## Fresh training contract and launch

All learned tensors are trainable; Adam LR 1e-4, fp32, full BPTT, no clipping,
effective batch32/micro4/evaluation micro4, at most two CPU threads. Preserve
official BSDS500 identities (train200/val100/test200). One sequential CUDA worker.
Fresh initialization and all stream/RNG seeds are recorded in checkpoint
`provenance`; the profile and production use the same seeds but share no state.

Target **5200 updates / 166400 episodes / 400 updates per task**, validation
at2600/5200. A single disposable 13-task update cycle and complete 35-cell
evaluation timing look pin this target if it fits. Otherwise reduce before
production by complete13-update cycles, recording honest per-task exposure and
midpoint/terminal validation steps. Selection is equal-task validation mean AUC,
then equal-task chance-normalized BA; earlier ties. No previous winner/baseline.
Final selected/terminal tests use paired fresh streams:128/cell, Krauzlis200.
Identical selected and terminal models can reuse the verified same test result.
All35 cells, N0 specificity and event subgroups remain in JSON/markdown reports.

The parent injects `assets/budget.json` at actual pod creation: numeric
`cap_started`, `hard_deadline=cap_started+14400`,
`deadline=hard_deadline-600`, `retrieval_reserve_seconds=600`,
`wall_cap_seconds=14400`. No prefilled epoch, stale deadline or automatic renewal.
The enclosing four hours include setup/profile/training/evaluation/retrieval.
Pricing, $3 maximum and independent provider stop enforcement belong to parent.

After parent extraction (`tar --no-same-owner`) and dependency installation:

```sh
cd /workspace/vawm_convdecoder_fresh/repo
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -u -m SecondPass.SpatialRecurrentConvDecoder.worker supervise \
  /workspace/vawm_convdecoder_fresh/run \
  --budget /workspace/vawm_convdecoder_fresh/assets/budget.json
```

Build locally with `python -m SecondPass.SpatialRecurrentConvDecoder.bundle
--output /Users/jonathanmorgan/VAWMRuntime/cloud_convdecoder_fresh_01`.
The immutable archive has `repo/`, requirements and a per-file verified
`deployment_manifest.json`, all500 image hashes verified, and **no checkpoint,
credential, historical metric or budget file**. `bundle_receipt.json` publishes
the absolute archive path and SHA256. Source imports close transitively over
the existing harness; unused historical adapter functions are never invoked.

Run the focused fresh-state test with the CPU command above, substituting
`test_fresh.py` (do not run predecessor migration tests). It performs a real
native-rendered synthetic-task optimizer update and verified full-state save,
checks advancing Adam/scheduler/streams/counters and fresh reproducibility.
`initial.pt`, `progress.jsonl`, verified periodic checkpoints,
`persisted_progress_verification.json`, `allocation.json`, `live_status.json`
and `cloud_completion.json` provide production evidence. Production refuses
restart/overwrite and cannot accept a checkpoint or source-pointer argument.
