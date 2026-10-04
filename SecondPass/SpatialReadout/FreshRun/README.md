# Fresh-only final spatial ConvGRU

This separate harness reuses the proven fresh-only `SpatialRecurrentConvDecoder`
worker and dependency-closed packager, but constructs the existing, unchanged
`SecondPass.SpatialReadout.model.SpatialReadout(task_classes())` directly.
It does **not** reuse the historical ConvGRU experiment's learned state.

Architecture: causal stack-3 centered RGB → CNN with three spatial KDA modules
→ 160-to-64 spatial projection → final spatial ConvGRU → flatten 64×7×7
→ dense 256/ReLU → native task head. There is no comparator, transformer,
pooling readout, CLS token, auxiliary supervision, or architecture modification.

## Fresh-state contract

- Every parameter is freshly initialized by the existing constructor, trainable,
  and fp32. Initialization does not load any checkpoint or call migration or
  `load_state_dict`. No predecessor/resume/checkpoint CLI input exists.
- One new Adam over every parameter: LR `1e-4`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, zero weight decay. Empty moments and step counters at initialization.
- Effective batch **32**, microbatch **4**, native cross-entropy, full BPTT,
  no clipping, no curriculum, no AMP or TF32. CPU threads at most two.
- Unchanged **13 tasks / 35 primary cells** and official BSDS500 source identity
  split **200 train / 100 validation / 200 test**, shared across photo tasks.
- Fresh scheduler, streams, selection history, and zero update/episode/frame
  counters. No imported optimizer, RNG, streams, or inherited validation score.
- Disposable profile gets 13 native updates and all 35 evaluation timing cells.
  Production is a separate process constructing the whole model/Adam/scheduler/
  streams again from step zero. Profile checkpoints are never initialization
  inputs. Current-run checkpoints are read only for persistence verification
  and final selected-model evaluation, not for initialization or resumption.
- Provenance is embedded in every checkpoint. `verify_progress` checks fresh
  initial Adam/streams, optimizer advancement, scheduler/exposure agreement,
  and changes in CNN, KDA, spatial projection, `spatial_gru`, readout, and heads.

New namespaces, distinct from the recent no-CLS trials:

|Purpose|Seed/namespace|
|---|---:|
|Whole-model/Python/NumPy/CPU initialization|98192763|
|CUDA RNG|98292763|
|Balanced scheduler|98392763|
|Train native streams|98492763|
|Validation native streams|98592763|
|Final native streams|98692763|

Stream seed is `namespace * 100000 + task_stream_id * 1000 + cell_index`.

## Exposure, selection and budget

Target **10,400 updates / 332,800 episodes**, **pending actual preproduction
measurement**. The existing conservative measured planner may reduce exposure
by complete 13-task cycles before production; it never extends the budget.
The target is not a throughput or convergence claim.

Parent supplies the immutable budget from the **new pod creation** timestamp:
`wall_cap_seconds=28800`, `hard_deadline=cap_started+28800`,
`deadline=hard_deadline-600`, and `retrieval_reserve_seconds=600`.
Setup, disposable profile, training, evaluation, and retrieval share this new
8-hour cap. Parent owns deployment, independent cloud stop, and artifact
retrieval; this package does not provision or restart any cloud resource.

Selection remains validation-only equal-task mean AUC, then equal-task
chance-normalized BA, retaining earlier ties. Two prospective validation looks
(midpoint and terminal allocation); 64 episodes/cell, Krauzlis 100. Selected and
terminal fresh final tests use 128 episodes/cell, Krauzlis 200, with same-model
reuse only after equality verification. Preserve recognition N0 specificity/FPR
separately, all task/condition results, and Krauzlis event subgroups.

## Build and CPU verification

From the source repository:

```sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python
$PY -m pytest SecondPass/SpatialReadout/FreshRun/test_fresh.py -q
$PY -m SecondPass.SpatialReadout.FreshRun.bundle \
  --output /Users/jonathanmorgan/VAWMRuntime/cloud_convgru_fresh_01/package
```

`bundle_receipt.json` identifies the immutable deployment directory/archive,
archive digest and counts. The package explicitly roots the four FreshRun
files, `SpatialReadout/model.py`, and the unchanged task catalog, recursively
closing only local Python imports. Shared historical helper *source code* is
included where imported; no historical worker is launched and no migration
function is called. Only manifest-listed original BSDS JPEGs and their manifest
are data assets. Every packaged file/archive member is hash-verified. No
predecessor/cancelled-attempt checkpoint, optimizer/stream/RNG state, old metric,
credential, or budget is included. The builder rejects non-photo asset paths.

Tests cover exact whole-model equality to direct seeded construction; failure
on checkpoint reads/state restoration during initialization; empty Adam and
untouched streams/counters; a real native-renderer CPU update at batch32/micro4
changing encoder, all three KDAs, spatial ConvGRU, readout and active head;
saved provenance and optimizer verification; whole-session reset; budget and
planner arithmetic; distinct seeds; rejected predecessor CLI/package inputs.
Synthetic planner timings are unit fixtures, not measured GPU evidence.

## Parent-owned deployment entry point

After extracting the archive, install `requirements.txt`, create `assets/`,
and inject the actual new pod-creation budget there. From the extracted `repo/`:

```sh
python -u -m SecondPass.SpatialReadout.FreshRun.worker supervise \
  ../run --budget ../assets/budget.json
```

The supervisor performs the disposable CUDA profile, pins allocation, and
starts fresh production. It requires exactly one CUDA GPU for profile/run.
It refuses to restart a directory with existing `initial.pt`; no continuation
is supported. For saved-state verification only:

```sh
python -m SecondPass.SpatialReadout.FreshRun.worker verify ../run
```

CPU checks establish executable fresh-state and update behavior, not GPU
throughput, convergence, trained accuracy, or a completed production run.
Historical sources and artifacts remain untouched.
