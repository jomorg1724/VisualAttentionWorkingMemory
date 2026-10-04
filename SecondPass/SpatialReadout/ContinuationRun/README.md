# Unchanged ConvGRU continuation v1

One authorized continuation of the **fresh-run terminal at cumulative 7,739**,
not a fresh initialization, model migration, profile, or rewind to the winner.
Target **104,000 additional updates**, cumulative **111,739**, batch32/micro4.
All13 tasks/35 cells, fp32/full BPTT, all trainable, Adam1e-4/no clipping,
TF32 disabled and native scheduling/streams are unchanged.

## Parent-owned launch

Install only this new subtree into the existing frozen repository; do not modify
FreshRun. Parent owns old-worker/supervisor exit verification, the same A40,
provider monetary/deadline enforcement, detached ownership, quota inspection and
artifact retrieval. This worker never invokes a cloud API or starts another arm.
From `/workspace/vawm_convgru_fresh/repo`, using its verified CUDA interpreter:

```bash
python -u -m SecondPass.SpatialReadout.ContinuationRun.worker supervise \
  /workspace/vawm_convgru_continuation_01/run \
  --budget /workspace/vawm_convgru_continuation_01/assets/budget.json \
  --source /workspace/vawm_convgru_continuation_01/assets/source_checkpoint.json
```

Parent detaches **this supervisor once**, with OMP/OPENBLAS/MKL thread limits2,
stdout/stderr retained. No automatic restart. It reuses `cloud.supervise` and
launches only `run`, never `profile`. The worker takes the exact shared lock
`fresh.ROOT.parent/convgru_fresh_worker.lock`, which at this deployment is
`/workspace/vawm_convgru_fresh/convgru_fresh_worker.lock`. Do not copy the entire
repository to a new root: the original source hashes and shared lock must retain
those paths. New run directory has exclusive `activation.json`; existing runs
cannot be overwritten. Direct `run` is internal and requires generated config.

The existing parent files are accepted directly. `source_checkpoint.json`:

```json
{
  "path": "/workspace/vawm_convgru_fresh/run/terminal.pt",
  "bytes": 16742235,
  "sha256": "7b87e8f21e4a972cbb78f8d657786e38c2aab29bf313329167575f9f564d5ff1",
  "verified": true,
  "step": 7739
}
```

The digest/step/size are pinned in this version: another checkpoint/model is
rejected. Source siblings must remain present: `report.json` with matching
signal-stop terminal, `checkpoints.jsonl`, `progress.jsonl`, completed
`validation_*.json` and the validation-selected checkpoint referenced in state
(currently `validation_checkpoint_005200.pt`). Completed source final-test timing
files, if any, are also included. No prior held-out scores affect selection.

`budget.json` (the supplied values, never regenerated at launch):

```json
{
  "cap_started": 1790662486.4311879,
  "hard_deadline": 1790792086.4311879,
  "deadline": 1790791486.4311879,
  "wall_cap_seconds": 129600,
  "retrieval_reserve_seconds": 600,
  "max_usd": 20,
  "origin": "Explicit user-authorized unchanged-model continuation; cap starts at infrastructure transition including setup, training, evaluation and retrieval",
  "additional_updates": 104000
}
```

The budget includes setup, optimization, evaluations and retrieval. Parent's
provider guard owns $20 enforcement; no price is invented. Optional
`hourly_rate_usd`/`fixed_cost_reserve_usd` permit an additional local price check.
The worker deadline is600s before the hard deadline. Every budget field is
matched to config. Allocation uses all saved per-cell production timing means,
exact inherited future scheduler queues, **1.20 optimizer margin**, slowest
recorded complete evaluation seconds/episode per cell with1.35 margin,
10 validations,2 final tests and900s other overhead. It fails before production
if all104,000 do not fit, without reducing updates or profiling on GPU.

## State and evaluation guarantees

- `Session` subclasses FreshRun.Session. Constructor reinitialization is only
  allocation: all model/named Adam/scheduler/native stream/state are immediately
  restored and compared exactly; Python/NumPy/CPU/CUDA RNG restored **last**.
- `resume_integrity.json` records full-state equality, origin and exact next
  scheduler/native microbatch draw equivalence. The draw probe is rolled back,
  consumes no production episodes and performs **no model inference**.
- Old exposure, optimizer seconds and selection history are carried unchanged;
  new `source_step`, `additional_step`, cumulative step and added optimizer
  seconds are explicit. Original config remains under `source_config` after
  verification; the original terminal is copied byte-for-byte to
  `parent_terminal.pt`, its receipt to `source_checkpoint.json`. Carried best is
  independently verified/copied to `parent_best.pt`; best-step/key/history stay
  unchanged until a strictly better validation result.
- Validation occurs at **additional10,400/20,800/.../104,000**, cumulative
  18,139/28,539/38,939/49,339/59,739/70,139/80,539/90,939/101,339/111,739.
  Exactly10 planned looks, val64/Krauzlis100, existing namespace98592763.
  No history-length guard; no unscheduled substituted early-cap validation.
  Selection is unchanged equal-task AUC then equal-task chance-normalized BA,
  earlier ties retained. Parent best5200 remains eligible.
- Final namespace **98792763** is injected into the actual inner evaluator's
  `SuiteStream` global and metadata. It differs from the fresh-run final
  namespace98692763. Terminal and selected models use paired fresh final
  draws,128/cell and200/Krauzlis, all35 cells. An identical same-step winner may
  reuse the terminal result only after exact model equality. Early-cap results
  are explicitly incomplete acquisition, not full-target success. Signal stops
  checkpoint at the update boundary and skip final evaluations.
- Frozen FreshRun checkpoint and cloud.update implementations are reused via
  versioned cloned-function globals, correcting only provenance/telemetry and
  continuation verification. Checkpoints are immutable, atomic, digest/readback
  verified. First **additional1 and13** trigger automatic named-Adam advancement,
  parameter-change, exposure, stream, history and optimizer-clock checks;
  terminal and normal completion repeat persisted verification. First13 timing
  is compared with matched saved cell costs without discarding progress.

The source ended partway through a shuffled task cycle. Preserving its exact
queues gives7,999–8,001 added updates/task (104,000 total), not exactly8,000 each.
`allocation.json` contains the exact per-task/cell counts. No queue is rounded,
reset, reshuffled early or normalized to manufacture equal counts.

## Disk/mirror policy

Parent selected immutable cadence **260 additional updates**. Save first1/13,
all10 validation checkpoints, terminal and copied parent/best as well.
**No deletion of any files**, old or new; therefore no mirror/deletion race.
At the verified source size the preflight reserves8,131,148,709 new bytes,
including checkpoint growth/temporary/JSON margin. Filesystem free-space is
checked automatically, but parent must also verify the20GB provider quota
(filesystem free space can misleadingly describe the host's larger filesystem).
Original run is approximately10.18GB. Do not silently change cadence or delete
history to make a run fit. Supervisor publishes the existing
`cloud_completion.json` manifest/mirror contract plus explicit added/cumulative
counters. Retain the parent directory until retrieval succeeds.

## CPU verification (no accelerator launch)

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python \
  -m unittest SecondPass.SpatialReadout.ContinuationRun.test_continuation -v
```

Tests limit Torch intra/inter-op threads2. They test exact restore/next native
draw, real terminal7739 CPU restoration without model forward, budget/horizon
rejection, the actual evaluator stream sentinel, scheduled validation independent
of inherited history, and persisted first1/13/terminal Adam/checkpoint/finalization
using explicitly synthetic CPU gradients/evaluation results. These fixtures are
not reported scientific results and never saved into production directories.
Real-checkpoint tests skip on hosts without the optional local terminal copy;
parent must run the tests in its deployment interpreter too. CUDA equality and
production-backend learning are automatically checked at launch/first saved
updates, not claimed by CPU tests.

Read-only verification of an existing run:

```bash
python -m SecondPass.SpatialReadout.ContinuationRun.worker verify \
  /workspace/vawm_convgru_continuation_01/run
```

Inspect `resume_integrity.json`, `persisted_progress_000001.json`,
`persisted_progress_000013.json`, `first_cycle_timing.json`, `progress.jsonl`,
`report.json`, `supervisor_result.json` and `cloud_completion.json`. A live
process alone is not proof of a saved optimizer update.
