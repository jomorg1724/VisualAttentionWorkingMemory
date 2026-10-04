# KDA encoder with final spatial ConvGRU — one authorized warm-start branch

[Fixed brief](BRIEF.md) · [journal](../../LabJournal/spatial-readout-convgru.md)

The model keeps the CNN and all **three** spatial KDA modules. The former
160×7×7 flatten→256 projection and global GRU are removed. The final spatial
field now goes through 160→64 learned 1×1 convolution, a 64-channel 3×3
ConvGRU at 7×7, then **only the terminal state** is flattened to 3,136 and
projected to 256 with ReLU and the inherited task head. There is no pooling,
extra normalization, feedback, task teaching change, clipping or second arm.

`model.py` documents the write/reset convention and initialization. `state.py`
verifies the exact source pointer, size, SHA256, step3393/108576 episodes,
source ledger and native stream identity. Legacy Adam IDs are first resolved
to the verified old constructor's names, then transferred by exact name and
shape into the new parameter order. Fresh parameters have empty Adam state.
CPU/Python/NumPy/MPS RNG, condition queues and task-local streams are carried;
initialization uses an isolated CPU generator. Prior selection remains in a
historical parent record, never a candidate in the new ranking.

This is not a matched causal comparison: CNN/KDA/heads retain prior training
while spatial recurrence/readout are fresh. Historical scores are context,
not a paired fresh control. No superiority claim is justified by launching.

## Entry points

**Host-specific durability finding:** a real CPU-only launchd probe failed to
read/import this Desktop repository with `Operation not permitted` (TCC).
No macOS permissions were changed. A byte-verified local source/dataset/parent
copy under `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/repo` passed the
same launchd probe: supervisor parent PID1 and actual hard-cap child kill.
The job was removed afterwards. Production artifacts therefore live at
`/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run`; use that runtime root
as the working directory for the commands below. The repository contains
links/receipts, not a second training arm. Original source/checkpoints/photos
remain untouched. Portable migration changes lookup paths only, retaining
the source checkpoint's exact digests and internal state.

Runtime: `/tmp/vawm-task-suite-venv/bin/python`. From repository root:

```sh
PY=/tmp/vawm-task-suite-venv/bin/python
RUN=/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run
$PY -m SecondPass.SpatialReadout.launch prepare "$RUN"
$PY -m SecondPass.SpatialReadout.launch profile "$RUN"
$PY -m SecondPass.SpatialReadout.launch configure "$RUN"
```

Preparation is CPU-only. Profiling starts the **immutable 28,800-second cap**,
executes one disposable effective-batch32/microbatch4 update in each of the
35 cells and measures forward-only validation timing for every cell. Its
subprocess has a 1,800-second hard profile bound. All profile model, Adam,
stream, scheduler and RNG progress is discarded for production. Profile
scores are not scientific evaluations or selection candidates.

The planner simulates the exact carried scheduler, targeting 165 complete
13-task cycles / 2145 added updates. It can reduce only before production to
whole 13-update cycles. It uses 1.25× measured optimizer time, 1.35× measured
evaluation time, two full validation looks, worst-case two full final tests,
900 seconds overhead and an additional 300-second review reserve. It records
a slower 1.5× optimizer-cost sensitivity rather than promising a finish.
The actual pinned allocation is `config.json`; never infer it from the target.

Two full validation looks use n64/cell and Krauzlis n100. Equal-task mean AUC
then mean chance-normalized BA select, earlier ties. No migrated baseline.
Final selected and terminal tests use the new test-only namespace94692763,
n128/cell and Krauzlis n200, with identical-model deduplication. N0 specificity
is separate; original strata and Krauzlis target/foil/catch counts remain.
Official BSDS500 source splits are unchanged, not novel image identities.

## Independent review and durable production

The implementing subagent returns after tests/profiling/configuration. It does
not wait for approval or launch production. Parent reviews the actual source
and pinned config, then writes **`RUN/REVIEW_APPROVED.json`** with:

- `approved: true`
- nonempty independent `reviewer`
- `config_sha256`: exact SHA256 from `REVIEW_READY.json`
- `source_hashes`: exact mapping from that same receipt/config

Then the parent immediately runs:

```sh
$PY -m SecondPass.SpatialReadout.launch launch "$RUN"
```

This bootstraps a uniquely named `gui/UID/org.vawm.spatialreadout.…` launchd
job, not a Hermes-owned background optimizer. `RunAtLoad=true`,
`KeepAlive=false`, no automatic restart; the plist is in the run directory,
not installed as a login-time LaunchAgent. The launchd-owned supervisor
launches **one** optimizer subprocess and kills its exact process group at
the unchanged absolute deadline. `caffeinate -i -w SUPERVISOR_PID` prevents
idle sleep during execution. A shared spatial-worker advisory lock and executable-aware
process scan prevent another local repository worker from overlapping.
Review consumed too much feasibility slack? Launch fails closed, without
silently reducing the pinned allocation or renewing the cap.

`launch_receipt.json` reads back launchctl state, but **liveness is not training**.
After launch the parent must observe advancing fsynced `progress.jsonl` and
independently load `migration.pt` and a post-update full checkpoint. Verify
SHA256, all carried tensors/states, unchanged scheduler/stream/RNG at migration,
and changed ConvGRU weights with initialized/advanced Adam after optimization.
The first two updates and every 13th update are checkpointed independently of
validation. `production_supervisor.json` records OS process identities;
`production_supervisor_result.json` records exit/cap outcome. `report.json`,
`REPORT.md`, and complete/partial `test_*.json` distinguish final coverage.
No report is fabricated if a process exits early.

## Tests

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  $PY -m pytest SecondPass/SpatialReadout SecondPass/TaskSuite/test_suite.py \
  SecondPass/JointTraining -q --ignore=SecondPass/JointTraining/runs \
  --ignore=SecondPass/SpatialReadout/runs
$PY -m compileall -q SecondPass/SpatialReadout
```

Tests cover recurrence shape, equation, zero biases, causality and early-frame
gradients; exact real-source migration; shuffled destination Adam name mapping;
stream/scheduler/RNG and checkpoint roundtrip; a real CPU optimizer update and
fsynced telemetry; cell coverage and cycle budget; selection/test namespace;
review hash binding; worker identity handling; and a real hard-cap subprocess.
Do not run the separate no-mistakes publication pipeline: this task explicitly
forbids commits/pushes and has a parent-owned independent read-only review.
