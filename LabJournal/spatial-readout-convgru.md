# Final spatial ConvGRU after the KDA encoder

## Matched temporal-access results — completed 2026-09-25

User authorized the proposed matched-capacity/supervision final-only versus
time-separated diagnostic. All12 fits were frozen before512 fresh held-out
episodes/task; old train1024/validation256 features were reused unchanged.

|Representation|Orientation final-only / separated BA|Binding final-only / separated BA|
|---|---:|---:|
|Whole spatial ConvGRU3136 (primary)|63.09% /61.33%|85.74% /87.70%|
|Readout256 (secondary)|49.80% /51.17%|62.50% /73.24%|
|Early visual reference|90.82% /93.36%|100% /100%|

Deployed49.22%/50.00%. Primary separated-minus-final BA intervals include zero
for both tasks; binding primary AUC has a small positive difference. Secondary
readout256 binding shows temporal-access BA gain10.74pp [5.66,15.63]. Strong
early D0 performance does not require external saved sample-time activations.
Useful comparison information is accessible from the final ConvGRU, particularly
binding, but deployment does not exploit it. The structured diagnostics still
use auxiliary angle/cue supervision and supplied circular relations. Across-layer
feature and fit dimensions differ, preventing causal attribution to compression.
D0 native stack3 still exposes sample imagery: no clean-retention conclusion.

All six final-only predictors/component outputs were invariant to removal and
NaN corruption of earlier diagnostic timesteps. Native wrapper parity0, unchanged
checkpoint/source/previous files, fresh raster/ID disjointness and saved-fit replay
verified. Parent independently recomputed all14 decision rows and hashes. Actual
fit/extract/verify/report70.93s within1200s; worker absent. No deployed updates or
further run. [Full methods/results](../SecondPass/SpatialReadout/TemporalAccessDiagnostic/REPORT.md).

## Feature accessibility/comparator results — completed 2026-09-25

Both native D0 tasks completed1024 fit/256 validation/512 held-out test episodes.
Main checkpoint6760 and source hashes remain unchanged. Exact extraction-wrapper
logit equality, separate fit/selection/test groups, saved-fit prediction replay,
and parent recomputation of held-out comparator metrics verified. Frozen analysis
and reporting finished360.82s into the1800-second allowance. Null process-tracker
exit was not worker failure; process is now defunct, not active compute.

|Measured quantity|Cued orientation|Spatial binding|
|---|---:|---:|
|Deployed held-out BA|48.63%|50.78%|
|Label-only relation comparator BA|49.22%|56.25%|
|Auxiliary-supervised structured diagnostic BA|91.60%|100%|
|Early sample-angle mean axial error|1.33°|1.39°|
|Early report-time probe-angle error|5.89°|2.66°|
|Final256 report-time probe-angle error|30.06°|22.14°|

Cue location decodes100% at report across all four measured layers. Signed cue
decodes99.8–100%. Thus weak sign effects on deployed choices do not reflect a
complete lack of accessible cue information. The label-only comparator's paired
gain intervals include zero: this is not a demonstrated native-label rescue.
The successful structured diagnostic uses auxiliary angle/cue supervision,
supplied circular relations, fixed geometry/timing and stored early sample-time
activations. All operational inputs are predicted from frozen features; decoder
fit and task-label calibration use disjoint training halves. This establishes
early information accessibility/composability, not final-state sufficiency,
spontaneous rule learning or clean retention. Layerwise decoding is not a causal
loss measurement because representations/probe feature dimensions differ.

Next unrun distinction: matched diagnostic readouts with final-state-only versus
time-separated feature access, keeping supervision and capacity comparable.
No architecture, deployed teaching or additional training was launched.
[Report and method](../SecondPass/SpatialReadout/FeatureDiagnostic/REPORT.md).

## Authorized follow-up: frozen feature accessibility and comparison

User approved the proposed tests after the frozen failure investigation. A
researcher is implementing/running independent analysis readouts for cue
location/sign and all local sample/probe orientations, then a feature-only
diagnostic comparator on native D0 cued orientation and spatial binding.
Terminal6760 stays frozen; no deployed optimizer or task modification. New
finite1800-second local cap starts at first accelerator profile/extraction and
covers fits and report, with independent base-episode fit/validation/test splits.
This is dispatch, not verified extraction/results. [Brief](../SecondPass/SpatialReadout/FeatureDiagnostic/BRIEF.md).

## Frozen failure investigation

Completed bounded source/task audits and15 frozen diagnostic conditions on160
unique base episodes; weights unchanged. D0 failure plus location-driven response
bias and weak sign/evidence counterfactual response prioritize cue-conditioned
comparison acquisition over a retention-only explanation. Native renderers and
exercised gradient paths passed checks; training loss remains near uninformed on
failing tasks. Joint optimization imbalance is measurable but not causally isolated.
No new training or architecture change. Read [consolidated findings](../SecondPass/SpatialReadout/FailureAnalysis/FINDINGS.md),
[behavioral diagnosis](../SecondPass/SpatialReadout/FailureAnalysis/DIAGNOSIS.md),
[task audit](../SecondPass/SpatialReadout/FailureAnalysis/task_audit.md), and
[training audit](../SecondPass/SpatialReadout/FailureAnalysis/training_audit.md).

## Triple continuation final results — 2026-09-25

Completed06:37:41 UTC, all5070 additional updates /162240 fresh episodes;
cumulative ConvGRU6760 /216320 episodes. Additional optimizer work14.023h.
Supervisor exit0; no failure or cap trigger; worker exited. Terminal SHA256
`1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841` verified.
Both final evaluations have35 distinct cells and13 task summaries.
Selection remains5486, using mean validation AUC, not final-test ranking.

| Task | Selected5486 BA | Terminal6760 BA |
|---|---:|---:|
| Motion direction |63.28%|100.00%|
| Signed orientation |99.22%|92.19%|
| Contrast |100.00%|99.22%|
| Spatial frequency |100.00%|98.44%|
| Chromatic increment |100.00%|100.00%|
| Contour grouping |79.69%|82.81%|
| Natural spectral detail |99.22%|94.53%|
| Ring-cued orientation |54.69%|53.91%|
| Spatially cued orientation |53.12%|49.41%|
| Cued motion duration |25.20%|25.00%|
| Krauzlis target/foil |51.42%|50.00%|
| Spatial binding |49.61%|50.00%|
| Nonempty recognition |48.52%|52.00%|

Each condition n128, Krauzlis n200; eligible-condition mean BA. Chance25%
for motion tasks,50% otherwise. Empty recognition specificity100% for both,
separate from nonempty performance. Terminal Krauzlis still all-positive
(target hits, foil false reports, catch false reports each100%); selected event
rates vary by condition and remain in test_selected.json, not collapsed into hits alone.
More unchanged exposure produced clear basic-motion and contour acquisition,
but no convincing spatial-memory acquisition. Strong terminal ranking with weaker
orientation/natural-spectrum choices is not evidence of erasure. No convergence
or architecture-superiority claim. No additional experiment launched.

Artifacts: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/`
`report.json`, `REPORT.md`, `test_selected.json`, `test_terminal.json`,
`production_supervisor_result.json`. All prior snapshots below are historical.

## Unchanged triple-length continuation — launched 2026-09-24 15:35 UTC

**RUNNING, persisted production verified at15:38:50 UTC.** Explicit authorization: “lets triple the training run” and “and continue training”. Exactly5,070 additional updates /162,240 fresh episodes, reaching6,760 cumulative ConvGRU updates /216,320 episodes. Each task receives390 additional updates /12,480 episodes. This is unchanged acquisition, not a new architecture or superiority comparison.

- Source: preserved `run_qos_repair/terminal.pt`, step1690, SHA256 `91bf029acafaf1a5e67fa3bf7ff659234deba39356c92f0cb36b0b05d91c02b5`. Predecessor normal exit0, all35 final cells, selected/terminal equality and old PID absence verified.
- Implementation: `SecondPass/SpatialReadout/continuation_v2.py`, reusing the unchanged worker/evaluator/supervisor through isolated function scopes. Source/runtime byte parity verified; no old source or artifact edited. The all13tasks/35cells recipe remains Adam1e-4, batch32/micro4, fp32 fullBPTT, all trainable, no clipping, at most2CPU threads, one local MPS worker.
- Migration readback preserved model, all named Adam states, task/condition queues, native streams, all RNG states, cumulative exposure/optimizer seconds, both historical selection looks and validation1690 fallback. Only authorized config/deadline bookkeeping changed. First checkpoint1691 proves44 changed tensors, including all4ConvGRU tensors, active Adam advancement and exact inactive-head preservation. ConvGRU Adam1690→1691; inherited CNN/KDA Adam5083→5084.
- Budget is pinned from **all1,566 repaired production updates across all35cells**, per-cell mean costs, and pooled completed per-cell evaluation timing from both validations and the unique deduplicated final test. Exact carried scheduler simulated for all5,070updates. Projected optimizer13.9667h; total19.0754h including1.25× training,1.35× evaluation, four looks, worst-case two final tests and900s startup/I/O/report. At1.5× training cost, total22.5671h; projections are not a completion guarantee.
- Cap begins at activation **2026-09-24T15:35:13.140134+00:00** and ends **2026-09-25T15:35:13.140134+00:00**, epoch1790350513.1401339. One-shot launchd ownership, `ProcessType=Interactive`, `KeepAlive=false`, existing absolute-deadline supervisor; no automatic restart or extension.
- Validation-only selection: equal-task meanAUC, then mean chance-normalizedBA, earlier ties. Four scheduled complete looks at2951,4225,5486,6760;64/cell and100Krauzlis. Historical1690 winner stays eligible. An early cap stop does not add an opportunistic look; preserve the last eligible winner. All task curves remain in full validation JSONs; inspect motion BA/AUC separately because baseline motion regressed845→1690 despite rising aggregate AUC.
- Fresh final-only namespace94792763 is injected into the actual stream constructor, not just report metadata. CPU sentinel tested the evaluator entry point without opening final stimuli. Final128/cell and200Krauzlis for selected/terminal; deduplicate only after model equality. N0 specificity and Krauzlis target/foil/catch denominators remain separate.
- Focused CPU verification:10 continuation tests plus4 existing protocol tests passed in the actual runtime; compileall passed. Tests exercise exact migration/config rejection, native next-draw replay, carried Adam update, fixed budget, immutable cap/restart refusal, actual evaluator namespace, all4 scheduled looks, baseline fallback and early-cap behavior.
- First complete task cycle1691–1703: **210.4094s measured versus178.0014s matched repaired-production estimate**, ratio1.18207 (18.2% slower, inside the pinned25% training margin). Checkpoint1703:54,496 cumulative ConvGRU episodes; SHA256 `265f2657312faba374559ac6cb78ba1cb406be1fa275b79c11ae409638bb52c1`. No new validation has completed yet.

Active artifact directory: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2`.
Job: `gui/501/org.vawm.spatialreadout.continuation-v2-6fe2d49ea188`.
Supervisor88230 (PPID1); worker88240. Evidence: `continuation_verified.json`, `migration_integrity.json`, `first_resumed_update.pt`, `checkpoint_001703.pt`, `progress.jsonl`, `latest_checkpoint.json`, `production_supervisor.json`, `budget.json`, `config.json`, `locked_source/`. Automatic final outputs: `report.json`, `REPORT.md`, `test_terminal.json`, `test_selected.json`, `production_supervisor_result.json`. No cloud, commits, or further launch gates. All earlier entries below describe the preserved predecessor.

## Final outcome — verified 2026-09-24

**Completed** at12:01:55 UTC,1690/1690 additional updates /54080 episodes,
130updates/4160episodes per task. Selected and terminal checkpoint1690 are identical;
final evaluation is deduplicated with model equality verified. All35 unique primary
cells and all13 task summaries present. Supervisor exit0, no hard-cap trigger;
5.683 optimizer hours including pre-repair slow work, about92minutes before cap.
Terminal SHA256 `91bf029acafaf1a5e67fa3bf7ff659234deba39356c92f0cb36b0b05d91c02b5` verified.

Final balanced accuracy (mean across eligible conditions):

| Task | BA |
|---|---:|
| Motion direction |25.00%|
| Signed orientation |100.00%|
| Contrast |100.00%|
| Spatial frequency |98.44%|
| Chromatic increment |100.00%|
| Contour grouping |64.84%|
| Natural spectral detail |99.22%|
| Ring-cued orientation |51.56%|
| Spatially cued orientation |51.56%|
| Cued motion duration |25.20%|
| Krauzlis target/foil |50.00%|
| Spatial binding |50.39%|
| Nonempty scene recognition |49.57%|

Chance25% for motion tasks;50% for other tasks. Final n128/cell, Krauzlis n200/cell.
Empty recognition specificity100% is separate, not successful nonempty recognition.
Krauzlis predicted positive on every trial: target hits, foil and catch false reports
all100% in each condition. Motion choices strongly favored one class despite AUC0.714.
Validation mean AUC0.7203 at845 ->0.7287 at1690; motion validation BA51.56%->25%,
so aggregate selection hides a motion regression. These are two looks, not convergence.
The replacement did not demonstrate spatial-memory acquisition under this exposure.
No matched global-GRU continuation control; gains cannot be causally assigned to ConvGRU.

Evidence: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_qos_repair/`
`report.json`, `REPORT.md`, `test_terminal.json`, `test_selected.json`,
`production_supervisor_result.json`. The worker exited; no further run launched.
Historical launch/repair snapshots follow.

## Scheduling repair — 2026-09-24 07:17 UTC

Current status: **training under repaired launch policy**, worker14390, supervisor14367.
User requested fixing the measured slowdown. Original profiling was interactive,
but production used launchd `ProcessType=Background` and ran at priority4.
Live `taskpolicy -B` did not restore throughput. Clean SIGTERM stopped at update124
with all3968 episodes preserved; no failure and no final tests were run at the stop.
The separate `qos_repair.py` wrapper resumes the exact terminal payload through the
unchanged production loop, using launchd Interactive scheduling. CPU/state readback
verified equality of model, Adam, optimizer names, scheduler, native streams, RNG
and migration. Target1690, both validation looks, full BPTT, fp32, threads<=2,
batch32/micro4 and original deadline13:33:53.442736 UTC remain unchanged.

Active artifacts: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_qos_repair`.
`repair_verification.json` verifies transition124->125; `throughput_repair_verified.json`
records a complete13-task cycle131–143 at0.945× original profiled time. Twelve
conditions with pre-repair matches show3.71× speedup: orientation7.47->2.10s,
motion-durationD0 40.52->11.96s, KrauzlisB12 120.08->31.30s. These are live update
timings, not task accuracy. At status readback, step145 /4640 episodes and checkpoint143.
No cap renewal, optimizer reset, additional training arm or lost updates.
Original run artifacts remain intact; the launch record below is historical.

Status: **PRODUCTION STARTED.** Parent reviewed actual implementation and verified
runtime source parity, approved the exact pinned config, and launched the one-shot
launchd job. At 2026-09-24 05:50:16 UTC independent CPU checkpoint verification
confirmed branch update1 /32 episodes, changed ConvGRU weights and Adam state,
and exact inherited migration. Worker97728; supervisor97682. Evidence:
`/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run/independent_production_verification.json`.
Deadline remains 2026-09-24 13:33:53.442736 UTC; target1690 new updates.

## Pinned preproduction outcome

- **1690 additional updates /54080 episodes**,130/task and4160 episodes/task.
  The requested2145 target did not fit the conservative eight-hour allowance;
  reduction is pinned before production in complete13-update cycles.
- Real35-cell profile took **661.729seconds**. Measured effective-batch32
  update costs ranged2.194–51.287seconds. All profile state is discarded.
- Expected optimizer work **18178.645seconds /5.050hours**, with1.25× optimizer
  allowance. Each full validation budgets654.997seconds; each final test
  1309.994seconds, already including1.35× evaluation margin. Two looks at
  branch845/1690, worst-case two final tests,900seconds overhead.
- Total pinned remaining projection **27553.288seconds**. The1.5× optimizer
  sensitivity is32097.949seconds and exceeds the cap: slower sustained
  throughput is an acknowledged completion risk, not a guaranteed finish.
- Cap started **2026-09-24 05:33:53.442736 UTC**, deadline **13:33:53.442736 UTC**.
  Strict launch feasibility currently requires launching by **05:54:40.154780
  UTC**; no cap renewal or postlaunch allocation change.
- Final full CPU suite after fixes: **78 passed in32.77seconds**;
  `checks/cpu_tests_final_passed.txt`. One intervening collection attempt
  followed the run-directory symlink into archived tests; exclusions for both
  run trees fix discovery, without deleting historical snapshots.
- Independent CPU readback verified the exact migration model/Adam/names/
  scheduler/streams/RNG and all eight fresh tensors changed after35 disposable
  updates, each fresh Adam step35. See `independent_profile_verification.json`.
- Before pinning production, final namespace was prospectively changed from
  94592763 (used by a tiny untrained CPU interface check) to **94692763**.
  The interface test now uses validation. Profile sources/results remain in
  `locked_source` and `profile_config.json`; the explicit seed-only amendment,
  pre-amendment preparation and `production_locked_source` retain the audit
  trail. No training/evaluation cost path, model or cap changed.

Canonical ready marker: `SecondPass/SpatialReadout/REVIEW_READY.json`.
Run alias: `SecondPass/SpatialReadout/runs/final_convgru_01` (symlink to the
external durable run). Config SHA256:
`c3e7369ebab2529f2cbbd3398a539d062b5b04d4228c281ef8c82d6e57378c89`.
No local optimizer worker remains after profiling, and there is no production
checkpoint or scientific validation/final-test result yet.

## Question and authorization

The user requested: “Yes, lets do conv gru. Lets build out your idea here and
start a new training run.” The approved change keeps the original CNN and all
three KDA encoder modules. It moves compression after the final recurrence:
160→64 1×1 spatial projection → 64-channel 3×3 ConvGRU at 7×7 → terminal
flatten3136→256 ReLU → inherited task heads. No global pooling, extra
normalization, controller, feedback, teaching changes or second arm.

The old global-GRU v4 run is stopped. Its latest pointer was independently
size/SHA256/state checked at **step3393 / 108576 episodes**, source digest
`51e37b64a69a6e5e7d5b7d6be8dccee28dd9223ebe573a4c59cc67638bb8d413`.
The old cap is expired and no attempt is made to resume it. Historical live
labels in earlier journal snapshots are superseded by this status.

## Migration and scientific contract

- All compatible CNN/KDA/heads and Adam moments are carried by exact name and
  shape. The old source-defined parameter order resolves legacy Adam IDs;
  new destination ordering does not determine compatibility.
- All eight new projection/ConvGRU/readout tensors are initialized with isolated
  seed94592763, ordinary PyTorch Conv/Linear weight initialization and zero
  recurrent biases. Their Adam state starts empty. Removed old feat/GRU state
  remains preserved in the original checkpoint.
- Native task-local streams, condition queues, scheduler and CPU/NumPy/Python/
  MPS RNG are preserved. Prior selection is historical, not a new-architecture
  candidate. Branch updates/exposure are counted separately from parent3393.
- Every learned parameter trains with one Adam LR1e-4, betas(.9,.999), eps1e-8,
  no decay, no clipping, fp32 and full BPTT. The new model has 1,383,028 learned
  parameters. Task semantics and source identities remain unchanged.
- Target2145 additional updates at effective32/micro4, contingent on measured
  feasibility pinned before production in whole13-update cycles. All13 tasks
  and35 cells remain. Two full validation looks n64/Krauzlis100; selected and
  terminal final tests n128/Krauzlis200 in fresh namespace94692763, deduplicated
  only for the same model. No optional migrated baseline.

This is a targeted warm-start architecture experiment, not a matched causal
comparison or proof of superiority. Older sensory/KDA/head experience differs
from the new recurrence/readout. Fresh test draws still use the same official
held-out source identities. N0 specificity and Krauzlis subgroups stay explicit.

## Actual CPU evidence and operational findings

[Implementation and launch instructions](../SecondPass/SpatialReadout/README.md)

- Test-first failures were observed for missing architecture, migration,
  worker/checkpointing, allocation/guards and durable launcher, then resolved.
- Combined new/suite/old-harness checks: **78 passed in35.30s**;
  [captured output](../SecondPass/SpatialReadout/checks/cpu_tests.txt).
  After final launcher metadata fixes, all12 new CPU tests passed again.
- Native renderer/source/replay verifier: **35/35 cells,13 tasks**, zero
  updates, fresh original KDA CPU forward smoke passed;
  [receipt](../SecondPass/SpatialReadout/checks/suite_verification.json).
- Tests include a real disposable CPU optimizer update, fsynced telemetry,
  full-state digest/readback and changed ConvGRU tensors/Adam; this is not
  production exposure or performance evidence.
- The first real launchd CPU probe could not import the Desktop repository;
  a second diagnostic established `PermissionError: Operation not permitted`
  before source import. Both test jobs were removed. Failed receipts remain
  in `SecondPass/SpatialReadout/checks/launchd_cpu_01` and `_02`.
- No macOS privacy permissions were changed. A byte-verified local copy of
  sources, BSDS500 files and the exact parent checkpoint was built under
  `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01`. The same launchd CPU
  ownership/cap probe then passed: supervisor PID89604, parent PID1, child89623
  killed at the test deadline with returncode−9. The test job was removed;
  `launchd_cpu_verified/verified.json` and `cleanup.json` retain evidence.

## Durable execution and artifact locations

The production runtime root is
`/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/repo`;
its run directory is
`/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run`.
This is one local experiment with an explicit local runtime copy, not another
training arm. All original sources, datasets and checkpoints remain unchanged.

`preparation.json` verifies source identity and a full CPU migration checkpoint.
The immutable eight-hour budget started at first MPS profile, epoch
**1790228033.4427361**, with absolute deadline **1790256833.4427361**. Profiling,
review, production, validation, final tests and reporting all consume it;
it will not reset. The profile is capped separately at30minutes and all its
weights, optimizer, RNG and stream progress are discarded.

After profiling, `config.json` and `REVIEW_READY.json` pin actual exposure/costs.
Parent must write `run/REVIEW_APPROVED.json` for the exact config/source hashes
before the ready launcher bootstraps the one-shot user launchd job. It has no
KeepAlive/restart, an absolute-deadline supervisor and one optimizer child.
A launch receipt alone is not training: advancing persisted progress and an
independently loaded changed-ConvGRU checkpoint are required.

Final automatic outputs are `report.json`, `REPORT.md`, `test_terminal.json`
and `test_selected.json`; partial coverage is labeled explicitly. No validation
or held-out competence result is available at this preparation/profile snapshot.
