# Authorized local fresh-weight KDA joint training

**New authorization / QUEUED, 2026-09-23 07:46 UTC:** another **2,145 updates / 68,640 episodes**, reaching **5,005 / 160,160**, is durably queued after v3 finishes validation2860, both complete final evaluations and normal OS exit. The unchanged recipe adds 165 updates / 5,280 episodes per task. v3 is still the sole MPS worker (PID51433), last verified in `final_test_terminal` at step2860; it was not interrupted. The new eight-hour cap has **not started**. Parent owns queue **`proc_af5aedcba7d9` / PID11016**, transferred with completion delivery; queue expiry is **09:13:50.062308 UTC**. Receipt and exact-state/budget gates fail closed. No v4 optimizer progress is claimed. CPU tests: **49 passed**, compileall passed. See [v4 protocol](AMENDMENT_V4.md) and [queue handoff](runs/fresh_kda_joint_01_continuation_v4_8h/handoff.json). Earlier running/completed descriptions below are historical snapshots.


**Current protocol: [continuation v2](AMENDMENT_V2.md).** The user retained the
original four-hour cap. The initial evaluation-heavy 156-update allocation was
an error; a separately versioned continuation now pins **715 cumulative
updates / 22,880 episodes / 1,760 per task**, preserving the complete step-117
state. Original hashed sources below are unchanged. Run directory:
`runs/fresh_kda_joint_01_continuation_v2/`; parent-owned process
`proc_e7bdb8db0d3f`. Resumed updates and checkpoint 118 are verified.
Deadline remains epoch `1790074076.5856378`, 2026-09-22 10:47:56.585638 UTC.
Completed validation looks 39/78/117 are retained and one new full-cell look is
planned at 715. Final evaluation is 128 per ordinary cell / 200 per Krauzlis
cell for both selected and terminal checkpoints, all 35 cells retained;
fresh final-only seed namespace 94292763 prevents reuse of potentially
forwarded v1 test draws. See the amendment for timing, stop receipts, a repaired
pre-optimizer launch failure, 23 passing tests and all unchanged settings.
`progress.jsonl` carries the original and resumed training-only curve points
with cumulative exposure. Final results remain pending. Do not treat the
initial v1 allocation documented below as the active horizon or rerun its
launch commands. The v2 run also refuses automatic restart/overwrite.

## Historical v1 contract and launch record

This directory owns the new training harness only. The user's explicit local joint-training authorization supersedes the **historical assembly-only** `training_authorized: false` field in the unchanged TaskSuite catalog. No older checkpoint is loaded, and no cloud or comparison arm is launched.

## Scientific contract

- Reuse `SuiteStream`, its unmodified 13 tasks / 35 cells, and `AccumulatorBaseline(task_classes(), stack=3, center=True, accumulator='kda')`.
- Fresh whole-model seed **94182763**, fresh empty Adam, all parameters trainable, a single learning-rate group: lr `1e-4`, betas `(.9,.999)`, eps `1e-8`, weight decay `0`. **No clipping**. Nonfinite loss/gradients stop safely; no unhealthy checkpoint is published.
- Float32 full BPTT; one MPS worker, CPU threads at most two. One task and condition per update. Mean CE over the effective batch, accumulated in microbatches without new objectives or architecture changes.
- Shuffled 13-task cycles give equal update/episode exposure; each task has its own shuffled balanced condition cycles. Native stream queues persist between updates, including incomplete label/side balancing queues. Metadata never enters the model; task ID externally chooses a head.
- Batch target 64 / microbatch 4. `config.json` freezes the actual batch, complete cycle horizon, four validation looks, and evaluation counts **before** production, based on full optimizer-update timings. Longest condition for every task is profiled; forward evaluation costs are measured independently. Profile updates use a different seed and a longest-condition schedule incompatible with the production allocation; they are explicitly disposable. Production restarts all weights/Adam/streams fresh once, not from profile or historical weights.
- The **14,400-second nonrenewable wall cap starts before the first accelerator profile**. Profile, configuration, production, validation, final evaluation and report all share `budget.json`'s absolute deadline. There is no automatic restart or extension. A separate supervisor kills only its child process group at that deadline. The worker stops training earlier to reserve both final test passes and finalization. `caffeinate -i` lasts only for this worker.

## Selection and reporting

Each planned look constructs a new `SuiteStream('val')`, giving fixed validation draws. All 35 cells are included. Selection maximizes lexicographically **minimum chance-normalized task BA**, then **equal-task mean AUC**; within-task cell means are equal-weighted, recognition N0 excluded, exact ties keep the earlier look. The final test split is not used for selection.

The preferred planned allocation is validation 64 per ordinary cell / 100 per Krauzlis cell and final test 256 / 200. If that does not fit measured throughput, the reduced allocation is pinned in config before production. Final selected and terminal checkpoints get separate complete test evaluations; the result is reused only if they are the same checkpoint. A hard deadline can yield explicitly incomplete results rather than a fabricated complete evaluation.

Metrics include confusion, class recall, accuracy, BA, binary or macro OVR AUC using the existing `WorkingMemory.BatteryAudit.observers.auc_binary` (ties half). Absent classes yield null BA/AUC. N0 has specificity/FPR, not accuracy/BA/AUC. Krauzlis reports target hits, foil false alarms, catch FPR, event and side denominators. Sampled metadata strata and sensory / orientation-auxiliary / spatial summaries are separate.

No source-independent confidence intervals or paired-delay claims: delay-cell streams are independent. BSDS500 source identity is guarded by the suite snapshot and a frozen manifest identity across both photo tasks. Suite test generators were previously used for stimulus verification, not model-selection measurements.

## State and observability

`runs/fresh_kda_joint_01/` contains ignored local run artifacts:

- `budget.json`, read-only `config.json`, `locked_source/`: fixed deadline and production identities.
- `profile.json`, `profile_progress.jsonl`: measured **full** forward/backward/Adam/sync update costs, including all 16 microbatches at effective 64, inference timings, memory and disposable-lineage checkpoint verification.
- `checkpoint_000000.pt`: fresh lineage, empty Adam, no trained source.
- `progress.jsonl`: one fsynced record per completed optimizer update, loss/norm/timing, cumulative episodes/frames, per-task and per-cell exposure. First update records sensory/KDA/GRU/active-head gradients and parameter deltas; inactive heads normally have no gradients.
- `live_status.json`: atomic phase, PID, progress, deadline and latest verified checkpoint pointer.
- `checkpoints.jsonl`, `latest_checkpoint.json`: exact-target save/readback receipts. Full state includes model, optimizer, shuffled scheduler, SuiteStream/native RNG queues, CPU/MPS/NumPy/Python RNG, step, cap deadline/elapsed, selection history, exposure and config. Checkpoints after updates 1 and 2, every 13 updates independently of validation, each validation, and terminal.
- `validation_*.json`, `test_terminal.json`, `test_selected.json`, `report.json`, `REPORT.md`: automatic bounded evaluation and final report. Partial evaluation files retain completed cells if interrupted. `supervisor_result.json` records ordinary completion or hard-cap termination.

No claim of learned performance is warranted before those evaluations complete. This is one bounded fresh-weight run, not a multi-seed generalization study.

## Commands (repository root)

Runtime: `/tmp/vawm-task-suite-venv/bin/python` (PyTorch 2.2.2, NumPy 1.26.4).

```sh
/tmp/vawm-task-suite-venv/bin/python -m pytest SecondPass/JointTraining/test_training.py SecondPass/TaskSuite/test_suite.py -q

# Only once: establishes the original nonrenewable wall-clock deadline.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 /tmp/vawm-task-suite-venv/bin/python -u -m SecondPass.JointTraining.launch profile SecondPass/JointTraining/runs/fresh_kda_joint_01

# Freeze measured recipe/source before launching production.
/tmp/vawm-task-suite-venv/bin/python -m SecondPass.JointTraining.launch configure SecondPass/JointTraining/runs/fresh_kda_joint_01

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 /tmp/vawm-task-suite-venv/bin/python -u -m SecondPass.JointTraining.launch run SecondPass/JointTraining/runs/fresh_kda_joint_01
```

The launch process must be started as a tracked background process and handed off to the parent session. Do not rerun a live or completed run: production refuses to overwrite checkpoint zero, and profiling refuses to renew an existing budget. Tests exercise scheduler balance/exact replay, N0/absent-class/Krauzlis scoring, full-state restoration including next sample and optimizer/RNG, budget guards, real CPU backward/Adam/fixed validation draws, expired-run zero updates, and a real child process killed by the supervisor deadline. The MPS profile separately verifies exact MPS RNG save/restore.
