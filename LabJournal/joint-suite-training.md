# Fresh local KDA joint-suite training — 2026-09-22

**V4 RECOVERED / RUNNING — verified 2026-09-23 08:14:42 UTC.** After normal v3 completion and the preserved pre-activation queue failure, explicit recovery passed all predecessor and unchanged feasibility gates. Independent CPU readback verified exact model/Adam/scheduler/stream/RNG migration from terminal 2860, then checkpoint 2861 with 42 changed model tensors and 42 advanced Adam states; persisted progress reached **2870 / 91,840 episodes**. Parent owns handed-off supervisor **`proc_b345f6fe2730` / PID15739**, sole MPS worker **15763**. The new cap began **08:13:26.181166 UTC** and expires **16:13:26.181166 UTC**, with no renewal. Target remains exactly **2145 added updates / 68,640 episodes**, endpoint **5005 / 160,160**. **66 CPU tests passed**; final evaluations remain pending. Monitor run `recovery_result.json`, `live_status.json` and `latest_checkpoint.json`; original `queue_result.json` deliberately preserves the blocked attempt. [Recovery details](../SecondPass/JointTraining/RECOVERY_V4.md). All earlier statuses below are historical snapshots.

**V3 COMPLETE / V4 pre-activation BLOCKED, 2026-09-23 08:12 UTC:** v3 finished normally at 08:04:35 UTC, terminal2860 /91520 episodes, selected2314, both complete35-cell final evaluations. Original PIDs51429/51433 are absent. The queued v4 attempt failed closed at08:04:36 on a changed raw process identity and started no cap or worker. The observed changed identity was not logged; a normal-exit zombie transition was reproduced separately, not proven for the incident. An explicit versioned recovery now passes66 CPU tests, re-verifies full predecessor state and retains all blocked evidence. Unchanged all-final-timings feasibility is28637.609s within28800s; no target/estimator/cap change. Actual recovered optimizer progress is not yet claimed. See [incident and recovery](../SecondPass/JointTraining/RECOVERY_V4.md). Earlier queue/live descriptions below are historical.

**New authorization / QUEUED, 2026-09-23 07:46 UTC:** another **2,145 updates / 68,640 episodes**, reaching **5,005 / 160,160**, is durably queued after v3 finishes validation2860, both complete final evaluations and normal OS exit. The unchanged recipe adds 165 updates / 5,280 episodes per task. v3 is still the sole MPS worker (PID51433), last verified in `final_test_terminal` at step2860; it was not interrupted. The new eight-hour cap has **not started**. Parent owns queue **`proc_af5aedcba7d9` / PID11016**, transferred with completion delivery; queue expiry is **09:13:50.062308 UTC**. Receipt and exact-state/budget gates fail closed. No v4 optimizer progress is claimed. CPU tests: **49 passed**, compileall passed. See [v4 protocol](../SecondPass/JointTraining/AMENDMENT_V4.md) and [queue handoff](../SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/handoff.json). Earlier running/completed descriptions below are historical snapshots.


**New authorization / running, 2026-09-23:** the user explicitly requested “go ahead and set up a 8 hour training run for more trsining and more updates”. The versioned v3 continuation resumes **terminal 715**, not historical selected 117. It pins **2,145 additional updates / 68,640 additional episodes** (5,280/task), reaching **2,860 total updates / 91,520 episodes** (7,040/task). One local MPS worker; no architecture, loss, stimuli, sampling, optimizer or batch changes.

The new 28,800-second allowance starts **01:08:50.062308 UTC** and ends **09:08:50.062308 UTC on September 23** (epoch **1790154530.062308**), including migration and finalization. No renewal. At **01:12:32 UTC**, optimizer progress was independently read through **733 / 23,456 episodes**; checkpoint **728** was SHA-verified and CPU-loaded. Migration preserved model, all Adam state, sampler/condition queues, native streams and Python/NumPy/CPU/MPS RNG exactly. First production update 716 is retained, not discarded.

Validation-only selection is prospectively **equal-task mean AUC**, then mean chance-normalized task BA, earlier ties. Baseline validation715 is reused; historical winner117 remains historical. Four new all-35-cell looks: **1248, 1781, 2314, 2860** (64 ordinary / 100 Krauzlis). Final selected and terminal tests use 128 ordinary / 200 Krauzlis in fresh final-only namespace **94392763**, deduplicating only identical models. Prior tests were seen; new draws reuse official test source identities, so this remains exploratory—not a wholly untouched population or convergence claim.

Parent owns CPU guardian **`proc_5cbdb07600dd` / PID51946**, attached to live supervisor **51429** and the sole MPS worker **51433**. The first tracking handle falsely reported exit after stdout/stderr redirection; OS identities and real updates disproved a worker stop. The guardian fixes completion delivery without restarting training or moving its deadline. Automatic `report.json`, `REPORT.md`, both final-cell JSONs and cumulative validation curves will be written under `SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v3_8h/`. Earlier completed715 results and source artifacts are retained; all older authorization/live statements below are historical.


## Completed outcome

Finished **2026-09-22 10:04:27 UTC**, normally, at the pinned **715 updates /
22,880 episodes / 1,760 episodes per task**. Total wall time was 3.275 hours,
including 1.968 hours of optimizer work; the original four-hour cap was not
extended. Both selected and terminal tests completed **35/35 conditions**,
4,696 episodes per checkpoint. Parent independently hash-verified and loaded
terminal 715, with its 66 Adam parameter states. No further run was launched.

**Terminal 715** achieved contrast 100%, chromatic increment 99.22%, natural
spectral detail 92.97%, and spatial frequency 63.28% balanced accuracy. Motion
and the spatial/sequence tasks remained near chance. **Official selection is
still step 117**, largely near chance: the minimum-task-first validation rule
preferred its better worst-task floor despite terminal mean validation AUC
improving from 0.5203 to 0.6296. Do not retroactively substitute the terminal
checkpoint as the validation-selected winner.

Full task and condition tables, both checkpoints, empty-set specificity and
Krauzlis target/foil/catch denominators are in
[completed results](../SecondPass/JointTraining/RESULTS.md). Both models always
answered positive on sampled Krauzlis events, so their 100% target hit rate
comes with 100% foil/catch false positives. Empty-set specificity was 100%,
but nonempty recognition remained at 50% BA.

This is selective sensory acquisition, not full-suite competence or a
convergence/capacity finding. Preserve terminal optimizer progress. Any longer
training and prospective selection-rule correction requires a new explicit
allowance; unused cap margin does not authorize automatic additional updates.
The amendment and launch statements below are historical snapshots.

## Current protocol amendment: same deadline, greater acquisition

The user explicitly chose **“Keep the existing four-hour limit.”** The initial
156-update plan was an allocation mistake: evaluation precision and overly
conservative reserves displaced training. It is superseded by the separately
versioned [v2 amendment](../SecondPass/JointTraining/AMENDMENT_V2.md), not by a
new budget or model. Original source/artifacts remain preserved.

**Pinned target: 715 cumulative updates / 55 task cycles / 22,880 episodes,
1,760 per task**, continuing exactly from step 117 / 3,744 episodes. Measured
equal-task cost was 10.2225 seconds/update; planning uses 20% training and 25%
evaluation inflation. Completed validation looks 39/78/117 are retained; one
new 35-cell look is planned at 715. Final selected and terminal evaluation
retain all 35 cells, ordinary counts reduced 256→128, Krauzlis kept at 200.
Validation-only selection and all training/scientific settings are unchanged.

**Verified resumption:** snapshot at 07:49:04 UTC reached 122 updates / 3,904
episodes; post-resume checkpoint 118 was independently SHA-verified and loaded.
Migration readback proved exact model/Adam/scheduler/stream/RNG preservation.
Run: `SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v2/`.
Parent owns **`proc_e7bdb8db0d3f`**, supervisor 78932 / worker 78937; one actual
joint-training Python worker, both original PIDs absent. The original worker's
`-9` exit was intentional deferred-finalization handover, not training failure.
A first pre-optimizer v2 launch hit a `caffeinate` helper false positive; its
receipts are archived, the executable-aware guard was regression-tested, and
the successful launch did not reset training or renew time. **23 tests pass.**

Deadline remains **2026-09-22 10:47:56.585638 UTC**, epoch
**1790074076.5856378**, original origin **1790059676.5856378**. Old test draws
may have been forwarded during handover, but no final-test metrics were used
for planning/selection; v2 final tests use fresh namespace 94292763 with the
same official test source split. N0 specificity/FPR and Krauzlis event/side
denominators remain explicit. Cumulative training curves use the carried and
continuing `progress.jsonl`; automatic final reports are in the v2 directory.
Final held-out performance is pending. More exposure is not a sufficient
convergence claim or overnight-equivalent run.

The sections below are the preserved **initial v1 allocation and launch
record**, not the current horizon or process ownership.

## Authorization and current status

The user said: **“lets go ahead and start training a new model locally. Train the model here and lets see how it does on the suite.”** This supersedes the suite's preparation-only status for **one fresh-weight local KDA run**. **Production training is now verified running.** At the parent's 06:59:57 UTC progress snapshot, 21 optimizer updates / 672 episodes were persisted, and every task had received an update. No held-out performance is available yet.

No cloud, second architecture arm, warm start, new task, changed stimulus, or automatic budget extension is included. Existing checkpoints and unrelated worktree files are preserved.

## Model and data

Reuse the existing `AccumulatorBaseline` with multiscale spatial KDA, centered inputs, causal stack of three available/padded frames, 256-unit GRU and the suite's 13 distinct heads. Initialize all model parameters from a recorded seed; all learned parameters remain trainable. No old model/Adam/stream state enters production initialization. Sensory episodes retain exactly two presented frames.

Use [TaskSuite](../SecondPass/TaskSuite/README.md)'s full 13-task, 35-condition catalog with unchanged renderers. Genuine BSDS500 is present locally with 200/100/200 source-separated splits. Analysis metadata is never passed to the model; only the task key selects the output head.

Live hardware check: arm64 macOS, Apple MPS available in PyTorch 2.2.2, no CUDA. Runtime `/tmp/vawm-task-suite-venv/bin/python` supplies compatible NumPy 1.26.4. One accelerator worker, at most two CPU threads. These are not the earlier Windows/CUDA platform; bitwise cross-platform equivalence is not assumed.

## Initial optimization contract

- Adam, one learning-rate group for all parameters: LR 1e-4, epsilon 1e-8, betas (.9,.999), weight decay 0. Full sequence BPTT and fp32; no inherited clipping-at-1 recipe.
- One task per optimizer update with mean cross-entropy; shuffled 13-task cycles give equal task-update exposure. Shuffle/cycle each task's declared conditions independently. This is joint training from the beginning, not ring-first or delay curriculum.
- Target effective batch 64, with microbatching to control full-BPTT memory. Measured throughput may require batch 16 or 32, pinned before production. The actual batch, microbatch, cycle count and episode horizon belong in the run configuration, not a guessed throughput claim.
- Nonfinite gradients/loss trigger an explicit safe stop and preservation of the latest healthy checkpoint. Early chance performance alone is not a stop rule or a capacity conclusion.
- Record completed optimizer updates, per-task/cell episodes and frame exposure, loss, gradient norm and elapsed time. Save full model/optimizer/sampler/stream/RNG/selection state independently of validation, with read-back verification.

## Finite compute and evaluation

The assistant selected a **14,400-second (four-hour) local wall cap** for this initial run, starting at the first accelerator profile and including profiling, production, validation, final evaluation and reporting. This is a new finite run allowance, not reuse of an older ledger, and has no automatic renewal. Pin a feasible complete-cycle exposure from measured full optimizer costs and evaluation overhead before production; reserve explicit finalization time.

Target four planned validation looks, all 35 cells: 64 episodes/cell except 100 per Krauzlis baseline. Target final selected/terminal tests: 256 episodes/cell except 200 per Krauzlis baseline; reuse results if selected and terminal are identical. Reduce counts **before production** if profiling requires it, and disclose reductions. Validation uses repeatable validation draws, never test-based selection.

Select by minimum chance-normalized task BA, then equal-task mean AUC, with earlier ties. Average eligible cells within task first. Exclude recognition N0 from BA/AUC; report its specificity/FPR. Report each task/cell, confusion/class recall, sensory difficulty strata and Krauzlis target hit/foil false-alarm/catch false-positive rates. Also report the sensory, orientation-auxiliary and spatial groups separately. No source-independent uncertainty or paired-delay claim is implied.

## Verified launch and pinned exposure

- Harness: [JointTraining](../SecondPass/JointTraining/README.md).
- Run: [`fresh_kda_joint_01`](../SecondPass/JointTraining/runs/fresh_kda_joint_01/), with pinned `config.json`, `progress.jsonl`, `live_status.json`, `latest_checkpoint.json` and `handoff.json`.
- Durable parent-owned process: `proc_9af99978acf5`; supervisor PID 70215 and worker PID 70221. Parent process poll confirmed running after handoff.
- Batch-64 MPS profiles covered all 13 tasks at measured 4.18–98.43 seconds/update; these profile weights and optimizer states were discarded. Effective production batch is **32**, microbatch **4**. Horizon fixed before launch: **156 updates / 12 complete task cycles / 4,992 episodes**, or **384 episodes per task**. No learning-rate changes or clipping.
- Parent independently SHA-256 verified and CPU-loaded `checkpoint_000013.pt`: step 13, 416 episodes, 4,608 frames, one Adam parameter group and 66 populated parameter states. It also independently loaded checkpoint zero and confirmed empty Adam state. First-step diagnostics show nonzero gradients and parameter changes in sensory, KDA, GRU and the active head; inactive heads correctly have no update on that task.
- Researcher reports 17 passing tests. Parent inspected worker/scheduler/checkpoint/metric paths; it did not rerun training or accelerator profiles.
- Cap ends **2026-09-22 10:47:56 UTC** (epoch `1790074076.5856378`). Automatic validation is pinned at steps 39/78/117/156; final selected and terminal test sizes are as specified above. Supervisor enforces the original deadline; no restart or extension.
- **Exposure limitation:** this is a short joint-learning pilot, not an adequate acquisition horizon established by prior evidence. The complete evaluation reserve is substantial (approximately 5,734 seconds for finalization, plus four planned validation looks). Low final scores would not establish an architecture or memory-capacity limit; report the small training exposure prominently.

Pending: planned validation snapshots and final selected/terminal held-out results, plus completed-run journal updates. Live training losses are not validation accuracy.

This is an initial empirical joint-learning run, not a sufficient acquisition horizon, a demonstration of general working-memory capacity, or biological validation.
