# v4 pre-activation identity incident and explicit recovery

## Evidence and cause qualification

The original CPU queue (`proc_af5aedcba7d9`, PID11016) exited with code2 at epoch1790150676.198947 (2026-09-23 08:04:36 UTC), reporting `Predecessor PID identity changed; fail closed` at `continuation_v4_queue.py:258`. It compared the entire `ps -p PID -o lstart=,command=` string. The receipt did **not** save the observed changed string or which PID changed, so the historical transient cannot now be identified conclusively.

V3 completed normally at terminal2860 /91520 episodes; selected2314 is historical, not the continuation source. Both 35-cell finals and validation2860 are complete. The matching supervisor receipt has returncode0 and hard_cap_triggered=false. Both original PIDs51429/51433 and failed queue11016 were absent during recovery preflight. No v4 budget, activation, supervisor, migration or progress artifact existed.

Ranked explanations were (1) a normal-exit zombie command transition, (2) ps formatting/timezone change, (3) actual PID reuse. A CPU-only subprocess on this host reproduced normal exit changing the exact identity from `Wed Sep 23 15:07:02 2026 /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python -c import time; time.sleep(0.3)` to `Wed Sep 23 15:07:02 2026 <defunct>`, with OS statusZ, then absent after reap and exitcode0 (probe PID14482). This establishes a failure mechanism, **not** the unrecorded historical value. Deterministic tests reproduce the original queue blocking both a normal-exit transient and real PID reuse. No live changed identity is accepted or normalized by recovery.

## Narrow versioned repair

`continuation_v4_recovery.py` is an explicit one-shot **pre-activation** path, not an automatic retry or a replacement identity guard. All original v1–v4 sources, including the failed queue, remain unchanged. The same queue lock is acquired. It rejects any existing budget/activation/config/supervisor/migration/progress; an expired queue; a changed manifest/handoff; another joint worker; any original predecessor or queue PID still present (including a zombie or reuse); incomplete/failed completion receipts; or changed source/config/budget/checkpoint hashes. Live-identity errors now retain observed identity values.

Only after full verification does it hash-copy the blocked attempt into read-only `blocked_attempt_01/`, including frozen sources and old handoff/manifest/result/status. The original blocked receipts also remain byte-identical **in place**. A new recovery manifest includes the new module hash. The frozen original activation function is reused verbatim with only its two queue status/result outputs routed to `recovery_status.json` / `recovery_result.json`; no process, receipt gate, scientific recipe, planner, clock or worker behavior is replaced. `recovery_verification.json` retains independently checked predecessor artifact hashes and the archive hashes. No already-started cap can be reset. No recovery restart is accepted.

## Unchanged fixed-exposure feasibility

Preflight recomputation pooled all2145 v3 optimizer timings, four complete validation looks and **both completed final evaluations**, using the prespecified per-cell mean seconds/episode estimator and exact inherited scheduler queues:

- Additional2145 updates /68640 episodes; endpoint5005 /160160;165 updates /5280 episodes per task.
- Optimizer21182.0404487755 seconds; with15% margin24359.346516091824 seconds.
- Four validations1839.131315897344 seconds; two finals1839.131315897344 seconds; startup/checkpoint/report600 seconds.
- Total28637.609147886513 seconds <=28800. No exposure, estimator, precision, cadence or cap adjustment.
- Maximum-observed-cell evaluation sensitivity28831.80725757594 seconds exceeds the cap. The mean fit is not a guarantee of finishing.
- Original queue expiry remains2026-09-23 09:13:50.062308 UTC. A new28800-second cap starts only at actual activation, before migration/MPS work, with no renewal.
- Terminal SHA256 `2b38207b0700ee45319c4671f68764b71d7858e33d3a1e6eb5ed51e17df8bc62`.

## Executed validation

The recovery regression first failed against the original queue with `ValueError: Refusing queue restart` (1failed,2passed), then passed using the explicit versioned recovery.17 focused tests now cover preserved evidence, the original transient/reuse symptom, rejection of live original/zombie/reused/queue PIDs, existing cap/activation/migration, expiry, missing final, failed predecessor, corrupt checkpoint, changed manifest, remaining joint worker, duplicate recovery, and the real activation path through a mocked subprocess boundary. They verify the cap exists before worker launch and original blocked receipts are unchanged.

Actual runtime: `/tmp/vawm-task-suite-venv/bin/python`.
`python -m pytest SecondPass/JointTraining/ SecondPass/TaskSuite/test_suite.py -q --junitxml=SecondPass/JointTraining/checks/v4_recovery_cpu_tests.xml`: **66passed in30.39s**. `compileall` passed. Focused static dangerous-command scan found no matches. No ruff executable was installed. Repository-wide `git diff --check` reports pre-existing CRLF/trailing-whitespace changes, including protectedAGENTS.md; these were not repaired. Focused JointTraining diff check passed. No commit, push, cloud work, new task, stimulation, disposable GPU probe, or AGENTS.md write occurred.

## Operational status

**Recovered and independently verified training at 2026-09-23 08:14:42 UTC.** The parent now owns `proc_b345f6fe2730` (explicit `process_manage handoff` returned `handed_off`), supervisor15739 and sole MPS worker15763. Attached stdout/stderr remain active. The cap began **08:13:26.181166 UTC**, before migration, and ends **16:13:26.181166 UTC** (epoch1790180006.1811662). No cap reset/renewal.

Independent CPU readback verified migration model/Adam/scheduler/native streams/RNG exactly against terminal2860. First saved update2861 changed42 model tensors and42 Adam states, advancing42 Adam parameter step counters; scheduler replay matched the exact inherited next task/cell and native stream state advanced. `checkpoint_002861.pt` SHA256 is `19f4434754173005b4662685dcf0ebc593a8e97430bdaf2ceca9bdc8275197bf`. Persisted progress reached2870 /91840 episodes at this snapshot. All archived and original blocked files and frozen sources verified unchanged. See `recovery_independent_verification.json` and `recovery_parent_handoff.json` in the run directory; the executable CPU audit is `checks/verify_v4_recovery.py`.

Root `queue_result.json` intentionally continues to describe the blocked historical attempt; monitor `recovery_status.json`, `recovery_result.json`, `live_status.json`, `progress.jsonl` and `latest_checkpoint.json` for the recovered run. Final evaluations and `report.json` / `REPORT.md` remain pending; resumed training is not a completed-run claim.
