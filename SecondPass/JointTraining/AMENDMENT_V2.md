# Protocol amendment v2 — training exposure within the original four hours

**Authorization:** the user explicitly selected **“Keep the existing four-hour limit.”** This amendment supersedes only the initial training-exposure/evaluation allocation. It does not authorize an overnight-equivalent run, cap extension, new model, reset, additional accelerator worker, changed task/loss/LR or curriculum.

## Allocation mistake and correction

The initial planner chose **156 updates / 4,992 episodes / 384 episodes per task**. It gave evaluation precision priority over acquisition, used conservative profile estimates, reserved four full validation looks and roughly 5,734 seconds for finalization, and rounded training to multiples of twelve task cycles. Fitting the wall cap was not enough: this was an unnecessarily small acquisition allocation. It should have reduced optional evaluation precision/cadence before constraining training so heavily.

The separately versioned `continuation_v2.py` continues the **same fresh-weight lineage**, not a new experiment. The original `worker.py`, `core.py`, `launch.py`, task renderers and all their frozen source hashes are unchanged. Original artifacts remain in `runs/fresh_kda_joint_01/` as the superseded pilot allocation. The continuation lives in `runs/fresh_kda_joint_01_continuation_v2/`.

## Unchanged hard budget

- Origin: epoch **1790059676.5856378**.
- Absolute deadline: epoch **1790074076.5856378**, **2026-09-22 10:47:56.585638 UTC**.
- Total allowance remains **14,400 seconds**, including original profiling, training, handover/debugging, continuation, validation, final tests and reporting.
- The supervisor retains the original absolute hard kill deadline. The worker checks a final-evaluation reserve before each update. A slowdown may cause fewer updates or explicitly incomplete evaluation; it never extends the cap or fabricates completion.

## Pinned amended allocation

The pre-continuation configuration pins **715 total updates / 55 complete 13-task cycles / 22,880 cumulative episodes**, including the **117 updates / 3,744 episodes** carried from v1. That is **1,760 episodes per task**, versus the original 384, and 598 additional updates after the migration checkpoint. Batch 32, microbatch 4, full BPTT, fp32, all trainable parameters, Adam LR 1e-4, one group and no clipping remain unchanged.

The plan uses only measured timing/exposure, not test performance:

- Measured equal-task/condition-adjusted update cost: **10.2224853 seconds**, inflated by **1.20** for planning.
- Last complete 35-cell validation: **463.1577632 seconds**; per-cell `seconds / n` extrapolates amended evaluation sizes, inflated by **1.25**.
- One new full validation: **578.7964223 seconds** reserved.
- Each final selected/terminal pass: **1,157.5928446 seconds** reserved.
- Combined final validation, two tests and save/report reserve: **3,133.9821115 seconds**; initial additional implementation/startup margin **420 seconds**.
- Projected total remaining cost at pinning, including that margin: **10,889.6375863 seconds**. A pre-optimizer guard correction consumed part of the margin; recheck before successful launch retained about **308 seconds** beyond the guarded work estimate, without changing the target or deadline.

`config.json` pins exact per-cell training exposure by replaying the preserved scheduler **without drawing stimuli or touching the accelerator**. Single-cell sensory/ring tasks receive 1,760 episodes each; four-delay tasks have 416–448 per cell; Krauzlis baselines 576–608; recognition cells 128–160. Condition queues continue unchanged; no condition is dropped or reweighted.

## Selection and final evaluation

Completed full-cell validation looks **39, 78 and 117** and their selection history remain valid. The one new planned look is **715** (or the earlier termination boundary if the wall reserve or an operator signal stops training). Validation stays at **64 ordinary / 100 Krauzlis episodes per cell** on the original fixed validation draws, preserving comparability and the original validation-only lexicographic selection rule.

Final ordinary-cell precision is explicitly reduced **256 → 128**; Krauzlis remains **200** per baseline. Both terminal and validation-selected checkpoints cover **all 35 cells**. Reuse is allowed only when they are the same checkpoint. N0 specificity/FPR, Krauzlis target/foil/catch and side denominators, class confusion/recall, and task/cell reports remain separate. There is no pooled concealment or test-driven selection.

Because v1 automatically enters final evaluation after saving its terminal checkpoint, old test draws **may have been forwarded** during the handover window. No old final-test metrics were inspected for this amendment. Final evaluation therefore uses a fresh test-only seed namespace **94292763**, retaining the unchanged official test source-identity split and native renderers. Training and validation namespaces do not change. Earlier suite stimulus verification also used the original test generators; that historical disclosure remains.

## Verified state-preserving handover

1. SIGTERM requested a safe boundary from original worker **70221**. Its in-flight full validation at step 117 finished. All 117 completed updates were retained.
2. Once `terminal.pt` was atomically written, the exact worker was suspended, the checkpoint CPU-loaded and validated, then the exact worker was terminated to defer v1 finalization. Original supervisor **70215** exited. Worker return code **-9** / supervisor shell exit **247** is an **intentional protocol handover**, not a scientific training failure or hard-cap event.
3. Source terminal SHA-256: `ac5578ead3d648f660b193aa8a69c3ee1c2d358708346a719d1deb23ad2aa032`. Model, Adam, task scheduler, native stream queues, CPU/MPS/NumPy/Python RNG, exposure, optimizer clock and selection history are preserved.
4. The v2 migration save/readback verifies exact equality of model, optimizer, scheduler, streams and RNG against the source before any new update. `resume_integrity.json` records the check.
5. An initial **pre-optimizer** v2 launch failed because command-substring process detection misidentified macOS's `caffeinate` helper as a second worker. No migration/update ran in that attempt. `failed_launch_01/` preserves its source/config/supervisor evidence. Executable-and-interpreter-argument matching fixes the root cause; concurrency detection and the exclusive worker lock remain enabled.
6. Successful v2 supervisor **78932** / worker **78937**, tracked process **`proc_e7bdb8db0d3f`**, was transferred to the parent with completion notification. There is one actual joint-training Python worker; original PIDs are absent.
7. The verified launch snapshot advanced through **122 updates / 3,904 episodes**, with post-resume **checkpoint 118** SHA/load verified and **66 Adam parameter states**. Use current `live_status.json`, `latest_checkpoint.json` and `handoff.json` for later progress, not this dated snapshot.

`progress.jsonl` carries all original 117 rows and appends resumed updates with cumulative episodes, optimizer seconds, task/cell exposure and training loss. It is the persistent training-only learning-curve source; resumed curves do not reset their x-axis. Automatic final outputs are `report.json`, `REPORT.md`, `test_terminal.json`, `test_selected.json`; incomplete coverage is explicit and makes the CLI non-successful.

## Verification and interpretation

**23 tests passed** in the actual Torch 2.2.2 / NumPy 1.26.4 runtime: exact resumed next Adam update, streams/scheduler/RNG; budget/deadline refusal; live-cost cycle planning; fresh test namespace; real CPU continuation and all-cell selected/terminal evaluation; N0 and Krauzlis denominators; guard false positives; existing suite/harness regression tests. No new accelerator profile or comparison model was run.

This is **more substantive acquisition within the user-retained four-hour cap**, not evidence of sufficient convergence, overnight-equivalent training, general working-memory capacity or biological validity. Final held-out performance is still pending.
