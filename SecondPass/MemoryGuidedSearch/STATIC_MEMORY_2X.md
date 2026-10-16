# Static memory-guided search at twice the sampling weight

User-authorized experiment, October 14, 2026.

## October 16 continuation and extension

Resumed later October 16 at explicit request in `LocalRuntime/static_memory2x05`
from latest 173,247 toward the unchanged 193,800 endpoint. Exact restore and
matching disposable next update passed; saved 173,287 independently verifies
advancement of all 92 learned parameter tensors and Adam states, with the
weighted schedule preserved. Best 169,870 remains eligible. The unused 52,205
seconds of local allowance are carried forward; supervisor deadline is October
17 at 02:39:40 PDT. Next monitoring is 173,627, then every 380 updates. Stop
requests now belong in `static_memory2x05/stop_requested.json`.

Paused at the user's request at update 173,247, with 20,553 updates remaining
to 193,800. Both checkpoints were CPU-reloaded and their optimizer and resume
states verified. Worker and supervisor exited cleanly; GPU lock is free. Best
remains 169,870 with monitoring mean AUC 0.9463159310. Latest monitoring at
173,223 gave mean AUC 0.9435347554 and static memory-search accuracy 53.3854%.
Unused local allowance is 52,205 seconds; resume only on a new request. Receipts:
`LocalRuntime/static_memory2x04/pause_verified.json` and `continuation_ready.json`.

User requested another 19,000 updates added to the endpoint and resumption.
`LocalRuntime/static_memory2x04` resumes latest 170,183 and targets 193,800,
leaving 23,617 updates at launch. Best 169,870 remains eligible. All training
settings and the 2:1 sampling mixture remain unchanged, with the unfinished
weighted bag, optimizer, streams and RNG state restored exactly.

Saved update 170,223 independently verifies advancement of all 92 learned
parameter tensors and all 92 finite aligned Adam states by expected task counts.
Original checkpoints and source hashes are unchanged. See the run's
`independent_production_verified.json`. Monitoring starts at 170,563, then every
380 updates. The new 16-hour local cap runs from October 16 at 09:23:32 PDT to
October 17 at 01:23:32 PDT, including evaluation. No automatic restart or
extension. Stop requests belong in `static_memory2x04/stop_requested.json`.

## October 15 continuation

Stopped cleanly for the night at the user's request at update 170,183, leaving
4,617 updates to the 174,800 target. Best checkpoint 169,870 has monitoring mean
AUC 0.9463159310 and static memory-search accuracy 54.6875%. Both checkpoints
were CPU-reloaded and their optimizer, weighted scheduler and resume state
verified. Worker and supervisor exited; the GPU lock is free. Remaining local
allowance is 14,669 seconds. Full final evaluation remains unfinished. Resume
only on a new request. See `static_memory2x03/night_stop_verified.json` and
`continuation_ready.json` in `LocalRuntime`.

Resumed later October 15 on explicit request in `LocalRuntime/static_memory2x03`,
from latest 149,730 toward the unchanged 174,800 endpoint. All learning and RNG
state, including the partially consumed weighted task bag, restored exactly.
Saved update 149,770 independently verifies all 92 parameter tensors and Adam
states advanced correctly. The remaining 49,510-second local allowance is
carried forward, with supervisor shutdown at October 16, 02:43:53 PDT. Next
monitoring is 150,110, then every 380 updates. Stop requests now belong in
`static_memory2x03/stop_requested.json`.

Paused at the user's request at update 149,730: 5,273 updates completed in this
continuation, with 25,070 remaining to 174,800. Worker and supervisor exited
cleanly and the GPU lock is free. Latest and best checkpoints were CPU-reloaded
and verified, including all 92 Adam states and the weighted scheduler. The best
remains update 139,840, monitoring mean AUC 0.9448987979. Last monitoring at
149,397 gave mean AUC 0.9397885881 and static memory-search accuracy 46.09%.
Resume only on explicit request; 49,510 unused allocation seconds are recorded
in `LocalRuntime/static_memory2x02/pause_verified.json`.

The user requested resumption and another 19,000 updates added to the previous
155,800 endpoint. `LocalRuntime/static_memory2x02` resumes latest update 144,457
and targets 174,800: 30,343 further updates. The unfinished weighted task bag,
condition queues, native streams, optimizer and all random-number states are
restored exactly. The 2:1 mixture and model/training settings are unchanged.
Best update 139,840 (monitoring equal-task mean AUC 0.9448987979) remains eligible.

Independent verification at saved update 144,497 confirmed all 92 learned
parameter tensors changed, all 92 Adam states advanced by the expected task
counts, and the weighted schedule matched the restored sequence. Original
checkpoints and source hashes are unchanged. See
`LocalRuntime/static_memory2x02/independent_production_verified.json`.

Monitoring begins at 144,837 and repeats every 380 additional updates. The
16-hour local allocation runs from October 15 at 08:43:20 PDT until October 16
at 00:43:20 PDT, including evaluations. No automatic restart or extension.
Request a clean stop through `static_memory2x02/stop_requested.json`.

The equal-task run stopped cleanly at update 144,368. Its best checkpoint at
136,800 (monitoring equal-task mean AUC 0.9437480324) is the starting point for
this separate experiment. Both checkpoints, the best validation results and
original source files are preserved with hashes in
`Milestones/best_136800_before_static_memory_2x_20261014/manifest.json`.

## Training mixture

Each shuffled 20-batch block contains two `memory_guided_search` batches and
one batch for each of the other 18 tasks. Batch size stays 32. Static memory
therefore occupies 10% of training presentations; each other task occupies 5%.
The former equal-task mixture allocated 1/19 to each task. “Twice” means a 2:1
ratio relative to each other task, rather than doubling the total batch size.

Exactly 19,000 additional updates comprise 950 complete blocks: 1,900 static
memory updates (60,800 presentations) and 950 updates (30,400 presentations)
for each other task. This is 608,000 presentations in total. Condition queues
within tasks retain their original balancing and native stimulus laws.

## Restoration and selection

Weights, all 92 Adam states, native stream counters and all random states are
restored from the archived best checkpoint. The scheduler RNG and condition
queues are preserved. The old task bag is replaced with the new 20-entry
weighted bag at the first draw. Weighted scheduler state is checkpointed so
subsequent resumes preserve any partially consumed bag.

Architecture, losses, Adam learning rate 3e-4, batch32/micro4 and clip100 remain
unchanged. Validation remains the original 71-condition monitoring panel,
32 trials per condition, selected by equal-task mean AUC every380 updates.
The archived baseline remains eligible. Full437-condition validation64 and
test256 are scheduled after training, within the allocation.

## Runtime and verification

`LocalRuntime/static_memory2x01` starts at136800 and ends at155800; progress
also reports `new_updates`, with a target of19000 additional updates. A single
finite16-hour local allocation includes training and evaluation. The supervisor
does not automatically restart or extend the allocation. A clean stop is
requested through `static_memory2x01/stop_requested.json`.

The isolated `weighted_resume.py` and `launch_weighted.py` leave original
trainers unchanged. Two contract tests verify the exact19,000-update quotas,
condition coverage, preservation of learning state and deterministic continuation
from a partially consumed weighted scheduler. Production verification records
actual saved parameter/Adam advancement, task ratios and original hashes.
