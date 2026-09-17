# 21. Prospective sensory-conditioned attention queries

[Current status](../CURRENT_STATUS.md) · [Implementation and live report](../../WorkingMemory/ProspectiveQuery/report.md) · [Launch receipt](../../WorkingMemory/ProspectiveQuery/launch_receipt.json)

## Why this experiment

The original pre-update attention lets previous spatial memory query current sensory and memory value banks. That architecture improved delayed orientation but immediate motion remained weak, and focused motion-only continuation did not acquire the cued duration task. The intervention here tests one narrower architectural hypothesis: routing may be limited because the query is formed only from prior memory and cannot prospectively incorporate the current cue/sensory field.

## What changed

One scalar sensory residual was added to the existing query input:

`Q_t = W_Q(LN(R_{t-1}) + P + e_m + gamma * LN(H_t))`.

The learned scalar `gamma` starts at zero. Thus the migrated model begins exactly at the original query function but can learn whether current sensory context should alter routing. The same normalization and inherited attention projections are retained. No keys, values, attention biases, recurrence, comparator, pooling, task, stimulus, loss, optimizer setting, seed, exposure or evaluation rule changed.

Migration begins independently from the intact attention8400 checkpoint (`e37602…bc9`), not from the stopped five-task run or failed motion-only continuation. Strict checks reject any already migrated or incorrect parent.

## Pre-launch evidence

Focused checks found exact zero difference at `gamma=0` for queries, attention output and inherited full-model output. Gamma nevertheless had a nonzero gradient at zero. Enabling the sensory term changed both queries and routing under controlled sensory changes. Source/locality parameters and the planned 4,000-update / 160,000-episode exposure remained fixed. These are implementation checks, not evidence that training will improve a task.

## Live execution

Fresh RunPod pod `dqi13o2x3qkvos` uses one A100-SXM4-80GB at $1.59/hour with the inherited PyTorch 1.13.1+cu117 runtime. Profiling completed and production advanced beyond parent step 8400; at 2026-09-15 01:04:25 UTC it had reached step 8419 / 760 fresh episodes with finite metrics and `gamma=0.00198899`. The supervisor and worker were alive and GPU utilization was observed.

The first setup attempt exposed two operational defects before training: CRLF shell line endings and omitted BSDS500 runtime images. The script was normalized, the exact fixture payload was hash-verified, and the same run ledger and original deadline resumed. No training exposure was consumed by either failed setup/profile attempt, and no scientific setting or budget deadline changed.

Only validation steps 9200, 10000, 10800 and 11600 can be compared to the historical biased control. That run lacks step 12400 and final held-out results, so prospective terminal/final results are explicitly unmatched exploratory endpoints.

## Diagnostics and next decision

The bounded local frozen diagnostics completed without local GPU training or checkpoint mutation. In 32 paired movies, changing only the valid cue rings changed the frozen model's prediction in 15 cases (46.88%, Wilson 95% interval 30.87–63.55%) and measurably changed attention routing. However, paired original/counterfactual accuracies were 21.88%/34.38% and mean correct-class probabilities stayed near 0.25. The baseline therefore detects cue changes but does not demonstrate successful use of the cued duration rule.

The independent split linear probe used 256/64/128 base episodes and decoded the longest-duration patch direction from local final-moving-frame sensory features at 25.98% on held-out data, versus 25% chance. Target-patch accuracy was 23.44%. This specific summary provides no evidence of accessible duration direction before pooling, but probe failure does not establish erasure or absence from other times, spatial summaries or nonlinear codes. [The diagnostic receipt](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/diagnostic_summary.json) records the exact split and results.

The cloud watcher owned incremental hash-verified checkpoint retrieval.

## First matched validation: step 9200

The first planned validation completed and training resumed toward step 10000.
All 1,836 examples match the historical control exactly on labels, IDs and
metadata. [The full per-condition comparison](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/validation_9200_report.md)
shows substantial point gains for orientation and spatial binding, mixed
recognition with clear load-24 losses, and motion tasks still near chance.
The prospective selection rank is better at this look, but this is live
validation rather than selected or final held-out performance.

## User stop and final disposition

The user stopped the run because the only completed matched validation did not
improve the primary cued motion-duration failure. Logged training reached step
9430 / 41,200 fresh episodes; durable checkpoint 9216 contains 32,640 episodes.
The 214 later updates / 8,560 episodes are preserved in metrics but not in a
checkpoint. No step-10000 validation, checkpoint selection or final held-out
evaluation occurred.

All 29 indexed artifacts were retrieved and verified. Pod
`dqi13o2x3qkvos` reached provider state `EXITED`, was deleted, then returned
HTTP 404; the account pod list is empty and current spend is zero. See the
[completion report](../../WorkingMemory/ProspectiveQuery/completion_report.md)
and [cleanup receipt](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/cleanup_receipt.json).
