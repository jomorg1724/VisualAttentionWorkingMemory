# User-requested equal-exposure continuation v4

## Request and interpretation

On 2026-09-23 the user said: “We should go ahead and plan another long training run. Lets continue training for the same amount of iterations”. This is another 2,145 optimizer updates, matching v3's additional exposure, not a repeat of the cumulative 2,860 updates. The assistant disclosed a new eight-hour cap and sequential handoff after v3 final evaluation.

## Scientific contract

- Predecessor: `runs/fresh_kda_joint_01_continuation_v3_8h`, currently finishing acquisition and its planned evaluations. Do not interrupt it or change its files, sources, deadline or monitoring.
- Source: verified **terminal2860**, preserving model, all compatible Adam state, scheduler and condition queues, native task streams, Python/NumPy/CPU/MPS RNG and cumulative exposures/optimizer time. Do not rewind to the validation-selected checkpoint.
- New versioned directory: `runs/fresh_kda_joint_01_continuation_v4_8h`.
- Exact additional acquisition: **2,145 updates / 68,640 episodes**, 165 updates / 5,280 episodes per task.
- Cumulative endpoint: **5,005 updates / 160,160 episodes**.
- Unchanged 13-task /35-cell suite, native stimuli/cues/labels, batch32/micro4, mean cross-entropy, all parameters trainable, Adam1e-4 one group, inherited betas/epsilon/weight decay, no clipping, fp32 full BPTT, equal shuffled task cycles and continued condition queues.
- One local MPS worker, CPU threads at most2. No cloud, second arm, architecture change, task teaching, stimulation or new probes.

## Scheduling and budget

Prepare CPU-only implementation/tests now. A durable CPU queue starts v4 only after v3 normal completion, complete validation2860 and all35-cell selected/terminal final evaluations (valid deduplication allowed), verified terminal checkpoint and actual predecessor worker/supervisor exit. The queue must distinguish launch tracking EOF from OS process exit and must fail closed if predecessor fails or ends incomplete. Wait deadline is v3's existing deadline plus a bounded cleanup grace; no endless queue. No added accelerator work while v3 is active.

A NEW finite **28,800-second cap** starts on actual v4 activation before source migration/first MPS work, not at queue creation. This does not modify v3's cap. Preserve that v4 origin on all subsequent paths; no automatic extension/restart. A normal completed predecessor may permit the new explicitly authorized run before the old maximum deadline, provided receipt/OS-exit checks pass. Do not globally disable the old v3 deadline guard; any new migration rule belongs only to the separately versioned v4 implementation.

Use all relevant existing v3 per-cell timing measurements and exact future scheduler queues to verify fit of the fixed exposure including four validations, two final tests, checkpoint/report overhead and safety margin. No disposable GPU profile. If the requested exposure cannot fit, stop before production and report it rather than silently reduce the target or renew the allowance.

## Validation and final results

- Reuse the completed terminal2860 validation as baseline; retain prior validation histories with their historical roles.
- Four new all35-cell validation looks: **3393,3926,4459,5005**.
- Keep v3 selection: equal-task mean validation AUC, then mean chance-normalized BA, earlier ties. Include the parent baseline; preserve prior selection records separately.
- Ordinary validation64 / Krauzlis100; finals128 /200, as in v3; empty recognition specificity separate and Krauzlis event/side subgroups retained.
- Use new final-only namespace **94492763**, verify not already used in this lineage. Fresh draws reuse official source identities and previously seen evaluation populations; disclose exploratory status. Prior test scores do not choose exposure, hyperparameters or checkpoint.
- Automatic final report, terminal/selected tests, cumulative validation curves and preserved checkpoints.

## Implementation and handoff acceptance

Researcher implements and tests actual v4 migration/queue/budget/next-update replay, including exact model/optimizer/sampler/stream/RNG preservation and single-worker enforcement. Keep v1/v2/v3 frozen source hashes unchanged. Do not edit `AGENTS.md`: a parent attempt was blocked by protected-file approval timeout; it must not be retried or bypassed. The user's direct request governs this experiment and this run-specific brief records the contract.

Start the CPU-only queue under a durable bounded process with completion notification and attached stdout/stderr. Return queue handle/PID, paths, current predecessor phase, queue expiry, activation conditions, test output and automatic handoff/status receipts. The child must not remain polling/waiting for predecessor completion: verify the durable queue is live and return. If predecessor has already completed safely, verify actual new persisted optimizer progress and first checkpoint before claiming training.

Update experiment documentation and LabJournal honestly: prepared/queued is not running. Parent independently verifies queue artifacts/process identity and, on activation, the new training checkpoint. Do not create a second MPS worker, terminate unrelated processes, commit or push.
