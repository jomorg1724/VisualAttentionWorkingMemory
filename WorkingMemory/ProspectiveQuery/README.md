# Prospective sensory-conditioned queries (version 1)

This is a prospective, separately versioned **architecture-only** intervention.
Its RunPod experiment was user-stopped after the first matched validation,
retrieved, and deleted. It started from the exact original-bias
`attention8400` parent (`e37602…bc9`) and uses the unchanged five-task
`BiasedTraining` protocol.

The only architecture change is

`Q = Wq(LN(old memory) + position + memory-source embedding + gamma * LN(current sensory field))`.

`gamma` is one learned scalar, initialized to zero. Zero preserves the original
function while allowing a direct gradient through the current-sensory query
path. The same affine `query_norm` is reused, so no second normalization or
other learned tensor is added. Keys, values, Q/K/V/O weights, position/source
embeddings, learned source and locality biases, E/I recurrence, comparator,
pooling, heads, stimuli, losses, seeds, optimizer hyperparameters and
evaluation/selection protocol are unchanged.

Migration accepts only the original `spatial_preupdate_joint_attention_v1`
step-8400 checkpoint with the recorded SHA256. It retains every parent tensor
and compatible Adam state, creates the same deterministic five semantic heads
as `BiasedTraining`, adds only `attention.gamma`, and rejects a migrated
checkpoint.

The planned exposure is 4,000 updates. Every update uses five task
microbatches of eight episodes, averages the five cross-entropies, clips once,
and takes one Adam step: 160,000 fresh episodes total. Validation occurs after
each 800 updates and uses the original minimum normalized task BA then mean
task AUC rank.

Exactly one arm is authorized: `prospective_query`. There is no new or resumed
control training. Direct control comparison is limited to the historical
original-bias validation points at 9200, 10000, 10800 and 11600. That control
was user-stopped before step 12400 and before final held-out tests. Prospective
step-12400 validation, selected held-out, and terminal held-out results must
therefore be labeled **unmatched exploratory** unless a matched control later
exists. The validation/test generators, seeds, sizes, metrics and selection
rule remain unchanged.

`sweep.py` resumes only an existing run with the same absolute deadline,
source-hash ledger, parent, fixed exposure and sole arm. Completed jobs are
read from their exact job/result cache. Training resumes from the newest
hash-indexed checkpoint not beyond the requested target. Both supervisor and
worker reject corrupt, missing, duplicate or unindexed checkpoints; an
existing checkpoint is never silently skipped. The original start time and
deadline are retained, so restart does not renew a budget.

Each microbatch again fails immediately on nonfinite loss. Logged
`gamma_gradient_preclip` is captured before global clipping. Diagnostics named
`*_cumulative_preclip` explicitly include gradients from preceding task
microbatches in that five-task update; the current sensory-field gradient is
task-local.

`prepare_bundle.py`, `sweep.py`, and `worker.py` do not provision cloud
resources. `cloud_provisioning.json` is the credential-free live runtime
receipt. `setup_remote.sh` reads its
remote root, bundle hash, deadline and environment paths, refuses duplicate
supervisors, and supports same-ledger supervisor restart. `watch_remote.py`
uses its SSH path references at runtime, incrementally syncs only indexed
checkpoints, verifies the non-checkpoint archive and complete artifact index,
and stops retrying after the original deadline plus bounded retrieval grace.
It does not stop/delete a pod; an external coordinator must do so after a
verified receipt. No reporter is added; comparison labels live in the
aggregate/results to avoid presenting unmatched tests as controlled results.

`diagnostic.py` is analysis-only. It evaluates frozen `BiasedTraining`
checkpoint 10000 on paired motion movies whose evidence rasters are identical
outside the changed valid cue rings. Its optional frozen pre-pooling probe uses
independent train/validation/test streams, train-only scaling, validation-only
ridge selection, and location-local final-motion sensory fields. Neither path
changes deployed weights.

`construction_checks.json` records the focused CPU invariant tests.
`diagnostic_smoke.json` and `diagnostic_probe_smoke.json` are one-pair/tiny-split
execution checks only; their scores are not experiment results.

[`report.md`](report.md) and [`launch_receipt.json`](launch_receipt.json) record
the controlled protocol, hashes, initial live evidence, setup corrections and
comparison boundary. Full local diagnostic results are stored under the dated
run directory when complete.
