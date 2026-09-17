# Prospective-query experiment: user-stopped partial result

The user stopped this experiment after the first planned matched validation
showed no improvement on the primary cued motion-duration failure. Training
ended at logged step 9430: 1,030 additional updates and 41,200 fresh episodes,
25.75% of the planned exposure. The latest durable checkpoint is step 9216:
816 additional updates and 32,640 fresh episodes. The 214 later logged updates
/ 8,560 episodes are not represented in a checkpoint and must not be conflated
with the durable endpoint.

The worker received SIGTERM at 2026-09-15 02:02:05 UTC. The supervisor
finalized its ledger and complete artifact index, so its machine-readable
status is `failed` with return code -15; scientifically and operationally this
is a **user-stopped partial run**, not an unexplained training failure.

## Only completed matched validation

[Step-9200 full table](runs/prospective_20260914_175602/validation_9200_report.md)
uses 1,836 examples exactly matched to the historical control. The prospective
rank was `[-0.0260, 0.6422]`, versus control `[-0.1016, 0.6186]`.
Orientation and spatial binding improved substantially, but recognition was
mixed and clearly worse at load 24. Cued motion duration remained at or below
chance (normalized BA -0.0260, mean AUC 0.4946), and Krauzlis motion remained
at chance (normalized BA 0, mean AUC 0.4964).

The architectural intervention therefore did not address the primary known
motion failure at the completed checkpoint. Its orientation/binding gains are
secondary partial-run observations, not grounds to select the model. No
step-10000 validation, checkpoint selection or final held-out evaluation was
performed.

## Stability and learned scalar

Training remained finite through the logged endpoint. The last 100 updates had
mean loss 0.6721, composite five-microbatch accuracy 0.5908, mean pre-clip
gradient norm 1.9637 and clipping on 89% of updates. `gamma` moved from zero to
-0.01851 at the durable checkpoint and -0.01850 at the logged endpoint.
Individual per-task losses were not persisted; only the mean of the five task
microbatch losses was logged each update.

## Retrieval and billing closure

[Retrieval](runs/prospective_20260914_175602/retrieval_receipt.json) verified all
29 indexed artifacts, including every indexed checkpoint, complete step-9200
predictions/summary, metrics and ledgers. Archive SHA256 is
`4579749c01cd5c228e2b83a9c1a8ce53679d1243ab80314dc60752b7d43057b1`.

[Cleanup](runs/prospective_20260914_175602/cleanup_receipt.json) records provider
status `EXITED`, successful deletion, post-delete HTTP 404, an empty pod list
and `currentSpendPerHr=0`. The pod existed for 4,008.25 seconds (66.80 minutes);
at the listed $1.59/hour rate this is an estimated $1.7703, not an invoice.
The watcher completed successfully and no monitor remains active. No
replacement infrastructure or experiment was launched.
