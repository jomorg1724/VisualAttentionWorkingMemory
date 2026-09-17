# 19 — Five-task acquisition with the original attention biases restored

[Index](../README.md) · [Current status](../CURRENT_STATUS.md) · [Previous bias-removal result](18-unbiased-attention-spatial-battery.md)

**Closed 2026-09-14T23:44:02.980959+00:00: user-stopped partial run, all73 artifacts verified, pod stopped/deleted at 2026-09-14 05:52:23 UTC and post-deletion pod list empty. Monitor paused; no remaining project model workers. No final held-out evaluation was performed.**

[Verified cleanup](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json) · [Verified retrieval](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/retrieval_receipt.json) · [Partial validation report](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md).

The chronological launch/status notes below are historical. Neither the pod nor the local motion worker is active. Logged11923, durable11776, selected-so-far10000 and last-validation11600 are separate endpoints; no shutdown changes their interpretation.

## Original rationale and launch history


The user cancelled the five-task bias-removed run, then requested restoring the original source and distance biases and launching again. The completed old-task comparison showed severe delayed-memory regressions after their joint removal. That result motivates retaining the original computation for the separately taught spatial tasks, but does not prove the individual function of either bias term.

Start independently from the intact trained attention 8400 parent. Preserve its learned source bias, locality penalty, Q/K/V/O projections, embeddings, sensory encoder, E/I memory and comparator. Do not warm-start from the damaged old-task terminal model or cancelled five-task weights. Initialize the same five fresh heads and task-local streams using the prior new-battery policy. Implementation: [BiasedTraining README](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/README.md) and [model](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/model.py). Both learned bias terms remain trainable, with 123 compatible Adam states retained. Fresh head initialization is identical to the cancelled arm; its partially trained weights are not reused.

The [five-task protocol](../../WorkingMemory/SpatialTaskBattery/PROTOCOL.md), [source rationale](../../WorkingMemory/SpatialTaskBattery/SOURCES.md), rendered stimuli, labels and 27-condition evaluation matrix stay fixed. Use five batch-8 family microbatches per update, averaged family losses, full BPTT/fp32, one clipped Adam step and the inherited learning-rate policy. Target 4,000 updates / 160,000 episodes. At launch, the existing healthy L40 pod `ep8bmcjxz9k66l` was retained, with the original **2026-09-14 09:36:25.001 UTC deadline**; this restart does not renew time or authorize another arm.

At initial launch no results were available. The cancelled bias-removed acquisition arm stopped at logged +150 updates / 6,000 episodes and durable +48 updates / 1,920 episodes. It is not an equally trained baseline; differences from its partial scores cannot identify the causal effect of restored biases. The completed old-task comparison remains valid for its distinct task battery and exposure, not this new acquisition question.

[Prior cancellation receipt](../../WorkingMemory/UnbiasedAttention/Cloud/cancellation_receipt.json) · [Existing pod record](../../WorkingMemory/UnbiasedAttention/Cloud/cloud_provisioning.json). [Launch receipt](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/launch_receipt.json) verifies the advancing step, parent/source hashes and pinned exposure. Runtime is Python 3.10.12 / Torch 1.13.1+cu117, NumPy 1.23.1, SciPy 1.8.1 and Pillow 9.1.1. Remote supervisor 20111 / worker 20260 and incremental artifact watcher 43720 were active at launch; the watcher had verified two checkpoints. Profile cost is about 5.28 seconds/update and nine minutes evaluation, an estimate rather than completed work. No old-task rerun is included. Broad latent probes, residual experiments and other extra model tests remain paused/unrun.


## Live validation and authorized local branch (2026-09-14T04:42:55.701953+00:00)

Root's latest remote inspection reports active cloud training at global 10613; checkpoint 10000 is the latest completed validation. After 64,000 joint episodes, family-average validation is binding 100%, orientation 54.2969%, duration 25.3906%, Krauzlis 50%, recognition 69.7917% excluding load 0. These are provisional validation values, not final held-out results. The user authorized a local [motion-only continuation](20-single-task-motion.md) from this checkpoint, with the same architecture/tasks and only the motion loss supplying gradients. The joint cloud run remains unchanged; the local target is 32,000 episodes under a separate 7,200-second cap and is not yet verified launched. Uneven acquisition motivates the test but does not prove multi-task interference.


## User requests retrieval and shutdown (2026-09-14T05:51:01.733402+00:00)

The user asked to retrieve the current cloud results and remove the pod for the night. The researcher stopped supervisor20111 and worker55057; both were confirmed absent before freezing `biased_results`. Retrieval is in progress. The exact logged endpoint, durable checkpoint, last completed validation and archive verification will be filled from the saved receipt, without inventing completed exposure or final-test scores. Root will delete the pod after successful retrieval and provide cleanup evidence. The local SingleTaskMotion experiment is already complete. No additional model evaluation is authorized by this shutdown request.


[Saved stop receipt](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/user_stop_receipt.json) records 2026-09-14T05:50:45.027845+00:00: last logged global **11923**, +3,523 updates / **140,920 episodes**; last durable checkpoint **11776**, +3,376 updates / **135,040 episodes**. The remaining147 logged updates (5,880 episodes) are not checkpointed and cannot be represented as a saved trained model. Planned exposure was160,000 episodes. Latest completed validation is11600. **No final held-out evaluation was performed.** GPU process listing is empty and73 files are frozen for retrieval. Subsequent retrieval and API cleanup are complete; see the closure at the top.


## Retrieved partial results: last completed validation 11600

[Retrieval receipt](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/retrieval_receipt.json) verifies all73 saved files, including checkpoints restored from hash-verified incremental transfers. Watcher43720 has exited. The archive SHA-256 is recorded in that receipt. Subsequent [API cleanup](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json) confirms pod deletion at05:52:23UTC and an empty pod list.

Validation-selected so far is **10000**. The following scores are **last completed validation11600**, not the selected checkpoint's results, not the durable11776 checkpoint's results and not a final held-out test. Non-Krauzlis cells use64 examples; Krauzlis cells use100. The run stopped before its planned full exposure and final test.

| Last validation condition | BA | Accuracy | AUC |
|---|---:|---:|---:|
| image_recognition_N0_H3 | undefined | 100.00% | None |
| image_recognition_N0_H4 | undefined | 100.00% | None |
| image_recognition_N0_H5 | undefined | 100.00% | None |
| image_recognition_N12_H3 | 82.81% | 82.81% | 0.94140625 |
| image_recognition_N12_H4 | 82.81% | 82.81% | 0.947265625 |
| image_recognition_N12_H5 | 82.81% | 82.81% | 0.94140625 |
| image_recognition_N24_H3 | 71.88% | 71.88% | 0.7958984375 |
| image_recognition_N24_H4 | 71.88% | 71.88% | 0.8193359375 |
| image_recognition_N24_H5 | 73.44% | 73.44% | 0.8271484375 |
| image_recognition_N4_H3 | 96.88% | 96.88% | 0.99609375 |
| image_recognition_N4_H4 | 96.88% | 96.88% | 0.998046875 |
| image_recognition_N4_H5 | 96.88% | 96.88% | 0.998046875 |
| krauzlis_B12 | 50.00% | 57.00% | 0.41819665442676457 |
| krauzlis_B20 | 50.00% | 57.00% | 0.4500203998368013 |
| krauzlis_B28 | 50.00% | 57.00% | 0.5214198286413709 |
| motion_duration_cued_D0 | 25.00% | 25.00% | 0.51953125 |
| motion_duration_cued_D12 | 25.00% | 25.00% | 0.486328125 |
| motion_duration_cued_D24 | 25.00% | 25.00% | 0.49674479166666674 |
| motion_duration_cued_D4 | 23.44% | 23.44% | 0.5198567708333333 |
| orientation_cued_D0 | 42.19% | 42.19% | 0.34375 |
| orientation_cued_D12 | 45.31% | 45.31% | 0.521484375 |
| orientation_cued_D24 | 46.88% | 46.88% | 0.4921875 |
| orientation_cued_D4 | 53.12% | 53.12% | 0.451171875 |
| spatial_binding_D0 | 100.00% | 100.00% | 1.0 |
| spatial_binding_D12 | 100.00% | 100.00% | 1.0 |
| spatial_binding_D24 | 100.00% | 100.00% | 1.0 |
| spatial_binding_D4 | 100.00% | 100.00% | 1.0 |

Binding and exact image-set recognition show clear acquisition on these validation examples, while cued signed orientation, cued motion duration and Krauzlis target/foil change remain weak. Recognition load0 has100% rejection but no positive examples, so BA/AUC are undefined and cannot enter ranking. High recognition accuracy is exact raster membership in source-disjoint photographic splits, not semantic recognition or a demonstrated human memory capacity. The task-specific findings must not be combined into an all-task success claim.

[Full partial-run report](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md) preserves all cells, metric types and report boundaries. Stopping and retrieving does not convert validation into final evaluation. The independently completed local motion-only failure is reported in [entry20](20-single-task-motion.md).


## Final closure

Root verified stop statusEXITED, deletionHTTP204 and `list_pods` returning zero pods at2026-09-14 05:52:23UTC. All73 artifacts were verified before deletion. Local SpatialTaskBattery, PreUpdateAttention and TrainingExposure Python workers were absent; the completion monitor was paused. [Cleanup receipt](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json). No compute or further evaluation remains active or authorized by this run.
