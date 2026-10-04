# User-cancelled three-layer KDA — artifacts preserved

User explicitly stopped this run to start the queued sixteen-head single-layer
KDA. Training stopped at **2,615updates /83,680episodes**, before the4,216target.
The only completed validation, at2,108, was50%BA in all three conditions and mean
AUC0.499388. There are **no completed final held-out tests** for this cancelled
run; do not present these live validation results as final acquisition evidence.

All71scientific artifacts were retrieved and hash-verified. Terminal checkpoint
was CPU-reloaded with55active Adam states; SHA256
`7b2768b5f4344b09a532b970a9e8c3b070263d5662da9a5203fb7922a47f0884`.
Pod `5hnvb87npqpqb4` was stopped and deleted after verified retrieval. No restart
or cap renewal. [Retrieval](CloudRuntime/user_cancellation_retrieval_verified.json)
· [Deletion](CloudRuntime/user_cancellation_cleanup_verified.json).

Earlier entries below are historical snapshots.

# Three-layer whole-sequence KDA run

**Training running and independently verified, 2026-10-02.** Single A40 pod`5hnvb87npqpqb4` at$0.49/GPUhour. Exactly3 pre-LN residual KDA layers,324,488fresh trainable parameters. Native GPU profiling pinned the full **4,216updates /134,912episodes**, matching the completed one-layer run: B12/B20 each1,405updates/44,960episodes, B281,406/44,992. First3profile warmup updates and all disposable profile state were discarded; next6steady updates estimated12,004.4optimizerseconds,15,954.0seconds including margins/reserves. No exposure reduction.

Production checkpoint3/96episodes was downloaded, digest-verified and CPU-reloaded. All55learned parameter tensors changed, all55named Adam states advanced; initial model matched the fresh seeded constructor with empty Adam/streams. First production cycle was1.083x matched steady profile. Startup snapshot observed4updates/128episodes; it is not a current-step or learning result. Validation at2,108/4,216 and fresh paired final tests remain pending.

The original cap began **2026-10-02T18:20:05.521299+00:00** and ends **2026-10-03T02:20:05.521299+00:00 /October2 7:20:05 PM PDT**, scientific cutoff10minutes earlier. Initial pod`m7yn5d2uew9ltc` suffered SSH timeout before any upload/optimizer, was stopped, could not restart due host GPU availability, then was deleted with failure receipts preserved. The replacement keeps the same original cap/budget; neither was renewed.

Authenticated independent shutdown guard and detached15second artifact mirror PID43158 are active. Final artifacts and both final checkpoint CPU reloads must verify before cleanup; only this fresh pod is deleted. Source archiveSHA256`20bfab525eebe9d6e760ef0ddde14306a5a724d2cc197d7eb0c2a5d7e45a72ae`. Startup checkpointSHA256`41890f91d9d7985fc299a4c2c9880bd4cb3832aa0a76e7b9bd6078d6e4ccadf1`.

Evidence: [persisted production](CloudRuntime/production_verified.json), [downloaded checkpoint](CloudRuntime/downloaded_checkpoint_verified.json), [replacement](CloudRuntime/replacement_verified.json), [guard](CloudRuntime/guard_verified.json), [budget](CloudRuntime/budget.json), [mirror](CloudRuntime/mirror_started.json). Four focused CPU checks passed (two model, two planner/criteria); the native GPU profile exercises the complete unchanged task. No repeated CPU native training campaign or broad audit was required.

The user requested depth3 after the one-layer model finished4,216updates/134,912episodes at50% balanced accuracy in every native condition. This run changes only depth: three pre-LN residual KDA blocks, with full-token outputs from layers1/2 and CLS-only decoding after layer3. Width128, two heads, head key/value64, shared10x10RGB patch projection, spatial/time positions and terminalCLS remain unchanged. No FFN, CNN, GRU or extra arm.

Every model/Adam/RNG/stream/counter starts fresh. Reuse task schedule and draw seeds to match the earlier run; this is fresh deterministic initialization, not saved-state transfer. Native B12/B20/B28,26/28degree changes, cues/rendering/teaching/event proportions, Adam1e-4/no clipping, batch32/micro4 and FP32/full temporal gradients remain unchanged.

Target4,216updates/134,912episodes, matching one-layer exposure, reduced only before production if steady native GPU profiling requires it. Disposable warmup/profile state is discarded. New singleA40 cap8hours/$5 including setup/profile/evaluation/retrieval; absolute guard and artifact mirror reused in an isolated new attempt. No automatic extension or extra depth sweep.

Validation-only checkpoint selection uses mean three-conditionAUC thenBA; final selected/terminal tests200fresh paired trials/condition. Provisional acquisition signal is >=70%BA in every condition; provisional solved target >=90%BA in every condition. Both are reporting criteria; healthy training runs through the pinned allocation.

[One-layer baseline](../SequenceKDA/RUN_STATUS.md). Launch and persisted optimizer evidence will be recorded here once available.
