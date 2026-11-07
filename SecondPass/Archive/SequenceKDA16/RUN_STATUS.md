# Completed — task not acquired

All4216updates/134912fresh trials completed. Selected2108 and terminal4216
both give50% balanced accuracy in all3conditions (all-positive decisions).
Fresh selected/terminal mean AUC0.476345/0.447496. All45 manifest artifacts
verified, both checkpoints CPU-reloaded with27Adamstates, and podpa0ko8f2qirisy
stopped/deleted on2026-10-03UTC. No restart or cap renewal.

[Final evidence](CloudRuntime/artifacts/REPORT.md) · [Retrieval](CloudRuntime/retrieval_verified.json) · [Deletion](CloudRuntime/cleanup_verified.json)

Earlier entries below are historical.

# Sixteen-head single-layer KDA — training, optimizer verified

User cancelled the preceding three-layer KDA and requested immediate start of
this queued model. New A40 pod **pa0ko8f2qirisy** is training from scratch.
**4,216updates /134,912episodes** pinned before production from native CUDA
profiling. Peak profiled GPU allocation12,662,468,096bytes; full FP32/BPTT fits
batch32/micro4. Validation at2,108/4,216; paired selected/terminal final tests
200trials per native B12/B20/B28 condition. Native teaching remains unchanged.

Exactly one global KDA,16heads with64-dimensional keys/values,128-wide RGB patch
tokens and terminalCLS128→256→2 decoder.737,170parameters/27tensors,65,536state
floats/trial. All learned parameters train with Adam1e-4/no clipping. Direct fresh
whole-model constructor, empty initial Adam, zero counters and new native streams
/RNG were verified; disposable profile state and old weights are not inherited.

**Production checkpoint3 /96episodes** downloaded, hash-verified and CPU-reloaded.
All27parameters changed and all27named Adam states advanced. SHA256
`0e84876c27da4a4db2a7d5ffdcd6a8c738da54b8fa48e86f17d836a5ed5641ca`.
At20:36:15UTC, mirrored production reached12updates/384episodes. This is actual
training progress; validation and final accuracy are pending.

NEW8h/$5 cap: **2026-10-02 20:33:14UTC → 2026-10-03 04:33:14UTC**;
hardstop **October2 9:33:14PM PDT**, scientific cutoff600seconds earlier.
Single A40 quote$0.49/hour plus$1storage reserve; no cap renewal.
Authenticated independent pod guard and detached mirror84656/PPID1 are active.
Full artifact retrieval, both final CPU checkpoint reloads and verified stopped-
pod deletion are automatic. Bounded local wake assertion supports retrieval.

Previouspod5hnvb87npqpqb4 stopped at2,615updates/83,680episodes. All71cancelled
artifacts and terminal55Adam-state checkpoint were verified before stopping and
deleting the old pod. Its final tests were not completed; no final-performance
claim is made. The user cancellation superseded the prior complete-run queue
prerequisite. Local CNN-GRU continues; TwoFrameRViT remains implemented/untrained.

[Native production evidence](CloudRuntime/production_verified.json) ·
[CPU checkpoint verification](CloudRuntime/downloaded_checkpoint_verified.json) ·
[Measured allocation](CloudRuntime/artifacts/allocation.json) ·
[Pod and immutable budget](CloudRuntime/pod.json) ·
[Independent guard](CloudRuntime/guard_verified.json) ·
[Current state](CloudRuntime/queue_status.json).

[Architecture](README.md) · [Protocol](protocol.json) ·
[Research journal](../../LabJournal/krauzlis-sequence-kda16-heads.md) ·
[Cancelled predecessor](../SequenceKDA3/RUN_STATUS.md).

## Earlier checkpoint validation

CPU-only frozen checkpoint400 evaluation completed on100validation trials per
condition. B12/B20/B28 balanced accuracy is50% each; mean AUC0.509996. This
is an early validation snapshot, not final held-out performance or a change to
official selection. Training, optimizer and native streams continue unchanged.
[Snapshot](CloudRuntime/early_validation/latest.json).
