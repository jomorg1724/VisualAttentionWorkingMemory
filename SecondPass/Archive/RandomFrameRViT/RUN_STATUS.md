# Training — random-frame-gradient gated motion RViT

**COMPLETED — 2026-10-03 04:43 UTC.** All 3,960 updates /120,000 presentations /12,000 unique. Fresh final BA50% all conditions; selected/terminal mean AUC0.48828/0.47909; last100loss0.68455. Verified final retrieval/checkpoint readback and deleted exactpod5us0rp5jwmg5bu. [Final report](FINAL_REPORT.md). Earlier live notes below are historical.

Fresh training is running on A40 pod `5us0rp5jwmg5bu` at $0.49/hour.
The production checkpoint at update 3 / 96 presentations was downloaded,
hash verified and CPU reloaded. All 94 Adam states advanced; every learned
parameter tensor changed from the independently reconstructed fresh initialization.
All fixed analytic buffers remain unchanged. No profile or predecessor state was inherited.

Exposure pinned before production: **3960 updates / 120000
movie presentations / 12000 unique movies**. Each pool has 1,000 movies
reused for ten shuffled epochs. Effective batch 32, microbatch 4; Adam 1e-4,
FP32, no clipping or TF32. Complete native sequences and terminal labels unchanged.

Independently choose one frame per trial on every presentation. Only its learned
encoder and recurrence contribute temporal parameter gradients, scaled by movie
length to estimate the full gradient without bias. Later memory Jacobians remain
differentiable with detached weights; the terminal decoder trains normally.
The architecture includes a learned direct memory carry, initialized at 0.98.

CUDA profiling measured about two seconds per full update, depending on native
sequence length. This is an absolute throughput measurement, not a matched
comparison of training methods. First snapshot: update 25 / 800 presentations,
loss 0.70018; no validation or task-acquisition claim yet.

New eight-hour/$5 creation cap: 2026-10-03T02:39:56.820999+00:00 to
2026-10-03T10:39:56.820999+00:00, **October 3, 3:39:56 AM Pacific**.
Science deadline 2026-10-03T10:29:56.820999+00:00; last 600 seconds reserved for retrieval.
Independent authenticated guard and off-pod mirror are active. No automatic extension.

The prior KDA16 completed all 4,216 updates, its final artifacts were verified,
and its pod was deleted before this launch. The other cloud RViT and local
structured-motion model retain their original caps and continue.

[Architecture and policy](README.md) · [Focused gradient proof](check_results.json)
· [Production evidence](CloudRuntime/production_verified.json)
· [CPU checkpoint reload](CloudRuntime/downloaded_checkpoint_verified.json)
· [Live mirrored artifacts](CloudRuntime/artifacts/)
