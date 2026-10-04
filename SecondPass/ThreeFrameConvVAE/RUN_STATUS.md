# Three-frame convolutional VAE: fresh local training

**2026-10-03 19:34 UTC — VAE COMPLETED9900updates; only best9500/latest9900 retained.**
300,000triplet presentations /30,000unique movies, same original allocation/cap. Fresh paired final tests complete (384independentmovies /1152triplets per model). Last100total loss0.00063257/recon0.00060685. Selected held-out balancedrecon0.00064548 versuscopy-middle0.00342465; temporal-differenceMSE0.00005794 versus0.00008455 (31.5%lower). Full-imageMSE0.00005657 approximatelycopy-middle0.00005637. No response model/accuracy tested. ParentCPUverified both full checkpoints and126Adam steps; guard exited normally. [Final report](FINAL_REPORT.md) | [Completion proof](LocalRuntime/completion_verified.json). No continuation or new run launched.

**October 3, 11:59 AM PDT — VAE RESUMED and optimizer progress verified.**
Restored complete update5600 model/Adam/RNG/streams/scheduler. Latest independently CPU verified saved update5700 /172,768triplet presentations, all126Adam states at the saved step; live update5784. Only `resume01/latest.pt` and `resume01/best.pt` exist. Original5600 filename promoted/removed after verification; preserved checkpoint content/state. Startupval5600 balancedrecon0.001003128 initializes available best; historical5500 weights unavailable. Same9900target and original4:49:10PM PDT hardstop, no renewal; cloudclosed. Resume worker15761/guard15751. [Recovery evidence](LocalRuntime/resume_verified.json).

**User-authorized cleanup completed:** removed1,518 old checkpoint/model files, reclaiming57.9GB (53.93GiB); free space now109GiB (~117GB). Only verified VAE `checkpoint_005600.pt` remains with full model/Adam/RNG/streams/scheduler. Original initial and best5500 checkpoints were intentionally deleted; any recovery adapter must support latest-only storage before launch. Code, logs and reports preserved. [Cleanup manifest](LocalRuntime/checkpoint_cleanup.json).

Current status, October 3: **stopped after disk-full checkpoint failure at
5,700 updates (9:31 AM PDT)**. Latest verified complete checkpoint is 5,600;
best validation checkpoint is 5,500. No complete final evaluation. Recovery
adapter is prepared but has not launched; user requested holding the restart
while investigating storage. Original deadline and target remain unchanged.

Storage audit: project plus VAWMRuntime contain 1,522 checkpoint/model files,
54.03 GiB logical size (54.04 GiB allocated). VAE artifacts account for 8.18 GiB,
delayed-frame GRU 8.73 GiB, joint training 8.68 GiB, and final ConvGRU 8.45 GiB.
Checkpoint files with October 2–3 modification dates total 33.20 GiB. Data
volume currently has 55 GiB free. macOS logs confirm actual system-wide disk
exhaustion around the failure; the exact temporary-space consumer is unknown.
No files deleted. Historical startup observations below are not current status.

Observed 2026-10-03 15:57 UTC. Production checkpoint 3 / 96 triplets independently CPU reloaded:
all 126 named Adam states advanced and all 126 learned tensors changed. Direct
fresh constructor, empty initial Adam and saved MPS/window RNG verified.
Live snapshot 1063 updates / 32224 triplet presentations.

Three ordered native RGB100x100 frames concatenate as nine channels, then a
residual CNN produces diagonal-Gaussian mu/logvar at 256x13x13. The latent-only
residual decoder reconstructs all three frames, without encoder skip connections.
9,507,913 fresh trainable parameters. Deterministic `encode(triplets)` means are
available for future response-model integration; this run has no response head.

Objective: equal foreground/background reconstruction means + 1e-4 mean Gaussian
KL. Contrast support uses target pixels, not labels or location/phase metadata.
This weighted reconstruction is an explicit adaptation of the
[VAE formulation](https://arxiv.org/abs/1312.6114), not the standard pixel ELBO.
Full-image MSE, foreground/background error and temporal-difference error are
reported separately, with gray and copy-middle baselines.

Same no-cue/single-stimulus movies, original locations/dynamics/29,37,45 lengths.
Sample ordered consecutive windows inside moving-dot frames 7..T-2; no crops or
pixel modifications. Generate 1,000 movies, ten shuffled epochs, then replace.
A fresh window is drawn on each movie presentation. Response labels are unused.

One Apple MPS worker, FP32, Adam1e-4/no clipping, batch32/micro4, two CPU threads.
Disposable 3 warmup + 3 steady profile pins 30 complete replay pools:
**9,900 updates / 300,000 triplet presentations / 30,000 unique movies**.
Production model/Adam/RNG/streams start fresh after profiling.
Validation 100/250/every500/terminal: 64 independent movies/cell, 3 windows/movie.
Fresh paired selected/terminal finals: 128 movies/cell, 3 windows/movie.
Select by mean balanced reconstruction; retain side-by-side reconstruction PNGs.
Windows from the same movie are grouped, not counted as independent trials.

First validation100: balanced recon 0.009875; gray 0.029108;
copy-middle 0.003412. It still trails copy-middle; early images
show blur/color artifacts, not yet faithful dot-motion reconstruction.
Full-image MSE 0.004273, KL 0.323425, temporal-difference error
0.001664. Early validation, not final results or evidence
that the downstream response task is solved.

New eight-hour LOCAL cap starts October3,8:49:10AM PDT; hardstop4:49:10PM PDT,
science cutoff4:39:10PM. Includes profile/evaluation/reporting, no extension.
Runtime `/Users/jonathanmorgan/VAWMRuntime/three_frame_conv_vae_local01/run`.
Independent launchd supervisor21106 / worker21185 / guard21104; local GPU lock.
Old CNN-GRU stopped/saved570updates/17,288presentations/all70Adamstates.
Cloud pods/queues stay closed. No inherited model or profile state.

[Architecture](README.md) | [Protocol](protocol.json) |
[Production evidence](LocalRuntime/production_verified.json) | [Launch](LocalRuntime/launch.json).
