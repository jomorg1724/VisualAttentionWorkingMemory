# Weighted-mean CNN–GRU: fresh local training

**CANCELLED BY USER — replaced with three-frame VAE.** Terminal570updates/17,288presentations CPUreloaded, all70Adam states at570; source, optimizer/RNG/stream progress and prior validation preserved. No finaltest and no restart. [Saved checkpoint proof](LocalRuntime/cancellation_verified.json). Earlier live notes below are historical.

**LOCAL TRAINING — 2026-10-03 14:44 UTC.**
User authorized fresh local training. Native MPS profile pinned all 2,310 updates /
70,000 presentations /7,000 unique movies before production. Actual production
checkpoint3/96 was CPU reloaded by the parent: all70 Adam states advanced, all70
learned tensors changed, direct fresh construction/empty initial Adam and saved
MPS RNG verified. Live snapshot35/1064 presentations, phase
`training`; early training only, validation pending.

Effective batch32/micro4/eval4, FP32/full BPTT, Adam1e-4 without clipping, two CPU
threads. Same fixed .5/.4/.1 weighted pixels → shared CNN → flatten → standard
GRU256, no cue or distractor. Generate1,000 trials, ten shuffled epochs, repeat.
Three warmup plus three steady native profiles passed with full-length backward.
Measured native update costs about5–8 seconds, first production/profile timing
ratio1.0303. Profile optimizer/weights/RNG/streams were discarded before production.

Fresh eight-hour LOCAL cap: October3,7:40:04AM→3:40:04PM Pacific; finalization
begins by3:30:04PM. Independent launchd supervisor89395, worker90280, guard89390.
Runtime `/Users/jonathanmorgan/VAWMRuntime/weighted_mean_conv_gru_local01/run`.
Previous local structured-motion run is complete/preserved. All cloud pods stay
closed and the old cloud automatic queue stays disabled. No paid compute.

[Launch](LocalRuntime/launch.json) · [Persisted production evidence](LocalRuntime/production_verified.json)
· [Local adapter](local_worker.py). Previous queue/cancellation notes below are historical.

**HELD FOR TOMORROW — user cancelled cloud jobs.** Automatic queue4518 and keep-awake4519 stopped; no rental or budget was created. Architecture, code, protocol and checks are preserved. Do not launch or restart today. Earlier automatic-queue notes below are historical.

Observed 2026-10-03 04:13 UTC. Parent queue PID 4518 and bounded keep-awake PID 4519
are running independently with PPID 1. State: `waiting_predecessor_cleanup`.
This candidate has **not started training**: no pod, budget, profile or production
optimizer evidence exists. Waiting consumes none of the next cloud allowance.

The queue waits for single-stimulus RViT pod `jxbmb44y9wamhl` to complete,
retrieve/verify its selected and terminal checkpoints, and be deleted. It then
creates one fresh A40 automatically, profiles, pins exposure and starts production.
The random-frame cloud pod and local structured-motion job retain their own caps.
No existing job was stopped or changed.

Architecture: fixed causal `0.5 X[t] + 0.4 X[t-1] + 0.1 X[t-2]` RGB mean,
repeating the first frame for missing history → shared residual CNN → 16×13×13
features → flatten 2,704 → one standard GRU with 256 hidden units → 256→128→2
terminal classifier. All convolutions have stride one; space-to-depth compresses
the spatial map. Model: 6,999,474 trainable parameters / 70 tensors.

Task is identical to the last diagnostic: one original-center stimulus, no cue or
second stimulus, native 29/37/45-frame lengths, original dot dynamics and labels.
Fresh model/Adam/RNG/streams, FP32 full BPTT, batch 32/micro 4,
Adam 1e-4 without clipping. Generate 1,000 trials, run ten shuffled epochs, repeat.
Target 2,310 updates / 70,000 presentations / 7,000 unique movies, prospectively
reduced to complete replay pools only if profiling requires it. New eight-hour/$5
cap starts at pod creation and includes setup/evaluation/retrieval; no extension.

Focused CPU check passed in 1.45 seconds: weighted arithmetic/startup, causality,
direct flatten-to-GRU shape, finite gradients and changes in all 70 learned tensors,
and a gradient to the first input frame. The fixture's one Adam step is not a
production launch. The actual fresh constructor and empty-Adam checkpoint readback
also passed. No extra GPU checks were run.

[Architecture](README.md) · [Protocol](protocol.json) · [CPU evidence](check_results.json)
· [Live queue](CloudRuntime/queue_status.json) · [Source bundle](CloudRuntime/package/bundle_receipt.json).
