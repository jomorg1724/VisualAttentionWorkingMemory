# Weighted three-frame mean feeding a CNN and standard GRU

## Question and decision

User requests the next queued experiment: can a causal decayed pixel mean provide
an easier motion representation for a conventional CNN–GRU? Run on the same
single-stimulus/no-cue/no-distractor task as the preceding cloud RViT. Fresh held-out
B12/B20/B28 accuracy and AUC will determine whether this candidate merits further
training. This is one exploratory seed; lower replay training loss alone is not
successful generalization.

## Implementation

At every timestep form `M[t] = 0.5 X[t] + 0.4 X[t-1] + 0.1 X[t-2]`.
The mean remains RGB 100×100. Missing history repeats X[0], so M[0]=X[0] and
M[1]=0.5 X[1]+0.5 X[0]. The weights are fixed and causal. A shared eight-block
residual CNN uses stride-one convolutions and three space-to-depth stages,
32→64→128→256 channels at 100→50→25→13 pixels. Replicated edge padding permits
the final 25→26→13 compression. A learned 1×1 convolution reduces 256 channels
to 16; flatten all 169 spatial positions directly into standard PyTorch GRU(2704,
256), one layer. The final hidden state enters a 256→128→2 FFN and ordinary CE.
There are 6,999,474 learned parameters / 70 tensors. No pretrained weights.

Temporal blending gives the CNN an asymmetric recent trail while the GRU integrates
across the entire movie. Blending may also obscure fine dot displacements; benefit
is a hypothesis to test. Encoder recomputation bounds activation memory without
truncating gradients. Every frame and every learned parameter participates in full
sequence BPTT; one final decision/loss per movie.

## Teaching, exposure and evaluation

Reuse the exact SingleStimulusStream adapter: one stimulus at (20,50) or (80,50),
no cue or foil, all 29/37/45 frames and original dot trajectories retained.
Change/no-change labels remain 57%/43%. No new objective, cue, privileged feature,
phase gating or curriculum. All model/Adam/RNG/stream state starts fresh.

Effective batch32/micro4, FP32/TF32 disabled, Adam1e-4/no clipping; 1,000 new movies
per pool reused ten shuffled epochs, then replaced. Target 2,310 updates / 70,000
presentations / 7,000 unique movies, matching the preceding diagnostic. All partial
batches are included with their actual-size loss normalization. CUDA profile pins
feasible complete 330-update pools before production. Validation100/cell at100,
250, every500 and terminal; selection mean AUC then BA. Fresh paired selected and
terminal finals200/cell, exactly114 change /86 no-change /0 foil.

## Actual queue evidence — 2026-10-03 04:13 UTC

Parent one-shot queue4518/PPID1 is waiting for exact predecessor
`jxbmb44y9wamhl` final retrieval and verified deletion; keep-awake4519/PPID1 is
bounded independently. It will provision a single A40 and perform automatic
prepare/profile/pin/production. Same-as-last-trial new eight-hour/$5 envelope starts
at next pod creation; waiting consumes no allowance. No current job was altered.
No new rental, profile, production optimizer or result exists yet.

One short CPU check passed in1.45 seconds: arithmetic/startup/causality, full short
sequence backward, all70 finite gradients/updates and an early-frame gradient.
Fresh-constructor/empty-Adam checkpoint readback passed. Source-only bundle has33
files and no checkpoint; existing guarded cloud transport was reused. Future
training status requires persisted production Adam evidence, not this queue note.

[Source/design](../SecondPass/WeightedMeanConvGRU/README.md) ·
[Run status](../SecondPass/WeightedMeanConvGRU/RUN_STATUS.md) ·
[Queue evidence](../SecondPass/WeightedMeanConvGRU/CloudRuntime/queue_status.json).

## Fresh local training — 2026-10-03 14:44 UTC

User now authorizes local training. Earlier cloud hold is superseded only for this fresh local attempt; paid pods remain closed. Adapted MPS checkpoint/RNG/synchronization and bounded local supervisor from the existing working local harness. One CPU adapter check passed in1.29s; native MPS profiling passed in the actual bounded attempt, with no separate accelerator testing campaign. All40 copied sources are frozen in a source-only runtime with no checkpoint inputs.

Profile3warmup+3steady nativefull32 updates and20val/cell pinsFULL2,310updates/70,000presentations/7,000unique, same1000×10replay. Firstprofile starts14:40:04UTC/7:40:04AM PDT, hard22:40:04UTC/3:40:04PM PDT; no renewal. Production is a separate fresh constructor/optimizer/RNG/streams. Checkpoint3/96 parent CPU reloaded with all70 Adam states/tensors advanced and saved MPS RNG; live snapshot17/544, mean losssofar0.68722, no validation yet. Independentlaunchd guard89390/supervisor89395/worker90280. Batch32micro4eval4/FP32fullBPTT/Adam1e-4noclip/twoCPUthreads. [Evidence](../SecondPass/WeightedMeanConvGRU/LocalRuntime/production_verified.json).

## Cancelled for VAE replacement

User cancelled local response learning. Saved terminal570updates/17,288presentations,
all70Adamstates at570 CPUverified; no finaltest. Model/optimizer/RNG/streams and
validation500 preserved. Fresh VAE inherits no weights.
[Receipt](../SecondPass/WeightedMeanConvGRU/LocalRuntime/cancellation_verified.json).
