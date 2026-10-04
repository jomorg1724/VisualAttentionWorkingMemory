# RViT input amplitude experiment

## October 3 — Local VAE–RViT input ×10 experiment

User requested stopping the local classifier and repeating training with inputs multiplied by ten. Original local run stopped cleanly at 227 updates / 6,928 trial presentations; best100 and latest227 remain preserved. Cloud weighted-mean CNN–RViT continues independently.

`SecondPass/VAERViTInput10` scales the 169×256 encoded tokens by exactly10 **after** spatial positions and token LayerNorm, immediately before the original recurrent block. Internal query/memory/FFN normalization remains unchanged. Consequently, the current-token residual is scaled while normalized attention queries are mostly scale invariant; this is an input-amplitude experiment, not an attention-temperature change.

Same VAE9900 encoder+mu transfer; fresh recurrent/decoder/positions, Adam, RNG and zero classification counters. All94 learned tensors train. Same seeds and train/validation/test namespaces as the original classifier for matched examples. Same no-cue/single-stimulus movies, final-trial crossentropy, full-sequence gradients, FP32/MPS, Adam1e-4/no clipping, batch32/micro1, twoCPUthreads, 1000 movies reused for10 shuffled epochs. Requested990 updates /30,000 presentations /3,000 unique movies, prospectively limited to feasible complete330-update pools by native profiling. A new finite8h local cap covers profile/training/evaluation/reporting; only best/latest checkpoints.

Focused CPU check passed: exact10×token equality, all94 finite parameter gradients and a nonzero earliest-frame gradient. New immutable runtime `VAWMRuntime/vae_rvit_input10_local01`; launcher/independent guard active. Native profiling is underway; no persisted production progress yet.


Native profile completed; production worker9023/supervisor5300. Before production, measured throughput pinned **330 updates /10,000 trial presentations /1,000 unique movies** (one full 1000-movie×10-epoch block), reduced from requested990 to fit the new8h cap. Validation100/250/330, final200trials per condition. HardstopOctober3,10:54:06PM PDT, science cutoff10:44:06PM. Same original seeds for matched baseline inputs. Production initialization active; saved optimizer proof pending.

**Production training independently verified:** saved update1 /32 trial presentations; all94 Adam states advanced and all94 learned parameter tensors changed. CPU checkpoint reload verified, input scale10 in checkpoint provenance. Initial production loss0.69538. This is training progress, not evidence of held-out acquisition.

October3,23:14UTC live snapshot: epoch2 weighted-cloud202/2310 updates /6128presentations, pool_index3/epoch0 (fourth fresh1000-trial set), last100CE0.685434. Validation100: BA50% allcells, meanAUC0.508908 (B12 .516116/B20 .480620/B28 .529988), all-change decisions. Local input×10:200/330 /6064presentations, pool0/epoch6, last100CE0.686339; validation100 BA50% allcells, meanAUC0.476404 (B12 .416157/B20 .555692/B28 .457364), all-change decisions. Both healthy; next validation250. Preliminary validation shows no acquisition yet.

**Completed October3,5:31PM PDT:** planned330updates /10000trial presentations /1000unique movies. Both retainedbest330/latest330 CPUverified withall94Adam states, inputscale10provenance; finalpairedtest200/cell complete, guardnormalexit. Selected/terminalsame model: BA50% allcells, meanAUC0.489996 (B120.467029/B200.506834/B280.496124), all-change predictions. Last100trainCE0.683109. No useful held-out change detection in this allocation. No localtrainingactive/newrunstarted.
