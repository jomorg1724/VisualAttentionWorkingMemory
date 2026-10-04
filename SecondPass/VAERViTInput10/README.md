# VAE encoder → input×10 → RViT

## October 3 — Local VAE–RViT input ×10 experiment

User requested stopping the local classifier and repeating training with inputs multiplied by ten. Original local run stopped cleanly at 227 updates / 6,928 trial presentations; best100 and latest227 remain preserved. Cloud weighted-mean CNN–RViT continues independently.

`SecondPass/VAERViTInput10` scales the 169×256 encoded tokens by exactly10 **after** spatial positions and token LayerNorm, immediately before the original recurrent block. Internal query/memory/FFN normalization remains unchanged. Consequently, the current-token residual is scaled while normalized attention queries are mostly scale invariant; this is an input-amplitude experiment, not an attention-temperature change.

Same VAE9900 encoder+mu transfer; fresh recurrent/decoder/positions, Adam, RNG and zero classification counters. All94 learned tensors train. Same seeds and train/validation/test namespaces as the original classifier for matched examples. Same no-cue/single-stimulus movies, final-trial crossentropy, full-sequence gradients, FP32/MPS, Adam1e-4/no clipping, batch32/micro1, twoCPUthreads, 1000 movies reused for10 shuffled epochs. Requested990 updates /30,000 presentations /3,000 unique movies, prospectively limited to feasible complete330-update pools by native profiling. A new finite8h local cap covers profile/training/evaluation/reporting; only best/latest checkpoints.

Focused CPU check passed: exact10×token equality, all94 finite parameter gradients and a nonzero earliest-frame gradient. New immutable runtime `VAWMRuntime/vae_rvit_input10_local01`; launcher/independent guard active. Native profiling is underway; no persisted production progress yet.

