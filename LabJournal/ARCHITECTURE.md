# Current architecture and historical references

[Index](README.md) · [Technical report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md)

## Current successful motion encoder

Input is `[B, 3 time, 3 RGB, 100, 100]`, reshaped in frame order to `[B, 9, 100, 100]`. A shared residual CNN forms fields with 32/64/128/256 channels at 100/50/25/13 spatial resolutions. GroupNorm/GELU and residual convolutions learn motion-sensitive features; space-to-depth stages reduce resolution. Three frames contain two temporal transitions with a constant direction.

The final `[B,256,13,13]` field is spatially averaged to `[B,256]`, then projected through `256 → 128 → 128`, with GELU between dense layers and L2 normalization at output. The shared CNN and projection have 4,738,528 trainable parameters. The whole model started fresh; no VAE/predictor weights were inherited. There is no recurrence or frame-growing cache.

For two clips, the graded contrastive objective targets a Euclidean representation distance proportional to the shortest angular difference. Ground-truth direction enters the loss only. At test time, the two encoded clips are compared by distance and one validation-selected scalar threshold. The [report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md) gives exact equations, smoothing, force interpretation and limits.

## Previous representations and decision models

| Model family | Mechanism | Evidence |
|---|---|---|
| Three-frame spatial VAE | Spatial Gaussian bottleneck and reconstruction of all three frames | [VAE journal](krauzlis-three-frame-vae.md) |
| Variational predictive model | One flattened 512-dimensional Gaussian bottleneck and fourth-frame decoder | [Predictive journal](variational-motion-prediction.md) |
| Frozen predictor + FFN | Two 512-dimensional means concatenated to a trained 256/64/2 head | [FFN journal](predictive-motion-change.md) |
| Whole-sequence KDA | Ordered patch sequence and terminal CLS response | [KDA journal](krauzlis-sequence-kda-design.md) |
| CNN–RViT / CNN–GRU | Learned temporal representation followed by recurrent decision state | [RViT](krauzlis-two-frame-rvit.md), [delayed GRU](krauzlis-delayed-frame-gru.md) |
| Motion-energy RViT | Analytic motion-energy filters plus learned recurrent decisions | [Motion-energy journal](krauzlis-simoncelli-heeger.md) |
| Spatial KDA / ConvGRU / E/I and attention | Earlier selective integration, memory fields and learned visual/memory interactions | [Catalog](EXPERIMENT_CATALOG.md) |

The [archived pre-update sensory/E/I/attention description](archive/ARCHITECTURE_through_20261004.md) documents an earlier explicitly transferred lineage. It is not the architecture of the new contrastive CNN. The [research history](RESEARCH_HISTORY.md) explains why these lineages must be distinguished.
