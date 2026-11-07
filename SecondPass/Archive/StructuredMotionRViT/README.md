# Simoncelli–Heeger motion front end for Krauzlis

**Fresh local training is running, 2026-10-03 UTC.** This candidate provides structured motion features to a fresh RViT. Native images, visual cues, B12/B20/B28 conditions, 26/28-degree changes, labels and event proportions remain unchanged.

[Technical architecture and proposal (PDF)](TechnicalDocument/architecture_proposal.pdf) · [LaTeX source](TechnicalDocument/architecture_proposal.tex) · [Run status](RUN_STATUS.md).

The source is Tangemann, Kümmerer and Bethge, [*Object segmentation from common fate: Motion energy processing enables human-like zero-shot generalization to random dot stimuli* (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/f7d3cef7ff579f2f903c8f458e730cae-Paper-Conference.pdf), with the [authors' implementation](https://github.com/mtangemann/motion_energy_segmentation) pinned to commit `997ec55adf6062d8f92d4adcd555fdaccc5a1200`. Their reported task is random-dot motion segmentation, not our cued change decision. This is an adaptation of their motion front end, not a reproduction of their complete segmentation model or its accuracy.

## Architecture

1. Convert each RGB frame to mean grayscale for motion analysis. Retain ordered previous/current RGB separately for appearance and cue information.
2. Apply the published five-scale image pyramid and analytic Simoncelli–Heeger CNN. Nine-frame spatiotemporal derivative filters, V1 energy and divisive normalization, followed by MT pooling, rectification and normalization produce 19 channels per scale. Spatially resize these maps to 100×100 and concatenate: 95 motion channels.
3. Concatenate the 95 channels with the six ordered RGB channels. A fresh learned fusion CNN compresses the 101-channel field into 13×13×256 features.
4. Reuse the existing RViT architecture: learned spatial positions; current visual queries with separate current-visual and previous-memory attention streams; recurrent 169×256 memory. Reduce each final token 256→16, flatten and decode through the existing FFN to two logits.

There are **7,297,650 freshly initialized learned parameters in 92 tensors**. Published analytic constants are 2,430 buffer values, not inherited learned weights. No pretrained segmentation weights or predecessor project checkpoint is loaded. All learned parameters remain trainable, with full temporal gradients.

## Disclosed adaptations

- The original analytic factory uses H,W,T convolution coefficient layout while the author's CNN consumes T,H,W. Our wrapper permutes the three separable V1 coefficient tensors once; every coefficient and sign is preserved. The upstream source files are untouched.
- Each temporal window contains the current frame and eight previous frames. Startup repeats the first frame. This makes the nine-tap filter causal and introduces a four-frame delay relative to its centered alignment. We do not append synthetic report frames or expose future images.
- Spatial maps from the five scales are bilinearly resized and concatenated before the learned fusion CNN. The authors' segmentation head is replaced by our appearance, memory and binary readout pathway. No task timing, ROI, cue location or privileged label enters the motion computation.
- Only the required author classes are extracted into [author_cnn.py](author_cnn.py), avoiding their unrelated MMFlow training dependencies. Original source files and hashes are preserved in [upstream/SOURCE.json](upstream/SOURCE.json).
- On Apple, fixed analytic filters execute on CPU with two threads and all learned components execute on MPS. Native MPS lacks `avg_pool3d.out`; an equivalent spatial-pool substitution still produced material differences after normalized motion processing. We retain the exact CPU calculation, with differentiable CPU/MPS copies and unchanged full gradients through the learned model. [Mixed-backend check](verification/mps_compatibility.json).

## Interface and verification

```python
from SecondPass.StructuredMotionRViT.model import StructuredMotionRViT

model = StructuredMotionRViT(checkpoint_encoder=True)
logits = model(images)  # FP32 [batch,time,3,100,100] -> [batch,2]
```

The streaming API returns logits, current RGB, recurrent memory and eight-frame grayscale history; all state must reset between trials. Whole-movie and streaming motion features agree.

[Three focused CPU checks](verification/model_cpu.json) passed in 4.21 seconds with two threads: analytic coefficient/layout parity and causal streaming; opposite-direction gratings at 0.375 pixels/frame produce different motion populations; a complete native 29-frame trial yields finite logits and a short backward pass reaches every learned parameter and the earliest input frame. The grating check is direction sensitivity, not native task accuracy.

The user subsequently requested local training and cancellation of the local CNN-GRU. That run stopped and saved at 2,575 updates / 82,400 trials. This model starts its entire learned pathway, Adam, RNG, counters and native streams fresh. Disposable profile state is discarded.

The native profile pinned **660 updates / 20,000 presentations / 2,000 unique movies**: two pools of 1,000 native movies, ten shuffled epochs each, then regeneration. Batch32/micro1, Adam1e-4/no clipping, FP32/full BPTT; tail batches use their actual count. Validation100/250/500/660 at100/cell, selected/terminal fresh final tests200/cell. No changed teaching or auxiliary supervision.

Production checkpoint3/96presentations was independently CPU reloaded: all92learned parameter tensors changed and all92Adamstates advanced, analytic buffers unchanged. [Evidence](LocalRuntime/production_verified.json). The new finite local8h cap conservatively begins with the first compatibility attempt at2026-10-03T01:15:49.255076Z and ends09:15:49UTC /October3 2:15:49AM PDT; scientificcutoff tenminutes earlier. Independentlaunchd guard and supervisor own the job. No cap renewal. Cloud runs retain their own allocations; task acquisition is not yet demonstrated.
