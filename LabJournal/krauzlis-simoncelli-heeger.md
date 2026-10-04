# Structured motion features before the Krauzlis decision

2026-10-03 UTC / October 2 PDT. **Fresh local training is running.**

[Technical architecture and proposal PDF](../SecondPass/StructuredMotionRViT/TechnicalDocument/architecture_proposal.pdf), with equations, diagram, parameter accounting and measured training settings.

## Authorized local training after implementation

The user requested starting this model locally first, cancelling the local CNN-GRU, and then writing the technical proposal. CNN-GRU stopped normally and saved at2575updates/82400freshmovies; no trained state was transferred. Its worker/supervisor exited and its launchd jobs were removed. Cloud runs continue under their existing caps.

Apple execution uses the unchanged fixed analytic frontend on CPU/two threads and all learned components on MPS. MPS lacks AvgPool3d; a pool-only substitution allowed execution but had material differences after normalization (random-movie maximum absolute difference0.164). The chosen CPU frontend avoids that discrepancy; a complete native29-frame MPS backward gave finite gradients for all92learned tensors and bitwise-equalCPU motion features. CPU/MPS copies remain differentiable.

Disposable profiling was discarded. Measured allocation prospectively pinned660updates /20000presentations /2000unique movies: two1000-trial pools each reused10epochs, then regenerate. Effectivebatch32/micro1, correct partialtails, Adam1e-4/no clipping, FP32/full BPTT, finalbinaryCE only. Validation100/250/500/660 at100/cell, finalselected/terminal pairedfresh200/cell. Freshwholelearnedmodel/Adam/RNG/streams/counters. Conservative newlocal8h cap starts firstcompatibilityattempt01:15:49UTC, hard09:15:49UTC /October3 2:15:49AM PDT; scientificcutoff tenminutes earlier. Guard20403/owner20405/worker22149 operate independently.

Productioncheckpoint3/96presentations independently CPUreloaded: all92learned tensors changed, all92Adamstates advanced, analyticbuffers unchanged, initialconstructor/emptyAdam/freshstreams verified. [Receipt](../SecondPass/StructuredMotionRViT/LocalRuntime/production_verified.json). This is an actual launch record; no task-acquisition result is claimed. The document was written after persisted production began.

The implementation-only discussion below records the earlier build stage; its no-training statements are historical and superseded by this explicit user request.

## Question and decision

The previous fixed-scene overfit succeeded, so we already have evidence that a neural model can represent and fit these decisions. The live fresh-data runs have not yet demonstrated acquisition. The working hypothesis for this candidate is that explicit spatiotemporal motion features may make reusable evidence easier to learn from a delayed binary report. This is a hypothesis, not a diagnosis of all previous failures.

The user specifically requested the Simoncelli–Heeger motion-energy CNN after asking whether a structured detector could help. We therefore used the actual author implementation underlying Tangemann, Kümmerer and Bethge's [NeurIPS 2024 random-dot segmentation paper](https://papers.nips.cc/paper_files/paper/2024/file/f7d3cef7ff579f2f903c8f458e730cae-Paper-Conference.pdf), rather than implementing a different hand-designed filter bank. The published generalization result concerns common-fate segmentation; it does not establish performance on our cued target/foil change task.

## Implementation and lineage

[StructuredMotionRViT](../SecondPass/StructuredMotionRViT/README.md) runs the paper's fixed analytic V1/MT CNN at five spatial scales over causal nine-frame grayscale windows. It supplies 95 motion channels alongside six ordered previous/current RGB channels to a fresh learned CNN, producing 13×13×256 tokens. Existing RViT visual/self and previous-memory attention, spatial state and 256→16 token reduction/flatten/FFN decode the final binary response.

All 7,297,650 learned parameters are fresh and trainable. The analytic mathematical coefficients are buffers, not frozen previously learned task weights. There is no segmentation checkpoint or inherited project checkpoint. Native stimuli, cues, B12/B20/B28 conditions, change magnitudes, labels and teaching are unchanged.

The source is pinned at author commit `997ec55adf6062d8f92d4adcd555fdaccc5a1200`; upstream source and hashes are preserved. We corrected the analytic factory's H,W,T coefficient layout to the CNN's T,H,W layout without changing coefficient values. Causal alignment shifts the centered filter response by four frames, repeating only the first frame at startup. Five-scale outputs are spatially resized/concatenated and fed to our task-specific decoder instead of the authors' segmentation head. These adaptations are documented rather than described as exact end-to-end reproduction.

## Evidence and next decision

[Three CPU checks](../SecondPass/StructuredMotionRViT/verification/model_cpu.json) passed in 4.21 seconds on two threads. They cover coefficient/layout parity, causal streaming and future-frame independence, distinct opposite-direction grating populations at native 0.375 pixels/frame, a finite complete native 29-frame forward, and finite nonzero gradients for all 92 learned parameter tensors plus the earliest input frame. These are implementation checks, not trained accuracy or evidence of task acquisition.

No new accelerator worker, profile, optimizer update, pod or training queue was launched. Existing RViT replay, cloud KDA and local CNN-GRU training remain unchanged with their original finite caps. On a subsequent training request, prepare the adapter and measure feasible exposure under an explicit finite allocation; preserve all previous artifacts and initialize the whole learned model fresh.

## Completed local run

All660updates/20,000presentations/2,000unique completed06:13UTC. Selected250freshmeanBA50.37%, terminal660meanBA50.67%, 200/cell. Originalruntime/artifacts preserved, no continuation. [Final report](../SecondPass/StructuredMotionRViT/FINAL_REPORT.md).
