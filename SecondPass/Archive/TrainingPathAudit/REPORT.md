# Loss, decoder and temporal-gradient check

2026-10-03 UTC. CPU-only, two threads; saved model copies, no changes to live jobs, weights, optimizer states, streams or source snapshots. [Script](check.py), [machine-readable results](results.json). The final diagnostic execution took10.87seconds; the preceding source review and an initial measurement pass are separate.

**The objective and loss normalization checks passed. Both RViTs showed severe attenuation of gradients to early frames on the tested complete native trial.** This is a concrete temporal-credit-assignment concern, not evidence that the task is unsolvable or that every failed model has the same problem.

## Objective, batching and decoding

All three workers use two raw logits and ordinary `F.cross_entropy(logits, integer_labels)`. Label1 means target change; label0 means foil-only change or catch, consistent with the renderer and evaluator. There is no extra probability transform before CE, auxiliary objective or reaction-time target. Softmax and argmax are used for evaluation. Manual negative-log-softmax and the closed-form logit gradient `(softmax - one_hot)/batch_size` matched the implemented computation.

Seven comparisons exercised the actual production update functions with a small controlled model: replay batch sizes13/14/32 with micro1/4, and onlineKDA batch32/micro4. Loss, parameter gradients and one Adam update matched a full-batch calculation. Partial tails use their actual count; gradients are zeroed once per update and accumulated across complete movies. Training backpropagates from the final decision through the sequence; memory/chunk state is not detached.

RViT decodes final spatial memory using LayerNorm, per-token256→16/GELU, flatten169×16 and an FFN to2logits. KDA decodes terminalCLS through LayerNorm and an MLP. Every learned parameter tensor had a finite, nonzero gradient in the native diagnostic, consistent with earlier persisted optimizer/update evidence. KDA's patch-query gradients are zero by design for terminal-only readout; its terminalCLS query and every frame's keys/values receive gradients.

## Full native temporal gradient

One identical native B12 target-change trial,29frames, was run through each saved checkpoint. These are FP32 forward/backward computations onCPU. Gradient norms were measured in FP64 afterward to avoid underflow from squaring very small FP32 values. No model or optimizer updates were performed.

|Model/checkpoint|Largest cue-frame input gradient / final-frame input gradient|First memory-state gradient / final memory-state gradient|
|---|---:|---:|
|Cloud RViT replay1500|1.33×10⁻²³|3.04×10⁻²³|
|Motion RViT100|3.07×10⁻¹⁴|4.43×10⁻²⁰|
|16-head KDA4100|0.8795|Not the same state interface|

All30000RGB input entries in each cue frame had nonzero gradients, so the issue is magnitude rather than an accidental detach. The first pass's FP32 norm computation rounded the cloudRViT cue norm tozero; the corrected FP64 measurement confirms it is tiny but nonzero. All learned tensors still receive updates because the parameters are shared across time and get gradients from later frames.

The RViT update has a residual from **current visual tokens**: `Y = X + (visual_attention + memory_attention)/sqrt(2)`, followed by a residual FFN. It has no direct identity residual from previous memory. The old-memory route repeatedly passes through LayerNorm and learned cross-attention projections. The measured state gradients shrink steeply across time, consistent with a weak long-range gradient path. Adding fixed motion features does not repair that memory path; a cue may influence subsequent motion windows as well, so its raw-image gradient ratio is not identical to its state-gradient ratio.

## Interpretation

Earlier checks established graph connectivity and changes to all learned parameters. They did **not** rule out vanishing temporal gradients, and the short two-frame gradient check was insufficient for that question. FullBPTT retains the graph; it does not guarantee a useful gradient magnitude at the early cue.

This is a descriptive finding on one trial per saved model, with different training stages. It does not prove information erasure, uniquely explain generalization failure or establish thatKDA learns meaningful motion. A residual or gated previous-memory carry is now a motivated architectural proposal for RViT; none has been implemented or launched by this check. Current runs continue unchanged.

With57%positive labels, a constant predictor assigning probability0.57 tochange has expectedCE0.683315 and50%balanced accuracy. Losses around that value are consistent with learning the class prior. They are not evidence that objective derivatives are broken.
