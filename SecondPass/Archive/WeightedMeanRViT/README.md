# Weighted RGB mean and recurrent visual transformer

This queued candidate starts all7,269,426 learned parameters in92 tensors fresh.
At every timestep, the fixed causal filter computes
`0.5 X[t] + 0.4 X[t-1] + 0.1 X[t-2]` in native RGB pixel space. Missing startup history
repeats frame zero; the first filtered frame is exactly the first raw frame. The
arithmetic is shared with the weighted CNN–GRU implementation, without importing
any of its trained weights or recurrent modules.

One sophisticated shared CNN processes each resulting RGB100×100 image. Its
32-channel stem and two residual blocks retain100×100 resolution; learned
space-to-depth stages produce64×50×50,128×25×25 and256×13×13 fields, each with two
residual blocks. All convolutions use stride1, with GroupNorm/GELU. Replicate
padding converts25×25 to26×26 before the final compression. The full256-channel
13×13 field reaches attention: `flatten(2).transpose(1,2)` gives169 spatial tokens
of width256, followed by learned row/column positions and LayerNorm.

Exactly one original `RecurrentVisualBlock` is shared across timesteps. Independent
eight-head visual self-attention and previous-memory cross-attention use current
visual tokens as queries and separate softmaxes. Their outputs combine with1/√2
scaling in a visual residual, followed by the original pre-LayerNorm256→1024→256
feedforward residual. The169×256 memory starts at zero for every movie. There is
no added gate, GRU, CLS, VAE encoder or full-movie attention pass.

Only the terminal memory supplies the binary decision: LayerNorm, per-token256→16,
flatten2704, then512→256→2. FP32 full BPTT trains the entire model. Optional
nonreentrant activation checkpointing on the CNN and recurrent block reduces saved
activations without detaching temporal gradients. The data adapter remains the
single-stimulus/no-cue/no-distractor task with exact29/37/45-frame movies; labels and
rendering are unchanged. The separate fresh worker owns measured exposure and
queueing; this implementation does not launch training or renew a compute budget.

`WeightedMeanRViT(checkpoint_encoder=True,checkpoint_recurrent=True)` loads no
checkpoint. `forward(images,task='krauzlis_cued_motion')` returns terminal logits;
`encode(image)` returns256×13×13 fields and `encode_tokens(image)` returns169×256
tokens for an already filtered image. `stream_step(current,previous_frames,memory)`
accepts two prior raw RGB frames in oldest-to-newest order and returns logits,
next raw history and next memory, without an internal cache or detach.

`python -m SecondPass.WeightedMeanRViT.check` performs one short CPU engineering
check of exact filtering/startup/causality, field/token shapes, one shared RViT,
sequence/streaming/reset parity, finite nonzero gradients and a disposable update
for all92 learned tensors, and an earliest-frame gradient. This evidence establishes
the implementation, not learned response accuracy.
