# VAE spatial means and recurrent visual transformer

This candidate uses the explicitly authorized trained VAE9900 encoding and a new
response architecture. It copies only the VAE's residual encoder and posterior-mean
head:4,754,912 parameters in62 tensors. The spatial positions, recurrent block and
readout are fresh:2,582,034 parameters in32 tensors. All7,336,946 learned parameters
in94 tensors train together. The VAE decoder, variance head, stochastic samples,
optimizer, RNG and stream state are excluded.

The source checkpoint is
`/Users/jonathanmorgan/VAWMRuntime/three_frame_conv_vae_local01/resume01/latest.pt`,
SHA256`bf92d8c908eb5f1e10188d4ed2b220e352a0b7b1d9435fc11cef0c5c754b1690`.
`load_pretrained_encoder(path,expected_sha256=...)` checks its digest and strictly
copies only `encoder.*` and `mu_head.*`. The constructor itself loads no checkpoint.
Its returned receipt records the copied tensor names/counts and source step.

For each timestep, the encoder receives the ordered raw triplet
`[X[t-2],X[t-1],X[t]]`; missing startup history repeats frame zero. Its deterministic
mean has shape `[B,256,13,13]`. The exact operation `flatten(2).transpose(1,2)` produces
169 spatial tokens of width256. Learned row/column positions and LayerNorm prepare
them for one shared recurrent transformer block, reused at every timestep. There
is no temporal averaging or full-movie transformer.

The original `RecurrentVisualBlock` provides separate eight-head visual self-attention
and previous-memory cross-attention, with independent projections and softmaxes.
Both use current visual tokens as queries; prior memory retains its spatial layout.
The attention outputs are summed with1/√2 scaling before the visual residual, then
a pre-LayerNorm256→1024→256 feedforward residual updates the169×256 memory. Memory
starts at zero for every movie; there is no added gate, CLS or decoder shortcut.
Only terminal memory is read through LayerNorm, per-token256→16 reduction, spatial
flatten2704, and the512→256→2 classifier.

Training uses FP32 and full temporal gradients through every encoder call, memory
update and readout. Nonreentrant activation checkpointing on the encoder and
recurrent block reduces saved activations without detaching history. The model
receives the existing single-stimulus/no-cue/no-distractor movies at their unchanged
29/37/45-frame lengths; the worker preserves their response labels and reporting.
VAE reconstruction losses are not used in this response model. The parent owns the
fresh bounded local launch and its measured exposure.

`forward(images,task='krauzlis_cued_motion')` returns terminal binary logits.
`encoder_mu(triplets)` and `encode(triplets)` return the trained deterministic spatial
means; `encode_tokens(triplets)` exposes their169×256 token representation.
`stream_step(current,previous_frames=None,memory=None)` returns logits, the next two
raw frames and next memory, with no internal cache or detach. Its history tensor
has shape `[B,2,3,100,100]` in oldest-to-newest order.

`python -m SecondPass.VAERViT.check` performs one short CPU engineering check. It
verified exact equality to the trained VAE mean, the ordered triplet/token layout,
fresh nonencoder weights, sequence/streaming/reset parity, finite nonzero gradients
and a disposable Adam update for all94 tensors, and an earliest-frame gradient.
This is implementation evidence; response learning requires persisted training
and held-out results.
