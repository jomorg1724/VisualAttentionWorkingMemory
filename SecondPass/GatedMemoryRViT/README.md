# Learned direct memory carry

The existing RViT replaces its memory through normalized attention/projection on
each frame. A native-movie diagnostic found severe attenuation of early gradients.
This variant retains its CNN, attention proposal and decoder, and changes only
the memory update:

`H_t = a_t * H_(t-1) + (1-a_t) * proposal(X_t, H_(t-1))`.

One shared linear gate observes normalized current and previous tokens and
outputs 256 sigmoid carry values per spatial token. Its weights start at zero;
biases initialize carry at 0.98. The direct old-memory path has no normalization
or projection. The initial 2% write rate trades faster writing for longer retention;
the gate learns a content-dependent rate. There are no phase/cue oracles.

Fresh constructors are provided for the two-frame and structured-motion RViTs.
They add 131,328 parameters: the motion version has 7,428,978 parameters across
94 learned tensors. No trained weights are loaded.

The [paired native gradient check](gradient_check.json) used identical fresh
shared weights, one complete B12 and B28 movie for each architecture, FP32
forward/backward and FP64 norm measurement. Early/final memory gradient ratios
changed from approximately 1e-16–1e-25 to 0.57 (29 frames) and 0.42 (45 frames).
All learned tensors received finite nonzero gradients in these full-BPTT checks.
This validates the gradient route, not task acquisition.

Gradient clipping caps excessive gradients; multiplying the total gradient
cannot restore relative contributions already attenuated across time.
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) balances multiple task
losses; the current task has one cross-entropy loss. The carry gate is related to
[gated recurrence and timescale initialization](https://arxiv.org/abs/1804.11188).

No separate full-BPTT training arm was launched. The carry architecture is used
by the user-requested [random-frame training variant](../RandomFrameRViT/README.md).
