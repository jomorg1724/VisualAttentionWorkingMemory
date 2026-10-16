# Stabilized shared attention on native Krauzlis

User-authorized October 6, 2026: implement the suggested attention stabilization,
normalization and global gradient clipping at 100, then train locally.

Fresh whole model, 379,444 parameters in 56 tensors. No checkpoint inputs.
Native stimuli, one shared attention head, local temporal comparisons and objective
are retained from LocalComparisonKrauzlis. No increased mnemonic iterations yet.

## Changes

1. Normalize projected queries and keys with L2 normalization (epsilon 1e-6).
   Attention scores are cosine similarity times `1+15*sigmoid(s)`, initially 8.
   The scale remains between 1 and 16. This bound is an engineering hypothesis.
2. Separate per-patch channel LayerNorm for write, reset and candidate preactivations
   in the spatial GRU; separate channel LayerNorm for the corresponding CLS GRU
   computations. Normalize before sigmoid/tanh, with learned affine parameters.
   Spatial normalization uses no statistics from other patches. Stored memory is
   not renormalized after its update.
3. Replace `100*alpha*value` spatial input with unscaled values and a bounded write
   factor `a=alpha/(alpha+1/N)` for N=100. Effective gate is `a*sigmoid(normalized
   write preactivation)`; update is `h + effective_gate*(candidate-h)`.
   Uniform attention gives a=.5; complete concentration gives a=100/101.
   The same alpha still selects visual and local comparison readouts. This changes
   both amplification and memory timescale and is not a one-factor ablation.
4. After complete batch gradient accumulation, clip global L2 gradient norm to 100
   before Adam. No post-clipping gradient renormalization; activation normalization
   takes place during the forward pass. Raw and clipped norms are logged.

## Training and interpretation

Adam3e-4, FP32, batch32/micro4, full temporal backpropagation; CE + .05 pixel
prediction loss. Fresh197xxx data streams and seeds; 1000 movies/two epochs then
refresh. Requested2244 updates, prospectively reduced in complete66-update pools
if necessary after native timing. One finite14400-second local cap including
profile and evaluation. No automatic extension, cloud, or additional arm.

First validation33, then every50 and terminal; 300 validation movies, select minimum
CE. Final3000 new test movies. Best/latest checkpoints include full optimizer and
RNG/pool cursor. Prior experiments and checkpoints are preserved.

Log per-update gate saturation (<.01/>.99), effective write strength, candidate
saturation, attention entropy/peak, and gradient norms. Validation reports late
target/foil attention using metadata only for monitoring, never model/loss inputs.
Success of this combined intervention would not identify which component helped;
failure would not disprove any single component. Original gradient evidence did
not independently establish that the multiplier100 caused collapse.

CPU_CHECKS.json verifies patch-local normalization, exact retention with zero write,
bounded memory, native full backward/Adam, and clipping an oversized gradient to100.
Check/profile weights are discarded before production.

Sources: [Layer Normalization](https://arxiv.org/abs/1607.06450),
[Query-Key Normalization](https://aclanthology.org/2020.findings-emnlp.379/).
Per-patch placement, bounded scale and attention write mapping are our adaptations.
