# Variational motion predictor

The fresh model sees exactly three ordered RGB100×100 observations and predicts
one fourth RGB frame. It has15,463,363 trainable parameters in138 tensors. The
posterior depends only on the past: `q(z | X0,X1,X2)`. Its single diagonal Gaussian
latent has512 elements, rather than a spatial Gaussian field. No trained encoder,
attention, recurrent transformer, input-copy route or encoder-to-decoder skip is
included.

The nine frame-major RGB channels enter the existing residualCNN with32/64/128/256
channels at100/50/25/13 resolutions. All convolutions have stride1; learned
space-to-depth stages provide compression, with bottom/right replicate padding
from25 to26 before the final stage. A1×1 convolution reduces the13×13 field to32
channels; flatten5408→Linear512→LayerNorm/GELU creates vector features. Independent
linear heads produce512-dimensional `mu` and `logvar`. Log variance is clamped to
[-8,4]; its head starts with zero weights and bias−4, giving an initial standard
deviation of exp(−2). All these weights remain trainable.

Training uses `z = mu + exp(0.5*logvar)*epsilon` with standard-normal noise. A learned
dense512→5408 projection reshapes to32×13×13, a1×1 lift produces256 channels, and
residual/pixel-shuffle stages restore25/50/100 resolution. The26×26 upsampled stage
crops its last row/column to25×25. A three-channel sigmoid predicts only the fourth
frame. The decoder receives only `z`. Nonreentrant encoder/decoder checkpointing
preserves gradients while reducing saved activations. Evaluation defaults to
`decode(mu)`; this deterministic image is **not** the predictive expectation of
samples passed through a nonlinear decoder.

The loss balances mean RGB-pixel MSE over motion support and background equally
for each example. Support is the union of pixels whose RGB contrast from0.5 exceeds
1/255 in any of the past three frames or the target. These target-derived weights
appear only in the loss, never in the posterior or decoder. If a region is empty,
the available region receives the whole weight. A mean diagonal-normal KL per
latent element adds a default coefficient1e-4; the worker can supply its external
warmup coefficient. Separate metrics report support/background/full-image MSE,
MSE on pixels where the fourth frame changes relative to the third, and KL. The
change metric is diagnostic, not an extra loss.

The Gaussian reparameterization follows
[Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
We frame history-only stochastic encoding with target prediction and a KL penalty
as a variational predictive bottleneck, informed by
[Alemi et al., Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410).
Our target-weighted MSE is an explicit surrogate objective; this implementation
does not claim a standard conditional-VAE ELBO. The512-dimensional vector,
architecture, support rule and numeric beta are engineering choices, not constants
or validated results from those papers.

`VariationalMotionPredictor(latent_dim=512,checkpoint_encoder=True,
checkpoint_decoder=True)` initializes the whole model fresh. `forward(past,sample=None)`
returns `prediction`, `mu`, `logvar`, `z`; `sample=False` selects deterministic mean
encoding. `encode(past,sample=False)` exposes the512-vector and `decode(z)` accepts
only such vectors. `losses(prediction,target,past,mu,logvar,beta=...)` returns scalar
`loss`, `recon`, `support_mse`, `background_mse`, `full_mse`, `change_mse`, `kl`.
The dataset and bounded local worker own motion generation, streams and exposure.

`python -m SecondPass.VariationalMotionPredictor.check` runs a single short CPU
engineering fixture. It verifies vector/future-frame shapes, standalone decoder
equivalence, temporal order sensitivity, stochastic and deterministic modes,
union-support weighting, finite nonzero gradients and one disposable Adam update
for all138 tensors, and gradients for all three input frames. Production optimizer
progress and predictive accuracy require separate run evidence.
