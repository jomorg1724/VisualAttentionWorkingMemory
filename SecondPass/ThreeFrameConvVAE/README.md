# Three-frame convolutional VAE

This fresh model receives three consecutive native RGB100×100 frames in their
original order and reconstructs all three. It learns a diagonal Gaussian
posterior over a spatial latent with shape `(256,13,13)`:43,264 latent elements.
The entire learned model has9,507,913 parameters in126 tensors. No learned weights,
optimizer state or data-stream progress come from the cancelled CNN–GRU.

The input reshapes frame-major into nine channels, without temporal averaging.
A32-channel stem and two residual blocks retain100×100 resolution. Three learned
space-to-depth stages produce64×50×50,128×25×25 and256×13×13 fields, each with two
residual blocks. The25×25 field is replicate-padded on its bottom/right edges to26×26
before the last compression. All convolutions have stride1; GroupNorm and GELU
support the residual layers. Independent1×1 posterior heads produce `mu` and
`logvar`. Log variance is clamped to[-8,4] for finite scales, with the usual zero
clamp gradient outside this range.

Training samples `z = mu + exp(0.5*logvar)*epsilon`, with independent standard-normal
noise. The decoder receives only `z`; it has no input or encoder skip connection.
Residual layers process the13×13 latent, then learned convolutions and PixelShuffle
restore26×26,50×50 and100×100 fields. The26×26 stage crops its last row/column to25×25,
inverting the encoder's spatial padding. Decoder residual layers operate at every
resolution. A final nine-channel sigmoid reshapes into three RGB reconstructions.
Nonreentrant activation checkpointing on the encoder and decoder bounds training
activation memory while preserving gradients through the whole computation.

Ordinary image MSE can reward the nearly uniform gray background of the dot movies.
The implemented reconstruction objective instead uses a fixed target-derived
support: a pixel is foreground if its maximum RGB contrast from0.5 exceeds1/255.
For every trial and every frame, foreground and background pixel-MSE means receive
one-half weight each. If one region is empty, the available region receives the
whole weight. Trials and frames are then averaged equally. Fixation and dots follow
the same pixel rule; no location, cue, phase, direction or class metadata enters it.
The loss is this balanced reconstruction plus `1e-4` times mean diagonal-Gaussian KL
per latent element; beta is fixed. Foreground/background MSE, ordinary full-image
MSE and KL remain separate metrics. This target-weighted objective is an explicit
reconstruction adaptation, not the standard unweighted pixel-Gaussian ELBO. No
motion, temporal-difference or response-classification objective is added.

The intended source is the existing single-stimulus movie adapter: no cue or second
stimulus, retained original locations/dot dynamics and29/37/45-frame movie lengths.
Its response labels are unused for VAE learning. The prepared worker draws one
uniform consecutive triplet per movie presentation within active frames7..T-2.
This sampling choice excludes fixation-only startup and the terminal report frame,
without changing the rendered movies. Evaluation uses three distinct windows per
movie and groups them as one underlying example. The worker owns fresh streams,
exposure and measured local feasibility. The parent owns
stopping the old local worker and the new bounded launch. This source implementation
and CPU fixture are separate from evidence of actual production optimizer progress.

`ThreeFrameConvVAE().posterior(triplets)` returns `(mu,logvar)`.
`encode(triplets,sample=False)` exposes deterministic256×13×13 means for a later
response architecture; requesting `sample=True` returns a reparameterized sample.
`forward(triplets)` returns `reconstruction`, `mu`, `logvar` and `z`; training is
stochastic and evaluation decodes `mu` unless `sample` is explicitly specified.
`decode(z)` accepts a standalone latent. The global `losses(...)` function returns
`loss`, `recon`, `foreground_mse`, `background_mse`, `full_mse` and `kl`.

Run `python -m SecondPass.ThreeFrameConvVAE.check` for the short CPU engineering
check. The fixture verified exact output/latent shapes, sampled versus deterministic
latents, latent-only decoder equivalence, sparse-support weighting, gradients for
all126 learned tensors and all three input frames, and one disposable Adam update.
It does not establish successful reconstruction learning or useful motion features.

The requested local target is10,000 optimizer updates, prospectively reduced to
complete330-update replay pools if needed. A pool contains1,000 fresh movies reused
for10 shuffled epochs, with new triplet windows drawn on each presentation. Adam
uses1e-4 without clipping at effective batch32/micro4. Validation has64 movies per
condition; fresh paired selected/terminal final evaluation has128 per condition,
each with three windows. These are planned settings, not achieved exposure.

The October3 recovery starts at saved update5600 under the original9900-update
allocation/deadline. User-requested retention is exactly `best.pt` and `latest.pt`,
replaced atomically; no numbered, initial, terminal or selected checkpoint copies.
Prior checkpoint files were deleted by explicit user instruction. The deleted
best5500 cannot be selected; restored5600 initializes the available-best pool
via the same validation protocol. Recovery state includes model, optimizer, RNG,
streams and scheduler; logs/reports remain.
