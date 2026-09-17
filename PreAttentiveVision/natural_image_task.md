# Natural-image complement to the two-frame sensory battery

Status: recommended task; not implemented or trained. This complements the sourced controlled-stimulus tasks in `two_frame_task_research.md`.

## Question and source

Can the encoder preserve changes in the distribution of contrast across spatial scales in natural photographs?

Tadmor and Tolhurst (1994), [Discrimination of changes in the second-order statistics of natural and synthetic images](https://doi.org/10.1016/0042-6989(94)90167-8), measured human discrimination of altered image statistics. Their account relates performance to local contrast within spatial-frequency bands. This supports a natural-image spectral-detail discrimination task, rather than arbitrary colored squares pasted onto photographs.

Tajima and Okada (2010), [Discriminating Natural Image Statistics from Neuronal Population Codes](https://doi.org/10.1371/journal.pone.0009704), provides a computational population-coding treatment and references the psychophysical experiments. It is a model paper, not new human measurements.

## Proposed two-frame adaptation

Use two versions of the same grayscale photograph crop, represented in three RGB channels. Preserve Fourier phase and vary radial spectral weighting. Predict which frame has the greater high-frequency weighting (two classes). Randomize presentation order and base weighting; equalize mean intensity and RMS contrast across the pair. Sweep the difference in weighting to measure a discrimination curve.

For the non-DC Fourier components of a zero-mean image J, use

`F_beta(fx,fy) = F[J](fx,fy) * (sqrt(fx^2+fy^2) / f_ref)^(-beta)`.

Then inverse transform and apply a common target mean and RMS contrast. The zero-frequency component is set separately. A smaller beta means greater relative high-frequency weighting for the same source image. This multiplicative adjustment does not imply that every photograph has an exact power-law spectrum. Frequency cutoffs and contrast must keep the final raster within range without unequal clipping.

The 100x100 adaptation needs source crops at least that resolution. The previous CIFAR images are only 32x32 before upsampling and are unsuitable for claiming a test of native 100-pixel fine detail. Split source photographs, not merely transformed variants, between training, validation and test. No new dataset has been downloaded for this proposal.

## What this adds

Grating frequency discrimination isolates a narrow-band feature. This task asks whether scale information survives in broadband, structured images. It measures a sensory representation and decoder together; it does not establish semantic recognition, biological normalization, attention, or visual working memory. It is a disclosed computational adaptation, not a replication of the complete human protocol.
