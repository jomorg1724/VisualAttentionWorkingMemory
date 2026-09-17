# PAV: two-frame change detection task

PAV is the first component of the larger visual attention and working-memory project. This task isolates visual encoding and simultaneous comparison. It does not implement visual attention, a working-memory delay, a search process or the rest of Guided Search 6.0.

Every trial supplies exactly two separate RGB arrays of shape **100 × 100 × 3**, scaled to [0,1]. A shared-weight encoder sees each image individually; the common decoder compares its outputs. Metadata and source class labels are never model inputs. No-change and change labels are balanced within each even batch. The four families and easy/medium/hard bins are sampled uniformly unless the caller requests a specific stratum.

## Preventing a single-image edit shortcut

For every trial, first generate an original scene A and an edited scene B **regardless of the label**. For no change, present A/A or B/B with equal probability. For change, present A/B or B/A with equal probability. Thus each frame position has an equal original/edited marginal under either label: an occlusion or editing artifact alone does not reveal the answer. The variant identities, proposed magnitude and observed magnitude are recorded for audit only.

Each observed frame independently receives global gain Uniform(0.97,1.03), brightness offset Uniform(-0.01,0.01), and pixelwise Gaussian noise with standard deviation 0.025, then clips to [0,1]. These same distributions apply to both labels and every family. No-change images therefore are not identical pixel copies. This benchmark asks whether the latent scene variant changed despite sensory nuisance; it is not semantic change detection and does not establish human psychophysical equivalence.

## Families

| Family | Scene and change | Easy / medium / hard magnitude |
|---|---|---|
| Gabors | Nine Gaussian-windowed sinusoidal patches, independently sampled orientation, contrast and phase; rotate one patch | Uniform 35–65° / 15–35° / 5–15° |
| Dots | Sixteen colored Gaussian dots at jittered separated positions; displace one dot in a random direction | Uniform 6–9 / 3–6 / 1.5–3 pixels |
| Colored shapes | Nine colored circles, squares or triangles; randomly choose one-object color or shape change | Color-channel delta 0.35–0.50 / 0.20–0.35 / 0.09–0.20; shape radius 9 / 7 / 5 pixels |
| Natural images | Official CIFAR-10 photograph; uniformly choose local color edit, local occlusion or whole-image translation | Local square side 20–30 / 12–20 / 6–12 pixels; diagonal translation components 5–7 / 3–4 / 1–2 pixels |

For shape-color edits, intensity clipping can slightly reduce the proposed change; metadata records actual channel delta. Shape-category changes have no scalar psychophysical equivalence with color changes. Natural local color edits alter one channel by 0.3 before clipping, while local occlusion uses a color that contrasts with the original patch mean. Reflected padding avoids a black translation border. Frame marginals remain matched even when edited images have visible artifacts. No edit selects its location from a learned feature or ground-truth object mask.

The Gabor image is `0.5 + contrast × exp(-(x²+y²)/(2×5²)) × cos(2π(x cosθ+y sinθ)/7+phase)` on a 21×21 support. The dot renderer uses continuous-center Gaussian profiles with sigma 1.8 pixels, so subpixel movements do not collapse to the same raster. Two dot frames establish a displacement; **they cannot establish a change in velocity, acceleration or movement direction**, which requires additional temporal observations.

## Natural source and disjoint splits

[The official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html) describes 60,000 color images of size 32×32, with 50,000 training and 10,000 test images. Cite Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009). This experiment bilinearly enlarges the source to 100×100; it does **not** invent native high-resolution photographs. Image class labels are unused.

The official binary archive is downloaded once from `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz` and checked against the page's MD5 `c32a1d4ab5d03f1284b67883e8d87530`. Only the six expected regular binary files of the expected size are extracted to `data/`; no pickle execution or arbitrary archive extraction is used. The official page does not state an explicit license grant. The downloaded source remains local research data; no external redistribution is part of this task.

- Train: official training image indices 0–44,999, with fresh sampled edits and nuisances.
- Validation: official training image indices 45,000–49,999.
- Test: official test image indices 0–9,999.

Base IDs are disjoint by this allocation; repeated trials of a source image within a split are not independent natural scenes. The source set can contain similar or duplicate underlying pictures, which this index split does not detect. Procedural synthetic train/validation/test draws use distinct seeds. A fixed seed and stream position reproduce exactly the same sampled trials; `state_dict` and `load_state_dict` preserve stream RNG and trial IDs for honest continuation. Benchmark runner configuration records its actual seeds and exposures. Example sheet seed: **2026091201**, validation split, two medium trials per family.

## Scope and interpretation

This is a deliberately small, transparent engineering benchmark with local changes and some full-image translations. It favors aligned pair comparisons and does not model eye movements, cluttered natural object correspondence, occlusion inference or long-term memory. A good score demonstrates usable two-frame features and the trained comparison procedure on these distributions. It does not by itself identify a neuroscience mechanism or validate other future components.

`stimuli.py` implements the stream. `test_stimuli.py` checks shapes/ranges, balanced labels, paired-variant law, non-identical no-change sensory draws, exact stream continuation and base-index split allocation. `stimulus_examples.png` displays two trials per family, with both input frames. The archive provenance is recorded in `data/cifar_provenance.json`.
