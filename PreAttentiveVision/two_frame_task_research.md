# Two-frame tasks for PAV: a compact primary-source shortlist

Research recommendation only: no stimuli, models or training were changed. The current user direction selects **exactly two ordered 100×100 RGB frames and four-way RDK direction discrimination**. The older unresolved-choice wording in `krauzlis_stimulus.md` is historical; the present direction does not authorize reconstructing a direction-change task from two static dot clouds.

Add the five complementary tasks below to that motion task. Together they probe orientation, contrast, spatial scale, chromatic differences and spatial grouping. They are computational adaptations of sensory psychophysics, **not replications of published human thresholds or cortical mechanisms**. Human two-interval experiments include presentation durations and interstimulus intervals; this model receives both images directly. We do not add a retention delay, attention cue, distractor-selection demand or memory claim.

The labels require ordered features. A decoder that sees only `abs(F2−F1)`, their mean and their product is invariant to swapping the frames, and cannot distinguish opposite motion directions or answer which interval contains the target. Preserve order in the common task decoder. For example, ordered concatenation or signed differences can coexist with symmetric interaction features. Task identity can select the output head; latent orientation, contrast, frequency, hue and contour membership must never be model inputs.

| Complementary task | Precisely what the two frames contain | Label and difficulty |
|---|---|---|
| Orientation comparison | A localized Gabor in each image; common scale/envelope/contrast, small signed orientation difference | Two classes: second orientation clockwise or counterclockwise from first; difficulty `abs(delta_theta)` |
| Contrast comparison | Same grating family at pedestal contrast `c` and incremented contrast `c+delta_c`, random interval order | Two classes: frame with higher contrast; difficulty `delta_c` conditional on `c` |
| Spatial-frequency comparison | Two gratings with common envelope/orientation/contrast but frequencies `f` and `f*2^delta`, random order | Two classes: frame with higher frequency; difficulty `abs(log2(f2/f1))` |
| Chromatic increment comparison | Two equal-geometry colored patches, one with a small positive increment along a fixed declared chromatic axis, random order | Two classes: frame with the larger chromatic-axis coordinate; difficulty increment conditional on base color |
| Contour-in-clutter choice | One Gabor-element field with a smooth aligned path; one orientation-scrambled field with matched element statistics | Two classes: frame generated with the aligned contour; difficulty path curvature/alignment jitter |

These labels are task-specific discrimination judgments, not an artificial universal change/no-change label. Balanced ordering makes binary chance 50%; mandatory RDK four-way chance is 25%. Do not pool their raw accuracies into one uninterpretable percentage.

## 1. Orientation: signed Gabor comparison

**Published grounding.** Zhang and colleagues used Gabor orientation and contrast discrimination in two randomly ordered intervals. Their methods specify 92-ms presentations, a 600-ms interstimulus interval, and judging the more clockwise or higher-contrast stimulus. For orientation, phase varied across presentations. [Zhang et al., 2010, *Rule-Based Learning Explains Visual Perceptual Learning and Its Specificity and Transfer*, methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC3842491/).

**Our adaptation.** Use independent carrier phases and fresh pixel noise, a shared Gaussian envelope and nuisance contrast/frequency, and a roving base orientation. Draw a nonzero local difference with magnitude below 45°. The signed axial difference is

\[
\Delta=\tfrac12\operatorname{atan2}\{\sin[2(\theta_2-\theta_1)],\cos[2(\theta_2-\theta_1)]\}.
\]

Define positive rotation explicitly in image coordinates; do not use a naive numerical subtraction across the 0/180° boundary. The class is the sign of this local difference.

**What it tests / failure to avoid.** Fine orientation information and comparison across phase changes. Independent phases prevent pixel subtraction from becoming a surrogate for rotation. Randomize the base angle so one frame's absolute angle does not identify its class. At 100 pixels, use a declared cycles-per-image scale; the published cycles/degree values cannot be copied without a field-of-view conversion.

## 2. Contrast: pedestal versus pedestal-plus-increment

**Published grounding.** Legge and Foley measured two-alternative contrast masking. A signal added to a masker was distinguished from the masker alone; equal signal/masker frequency and phase gives contrast discrimination, while zero masker contrast gives detection. This motivates varying baseline contrast rather than only distinguishing a bright pattern from a blank. [Legge & Foley, 1980, *Contrast masking in human vision*](https://opg.optica.org/josa/abstract.cfm?uri=josa-70-12-1458); [author-hosted paper](https://legge.dl8.umn.edu/sites/legge.psych.umn.edu/files/files/media/legge80_contrast_masking_in_human_vision.pdf).

**Our adaptation.** Two achromatic sinusoidal patches use a common baseline luminance, spatial frequency, envelope and orientation; one has contrast `c`, the other `c+delta_c`. Randomize interval order. Sample several pedestal levels, with all intensities inside gamut; record contrast numerically before quantization. A small independently drawn sensor-noise field may accompany each image, with the same distribution for both labels.

**What it tests / failure to avoid.** Whether normalization and downstream features retain contrast differences. Mean luminance, clipping, unequal envelope area and label-dependent noise must not identify the target. Do not remove the desired signal by independently RMS-normalizing the two inputs. An energy-based solution is legitimate here; it is precisely why this task complements orientation/grouping rather than substitutes for them. Report performance versus increment at each pedestal, without assuming the model must reproduce a human dipper curve.

## 3. Spatial frequency: which grating is finer?

**Published grounding.** Campbell, Nachmias and Jukes compared gratings of differing spatial frequency and found a strong role for frequency ratio, with contrast-related qualifications at high frequencies. [Campbell et al., 1970](https://opg.optica.org/abstract.cfm?URI=josa-60-4-555). A later primary study explicitly used two-interval frequency discrimination and showed that filtered noise can bias perceived spatial frequency. [*A New Perceptual Bias Reveals Suboptimal Population Decoding of Sensory Responses*, 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325184/).

**Our adaptation.** Draw roving `f`; pair `f` with `f*2^delta` and randomize order. Keep physical envelope size fixed within the pair, match mean and grating contrast, and independently randomize carrier phase. Vary orientation independently of the label. Use frequencies expressed in cycles/image and sufficiently below raster Nyquist to be meaningfully represented at 100 pixels.

**What it tests / failure to avoid.** Spatial-scale representation, especially information lost through pooling/aliasing. Do not scale the entire image to change frequency: that also changes envelope size. Avoid edge cycle-count or contrast cues caused by a hard rectangular window. Filtered-noise conditions are a possible later mechanistic question, not a required extra sweep for this first battery. No delay or memory-mask paradigm is proposed.

## 4. Color: a controlled chromatic increment

**Published grounding.** Krauskopf and Gegenfurtner measured color discrimination in an isoluminant plane with controlled adaptation, distinguishing changes along cone-opponent directions. Their experiment motivates controlling base color and luminance, rather than assuming RGB distance equals perceptual color distance. [Krauskopf & Gegenfurtner, 1992, *Color Discrimination and Adaptation*, author-hosted paper](https://www.allpsych.uni-giessen.de/karl/pdf/01.coldisc.pdf).

**Our adaptation, explicitly not a cone-isolating replication.** Use equal-size central patches on the same neutral surround. In a declared **linear RGB** space let `w=(0.2126,0.7152,0.0722)` define numerical luminance, and let `u` be the unit vector proportional to `(w_G, -w_R, 0)`. It satisfies `w dot u=0`. Draw a fresh in-gamut base color `r`; the two patch colors are `r` and `r+delta*u`, with `delta>0`, random interval order and no clipping. The label identifies the larger `u` coordinate. This is a defined red–green-like numerical direction, not a calibrated human L−M cone axis. A second blue–yellow-like axis can later use the same procedure, but is not needed to establish this initial task.

**What it tests / failure to avoid.** Retention of chromatic differences and avoidance of inadvertent grayscale encoders. Keep the increment independent of patch size, position and background. Apply any gamma conversion consistently; constant arithmetic mean RGB does not guarantee constant luminance. Do not call this human isoluminance without display spectra, cone fundamentals and observer calibration. A simple color-mean solution is a legitimate low-level capability, not evidence of color constancy or object recognition.

## 5. Contour grouping: aligned path versus scrambled orientations

**Published grounding.** Field, Hayes and Hess introduced path detection using oriented band-pass elements embedded in clutter, linking contour visibility to spatial relations between local orientations. This is the association-field paradigm, a behavioral grouping assay rather than proof of a particular cortical circuit. [Field et al., 1993, *Contour Integration by the Human Visual System*](https://doi.org/10.1016/0042-6989(93)90156-Q); [paper](https://dev.ipol.im/~blusseau/biblio/psychophysics/1993-field-hayes-hess--contour-integration-by-the-human-visual-system.pdf).

**Our adaptation.** Construct a fresh sparse field with a subset of equal-contrast Gabors placed along a smooth path and oriented near its tangents. Construct its comparison field with the same positions, frequencies, sizes, contrasts and multiset of orientations, but permute orientations among locations to disrupt alignment. Randomly assign which frame is structured. Random global rotation/location and matched local statistics make relationships among elements informative. The label is the generative structured-field identity; it is not a guarantee that every random control is perceptually contour-free.

**What it tests / failure to avoid.** Whether the encoder supports spatial grouping beyond isolated feature detection. Do not add more elements, brighter elements or a density gap around the path only in the target. At 100 pixels, keep the contour short enough and elements sufficiently separated to remain resolved. Accidental alignment in controls and very high tangent jitter can make some trials ambiguous; report difficulty instead of interpreting every error as absent grouping. This task does not implement attentional selection or establish that grouping is exclusively preattentive.

## Practical recommendation

Start with the mandatory ordered RDK task plus these five clearly separated outputs. If the local allocation demands a smaller first pass, postpone spatial-frequency comparison: orientation and contrast already use controlled gratings, while color and contour provide more distinct demands. Retain fresh held-out generator seeds and report family-specific accuracy with uncertainty and its primary difficulty variable. A generic change/no-change score would conceal which computation was learned.

For natural-image generalization, use the separately researched [natural-image spectrum task](natural_image_task.md). That extension tests richer image statistics; it should not silently replace these controlled sensory tasks. No VWM, attention, diffuser or later integration implementation is selected here.
