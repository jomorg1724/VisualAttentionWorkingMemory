# Correction: moving dots must be grounded in the Krauzlis paradigm

Status: source research only, **not implemented or launched**. The user rejected the initial single-dot displacement family as an arbitrary substitute for the intended moving-dot stimulus. That implementation must not be represented as a Krauzlis task. The unresolved decision is whether dots should use two short motion clips, or whether exactly two still frames remains a strict constraint.

## Most direct source: Zénon and Krauzlis (2012)

[Attention deficits without cortical neuronal deficits, Nature 489, 434–437](https://pmc.ncbi.nlm.nih.gov/articles/PMC3448852/) (DOI [10.1038/nature11497](https://doi.org/10.1038/nature11497)), Methods, “Attentional task”:

- Two patches move in opposite mean directions, at diagonally opposed locations. A static patch cues the relevant location.
- Patch width: 5–7 degrees of visual angle, chosen to fit recorded receptive fields.
- Dot direction: Gaussian around patch mean, standard deviation **16°**. This is angular dispersion, not a stated proportion of fully coherent dots.
- Dot lifetime: **8 frames, 107 ms**; display refresh **75 Hz**.
- Cue: **133 ms**, followed by **500 ms** with fixation only.
- Moving baseline: **800 ms plus a geometric delay**, mean **480 ms**, range **0–3520 ms**.
- One patch undergoes a **16–20° mean-direction change**, titrated by behavioral performance.
- The subject responds to a change in the cued patch and ignores the foil. Stimuli persist **650 ms** after change or until response.
- Dot luminance **50 cd/m²**; background **14 cd/m²**.

The methods refer elsewhere for additional renderer characteristics; this section does not independently specify dot diameter, density or speed. Those must not be silently invented or attributed to this paper.

## Fuller renderer specification: Arcizet and Krauzlis (2018)

[Covert spatial selection in primate basal ganglia, PLOS Biology](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005930) (DOI [10.1371/journal.pbio.2005930](https://doi.org/10.1371/journal.pbio.2005930)), Materials and methods, motion-direction change detection task:

| Parameter | Reported value |
|---|---|
| Direction distribution | Normal, standard deviation **16°** |
| Dot lifetime | **10 frames = 100 ms** |
| Speed | **15 degrees/second** |
| Density | **25 dots/degree²/second**, as printed |
| Circular aperture radius | **3–3.75°**, median **3.25°** |
| Patch eccentricity | **10–13°**, average **12°** |
| Patch mean directions | Differ by **90°** |
| Change timing | **1.0–4.3 seconds** after motion onset |
| Median direction change | **28° and 26°**, respective monkeys |
| Change direction | Clockwise/counterclockwise equally likely |
| Post-change response window | **1 second** |

The paper also includes single-patch trials, which remove the foil while retaining motion-direction change detection. This is a relevant simplification for an initial visual component, without adopting attention selection now. The reported density includes a time denominator; it must not be relabeled as 25 simultaneously visible dots/degree² without checking the stimulus convention.

## Related coherence paradigm: Lovejoy and Krauzlis (2010)

[Inactivation of primate superior colliculus impairs covert selection of signals for perceptual judgments](https://pmc.ncbi.nlm.nih.gov/articles/PMC3412590/) (DOI [10.1038/nn.2470](https://doi.org/10.1038/nn.2470)), Online Methods, “Behavioral tasks”:

This is a related **direction discrimination** task with a coherent-motion pulse, rather than the same direction-change task. Four stochastic motion patches occupy circular apertures of radius **4.25°**, centered at **8.2° eccentricity**. Dot lifespan is **two refreshes**. At refresh a dot either appears at a random position or moves **four pixels, approximately 0.2°**. Coherence is the proportion moving in the common direction; the remainder move in uniformly distributed random directions. The informative pulse lasts **160 ms** at both target and foil locations. It emerges from incoherent motion by assigning newly appearing dots to the coherent pool. Monkeys report the cued pulse's direction.

This source provides a principled signal/noise motion construction if the selected task becomes detecting or discriminating coherent motion. It must not be called a replication of the 2012 direction-change task.

## The two-frame issue is information availability

With exactly two dot-position images at times t₀ and t₁, a visual system can estimate one displacement field, approximately `(position₁ − position₀)/(t₁ − t₀)`, subject to correspondence ambiguity, aperture effects, finite lifetime and noise. It can discriminate direction or coherent versus incoherent displacement when the renderer creates those signals. A single static dot cloud generally supplies no direction information.

To determine whether motion direction changed, the system needs evidence of **two velocities**, one before and one after the change. Three temporally aligned position observations are the idealized mathematical minimum with known correspondence. Random-dot stimuli with limited lifetimes and angular dispersion require useful temporal samples on each side; the experiment must establish usable evidence rather than simply asserting that three frames suffice. Two isolated still images, one before and one after the event, do not supply the two motion estimates. Rendering trails, arrowheads or latent direction into a static image would change the sensory task.

Two legitimate options were therefore presented to the user:

1. **Two short before/after motion clips:** preserves the intended direction-change question and population-motion stimulus, but changes the literal two-frame input constraint. Clip length and encoding remain undecided pending the user's response.
2. **Exactly two frames:** uses a population random-dot displacement or coherence discrimination question, explicitly labeled a two-frame adaptation rather than Krauzlis direction-change detection.

The choice is pending. No new stimulus implementation, architecture or training run is authorized by this document alone.

## Parameters still requiring an explicit adaptation

The user requires 100×100×3 images. Degree-to-pixel scaling, aperture placement, dot size, luminance normalization and temporal sampling must be specified for that raster, rather than copying screen pixels from another monitor. For example, converting a reported angular speed requires both pixels/degree and seconds/frame: `displacement_pixels = speed_degrees_per_second × seconds_per_frame × pixels_per_degree`. That equation is dimensional conversion, not a selected parameter set.

The patch number, foil, cue and waiting-time structure belong to the original attentional paradigm; they can be omitted for the isolated PAV task only with clear disclosure. PAV remains one initial component of a larger visual attention and working-memory project. No implementation for those later components is selected here.
