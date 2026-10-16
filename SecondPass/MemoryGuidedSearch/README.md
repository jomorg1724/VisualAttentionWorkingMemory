## October 10 doubled extension active

User-authorized continuation02 resumes latest19000 and targets38000 cumulative updates. All learning state restored; old best17860 retained. Saved19019 independently verified all92 Adam states advanced with original checkpoints/source unchanged. Worker17136/supervisor35552. Finite16-hour local cap ends October11 05:25PDT, no automatic extension. Doom unchanged. See [extension protocol](EXTENSION.md) and `LocalRuntime/continuation02/independent_production_verified.json`. Earlier entries below describe previous allocations.

## October 10 continuation active

The explicitly requested continuation resumes run01/latest9120 in `LocalRuntime/continuation01`, preserving model, all92 Adam states, scheduler, streams and all RNGs. Original best4560 stays eligible; target19000 and the71-cell validation panel are unchanged. A new finite eight-hour local cap ends October10 at approximately16:44PDT. Exact CPU resume/next-update check and CUDA RNG restoration passed; saved9234 independently verified all92 finite aligned Adam states advanced. Worker43940/supervisor13152; no automatic restart or extension. See `LocalRuntime/continuation01/independent_production_verified.json`. Original run01 is preserved.

# Object-memory, pop-out and motion-memory search suite extension

## Next stopping point: planned variable-duration VDA extension

October 9: user supports adding variable-length VDA trials in the next suite
continuation, after the next stopping point. This is a recorded next-run design;
it does not change the current frozen training source, request a stop, launch a
successor, or extend the current allocation.

Retain the original seven-observation condition exactly: blank, cue, blank,
two baseline-array observations, two test-array observations. Add independently
varied baseline lengths S in {2, 8, 16} and test lengths H in {2, 4, 8}, keeping
the first three observations fixed. Total length is 3+S+H, from 7 to 27.
Apply to VDA4/9/16 with the existing theta conditions, cue validity/value rules,
any-location-change labels, raster adaptation, architecture, and loss. These
timings are a proposed grid, not implemented training conditions yet. Balance
within VDA tasks so extra conditions do not increase their task-level weight.

Render fresh native orientation/pixel noise on additional array observations;
do not manufacture duration by duplicating noisy frames or adding terminal
blanks. Preserve underlying scenes and phase-indexed noise prefixes for matched
duration comparisons. The original S2/H2 condition must remain pixel-identical
under its original seeds; new timing streams must be independently versioned.

The architectural question is more specific than total trial length. At time
t the explicit reference uses t-15 through t-8, the recent window t-3 through t,
and joint confidence weights both. The seven-frame condition has no reference
and an identically zero 735-dimensional comparison branch. Longer sequences
can first provide a partial reference containing cue/blank history; that is not
yet a clean baseline-versus-test comparison. For example S16/H4 ends at t=22:
reference 7-14 lies fully in baseline 3-18, the gap is 15-18, and recent 19-22
lies fully in test. Check actual confidence and branch values, not length alone.

At the checkpointed stopping point, preserve best/latest and full optimizer/
stream/RNG state, then prepare and verify the duration adapter before launching
the next requested continuation. Profile the longer conditions prospectively;
their larger frame count changes compute cost. Preserve original seven-frame
evaluation and keep the other 16 task laws unchanged.

Analysis: repeat paired cued-site inhibition and opposite-site stimulation,
including change at the cue, change elsewhere, and catch trials. Keep native
validity trials separate from the deliberately violated 100%-cue test. Report
hits, catch false alarms, balanced accuracy, and paired uncertainty by duration,
set size and change magnitude; inspect maps at baseline, change, and terminal
times. Test interventions during the test phase with clearly specified frame
windows; a fixed final-two-observation arm controls intervention duration.
Compare the full frozen observer against zeroing only the explicit comparison
contribution on identical long movies. Treat this acute ablation as a test of
checkpoint dependence, with its distribution-shift limitation, not proof of
exclusive mechanism. Longer viewing alone changes evidence and memory demands,
so duration effects by themselves cannot identify a benefit from that branch.

Interpretation: the existing VDA tradeoff cannot depend on the explicit
long-window comparison branch, which is zero there. It remains consistent with
shared spatial selection through the remaining pathways. It does not establish
that attention alone is sufficient, or explain why earlier task interventions
had different benefit results. The duration extension tests whether the
reciprocal tradeoff persists when the additional branch can contribute.

Prepared October 9, 2026. Three new tasks join the completed 16-task suite:
remember and locate an object, locate a feature singleton without a sample,
or remember and locate a motion direction. Each task has one four-class head.
Object stimulus families share one head, pop-out conditions another, and motion
conditions a third. Local continuation was explicitly authorized on October9. Live state belongs
to `LocalRuntime/run01/status.json` and its supervisor receipt. The RunPod Doom
experiment is separate.

## Trial and output

Memory search: one blank fixation frame, S identical central sample frames,
D blank frames, then H identical search-array frames. S and H independently
take values 1, 2, 4 or 8; D is 0, 4 or 12. Total length is 1+S+D+H
(3 to 29 frames), with no sequence padding or extra report frame.
Pop-out: one blank fixation frame followed by H identical array frames
(2 to 9 total); there is no sample or delay. Blank pixels are
128/255. Timesteps are computational observations, not calibrated human milliseconds.

Exactly one target is present. Labels are 0 top-left, 1 top-right, 2 bottom-left,
3 bottom-right. Four 28 by 28 pixel objects are centered at (25,25), (75,25),
(25,75), (75,75) in a 100 by 100 RGB array. The central sample has exactly the
same raster and scale as its array occurrence. There is no absence class,
rotation/scale invariance requirement, or nine/sixteen-location version yet.

## Conditions

Memory search has three families times two distractor conditions times four
sample durations times three delays times four array durations: 288 conditions.
Pop-out adds 24 conditions and motion-memory search adds 81.
The expanded suite has 19 tasks and 437 conditions;
the original 16 tasks and 44 conditions keep their task laws.

| Family | Distinct objects | Similar objects |
|---|---|---|
| Features | Four colors, shapes and textures, each unique in the array | Two values per feature, each appearing twice; no single feature identifies any target |
| Objects | Independently drawn Fourier-contour silhouettes | Four perturbations of the same parent contour |
| Fractals | Independent spectral fractional-noise textures | Correlated textures sharing a parent field |

Feature objects have randomized palette/shape/texture assignments and continuous
geometry variation per trial. The hard feature array has code combinations
000, 011, 101, 110: any one feature is ambiguous, while a conjunction resolves
the target. It does not require all three features simultaneously. Procedural
silhouettes can differ in area; equal-area normalization is not imposed on them.

The fractals are multiscale random textures generated with a power-law Fourier
amplitude spectrum, not Mandelbrot-set images. Their circular supports and full
pixel-value histograms match exactly within each array, preventing brightness,
contrast or occupied area alone from identifying the target. Similarity is a
generator manipulation, not a calibrated human perceptual-distance scale.
Minimum pairwise raster RMS is retained for audit. Distinct hashes guarantee
different pixels, not perceptual discriminability.

## Pop-out conditions

Four circular objects for color, or four bars for orientation. Exactly three
items have identical rasters and the fourth differs. Color differences are
12, 36 or 72 RGB opponent-amplitude units. Opponent colors vary around a random
midpoint around RGB (180,180,180), along (1, -0.2126/0.7152, 0), with randomly reversed singleton and
distractor assignments. These are not perceptually calibrated or physically
isoluminant stimuli; RGB rounding and display gamma matter. The randomized
role swap balances absolute target/distractor colors in expectation.
Orientation differences are 5, 15 or 45 degrees, with random base orientation
and difference sign. Each difference is crossed with H1/2/4/8, giving
2 features times 3 differences times 4 durations = 24 conditions.

Location is balanced and no sample identifies the singleton. Shape, size and
position geometry are held constant within each array. No size or motion
singleton, target-absent condition, or set-size manipulation is included yet.
Pop-out is the intended task paradigm; successful learning and human-like
search efficiency have not been established. The shared attention mechanism
is unchanged. The experimental motivation is the joint stimulus-driven and
memory-guided priority map in [Guided Search 6.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/).

## Motion-memory search

A central random-dot sample moves in one direction, disappears for a delay,
and is followed by four simultaneous random-dot patches. Select the location
whose motion direction matches the remembered sample. This is one additional
task, `motion_memory_search`, with the same TL/TR/BL/BR belief labels. Only RGB
movies enter the model; directions and generation seeds are analysis metadata.
No direction targets, flow supervision, or hand-coded matching readout are added.

- Sample duration S: 3, 6 or 12 frames.
- Blank delay D: 0, 4 or 12 frames.
- Array duration H: 3, 6 or 12 frames.
- Direction spacing: 20, 45 or 90 degrees.

All four factors are crossed: 81 conditions. One fixation frame precedes the
sample, so total length is 1+S+D+H, from 7 to 37 frames. Three observations give
two motion intervals; the static tasks retain their one-frame conditions.
The four directions are theta + {0, delta, 2*delta, 3*delta}, modulo 360, where
theta is uniformly random and delta is the condition's spacing. Their assignment
to locations is randomly permuted. All four candidates can be the sample; the
correct one is not chosen for being an angular outlier. Spacing is minimum
pairwise separation, not a claim that all three distractors are equally distant
from the selected target. At delta90 they are uniformly spaced around a circle.

All patches use 12 Gaussian dots (sigma0.65 pixels), 100% coherent translation
at 1.5 input pixels/frame, a 28x28 world and a circular radius12 aperture. Bright
Gaussian dots sit on the same gray128 background as other tasks. Uniform initial
positions translate on a periodic square; the circular aperture masks after
translation. The number of visible dots can vary as they enter or leave the
aperture. There are no dot-lifetime resets or direction-dependent initial layouts.
Screen-coordinate direction0 is rightward and90 is downward. Speed, density,
aperture, color and coherence do not identify which alternative is correct.

The sample and all four alternatives have independent initial dot layouts.
`Stream.paired_motion_queries` holds the whole search movie identical while
querying each location in turn. Even sample first-frame pixels stay identical:
only sample motion changes. This is a particularly strong check against static
image matching and array-only guessing. Fixed trial IDs give exact movie prefixes
across duration/delay conditions. Different splits use independent namespaces.
The old 18 tasks retain their v2 pixels and seeds; the added task uses stream42.

This follows the motion-memory principle of Mendoza-Halliday, Torres and
Martinez-Trujillo (2014), [Sharp emergence of feature-selective sustained activity
along the dorsal visual pathway](https://www.nature.com/articles/nn.3785), which
studied remembering motion direction in a macaque delayed matching task. Our
four-choice array, continuous random direction sets, pixel renderer, timing grid
and categorical belief output are explicit adaptations, not an exact replication.
No biological claim about our recurrent states follows from this similarity.

Four example GIFs show exact input movies, enlarged3x by nearest-neighbor for
viewing, at150ms per observation for illustration only. The two A90 examples
(query0 and query3) share identical arrays but have different sample motions.
PNG files and hashes preserve each actual frame, including blanks. GIF encoders
may merge repeated blanks while preserving their total duration.

## Timing comparisons

The sample and array duration factors are independently crossed. At a fixed
trial ID, family/difficulty and split, the exact scene and correct answer are
identical across sample durations, delays and array durations. Pop-out also
matches scenes across durations. This supports paired psychometrics without
changing object identities between timing conditions. Such repeats are within
a split, never across train/validation/test; uncertainty should cluster by scene
when pooling timing cells. Shorter timelines are truly shorter tensors, with
no hidden padding or additional observation before the terminal response.
Repeated images are additional recurrent processing steps, not physical motion.

## Memory dependence and novelty

All four array objects are eligible targets. Each four consecutive trials of a
condition uses a randomized permutation of the four labels. Each trial generates
a fresh array; identities are not fixed learned classes. Different generator
seeds separate train, validation and test. Feature grammars and generator families
remain shared across splits, so this tests fresh-instance generalization rather
than generalization to a wholly unseen object-generation family.

`Stream.paired_queries(cell, trial_id)` returns four trials with byte-identical
search arrays and four different central samples. The correct label changes
with the sample. This is the primary counterfactual evaluation control. The
ordinary training stream does not expose this matching metadata to the model.
Only image tensors enter the backbone; labels select the supervised answer and
metadata supports analysis. The task ID chooses a single head, not a target identity.

## Integration and authorized training

`dataset.Stream` implements the existing `(images, labels, metadata)` batch API,
versioned state restoration, and fresh 204xxx streams for all 19 tasks.
`model.ExpandedSuite` adds three Linear(64,4) heads, or 780 parameters. The shared
backbone, attention mechanism and original heads are unchanged. `initialize`
can copy VDA best16000 and its 86 Adam states; the six new parameter tensors
start without optimizer history. Total size is 382434 parameters and 92 tensors.

`launch.py` starts one detached worker and enforces an eight-hour hard limit.
`train.py` saves initial, best and latest checkpoints in a new run directory. It uses balanced
19-task cycles, balanced condition rotation, batch32/micro4, Adam3e-4, clip100,
terminal CE plus .05 pixel prediction. Original VDA value/3 CE weights persist.
Default ceiling is 19000 updates within one explicitly launched allocation of
at most eight hours. A fixed71-cell monitoring panel runs at baseline, update19, every380 and the
endpoint, using32 trials per cell. It includes all44 old conditions,12 object
search conditions (all families/similarities at two timing combinations), all6
pop-out differences at H4, and9 motion conditions spanning all spacings and
three timing combinations. Checkpoint selection always uses the same equal-task
mean AUC on this panel. It does not estimate the full timing psychometrics.
The selected model then receives all437 validation cells at64 trials each and
a fresh full-grid test at256 per cell, if the allocation permits completion.
A timeout saves progress and reports incomplete evaluation, never success.
The user authorized this launch on October9; no automatic extension or restart.
Each task receives one nineteenth of task-balanced batches. Conditions rotate
balanced within each task; the larger memory-search grid does not increase its
share. The v3 stream contract rejects earlier preparation states rather than silently changing
its trial timing. No earlier search training was launched. The19000-update value is a ceiling, not a guarantee of finishing within the
allocation. Initial-cycle timing and monitoring latency provide the live estimate.
No architecture, optimizer, learning rate, loss or data-law change accompanies
this launch; only the monitoring schedule is reduced from the preparation default.
Each complete19-update cycle saves model, Adam, scheduler, streams and all RNGs.
The parent file is never overwritten; `initial.pt` records exact migration.
Imported workspace source files are copied with hashes into the run for audit.

## Verification and figures

Eleven CPU contract tests cover all 393 new conditions, sample-to-array exact matches,
counterfactual cue switches, histogram and feature controls, balanced labels,
fresh split instances, old/new stream restoration, scheduler coverage, preserved
model/Adam state, and a disposable finite-gradient update. `verify.py` checks
forward outputs and normalized attention on all 393 new cells. No trained-performance
claim follows from these checks.

`examples/cue_switch_examples.png` compares samples A and B with an identical
array for each family and difficulty. The example folders export every actual
frame, including blanks, with hashes in `examples/manifest.json`. These are
stimulus illustrations; no attention or successful-model predictions are implied.
`examples/popout_examples.png` shows all six feature/difference combinations.
The current manifest selects v3 examples; earlier D4_H6 example folders remain
as historical v1 illustrations and are not current training conditions.

## Experimental basis

The sample-delay-search structure is inspired by
[Chelazzi et al. 1993](https://www.nature.com/articles/363345a0), the feature
competition by [Wolfe, Cave and Franzel 1989](https://pubmed.ncbi.nlm.nih.gov/2527952/),
and target-preview questions by [Vickery, King and Jiang 2005](https://doi.org/10.1167/5.1.8).
Fresh procedural objects, spectral textures, raster geometry, exposure counts
and categorical location beliefs are explicit computational adaptations.
