# PreAttentiveVision — component 1

This is the first component of the broader visual attention and working-memory project. It does not implement the later working-memory or attention components.

## Question

Which lightweight convolutional encoder preserves useful sensory information across neuroscience-grounded tasks using exactly two RGB100×100 images?

Current implemented-task work replaces the original dots task with random-dot motion direction classification: left, right, up or down. Each frame passes separately through shared encoder weights. The new common decoder preserves frame order, signed differences and local displacement correlations. See [Krauzlis stimulus sources](krauzlis_stimulus.md), [complementary two-frame task research](two_frame_task_research.md) and the [natural-image task proposal](natural_image_task.md). Additional task recommendations are not completed experiments.

## Active experiment: seven sensory tasks

The seven-task, five-encoder comparison is complete. Each model received 756 updates and 24,192 pairs; task-specific held-out results are in [the completed report](runs/multitask_20260912_141316/report.md) and `results_multitask.json`.

Each candidate uses one shared encoder across motion direction, orientation, contrast, spatial frequency, chromatic increment, contour grouping and natural-image spectral-detail discrimination. The common ordered decoder has a separate output layer for each task. Fresh procedural pairs and common streams across candidates support a practical comparison. Natural images use native-resolution BSDS500 photographs and official source splits.

Checkpoint selection uses validation mean task one-versus-rest AUC; test results are reserved for final reporting. Report task-specific accuracy, confusion matrices, AUC and difficulty curves. Motion chance is 25%; the six binary tasks have 50% chance. An initial one-seed budget compares practical learning under equal exposure; it does not establish a sufficient learning horizon or isolate individual architectural mechanisms.

## Current development: task allocation and retention

The user authorized the [task-allocation comparison](next_experiment_task_allocation.md). Both arms begin from the trained late-SE step-1512 checkpoint, preserving weights and Adam state. One continues uniform sampling; the other alternates contour batches with batches from the other six tasks. Matching task-local example streams isolate the schedule intervention. Equal total updates intentionally produce unequal per-task exposure.

Execution is recorded in `runs/allocation_20260912_160414/`, under the exact remaining 1008.2650083768344-second allowance, including profiling and evaluation. The target is 1008 additional updates per arm, reduced equally if measured costs require it. Validation selects by minimum task balanced accuracy, then mean task AUC; final evaluation reports all seven tasks and contour difficulty. Run receipts establish actual exposure and completion.

## Completed development: combine useful computations

The user authorized the [three-arm hybrid experiment](component_combination_research.md): continue the trained ConvNeXt unchanged, or add either the residual oriented-feature side branch or the late channel gate. These are separate siblings of the same trained parent, retaining compatible optimizer and sampler state. Each starts with the parent's input/output function; additional training may then change it. The original seven-task generators and ordered decoder remain common.

The [saved-score analysis](combination_analysis.md) found complementary contour errors but a motion cost from blanket probability averaging. The new experiment asks whether a compact feature addition can improve contour performance while preserving motion. A practical target fixed before the run is at least three percentage points of contour improvement versus the continued control, with no more than two points of motion regression. Report uncertainty and all seven task outcomes; this criterion is not a formal biological or statistical guarantee.

All three hybrid arms completed 756 additional updates from their common step-756 parent. Neither addition met the contour-improvement criterion. Continued training itself improved the control, and the late-SE arm retained strong performance on six tasks while contour balanced accuracy remained 63.4%. See [hybrid results](report_hybrids.html). These finite-budget results motivate the allocation comparison; they do not establish an architecture ceiling.

**Historical material below:** the first binary comparison was stopped for the user-requested stimulus correction. Its allocation, symmetric decoder, stimuli and partial validation are superseded protocol diagnostics, not results for the corrected task. No locked test was run. The five encoder implementations remain candidate components; no winner has been selected.

## Functional scaffold

[Wolfe's Guided Search6.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/) distinguishes early visual processing from the features that guide selection, and from subsequent target identification. This component measures an early visual representation's usefulness for a controlled comparison. It does not establish visual search guidance, selective attention or recognition in the complete GS6system.

The user requested two departures for the broader project: omit the diffuser, and provisionally treat activated long-term memory as synaptic weights. The latter is a modeling approximation rather than Wolfe's stated biological implementation. Future working-memory, attention and integration components remain separate development work.

## Comparison

Five compact, paper-inspired encoder adaptations share three output scales and channel widths so the decoder architecture can remain identical:

1. `aa_resnet`: residual local convolutions with filtering before downsampling.
2. `convnext_grn`: depthwise spatial filtering, pointwise channel mixing and global response normalization.
3. `inceptionnext`: multiple spatial kernel shapes/scales within an efficient convolutional block.
4. `mobilenet_se`: inverted bottlenecks and input-dependent channel gating.
5. `vone_resnet`: a V1-inspired Gabor/simple-complex feature front end followed by learned convolutions.

These are small adaptations for100×100inputs, not claims to reproduce published full-size model benchmarks. Exact equations, implementation choices, parameter counts and primary sources are recorded in `research.md` and `models.py`.

## Tasks and inference

The stimulus suite includes Gabor orientation changes, dot displacement, colored object/shape changes, and edits of natural images. Every example has exactly two frames. No-change examples still receive independent sensory nuisance. Pair construction balances the marginal distribution of each individual frame across the labels, preventing a changed-only visual artifact from becoming an easy single-image label cue.

Natural-image source identities are separated between training, validation and test. Synthetic examples are generated procedurally from disjoint seeds. Moving-dot trials test displacement/static differences, not motion acceleration or a change of motion direction that two frames cannot identify.

All models receive equal production update counts and fresh-pair exposure, the same training recipe and the same evaluation examples. Model selection uses validation; all five test results are reported. Parameter/throughput differences are disclosed rather than interpreted as isolated benefits of one operation. Performance and neuroscience grounding are discussed separately.

## Resource allocation

One localGPUworker runs at a time in fp32. An initial short, accounted throughput measurement fixes equal production exposure under a60-minute total compute ceiling, including profiling and evaluation. The exact configuration is frozen before production. This is an exploratory engineering screen, not a guarantee that any chosen exposure is sufficient for every architecture.

The measured allocation for this run is **one seed20261,1,536updates ×32pairs =49,152training pairs per model**,12,288per family. All five models receive the same generated pair stream. The32-update-per-model timing runs are separate, preserved profiling attempts and are included in the resource budget. Validation uses200pairs per family at each512updates; the best validation-AUC checkpoint receives800test pairs per family. A single threshold is calibrated on validation for each selected model; default0.5results are also retained.

Before production outputs, the practical target was set to macro balanced accuracy≥90% and everyfamily≥85%. This is a stated engineering target for this benchmark, not a biological criterion. Confidence intervals condition on this one trained seed; source-image clusters are used for repeated natural-image bases. Model selection uses validation rather than subsequent test scores.

Run artifacts: `runs/benchmark_20260912_135139/`. [Live aggregate](results.json) · [Rendered report](report.html). Completed status and actual scores are reported only when the worker supplies them.

## Files

- `stimuli.py` / `task.md`: generators, source identity, split and difficulty definitions.
- `models.py` / `research.md`: five encoders and mathematical/source rationale.
- `decoder.py`: common comparison decoder.
- Training/evaluation scripts and fixed configuration: execution and experiment accounting.
- `runs/`: logs, checkpoints, raw scores and summaries from actual runs.
- Final report and figures: added after the experiments finish.
