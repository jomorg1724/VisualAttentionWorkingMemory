# Tasks and metrics: read results in their own experimental context

[Index](README.md) · [Architecture](ARCHITECTURE.md)

## Two-frame sensory battery

Every PAV input is two RGB 100 × 100 frames. Shared encoder weights process each separately. The ordered decoder or causal accumulator predicts a task-specific answer; task identity selects an output head rather than being inferred as unrestricted natural-language instruction.

| Family | Tested sensory judgment | Chance BA | Important boundary |
|---|---|---:|---|
| Cardinal random-dot motion | Left/right/up/down mean motion from two ordered frames |25%| A disclosed100-pixel adaptation grounded in Krauzlis methods, not a full behavioral reproduction |
| Orientation | Signed Gabor orientation difference |50%| Later sample/probe match-change tasks use a different label rule |
| Contrast | Relative contrast judgment |50%| Nuisance and label construction belong to the saved generator |
| Spatial frequency | Relative frequency judgment |50%| Two-frame sensory discrimination, not remembered frequency capacity |
| Chromatic increment | Chromatic change/increment discrimination |50%| A controlled RGB task, not a complete biological cone model |
| Contour | Grouped contour versus matched alternatives in clutter |50%| A procedural grouping paradigm, not a measured human search threshold |
| Natural spectrum | Spectral-detail comparison from natural photos |50%| Source photographs split by official BSDS partition; new crops can reuse photos |

[Sensory sources/methods](../PreAttentiveVision/two_frame_task_research.md), [dot methods](../PreAttentiveVision/krauzlis_stimulus.md), [natural-image methods](../PreAttentiveVision/natural_image_task.md), and executable [seven-task generator](../PreAttentiveVision/neuroscience_stimuli.py) define details. The early stopped binary dot benchmark is superseded.

## Sequence tasks are not the original two-frame labels

The broad [memory battery](../WorkingMemory/TASK_BATTERY.md) introduces cue/rule formats, integration lengths, decision delays, distractors and retrospective probes. Its minimal-rule conditions assess whether the instructed task was acquired. A poor long-delay score cannot be interpreted as a pure retention limit if the corresponding short-delay task is also not learned.

The later focused motion task displays eight transitions whose directions can vary among four cardinal directions. The target is the direction shown for the greatest total duration, not the final direction or the direction of a single displacement. Blank frames after evidence test whether the answer can still be reported. Since the winner can in principle be computed before blanks, success can retain a decision rather than a detailed visual item.

Single-item orientation instead requires a remembered sample and an incoming comparison probe. D denotes inserted sample-to-query blanks; the identity query is a distinct visible frame. The existing renderer's `post_delay` means query-to-probe interval and was fixed to 0 in Retention. Do not relabel it as the manipulated sample delay.

## Exact current timing

Zero-based frame indices from the actual diagnostic wrapper:

| Phase | Single orientation D24 | Motion duration D24 |
|---|---|---|
| Instruction |0|0|
| Visual evidence |Sample1–2|Reference dots1, motion transitions2–9|
| Inserted blanks |3–26|10–33|
| Identity query |27|No separate identity query in this task|
| Probe / report |28|Report34|

At single D0, query/probe are 3/4. Motion evidence-to-report age is D+1 frames; orientation sample-to-probe age is D+2. Frame count is not a calibrated number of milliseconds. All frames, including blanks, pass through the model; there is no ordinary-path blank oracle.

## Spatial binding has different demands

The two-location task presents two oriented localized patches and tests preservation versus exchanging their locations. Full swaps preserve inventory, so inventory or probe-only shortcuts do not solve the balanced task. But remembering one location is sufficient to detect a full swap: high accuracy is not proof that two independent items were stored.

Binding uses different rendering, change sizes and query content from native full-field single-item orientation. A 99% binding score and 80% single-item score therefore do not constitute an inverted memory-load curve. New held-out centers test interpolation within the existing spatial layout, not unrestricted extrapolation. Four-case balanced blocks share nuisance structure and are clustered in uncertainty calculations.

## Metrics

For Kclasses and confusion matrix C with true classes in rows,

$$\mathrm{BA}=\frac1K\sum_{k=1}^{K}\frac{C_{kk}}{\sum_j C_{kj}}.$$

BA equals ordinary accuracy only when class counts are balanced. Motion chance is 25%; binary task chance is 50%. AUC chance is.5 in both. Macro one-versus-rest AUC averages each class's score-ranking ability against the rest; it does not equal the cross-class comparison used by argmax. A model can rank down examples reasonably but never make down the largest score.

A percentage-point change is an absolute accuracy difference:79.3%−70.3%=9.0 pp, not 9% relative improvement. Confidence intervals are conditional on the particular fitted models and data-generation distributions. Most runs have one training seed per arm; repeated episodes and multiple checkpoints are not independent seed replications.

Paired comparisons evaluate the same underlying examples in both conditions. Resample the underlying unit together: procedural base episodes, binding four-case blocks, canonical motion templates when applicable, or natural source photos. A fresh crop of an old photo is not a new source image. A new test seed makes a new draw, not automatically a wholly new population.

## Selection and evidence strength

- **Training:** examples used for gradient updates.
- **Validation:** planned checkpoint selection or probe/calibration choices; repeated looks make it a development set.
- **Held-out test:** separate examples scored after choices are fixed. Later reuse for exploratory diagnosis must be disclosed.
- **Probe fitting:** an analysis model trained on frozen representations; it does not change deployed weights unless a later explicit experiment does so.
- **Selected checkpoint:** weights chosen by the declared validation rule. **Terminal checkpoint:** final trained weights after the full allocation. They need not be the same.

Different experiments often reevaluate the same parent on different fresh draws. For example, parent motion percentages around 69–70% in successive studies are not contradictions. Use the same-experiment paired comparison; never subtract cross-run scores and attach a paired interval.

State reset, attention exclusion and feedback interruption are acute interventions outside ordinary training. A rescue implicates the altered computation but does not prove a trainable permanent change will help. A failed readout is not proof of absent information. Perfect empirical bootstrap intervals can collapse at 100%; they do not guarantee zero population error.

Current Stage1 uses fresh validation 53973001 and test 54973001. Its 2 pp validation screen is an engineering preservation rule, not proof of equivalence. Final simultaneous one-sided bounds provide a different, explicitly statistical assessment; uncertainty may remain with 512 examples per cell.
