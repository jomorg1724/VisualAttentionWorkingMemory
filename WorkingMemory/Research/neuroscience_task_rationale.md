# From sensory accumulation to working-memory demands

2026-09-12. **Design only.** No implementation or compute is authorized by this note. Scope follows the latest user correction: develop the successful opponent temporal model; set KDA and ConvGRU aside. This note supplies neuroscience rationale and interpretive controls. The parent design owns the exact seven-domain generators, labels and finite training/evaluation proposal.

## What we should ask next

The useful question is what the existing sensory system can learn when a task requires selecting, retaining and later using information. It is premature to conclude that we need a separate working-memory module, and it would be equally premature to label the existing fixed traces a general memory solution.

Use four related modules, sharing the seven established sensory domains where the task meaning remains clear. Avoid a full factorial crossing of every duration, cue, load and distractor type.

| Module | Crucial manipulation | What success would establish |
|---|---|---|
| Sensory/cue bridge | Train the new visible cue vocabulary and episode format with minimal retention demand | New stimuli, rules and readout are learnable |
| Selective temporal integration | Cue which evidence counts; distribute that evidence over segments | Rule-dependent accumulation over the observed interval |
| Delayed report / interference | Insert actual blank or irrelevant-image updates before the report | Retention of a learned decision or statistic despite delay/input |
| Retrospective feature probe / item load | Reveal the queried item and comparison only after encoding | Availability of item-bound sensory information, beyond a precomputed choice |

Warm-start the winning trained opponent model, then allow **all learned weights**, including the encoder and new cue/readout weights, to adapt. Keep its fixed quadrature kernels and retention coefficients fixed in this phase. This is supervised learning of new tasks and cue meanings, not a zero-shot examination of an encoder trained without those meanings. A frozen old checkpoint can remain a descriptive baseline, but failure of that baseline is not the main scientific result. No scratch restart is required merely because the task is new.

## 1. Sensory bridge: give new behavior a fair starting point

Visible context, item identity and report cues must have an explicit trained meaning. A colored frame, symbol or spatial marker is an input pattern, not an instruction a CNN already understands. Test the new formats at short duration and minimal delay during the same training program. Hold the symbol vocabulary constant initially; reserve novel cue styles or unseen object identities for an explicitly labeled transfer question.

The seven old scores establish competence on their old rendered distribution. Several changes could otherwise be mistaken for memory limits: shrinking Gabors to fit multiple items, rendering new identity markers over diagnostic pixels, changing contrast ranges, adding colored cue borders, or asking for an unfamiliar output label. The bridge should contain the same rendering, identities, probes and labels used in its harder counterpart, with only the intended retention demand reduced.

## 2. Selective integration: what the proposed examples really demand

Mante and colleagues trained monkeys to use a context cue to select motion or color evidence and integrate the relevant evidence toward a choice. Their neural and recurrent-network analysis motivates asking whether a common system can perform context-dependent selection and integration. Our signed-orientation and duration protocols would be adaptations of that functional question, not replications of the original task or its cortical mechanisms. [Mante et al., 2013, primary paper](https://www.nature.com/articles/nature12742).

For the user's motion example, the target statistic is total displayed duration per direction:

\[
C_d=\sum_j \Delta t_j\,\mathbf 1[d_j=d],\qquad
y=\arg\max_d C_d.
\]

With fixed frame intervals, duration can be measured by observed motion intervals. A static first frame does not itself specify a direction; the generator must define which subsequent displacement interval contributes to which segment. Segment reinitialization must not create an unlabelled jump that supplies a spurious motion direction. Tie handling is part of the task definition.

This task can be solved with four counters, without retaining every segment or an image. That is a useful accumulation capability. It is not a multi-item working-memory capacity measurement. Expose cases in which the total-duration winner differs from the last direction, the most frequent segment identity and the direction of the single longest segment. Match the final visible frame/direction and overall sequence duration across labels where feasible, so neither the endpoint nor a clock becomes a sufficient classifier.

For orientation, let the cue select a permitted rotation sign or subset of evidence. If the objective is total aligned angular change, a representative statistic is:

\[
A_c=\sum_j \mathbf 1[\operatorname{sgn}(\delta_j)=c]\,|\delta_j|.
\]

This is one gated sum for a precued sign, or two sums if the requested sign is revealed later. It does not require item-rich memory. For unoriented Gabors, orientation is modulo pi: signed increments need a declared unambiguous range, and total aligned rotation generally cannot be recovered by subtracting only the first and last orientations. The cue alone must not determine the response; the final response should depend on a comparison or threshold whose classes genuinely vary for each cue.

Carry the same distinction into the other sensory domains: retaining an accumulated scalar or a categorical segment count is computationally different from retaining the underlying contrast, frequency, color, contour structure or natural-image feature. Independent timing and content draws should prevent irrelevant evidence magnitude, total brightness or number of updates from determining the label.

A focused report should separate accuracy against total relevant evidence, recency, irrelevant evidence and evidence margin. The goal is to understand which statistic the trained model uses, not merely increase the difficulty until its score falls.

## 3. Delay and interference: retention of what?

A blank delay must advance the network through real blank-image inputs at the declared sampling interval. Merely pausing a Python loop leaves a discrete state unchanged and tests no retention dynamics. Distractors likewise enter through the normal image path; do not reset the model or selectively skip updates during their presentation. The report cue is visible and its timing is independent of the correct response.

Classic delayed-response work separates a brief visual event from its later report. Funahashi and colleagues used an oculomotor task and observed location-dependent delay activity in prefrontal cortex. This grounds a delayed-report manipulation; it does not establish that an artificial classifier with delay performance has the same neural mechanism. [Funahashi, Bruce and Goldman-Rakic, 1989](https://journals.physiology.org/doi/10.1152/jn.1989.61.2.331).

Delayed matching studies also motivate testing intervening visual input rather than relying exclusively on blanks. Miller, Erickson and Desimone studied remembered samples and intervening test stimuli, including prefrontal responses that retained sample information through interference. We borrow the requirement to preserve a relevant sample while processing other input. [Miller et al., 1996, author-hosted paper](https://ekmillerlab.mit.edu/wp-content/uploads/2013/03/Miller-et-al-1996.pdf).

Keep one basic delay axis and one interference comparison. Pair a blank interval with a duration-matched interval containing irrelevant sensory stimuli. Avoid interpreting worse performance under distractors as pure memory decay: masking, attentional capture and difficulty learning the ignore rule remain alternative explanations. Include the corresponding short-delay visible-cue condition so the rule and sensory burden are already exercised.

If the answer is determined before the delay, a model can store a choice bit or one of four categories. Name that result **decision retention**. It is valuable, but does not show the model retained the sensory evidence used to decide. For stronger content retention, the probe must introduce a comparison that cannot be decided before the delay.

## 4. Retrospective probe and item binding: the critical distinction

Griffin and Nobre used cues directed to locations in internal representations after stimulus presentation. Their findings motivate contrasting a precue, which permits selective encoding, with a retrocue, which selects from previously encoded information. Our cue timing, content and response requirements must be disclosed as adaptations. [Griffin and Nobre, 2003, institutional record and abstract](https://ora.ox.ac.uk/objects/uuid%3Aee268292-114c-46f8-b11b-416eafa5823b).

Start with a two-item contrast rather than a presumed four-slot limit. Give two items distinguishable, trained identity/location markers, independently sampled feature values, then remove them. After the first delay, reveal which item is relevant; after an optional matched second delay, present a new comparison probe. Both identity and probe value must remain unknown during encoding in the retrospective condition.

For example, store the orientations of items A and B; later cue B and ask whether a newly displayed probe is clockwise or counterclockwise from B's earlier orientation. Or store item-bound color values and later compare the cued item's color with a new increment/decrement probe. A model cannot precompute the final binary response when it sees the samples, because the probe is not yet known. Distractor-item values should sometimes imply the opposite correct response, permitting swap/binding errors to be distinguished from generic inaccuracy.

The broader seven-domain adaptation should preserve that logic. A stored motion **direction category** is a legitimate remembered feature but is still coarser than a continuous sensory value. A contour-presence bit is not the same as retaining which contour structure belonged to which item. For natural-image spectral comparisons, identity must remain explicit while the probe's spectral parameter is unknown at encoding. If the protocol uses independent crops within one source identity, disclose the added invariance demand and establish the same crop relation in the sensory bridge; a base-photo shortcut or an ambiguous cross-crop comparison could otherwise dominate the memory result.

Use precue versus retrocue with matched total delays to ask whether retaining both items before selection is harder than selecting one at encoding. A cue delivered at the instant of report is a postcue, not evidence that the model used a preceding interval to prioritize an internal representation. Do not inflate this initial comparison into a full capacity curve across every domain.

Bays and Husain measured memory precision and its allocation with changes in item load, illustrating why capacity should not be assumed to equal a fixed four-item threshold. Our two-item probe is an operational test of binding and retention, not an estimate of a universal human slot count. [Bays and Husain, 2008, primary manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC2532743/).

## Interpret the actual opponent architecture

The current implementation has two spatial traces per scale, with fixed .25 and .75 retention, followed by fixed energy computations and learned output/readout layers. The learned encoder processes only the current image; the accumulator does not feed state back into it. Training all learned weights can improve sensory/cue representations and readout, but does not turn these fixed trace coefficients into learned write/protect gates.

For two histories followed by the same sequence of blank or distractor images, their state differences obey:

\[
\delta F_D=.25^D\delta F_0,\qquad
\delta L_D=.75^D\delta L_0.
\]

This follows directly from the implemented recurrence, independently of how the common current images are encoded. The slow trace retains approximately 1% of a history difference after 16 identical subsequent updates, and about 0.01% after 32. These are attenuation factors, not predicted accuracy or biological time constants. A nonlinear readout may exploit small residual differences, and numerical precision and distractor statistics matter.

Consequently, a delay deficit could reveal the imposed leakage law rather than a lack of learned sensory features. Strong near-zero-delay performance followed by systematic delay loss would justify considering a more protective update mechanism. It would **not** establish that a separate anatomical-style working-memory module is necessary: changing the update rule inside a temporal component and adding a distinct memory component are different candidate explanations to test later.

## Inventory information routes before interpreting performance

- The core may receive pixels and explicitly observable cues only. Domain identity may select the appropriate output vocabulary, but cue sign, item identity, relevant segment, phase clock, latent duration totals and correct labels must not arrive through a privileged head selector or state gate.
- The current-frame bypass is legitimate; ensure the probe image, final frame, response layout and cue cannot answer the task without stored content. Match their marginal distributions across the relevant labels rather than requiring chance by assertion.
- Do not retain sample images, old logits, an external cue embedding, a previous decision or a hand-coded running sum outside the declared state. Those would be additional memory routes, even if called preprocessing.
- A generator-level exact statistic is useful for checking labels. A perfect oracle statistic supplied to a network is an upper bound with privileged input, not the performance of the visual model.
- Reset only between complete episodes. Labels used in supervised loss must never enter recurrent updates. Repeated frames, shared images, cue repetitions and delay checkpoints are not additional independent examples.
- New-context failure, underexposure, long-gradient optimization difficulty, crowding and probe ambiguity are alternatives to a storage limit. Use the sensory bridge and matched conditions to locate the failure before changing architecture.

The intended outcome is a map of capabilities: which sensory computations can be selectively accumulated, which derived decisions survive interference, and which item-bound values remain accessible to a later unknown query. That map provides evidence for the next component decision without predetermining it.
