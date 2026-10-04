# SOP: neuroscience analysis of visual attention and working memory

**Read this before analyzing the next trained model.** Treat the model as an experimental observer. Start with cueing psychophysics, allocation to cued versus uncued changes, and causal effects of inhibition and microstimulation. Architecture diagnostics support those questions; they must not replace them.

## Reference and scope

This protocol follows the experimental logic of **Jonathan Morgan, Badr Albanna, and James P. Herman, _A recurrent vision transformer shows signatures of primate visual attention_**, arXiv:2502.10955v1, February 16, 2025 [1–3]. The pinned PDF was downloaded and checked directly. It is also available on bioRxiv [4].

### What the reference paper actually establishes

| Source | Experiment or analysis | Protocol implication |
|---|---|---|
| Section 2.1; Figure 1 | Four Gabor locations; cue validity 25%, 50%, 75%, or 100%; wait/declare-change actions. | Cue validity means probability of change at the cued location, conditional on a change trial. It is not cue contrast. The 25% condition is spatially uninformative across four sites. |
| Section 4.1; Figure 3 | Response probability and response time versus change magnitude; comparisons across validity and between cued and uncued changes. | These are the primary cueing-effect panels. Good overall accuracy alone does not establish attention. |
| Section 4.2; Figure 4 | No-change spatial attention maps over time; attention versus change magnitude at cued and uncued sites; cued/uncued allocation time courses. | Show anticipatory allocation, persistence after the cue, and capture by changes at competing locations. |
| Section 4.3; Figure 5 | Artificial spatial bias toward S1 or S4; response-rate and response-time consequences; comparison of cue-time and change-time manipulation. | Use location-specific, timed perturbations with behavioral readouts. Figure 5 describes forcing the selected attention bias to one, not a calibrated biological current. |
| Section 4.3 | Sensitivity and criterion can both change under the same manipulation. | Report both; do not equate an effect on response probability with an isolated sensory improvement. |
| Section 4.4; Table 1 | Some alternative models perform the task without the same cueing effect or attention dynamics. | Task competence, cueing, and causal attention signatures are separate acceptance criteria. |

**Reference-task semantics matter:** in this paper, a change at an uncued location is still a real change that can be correctly detected. It is **not a false alarm** merely because it was uncued. A false alarm is an inappropriate change declaration, such as on a no-change trial or before the change. Some repository tasks instead require a report about a designated target and rejection of changes elsewhere. State which task is being analyzed before naming response categories.

The paper's artificial-bias result motivates the microstimulation analogue. **Spatial inhibition is an additional required analysis in this SOP, not a claim that the pinned paper contains a separately verified inhibition experiment.** Do not invent a pre-softmax ±50 manipulation, an inhibition figure, or implementation details not established by the source or code.

## 1. Questions and figure order

The results must answer, in this order:

1. **Cueing:** is a matched change detected differently when cued versus uncued, and does this effect scale with cue validity?
2. **Allocation:** how does attention shift between cued and uncued locations before and after changes of different magnitudes?
3. **Persistence:** does the cue's influence survive a stimulus-free interval, and how does delay affect behavior?
4. **Inhibition:** what changes when a specific spatial representation is suppressed at a specified time?
5. **Microstimulation:** what changes when processing is biased toward a specific spatial representation at a specified time?
6. **Mechanism:** which measured internal processes support those effects, and which explanations remain untested?

Training curves, tensor statistics, gates, state decoding, and global resets are supporting material. Do not begin with an exhaustive atlas of whatever was easiest to export.

## 2. Prerequisites and data to retain

Before evaluation, record the checkpoint identity/hash, model configuration, report rule, training exposure, cue semantics, stimulus geometry, delay and change ranges, and random seeds.

- **Inspect the report rule.** Does any change count, or only a target change? Does the cue encode location, validity, response sign, or multiple things?
- **Separate cue validity from cue visibility.** Contrast and jitter are robustness manipulations, not measurements of the classical validity-dependent cueing effect.
- **Establish supported conditions.** If the model was not trained with different validities, or no uninformative baseline is defined, say so. A counterfactual test may still be informative, but label its distribution shift. The reference paper explicitly tested uncued changes under a 100%-valid cue despite their absence during training; that does not erase the distinction between familiar and counterfactual conditions.
- **Do not silently change the task or retrain.** Keep the report meaningful when cues are manipulated. Propose required task changes explicitly.
- **Use matched scenes and random seeds across comparisons where possible.** Keep a shared trial ID; recompute correct labels under the declared task rule rather than blindly copying them across tasks.
- **Keep trial-level data:** scene/seed, cue location/type/validity/sign, all Gabor locations and signed rotations, change time and location, true report, response score, selected action at each available decision time, correctness, and trial-ending time.
- **Keep raw spatial arrays:** actual attention/readout coefficients, signs, gates, ROI masks, stimulus frames, layer/head/time identities, and coordinate transforms. Regional averages cannot reconstruct maps.
- **Keep perturbation records:** site, mask, layer/scale, affected representation/channels, operation, onset, duration, dose, achieved local effect, and sham condition.
- **Retain the trained weights or a durable accessible pointer.** Put scripts, metadata, condition tables, and artifact manifests in Git. Document where excluded large files live.

Inventory actual artifacts before interpreting them. A script that could generate a map does not mean the map was recorded.

## 3. Figure 1 — Behavioral cueing effect

### Required panels

**A. Trial and task schematic.** Show the cue, samples, delays, change, action opportunities, and correct report. Clearly distinguish a probabilistic spatial cue from an instruction defining the only relevant target.

**B. Psychometric curves.** Plot probability of declaring a change against change magnitude, split by:

- change at the cued versus a matched uncued location;
- cue validity, including an uninformative baseline when available;
- other prespecified sensory conditions that materially affect difficulty.

In the reference design the validity levels are 25%, 50%, 75%, and 100%; do not impose those levels on a different task without implementing and documenting them. Use measured response probabilities, not balanced accuracy mislabeled as a detection rate.

**C. Cueing effect.** Plot the cued-minus-uncued response difference at matched magnitudes and its dependence on validity. Include threshold shifts only if the response curves and sampled range support the fit. Distinguish threshold, slope, lapse, and response bias.

**D. Chronometric curves, when meaningful.** Plot response/trial-ending time against magnitude for the same conditions. Declare whether the statistic includes misses or nonresponses and how censoring is treated. The reference Figure 3 averages trial-ending times including waiting through the final timestep. A fixed-probe classifier has no measured reaction time; do not convert its confidence or runtime into one.

**E. Controls.** Show no-change false alarms and premature responses. For target-report tasks, separately show distractor-only false reports. For any-change detection, an uncued-change report is a hit, not a false alarm.

A perfect aggregate accuracy does not replace these panels. If the design cannot support the contrast, label it **not yet measured**, rather than substituting cue degradation.

## 4. Figure 2 — Allocation to cued and uncued changes

Use the same conditions as the behavioral analysis.

**A. Spatial maps over time.** Show actual measured maps aligned to stimulus positions, beginning with no-change trials as in reference Figure 4A. Separate cue validity conditions and mark cue onset, cue offset, stimulus onset, pre-change, change, and post-change epochs. Show representative trials alongside correctly labeled trial averages when available.

**B. Allocation versus change magnitude.** Plot attention to both the cued and uncued ROIs as a function of magnitude, separately when the change occurs at the cued location and when it occurs at the uncued location. This directly tests competition between predictive cue information and capture by a large uncued change.

**C. Allocation time courses.** Compare cued and uncued ROIs through the trial. Distinguish anticipatory selection before a change from change-driven capture after it. Include matched background where meaningful.

**D. Behavior–allocation relationship.** Compare allocation with the probability of reporting the same cued or uncued changes. Use held-out analysis if fitting an evidence-use or choice model. An internal coefficient is a correlate, not automatically final-decision attribution or causal importance.

Every map/caption must identify the quantity, coordinates, query/source convention, layer/head, time, averaging, normalization, and color scale. State whether ROI summaries are per-pixel means or regional totals. Do not use unequal ROI sizes to manufacture allocation differences.

For KDA, distinguish:

- **spatial gating maps**;
- **spatial maps of implicit readout coefficients from specified source frames**;
- **temporal profiles averaged within regions**;
- **spatial maps of behavioral intervention effects**.

These are different measurements. KDA implicit coefficients can be signed and are not normalized transformer attention probabilities. Preserve signed arrays; explicitly label absolute-magnitude summaries. Do not manufacture a spatial image from regional means.

## 5. Figure 3 — Retention of selection

Repeat key behavioral and allocation contrasts across delays, rather than showing only an accuracy ceiling. Ask whether predictive spatial allocation persists or is reactivated, whether it still benefits cued changes, and whether uncued changes capture processing differently after longer delays.

For stacked-frame inputs, mark early nominal blanks that still contain previous sample images in the input stack. Claims about recurrent retention require genuinely stimulus-free updates. Decode pre-probe state separately from state after new probe evidence enters.

State decoding may support the story, with held-out evaluation, feature counts, regularization, uncertainty, and controls. It is not a substitute for the cueing effect or evidence of exclusive memory localization. A flat ceiling curve does not identify a memory-decay time constant.

## 6. Figure 4 — Spatial inhibition

### Experiment

Compare sham with localized suppression at the cued site, a matched uncued site, and an off-target/control site. Vary onset (cue, sample/encoding, genuinely blank retention, change/probe), duration, and graded strength. Use the same underlying trials across conditions when possible.

### Required readouts

1. A spatial schematic and verification that the intended local representation was suppressed.
2. Baseline/sham versus inhibition psychometric curves for cued and uncued changes, stratified by cue validity where supported.
3. Effects on hit probability, no-change false alarms, sensitivity, criterion, and response time when available.
4. Site × epoch × strength comparisons; a spatial intervention-effect map if enough sites were sampled.
5. In target-report tasks, whether uncued-site inhibition reduces distractor interference; in any-change detection, whether it impairs reports of legitimate uncued changes. Do not assume these tasks predict the same beneficial effect.

Suppressing a signed feature vector toward zero is a computational intervention, not automatically physiological inhibition. Define its reference state and show its effect. Global memory reset and sensory occlusion are distinct controls and cannot replace local internal suppression.

## 7. Figure 5 — Spatial microstimulation

### Experiment

Compare sham with localized biasing/excitation at the cued site, an uncued site, and a matched control site. Match the inhibition study's epochs, trial identities, and behavioral conditions. Include graded strength as well as an interpretable limiting manipulation where appropriate.

In the reference paper, Figure 5 describes forcing spatial attention bias toward one location (alpha at that location equals one). This reallocates processing competitively; it is not a literal current injection. Check the actual implementation before claiming a particular logit operation or numerical constant.

### Required readouts

1. A spatial schematic and measured verification of the achieved allocation/activity change.
2. Baseline/sham versus stimulation psychometric curves for changes at stimulated and unstimulated sites.
3. Effects on sensitivity and criterion, not just accuracy or mean confidence.
4. No-change false alarms; distractor-induced false reports only where the report rule makes that category meaningful.
5. Timing specificity, dose response, and spatial intervention-effect maps when measured.
6. Response-time effects if the observer has genuine action timing.

Do not assume stimulation improves performance. It may bias reports, improve detection at one site while worsening another, or disrupt processing. If a feature-specific channel direction is used, estimate it on independent calibration data and include matched pattern controls. A spatial coordinate alone does not specify stimulation content.

## 8. Choosing a KDA intervention substrate

Declare the scientific target and perturbation substrate before running:

| Substrate | What changes | What must not be claimed |
|---|---|---|
| Input sensory features | Incoming local evidence | Direct manipulation of stored attention or memory |
| Local KDA emitted/readout features | What downstream processing receives at that update | Direct erasure of the same module's stored state |
| Local persistent KDA state | Stored local representation | Isolated attention reallocation without memory-content disruption |
| An explicitly implemented local routing operation | Influence of specified source inputs on readout | Equivalence to softmax patch competition unless actually established |

State whether perturbation precedes or follows the update/readout and verify downstream propagation. Calibrate doses against the measured representation's scale. Preserve spatial masks and transformations across resolutions; overlapping receptive fields mean a feature-map cell is not an isolated input pixel.

The goal is to test comparable **functional hypotheses**, not pretend architectures expose identical neural variables.

## 9. Statistics and interpretation

- Use paired outcome uncertainty for matched conditions. Separate trial uncertainty from variability across independently trained checkpoints.
- Report denominators and trial counts for each plotted condition. Ordinary bootstrap intervals of [1,1] at a ceiling do not establish certainty; use appropriate finite-sample binomial intervals where applicable.
- Estimate sensitivity and criterion from explicitly matched hit/false-alarm definitions and disclose corrections for rates of zero or one.
- Fit psychometrics only where the sampled range constrains them; mark extrapolated thresholds and unidentifiable parameters.
- Keep negative out-of-sample decoding scores. Document regularization, train/test separation, and feature/sample ratios.
- Separate internal correlation, behavioral evidence use, and causal perturbation. The same manipulation can change both sensitivity and criterion.
- Use sham, site, timing, and dose controls to distinguish selective effects from nonspecific damage.
- Label exploratory analyses and distribution shifts. A result from one model is not a general property of all KDA models or biological attention.

## 10. Deliverables and completion checklist

Deliver a short neuroscience-style results narrative in the figure order above, a figure atlas, vector/raster exports, trial-level data or durable artifact pointers, numerical condition tables, and reproducible evaluation/plotting scripts. Captions must identify the question, conditions, sample size, uncertainty, and limit of interpretation.

Use **measured**, **not yet measured**, and **unsupported inference** consistently. A missing checkpoint, unsupported cue condition, or absent array is a missing measurement—not permission to substitute an easier diagnostic or fabricate a panel.

- [ ] Cued-versus-uncued psychometrics and cue-validity dependence are shown or explicitly unmeasured.
- [ ] Any-change detection and target-specific reporting are not conflated.
- [ ] Attention allocation is shown across space, time, change location, and magnitude where data support it.
- [ ] Inhibition has matched controls and behavioral readouts, or is explicitly unmeasured.
- [ ] Microstimulation has matched controls and behavioral readouts, or is explicitly unmeasured.
- [ ] Sensitivity, criterion, and genuine response timing are distinguished.
- [ ] Delay interpretation accounts for frame stacking and probe leakage.
- [ ] Every numerical result traces to saved experiment records rather than the reference paper's outcome.
- [ ] The presentation answers the scientific questions before discussing model diagnostics.

## Sources and local paper

[1] Morgan, J., Albanna, B., & Herman, J. P. (2025). *A recurrent vision transformer shows signatures of primate visual attention*. arXiv:2502.10955v1. https://arxiv.org/abs/2502.10955v1

[2] Pinned full text: https://arxiv.org/html/2502.10955v1

[3] Pinned PDF, directly checked for the task, Figures 3–5, and Sections 4.1–4.4: https://arxiv.org/pdf/2502.10955v1 . Local copy: [Morgan_Albanna_Herman_2025_Recurrent_ViT_arxiv_2502.10955v1.pdf](SecondPass/papers/pdf/Morgan_Albanna_Herman_2025_Recurrent_ViT_arxiv_2502.10955v1.pdf). The PDF directory is excluded from Git; the public URL is the portable reference.

[4] Related bioRxiv version: https://www.biorxiv.org/content/10.1101/2024.11.09.622721v2 . Use the pinned arXiv version above for the section and figure references in this SOP.
