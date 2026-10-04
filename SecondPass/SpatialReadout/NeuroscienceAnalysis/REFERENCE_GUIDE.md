# Fresh-ConvGRU neuroscience analysis: reference and figure guide

**Scope:** proposed interpretation/checklist, not new results. No model evaluation, calibration or training was performed for this guide. The delegated analysis fixes checkpoint **35039**; its reported solved-task scope is seven sensory tasks, ring orientation, signed-cue orientation and spatial binding. Treat that as the supplied selection, not an independently verified fresh test. Exclude partial image recognition from “solved.”

## What carries over—and what does not

The current `../model.py` inherits the stack-three CNN and 25×25/13×13/7×7 local KDA fields, but replaces the old dense temporal GRU with a **7×7 ConvGRU**, compressing only afterward. There is no native spatial softmax. Downstream convolutions and GroupNorm spread/couple effects; a localized intervention is not an isolated retinal lesion. ConvGRU provides an additional memory route, so a null KDA-emission effect cannot establish absence of memory at that site.

Morgan et al. use **any-change detection** with probabilistic cues: an uncued change remains a legitimate hit (§2.1, pp. 2–3). Their Figure 3 (p. 4) plots response probability and trial-ending time across cue validity; Figure 4 (p. 5) shows allocation across space, time and change magnitude.[1] Here signed orientation reports **target rotation aligned with cue sign**; zero and opposite-sign target rotations are negative (`WorkingMemory/SpatialTaskBattery/stimuli.py:85–91`). Binding instead reports target exchange, with foil-only exchange negative; its retrocue arrives **after retention** (lines 92–96). Do not pool these report rules or interpret pre-retrocue target selectivity as anticipation of an unavailable instruction.

Figure 5 (p. 8) forces a selected Morgan spatial bias to one: blue targets S1 in A/B/D/E but S4 in C/F. Bias toward the **changing**, not invariably cued, location helps. Exact overwrite/renormalization details are unspecified; direct sensory residuals remain (Eqs. 8, 15). KDA retention α=1 is not this operation. Section 4.3's sensitivity/criterion and cue-time claims refer to unfinished supplemental references absent from pinned v1; do not fabricate their curves.[1]

**Old graphics inspected:** `../../KDA_paper/figures/11_psychometric_magnitude.png` is balanced accuracy versus unsigned magnitude, not positive-response probability. Its below-range fitted midpoints and ceiling bootstrap intervals cannot supply new detection thresholds. `22_attention_profiles_scale0.png` is final-read **absolute coefficient versus source frame**, averaged over regions/heads—not attention evolving at successive read times. Cued and uncued traces largely overlap. Its amber frames disclose stack leakage. `build_figures.py:220–227,296–310` confirms these definitions; missing spatial arrays cannot be reconstructed from regional means. Reuse presentation clarity, not historical measurements or conclusions.

## Figure checklist

### 1. Competence and selective evidence use

- **A: real matched trials.** Show frames, locations, cue sign, labels and checkpoint identity. Preserve pre-overlay sensory rasters when exchanging cue identity/sign; recompute labels. Sign reversal leaves unchanged-target negatives negative. Report both-members-correct and task-aligned logit/probability changes, not merely output changes.
- **B: response psychometrics.** Plot empirical fraction reporting positive against signed cue-relative target rotation; distinguish that fraction from mean model probability. Independently vary foil evidence and condition curves on it. Retarget the same physical change from relevant to irrelevant while preserving other sensory evidence. The latter curve measures foil-driven false reports, not a second detection threshold. Separately report unchanged-target and opposite-target responses.
- **C: selection contrast.** Estimate relevant-versus-irrelevant evidence weights and their interaction with cue assignment, with paired uncertainty. Native generators couple rotation multisets: independent evidence sweeps alter joint frequencies; label them controlled counterfactuals, not unchanged-distribution samples. Contrast/jitter sweeps establish cue robustness, not validity-dependent attentional benefits. Classical valid/neutral/invalid comparisons remain unavailable without an independent report instruction.
- **D: task-specific controls.** Binding requires target-versus-foil swaps preserving item inventory; signed-orientation labels do not transfer. Sensory tasks can establish feature competence, not spatial selection by themselves. This final-frame classifier has **no measured reaction-time policy**; confidence, frame count and runtime are not chronometrics.

### 2. Spatial allocation and retention

Show individual stimulus sequences beside actual per-update maps, then cue-aligned averages. Include no change, target change, foil-only change and competing changes where the generator permits them. Separate write/retention gates, signed implicit coefficients, absolute coefficients, emitted-feature magnitude and behavioral effect maps. Label layer, head, source/read time, mask, coordinates and common color scale; use equal-support per-site ROI means.

Ask whether cue counterfactuals alter maps before probe, whether differences survive genuinely blank updates, and whether competing-change magnitude predicts allocation **and evidence use**. These are hypotheses, not guaranteed Morgan-like patterns. Increased activation alone is neither selection nor final-decision attribution. The first two nominal blanks still contain samples in stack-three inputs; “late retention” must follow their removal. Zero-delay trials cannot provide this contrast. A ceiling delay curve identifies neither lossless storage nor a forgetting time constant.

### 3. Local inhibition: necessity with specificity

Primary substrate: **32-channel emitted field at 25×25**, after KDA output projection and before concatenation. Specify `O′=(1−γM)O`, zero as reference, bounded γ, smooth mask M and exact update window. Leave other scales unmodified directly; downstream consequences are expected. This suppresses emitted content, **not directly that module's stored state**.

Pair sham, cued, foil and matched-background sites on identical trials; counterbalance physical positions. Match mask support/edge truncation and report both fractional suppression and achieved perturbation RMS, since equal γ need not remove equal energy. Separate encoding, late blank and probe windows, recording pulse duration. For binding, encoding “eventual-target” effects are retrospective groupings, not pre-cue attention.

Plot complete response curves, catch/foil/opposite errors and site×epoch×dose contrasts. A selective target impairment beyond matched controls supports causal use of that local representation; broad degradation supports nonspecific disruption. Foil suppression need not help. Neither outcome identifies a finite memory capacity or exclusive storage locus.

### 4. Microstimulation analogue: content, not arbitrary positivity

Use the same sites/epochs with `O′=O+λMσd`. Freeze d and σ from **independent calibration**. Define d to have channel RMS one and σ as ordinary calibration activation RMS, making λ interpretable at full mask amplitude. Include sham, opposite direction and matched-norm random-direction pulses; retain achieved RMS and signed feature projection. Do not equate frames/model units with milliseconds/current or call the pulse Morgan's α=1.

A useful feature localizer must dissociate stimulus feature from cue, label and response. For orientation change, absolute orientation preference alone does not identify a rotation/comparison direction; disclose exactly what calibration varies. Test whether ±d produces predictable opposing choice shifts under cue/sign counterfactuals. A choice bias can demonstrate causal feature readout without improving sensitivity or attention. Spatial/temporal specificity exceeding random controls strengthens interpretation; dose-related global damage does not.

**Monkey comparison:** Cavanaugh, Alvarez & Wurtz tested SC stimulation against visual-cue, timing and superficial-layer alternatives; these controls argue against a mere phosphene explanation.[2] Their Figure 2A's change-time protocol motivates timed perturbations, not biological dose equivalence. Salzman, Britten & Newsome's MT study instead biased motion judgments toward independently characterized neuronal preferred direction (Nature 346:174–177; primary abstract verified).[4] That is the stronger precedent for **feature-direction** pulses. The solved sensory `motion_direction` task offers a feature-readout analogue, not automatically spatial cueing or the unsolved cued-duration task. A stronger joint MT/attention analogue needs demonstrated spatially cued motion competence; do not silently add it to solved-task claims.

## Statistical and claim gate

Use Wilson/exact binomial intervals for observed response rates, including ceilings; resample independent scene IDs with all matched variants kept together for contrasts. Report denominators. Keep sensitivity and criterion separate, with declared signal/noise classes and finite-rate corrections; do not pool opposite-target, foil-only and catch negatives invisibly. Fit thresholds/lapses only if sampled evidence identifies them. Trial uncertainty is not across-checkpoint generality.

Accept a **causal functional-selection signature** only when physically matched evidence use, cue dependence and controlled intervention specificity agree. Competence, feature decodability, capacity, internal correlation and neuroanatomical homology remain different claims. Nulls may reflect ceiling, ineffective direction, bypass pathways or insufficient power.

**Source-access boundary:** pinned Morgan PDF and existing page extraction/render were checked; two prior atlas images were inspected directly. Web search/extraction and browser access failed. Europe PMC primary bibliographic records/abstracts for the biological studies were retrieved directly; SC full-text/Figure 2 timing is supported by the repository's earlier `TechnicalReport/paper_evidence/verified_quotes.json`, not a newly recovered full text. No biological effect sizes were inferred.

## Sources

[1] https://arxiv.org/pdf/2502.10955v1
[2] https://www.jneurosci.org/content/26/44/11347
[4] https://doi.org/10.1038/346174a0
