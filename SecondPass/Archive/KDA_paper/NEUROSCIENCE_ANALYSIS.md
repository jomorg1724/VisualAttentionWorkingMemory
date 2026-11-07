# KDA: cueing, selective attention, inhibition and microstimulation

> The repository-wide, paper-grounded protocol is now [ANALYSIS_SOP.md](../../ANALYSIS_SOP.md). Follow that SOP for subsequent analysis; this file is the earlier KDA-specific scoping note.

## Scientific question

Does the observer preferentially use changes at the cued location, reject changes at uncued locations, and change that allocation predictably when a spatially localized model population is inhibited or stimulated?

The main presentation is behavioral psychophysics and causal spatial interventions, not a catalogue of tensor statistics. Architecture, optimizer curves and compute belong in supporting material.

## Figure 1 — Behavioral selection and cueing

Use physically matched displays and preserve the task's existing sign-glyph/report rule. Keep the observer frozen. Counterbalance locations and cue sign.

- Plot the probability of an aligned-change response against signed orientation change at the cued location. Separate or condition on changes at uncued locations. Show trial-level uncertainty, detection sensitivity and decision bias separately.
- For a change at the same physical location, compare trials where that location is cued versus uncued; keep the non-cued target unchanged where appropriate. Label the first curve as target detection and the second as distractor-driven responses/false alarms. Do not interpret the latter as an ordinary second target-detection threshold.
- Independently vary relevant and irrelevant change evidence to estimate behavioral weighting of cued and uncued changes. Report thresholds/slope for supported target psychometrics, false alarms to uncued changes, and lapse/bias only where identified by the data.
- Preserve matched trial identities for paired contrasts. A cue-contrast or cue-jitter sweep measures robustness to cue visibility/location, not by itself an attentional cueing benefit.
- Do not introduce 'valid/neutral/invalid cue' labels casually: this task's glyph also defines the report location and sign. Any such comparison needs an independently defined report target and an explicit task protocol; otherwise removal or relocation changes/obscures what response is requested.

## Figure 2 — Allocation to cued and uncued changes

Show individual trials with the actual image sequence and per-timestep spatial measurements, then cue-aligned population summaries.

- Split trials by cued change, uncued-only change, both changes and no change; retain sign, magnitude, correctness and report time.
- At every spatial scale, compare cued and uncued locations during cue, sample, blank and changed-probe periods. Keep heads distinct where the stored data allow it.
- Show KDA write/retention gates, signed and absolute implicit temporal coefficients, and local feature responses with their actual definitions. Region averages are supplementary, not substitutes for individual spatial maps.
- Distinguish the time being read from the earlier source time of a memory coefficient. KDA coefficients describe local memory reads, not native transformer-style query-patch-to-key-patch attention probabilities.
- Pair these internal measures with behavioral sensitivity to relevant versus irrelevant evidence. A larger activation or coefficient alone does not establish selective attention or causal relevance to the decision.

## Figure 3 — Spatial inhibition

Compare unperturbed/sham trials with transient local suppression at the cued site, an uncued site and matched background sites, on identical stimulus sequences.

Primary activity-perturbation site: the emitted 32-channel KDA field at a single specified scale, before concatenation into downstream features. Multiply activity inside a smooth spatial mask by a suppression factor. This is a model analogue of local activity suppression, not a biophysical inhibitory circuit.

- Start with the finest spatial scale; keep other scales untouched in the primary contrast.
- Test separately during encoding, late retention and probe processing. The first two nominal blank frames still contain sample images in the rolling three-frame input; do not label them pure memory maintenance.
- Plot baseline versus suppression psychometric functions, threshold/bias changes, distractor-driven false alarms and the spatial distribution of behavioral effects.
- Sweep a bounded range of suppression strengths and normalize spatial masks so off-target controls receive matched intervention support.
- An uncued-site intervention might improve, worsen or leave behavior unchanged. Do not draw the expected sign as a result.
- A whole-model KDA-state reset is NOT this experiment and must not be relabeled spatial inhibition.

## Figure 4 — Spatial microstimulation analogue

Apply a brief, calibrated additive pulse to the same local activity field, with sham and equal-magnitude off-target controls. Counterbalance stimulated location versus cue location and hold the actual stimuli fixed.

- A pulse requires a channel direction as well as a position. For feature-specific stimulation, identify feature-preference directions from an independent localizer or calibration split, not the scored trials. Include matched untuned/random-direction controls.
- Express dose relative to ordinary local activation scale and document pulse duration and spatial extent. Do not equate model units/frames with current or milliseconds.
- Plot shifts in the response psychometric function, sensitivity and criterion/bias separately; include no-change catch trials and distractor-driven false alarms.
- Test whether stimulation at a cued location differs from an equally stimulated uncued location, and whether the effect depends on encoding, retention or decision timing.
- An effect on a response is not automatically enhanced attention; a decision bias, corrupted representation or nonspecific disruption remains possible.

## Figure 5 — Causal allocation summary

Use paired trial-level differences to map inhibition and stimulation effects across space and time. Show target and foil effects on the same behavioral scale and report uncertainty over independent trial identities. Preserve within-trial repeated intervention conditions in resampling. Do not create trial counts by treating pixels, frames or heads as independent observations.

All predictions and interval estimates must come from real frozen-model execution. No training or paid/cloud provisioning is implied by this analysis brief.

## Available evidence versus missing measurements

Available in the current checkout:
- Completed local orientation-training results.
- Aggregate magnitude, delay, glyph-contrast/jitter and distractor-count psychometrics.
- Cued/uncued/background means of gates and implicit temporal coefficients.
- Whole-KDA-state reset and blank-gate clamps.

Not available in the checkout:
- Trial-resolved cueing psychometrics and change-conditioned attention allocation.
- Spatial inhibition or microstimulation results.
- Per-trial spatial-map NPZ files or trained model weights.

The existing saved global reset is neither a spatial inhibition experiment nor microstimulation. It cannot supply Figure 3 or 4. The current atlas is supporting descriptive material, not evidence that these causal comparisons have been completed.

## Required artifact to execute

The recorded checkpoint is:
`WorkingMemory/PlainBaseline/runs/local_kda_program_20260917/kda_s1/delayC/terminal.pt`

The originating run was on Windows; records name the repository root `C:\Users\jomor\Documents\VisualAttentionWorkingMemory`. Obtain the actual trained checkpoint (or an accessible source path), verify its recorded identity and model configuration, then run frozen evaluation without changing weights. Do not substitute a randomly initialized or newly trained model.

A concurrently prepared `SecondPass/TaskSuite/` is a separate future joint-training task suite, not a trained replacement for this checkpoint. Do not alter its work to manufacture these results.
