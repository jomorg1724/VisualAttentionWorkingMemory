## Stability investigation completed — October6,2026

**Main finding:** the regressed LocalComparisonKrauzlis checkpoint has lost target-directed attention and concentrates on a fixation-adjacent patch. Replacing its attention trajectory with the good checkpoint's trajectory restores performance without changing its weights. Backward-only interventions localize severe gradient amplification to the coupled attention/memory feedback paths. This is a more specific diagnosis than generic optimizer instability, but does not identify the exact original parameter update that triggered the transition.

### Original run status

The original run did not finish: last live1737/52672presentations, savedlatest1700, best600. Original worker and supervisor are absent, no completion receipt or final-test file. The interruption cause is not established. Its original deadline was not renewed, and production was not restarted. Both original checkpoint SHA256 values were verified unchanged after this investigation.

### Independent validation of the good state

Frozen checkpoint600 reproduces its300-trial validation BA74.908%, AUC.853801. On600 new diagnostic test movies (200 each B12/B20/B28, namespace196103/block990001), it obtains BA75.724%, AUC.858459. This supports the earlier learning result beyond the repeatedly monitored validation set. Checkpoint1700 obtains50% BA/AUC.555352 on those same600 movies. These600 diagnostic trials are separate from the original reserved3000-trial final evaluation, which remains unperformed.

### Causal attention tests

All rows below use the same300validation movies and frozen weights. The forced-attention implementation was copied from the original model with one explicitly identified override of alpha. Donor maps come from the other checkpoint on the identical pixel movie, not supplied labels or target coordinates. Attention maps are replaced at every frame, affecting both evidence selection and spatial-memory writes; this is an intervention on the full attention-controlled trajectory.

| Frozen condition | Balanced accuracy | AUC |
|---|---:|---:|
| Good600 |74.908%|.853801|
| Collapsed1700 |50.000%|.567251|
| Collapsed weights with good attention maps |75.731%|.865225|
| Good weights with collapsed attention maps |50.000%|.469468|
| Collapsed weights, only fixation cross removed |70.774%|.796410|

At the sampled final motion frame, mean attention mass in the target-region mask falls from88.206% at600 to0.686% at1700; foil mass is1.676% versus0.794%. Mean maximum single-patch attention rises from54.668% to87.571%. In12additional independent movies across all durations and both target sides, the collapsed maximum always lies at grid index45, center(54.5,44.5), whose receptive field covers the central fixation cross. This12-movie location check is descriptive, not a population-frequency estimate.

The fixation intervention replaces only the central4x4pixels with background gray on every frame; dots, cue and timing stay otherwise identical. Target attention then rises to80.142% at the sampled late-motion frame. Removing fixation is a diagnostic alteration of the task, not an endorsed task simplification. The attention swap and fixation removal show that useful sensory and decision machinery survives in the collapsed weights.

The mean across-trial standard deviation of the735selected comparison channels drops from.13230 to.00365 after collapse. Decision-score standard deviation drops from1.51812 to.01321. This is consistent with attention excluding the informative motion apertures and starving the classifier of varying comparison evidence.

### Gradient amplification

Original logs contain raw gradient norms308.46 at601,1036.03 at612,2089.04 at620, and670.03 at621, followed by deteriorating training loss and weaker CLS variability. An even larger spike19236occurred at408, before later learning, so a large norm alone is not a deterministic collapse rule.

Restoring model600, Adam state, exact saved pool9, batch order and RNG reproduces the601loss exactly. About82.65% of its squared gradient norm is in the spatialGRU. Prediction loss has no direct spatialGRU or attention parameter dependence in this implementation; its direct gradients act on motion estimation.

On that same32-trial batch, stop-gradient interventions change only backward paths. Forward logits are IDENTICAL to baseline in all four tests; no optimizer step is taken:

| Backward computation | Total gradient norm |
|---|---:|
| Original |308.457|
| Stop alpha gradient only in spatial-memory write |3.538|
| Stop previous spatial/CLS memory gradient into attention keys/query |2.076|
| Stop previous spatial memory gradient into visual value |23.380|

These tests localize amplification to the recurrent attention-memory loop. They do not validate detaching those paths as a learning rule. The model uses100*alpha-scaled spatial writes, so concentrated attention creates large relative write gain; that scaling is a plausible contributor, but it was not independently varied in these experiments.

### Matched continuation experiments

Each continuation starts at600 with the same Adam moments, all saved RNG state and original pool9/epoch/cursor/order. Exactly60updates/1808presentations per arm, ending660, stay within the SAME1000-movie pool. This excludes a fresh-pool transition as a necessary explanation for the original601–621 deterioration. Architecture, inputs, terminal CE+.05prediction objective and microbatch4 remain fixed. Only the listed optimizer setting changes.

A90-trial validation subset is checked every10updates. Endpoint evaluation uses the full300validation and same independent600diagnostic movies. All arms have finite43Adam states at660. One endpoint per arm is retained as arm/latest.pt; no production checkpoint is overwritten.

| Continuation | Endpoint validation BA / AUC | Fresh600 BA / AUC | Fresh late target attention |
|---|---|---|---:|
| Original lr3e-4, no clipping |74.235% / 0.839249|75.010% / 0.840575|81.69%|
| lr3e-5, no clipping |75.228% / 0.879777|78.233% / 0.871855|89.82%|
| lr3e-4, global gradient norm clip1 |71.549% / 0.856113|75.728% / 0.866755|90.02%|

The unchanged replay did NOT reproduce the original collapse in this60update window. Initial losses agree exactly, the second update differs by roughly1e-7, and differences grow thereafter (about.0016 by604). These are matched deterministic input/optimizer starting states, not bitwise identical GPU trajectories. This numerical sensitivity is relevant evidence, but is not a measured Lyapunov exponent or proof that a particular CUDA kernel is the cause.

The clipped arm retained target attention even while its PRE-clipping gradient norm reached7651; gradients delivered to Adam were capped at1. Lower lr retained the strongest endpoint fresh BA, but one60update trajectory per setting is not enough to establish statistical superiority or long-term stability. Both interventions were tested separately; their combination was not tested. The standard-rate control's survival means lower lr cannot be called a necessary or proven sufficient cure from these experiments.

### Interpretation and recommendation

The supported mechanism is an amplifying recurrent attention/memory training path, with a failed policy attracted to the irrelevant fixation stimulus. Learned motion/decision competence survives and can be recovered by restoring attention routing. This is not evidence of catastrophic destruction of the entire representation, nor a mere decision-threshold shift.

The most conservative next continuation candidate is the preserved lower_lr/latest.pt (step660, Adam3e-5), with frequent held-out checks and best-checkpoint retention. This is a recommendation only: no extended training, restart of the interrupted production job, new architecture or cloud allocation was launched. Removing fixation or permanently detaching gradients is not prescribed on the basis of diagnostic interventions alone.

### Reproducibility and artifacts

Stability/run01 contains protocol.json, original_run_audit.json, frozen_audit.json, first_replay_gradient.json, gradient_through_time.json, attention_swap.json, collapsed_peak_locations.json, gradient_routes.json, fixation_removal.json, parameter_changes.json, each arm's update log/results/latest.pt, summary.json, completion_verified.json and stability_summary.png. Scripts experiment.py, attention_swap.py, forced_attention_model.py and gradient_routes.py preserve the methods. The one-hour cap included the initial probe and all experiments. All diagnostic sessions exited successfully, source checkpoints are unchanged, and the GPU lock is free. A diagnostic PID was later reused by an Edge renderer; process completion was verified using tool exit receipts and process identity rather than treating PID reuse as a live worker.

![Stability investigation](../SecondPass/LocalComparisonKrauzlis/Stability/run01/stability_summary.png)

---

## Latest authorization — controlled stability investigation, October6

User explicitly requests investigation and experiments to locate the stability problem. Original LocalComparisonKrauzlis stopped unexpectedly at live1737, latest1700/best600; no completion receipt and original worker/supervisor exited. Cause of interruption not established, no original restart. New Stability/run01 preserves originals and runs frozen best/latest comparisons plus60 matched updates from600 on original pool9/order/Adam/RNG for each of: original Adam3e-4, Adam3e-5, and Adam3e-4 with global gradient norm clipping1. No model/data/loss changes. One sequential localGPU, one new finite3600second diagnostic cap including first replay probe, no extension/cloud/production continuation. Frequent90-trial sentinel checks, endpoint300validation and600fresh diagnostic trials (block990001), no official final-test stream. One separate diagnostic endpoint per arm; original best/latest untouched. First replay601 exactly reproduces loss; dominant gradient spatialGRU. See Stability/run01/protocol.json and runtime results. No architecture fix or future full run automatically authorized beyond these experiments.

Production independently verified: LocalComparisonKrauzlis worker18300, saved1/live12,43finite aligned Adam states CPU-reloaded; frozen source hashes match. Fresh2244update target,68000presentations/34000unique. First validation66,then every100. Original deadline12:29:23AM PDT October6, no renewal. No held-out result yet.

# Local-comparison Krauzlis — October5,2026

User says proceed with the suggestions following completed frozen-model failure diagnostics. Authorization: build and train one fresh local native Krauzlis successor with explicit per-location temporal comparisons selected through the same appearance/motion attention map. Preserve prior artifacts. No cloud or automatic follow-up arm.

SecondPass/LocalComparisonKrauzlis retains the prior sensory bank, appearanceCNN,width64,10x10grid,singleCLS-queryattention,spatialGRU,CLSGRU and native rendering. Adds reliability-weighted reference8frames and recent4frames with4intervening frames, evaluated causally at every timestep without event metadata.735comparison channels are computed before spatial pooling, selected by the SAME alpha as appearance, retained in a.8decay trace and concatenated with finalCLS for799→64→2 readout. This intentionally bypasses CLS compression for selected comparisons, not the shared attention map.

Fresh378675parameters/43tensors, all trainable; no predecessor/probe weight transfer. CE+.05pixel-prediction loss only. Native .375speed,26/28changes,cue/distractor,lifetime/direction noise,B12/20/28 unchanged. New196xxx streams/seeds, batch32/micro4, FP32Adam3e-4,no clipping,1000movies/twoepochs. Requested6600updates prospectively pinned to whole66update pools under a new finite14400s local cap. Validation66/every100/terminal on300trials, bestbyCE, final3000test with no threshold fitting. Full frequent evaluation costs included in profile; no external evaluator. Best/latest only, independent guard.

CPU checks passed all43gradients/Adam states/tensor changes, exact causal-window arithmetic, prefix causality, map normalization and preserved comparison channels. Eleven source files frozen. Supervisor20944 launched; runtime receipts, not this journal entry, establish actual profile and production state. No claim of generalization from tiny-fit/implementation checks.

Actual CUDA profile pins2244updates (34completepools,68000presentations,34000unique movies). Median longest-duration effective-batch update4.180s; validation cost0.0674s/movie. Original finite deadline12:29:23AM PDT October6 remains unchanged. Production worker18300 starts with fresh weights; no held-out result yet.
