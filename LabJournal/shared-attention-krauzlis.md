## Completed failure diagnostics — October5

Frozen native checkpoint1300 and simplified checkpoint4250 were examined. Neither checkpoint or original model weights changed. One local GPU was used for inference only; CPU fitted diagnostic readouts. All diagnostic processes exited; GPU lock independently checked free. Original training stays cancelled; no next run queued.

### Independent data and methods

Native probes fit on1,200 fresh native movies (400 per B12/B20/B28, block880001 in train namespace) and test on600 separate native movies (200 per duration, block880002 in test namespace). Paired aperture observations stay within the same split. Cue-switch tests use300 additional movies, block880003. Original training, validation and final-test streams are not reused. All600-trial probe results below are exploratory, fixed-probe estimates rather than a prespecified population-level causal study.

Linear probes use train-only standardization and logistic regression C=.1; nonlinear probes use32 hidden units, alpha.01 and internal training-only early stopping, at most180 iterations. No test-driven hyperparameter search. Motion probes receive aperture positions and event times, average four late-baseline and four late-postchange measurements, and use before/after/absolute-difference features. Local-change probe labels use simulator event metadata. These privileges are deliberate diagnostic isolation and are not an end-to-end task solution.

### Results

| Diagnostic | Held-out result |
|---|---:|
| Cue side from appearance during cue |100% BA|
| Cue side from spatial memory after gap, before change, after change, and at report |100% BA at each sampled time|
| Cue side from final CLS memory |99.3% BA|
| Local change from motion features before fusion/attention |81.4% BA; AUC.892,1,200 aperture observations from600 movies|
| Local change from independent aperture-centered pixel correlation |76.4% BA; AUC.820|
| Target change from motion features, supplied target side |77.0% BA; AUC.841|
| Target change from final CLS, fresh linear decoder |52.9% BA; AUC.555|
| Target change from selected spatial-memory before/after states |54.2% linear,56.1% nonlinear BA|
| Compose decoded cue with independently trained local motion decoder |82.9% BA; AUC.912|

Composed decoder bootstrap95% interval is79.7–85.8% BA (1,000 stratified movie resamples; conditional on fitted probes, not including retraining uncertainty). It selects the aperture using cue side decoded from final spatial memory; this is identical to oracle-side selection because cue decoding is100% on these600 movies. Selecting the wrong aperture yields28.7% BA, AUC.233. The composed pipeline uses privileged event timing and local-change supervision during probe fitting; it is not the frozen network's performance.

Spatial-memory probes pool local regions and compare explicit pre/post snapshots; their weak results do not prove that every possible decoder of the full memory would fail. They indicate that task-relevant motion comparison is much easier to extract from the earlier motion representation than from the tested recurrent representations. The local-motion probe has93.2% training BA vs81.4% test BA, so it is not perfectly generalizing; cue information availability is substantially clearer than motion reliability.

### Cue-switch and attention observations

On300 new matched movies, changing only the cue ring to the opposite aperture leaves the original model choosing change on every trial. Original BA50%, AUC.562; switched-cue BA50%, AUC.465. Mean absolute decision-logit change is.00228; on single-change movies, its direction is appropriate53.1% of the time, and no pair gets both required opposite choices correct.

During the cue, mean attention mass in the small target-region mask is22.8% versus0.64% at the foil, and target mass is higher on100% of trials. At the sampled last motion frame, target/foil masses are25.4%/24.5%, with target higher on49.8% of trials. This shows an initial visual cue response but no reliable target preference at that sampled late time. The cue-side probe still succeeds, so available memory content is not equivalent to functional cue-guided selection. This is a diagnostic observation, not a microstimulation/inhibition causal experiment.

### Why the simplified variant was easier

Frozen simplified checkpoint4250 achieves100% BA/AUC1 on600 new fast, clean six-frame movies. On corresponding clean movies slowed to native .375px/frame, its unchanged argmax classifier chooses no-change throughout (BA50%) but retains AUC.996. A separate600-movie calibration set supplies a single score threshold; on another600 fresh slow movies it achieves97.7% BA/AUC.998. No weights changed and no test threshold fitting occurred. Therefore slow speed alone does not erase the clean-stimulus motion signal; it causes substantial score calibration shift in that older model. This does not establish equal robustness under native dot replacement, direction dispersion, cue/distractor competition or longer sequences.

The old model's classifier receives direct before/after displacement-population differences and prediction-error features in addition to recurrent features. The new native model removed that direct decision path and requires local evidence to pass through fused features, spatialGRU, shared attention and CLS memory. Earlier success therefore did not establish that recurrent attention itself learned the comparison operation. Multiple sensory/task/architecture changes confound a direct causal old-vs-new comparison.

### Interpretation and next targeted change

These tests argue against an unreadable or completely forgotten cue, and against motion resolution removing all usable evidence. The strongest supported bottleneck lies in preserving/comparing/selecting motion evidence between the pre-attention representation and the final decision. Native motion remains noisier and harder than clean motion, so the results do not rule out sensory improvements. They do not uniquely identify fusion, GRU dynamics, attention routing, optimization, or training duration as the cause.

A focused successor would preserve explicit per-location temporal reference/comparison evidence through memory and let the same shared spatial-attention map select that evidence together with appearance. It should not treat the successful privileged probes as a finished model or infer that increased image resolution alone will fix learning. No successor has been implemented or trained by these diagnostics.

Evidence: SecondPass/SharedAttentionKrauzlis/Diagnostics/run01/{results.json,bottleneck_results.json,slow_calibration.json,records.json,features.npz,cue_switch_trials.json,composed_predictions.json,completion_verified.json}; reproducible scripts diagnose.py, finish.py, calibrate_slow.py. The first diagnostic process stopped at the simplified-model call due to a tuple-return adapter error; cached completed native tests were preserved and finish.py completed the remaining analyses. No original training model was involved in that error.

---

## User-directed stop — October5,7:55PM PDT

User explicitly cancelled SharedAttentionKrauzlis. Trainer30284 and CPU evaluator12188 stopped and identities verified exited. Last live update1339; preserved best1000/latest1300, all43 finite aligned Adam states independently CPU-verified. Latest completed monitoring validation1300: BA0.500000, AUC0.509180. No final test; incomplete user-cancelled run, not a spontaneous crash. No automatic restart, successor experiment, diagnostic training, queue or budget extension. Preserve source/checkpoints/logs. Runtime cancellation_receipt.json records ownership and stop evidence. Earlier active descriptions below are historical.

Production independently verified: worker30284, live step9, saved step1, all43 finite aligned Adam states CPU-reloaded; frozen source hashes match. Target2574updates/78000presentations/39000unique, deadline10:36:52PM PDT. No held-out result yet.

# Shared-attention native Krauzlis — October 5, 2026

User requested stopping current training and trying one RViT attention head with shared appearance/motion selection on native Krauzlis. Attention allocation and stimulation/inhibition analysis are conditional on learning.

CorrespondencePredictiveMemory stopped at live4324; best64/latest4250 preserved and independently CPU-verified,43 finite aligned Adam states each. Supervisor and guard exited; cancellation receipt lives in its run01. This is a user-directed stop, not a spontaneous crash. Only the earlier validation64 exists; no final held-out test was performed.

Successor: `SecondPass/SharedAttentionKrauzlis`. Fresh331635parameters/43tensors; local causal correspondence and appearance fused at100spatial positions; one CLS-query attention map routes all channels to spatialGRU and CLSGRU. Terminal CE plus.05pixel-prediction loss; no supplied direction/event metadata or global comparator bypass. This modifies RViT attention explicitly rather than simply changing heads=1 on the previous architecture.

Native cues, two patches, lifetime resets, .375px/frame,26/28degrees,29/37/45frames restored via the unchanged original renderer. Fresh1000-movie pools/twoepochs, requested6600updates under a new finite14400s local cap. Profile prospectively pins complete66update pools. All tiny-fit/profile weights discarded; production fresh. No cloud, automatic extension, second experimental arm or intervention campaign.

CPU checks passed all43gradients/Adam states, allweights changed aftertwo updates, native durations, causal normalized map and intervention-hook semantics. Source11files frozen. Supervisor22308 launched; actual production and pinned exposure must be read from runtime receipts. See model README for equations, limitations and references.

Actual CUDA profile pins 2,574 updates (39 pools,78,000 presentations,39,000 unique movies), batch32/micro4. Worst-duration update median3.656s; pool generation27.323s. Original finite deadline10:36:52PM PDT unchanged. Tiny-fit8/8 passed at80updates, training-only and discarded. Production starts fresh; first validation66, then250/1000/2000/2574. Final3,000 native trials.


October5, after validation1000: user objects to the long gap before the next result. Added an independent CPU-only saved-checkpoint evaluator every100updates (roughly5-6minutes), using the identical300-trial validation set and originaldeadline. Below-normal-priority monitor12188, oneCPUthread; training process30284 remains unchanged. Results in LocalRuntime/run01/frequent_validation; these monitoring results do not change production best-checkpoint selection. No extra training, GPU worker, new budget, or automatic extension. Initial evaluation cross-checks CPU scores against the existing CUDA validation when available.

## Latest authorization — post-stop failure diagnostics, October5

User explicitly requests the proposed tests to identify why native Krauzlis failed while the simplified variant worked. This supersedes the prior hold only for frozen-model analysis and small diagnostic probes. Run independent native train/test streams, motion-feature and pixel-reference probes, cue appearance/memory probes, matched cue switches, and a frozen simplified-model speed comparison. No original model/optimizer updates, successor training, checkpoint replacement or cloud. A bounded30minute local diagnostic process uses the free GPU for inference and CPU for small probes. Source and original checkpoints preserved; outputs under SecondPass/SharedAttentionKrauzlis/Diagnostics/run01. Diagnostics may use privileged aperture/event metadata explicitly to isolate stages; these are not task-performance claims.

