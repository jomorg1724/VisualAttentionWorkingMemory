# What remains unresolved, and what would move it forward

[Index](README.md) · [Current experiment](experiments/18-unbiased-attention-spatial-battery.md)

| Question | Existing evidence | Useful discriminating result |
|---|---|---|
| Does motion lack sufficient training allocation? | It had10% of updates and was excluded from selection; earlier contour responded strongly to allocation | Stage1 equal-update10% vs50% motion, with every orientation/binding cell retained in reporting |
| Can motion recover without sacrificing delayed orientation? | Attention improved singleD24 but worsened immediate motion | Fresh paired parent/selected/terminal evaluations; report regressions even if parent is selected |
| Is attention actively refreshing memory or protecting it from blank input? | Excluding memory values only during blanks collapses orientation | A targeted future manipulation separating recurrent value content from sensory-drive renormalization; current exclusion changes both |
| Is early temporal information filtered before attention? | Motion bypass gives a partial acute rescue; fixed fast/slow states contain earlier feature interactions | If Stage1 leaves a deficit, compare one direct learned temporal residual with a matched continuation from the same parent |
| Are remaining motion errors mostly decision bias? | Validation-fitted class offsets rescue D0 but transfer weakly toD24; AUC also falls | Per-class curves and fresh decisions alongside AUC; do not infer the answer from argmax alone |
| Does this model store multiple bound items independently? | Full-swap binding can be solved using one location | A later specifically authorized task that cannot be solved from one stored location; current high binding is insufficient |
| Is the useful orientation code stable over time? | Sample-trained angle decoder transfers badly; late-trained decoder succeeds | Distinguish offsets/scales from content geometry; present diagnostic supports changing accessibility, not a unique coding mechanism |
| Do conclusions replicate across initialization and broader stimuli? | Most comparisons have one seed and one procedural distribution | Replicated training/transfer when warranted; current intervals do not estimate seed variability |

## Completed Stage1 and remaining decision

Stage1 is complete at its fixed allocation. Parent, validation-selected and terminal models are reported across all 14 cells with uncertainty and exposure curves. The focused schedule improves motion more, while ordinary continuation improves orientation more; see [the completed result](experiments/16-training-exposure.md). Compare both total updates and motion examples; equal updates intentionally differ in family exposure and total frame workload. The focused arm at+800 updates and control at+4000 have equal motion counts but different nonmotion experience and delay assignment, so that view is descriptive rather than a perfectly isolated motion-exposure match.

The proposed Stage2 residual would pool earlier 72-channel opponent features from each scale, concatenate 216 channels on 13 × 13, normalize per site and use a zero-initialized 216→64 projection into attention's visual source. It adds 13,824 parameters and no persistent state. It would bypass some learned sensory mixing, not the entire encoder or attention. It is documented in [the design](../WorkingMemory/Research/training_exposure_and_temporal_residual.md); **it has not been trained and must not be reported as a result**.

If tested later, use a single schedule and common parent for residual versus no-residual. Improvement could reflect capacity or optimization as well as information access; residual success alone would not prove earlier layers had erased information. A classifier-only short-trace skip is unlikely to retain evidence through 24 blanks by itself, since old contributions to the slow trace shrink as.75²⁴≈.001, but nonlinear amplification and learned representations prevent treating that scalar decay as a model-level impossibility proof.

## Interpretive commitments

Preserve a substantial improvement without pretending it is an all-domain upgrade. Keep negative results, unsuccessful hypotheses, and older useful models. Change one justified factor when possible, and allow healthy training to finish its pinned allocation rather than declaring failure from an early checkpoint. A successful source-grounded diagnostic should influence the next implementation; it is not merely another check to collect.

## Paused latent analysis; attention maps now take priority (2026-09-14T00:45:26.831440+00:00)

The broader [LatentDynamics proposal](../WorkingMemory/LatentDynamics/RESEARCH.md) is paused at the user's request. The [queue receipt](../WorkingMemory/LatentDynamics/queue_receipt.json) confirms that its supervisor was stopped before extraction (`compute_started: false`). No probe or model experiment should be started as a consequence of the earlier proposal.

The current request is frozen attention visualization: condition-averaged spatial maps, overlaid separately on the immediate scene and the relevant remembered-scene reference. The researcher has completed [AttentionMaps](../WorkingMemory/AttentionMaps/index.html) on frozen attention 8400 after local training/evaluation exited, using 30.45 seconds of that same allowance. Each head must be split into **both** sensory and memory source banks; head identity does not hardwire source identity.

Average source-mass statistics already exist in [MechanismDiagnostic](../WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md), and the requested spatial capture has now [completed](experiments/17-attention-maps.md). A map illustrates learned routing, not whether an attended state carries an accurate orientation or duration code. Averaging and coordinate conventions must remain explicit. Broad UMAP/t-SNE/probe questions remain deferred.

## Dated addition: why do motion decisions swing? (2026-09-14T00:39:06.263187+00:00)

The focused model's last 800 updates reduce D0 held-out BA 65.625%→49.8047% and shift predictions from 113 down/123 up to 2 down/308 up on the same 512 balanced movies. AUC declines .88834→.86650, so this is not purely unchanged ranking with a different decision threshold. All motion updates in that final block were clipped, but the logs do not measure actual parameter changes or establish exploding states.

[The detailed note](experiments/16-training-exposure.md) records the per-class evidence and optimization settings. Shared-task interference, a representation/head tracking mismatch and noisy constant-step continuation are hypotheses. Gradient-direction comparisons, parameter-update measurements and branch drift would distinguish parts of those accounts; they have not been collected here. The now-paused latent visualizations would not identify an optimizer cause from attractive clusters alone. The completed forward-only attention maps are descriptive and do not by themselves identify the cause of training instability. No additional training or diagnostic GPU job is launched by this note.

## Current decision boundary after completion

The requested attention viewer is ready for inspection. The complete training comparison documents a motion/orientation trade-off and nonmonotonic focused motion decisions. Preserve both useful continuations; no new residual, optimizer sweep or model experiment is authorized by these open questions. Broad latent/probe analysis remains paused. Future causal or representation tests would require a new scoped decision after the user reviews the maps.

## Presentation correction is complete

The user subsequently rejected phase/time averages as the primary answer and requested every timestep of individual trials. The [corrected viewer](../WorkingMemory/AttentionMaps/index.html) now provides that view, with 48 movies and 912 frame observations; see the [receipt](../WorkingMemory/AttentionMaps/perframe_receipt.json). Earlier averages remain historical only. Strong local routing explains why an initial single-query map looked concentrated; it does not show absence of attention or establish remembered feature content. No additional model experiment is underway because of these questions.


## Current test: explicit attention priors and spatial task acquisition

The [new competitor](experiments/18-unbiased-attention-spatial-battery.md) tests whether the strong explicit source/distance priors constrain useful learned routing. Old-task continuation addresses performance preservation using the saved control; its reused test is exploratory. New spatial tasks teach selection and larger study sets, but have no trained biased counterpart, so their success cannot isolate a bias-removal benefit. Learned position/source embeddings remain: removing scalar priors does not make the model spatially neutral. Recognition load 0 tests rejection only and must never enter chance-normalized BA/AUC ranking. Broad latent probes and the temporal residual remain unrun.


The completed old-task arm now shows that removing both explicit attention priors sharply reduces delayed binding and orientation within this warm-start continuation. Which term contributes, whether relearning from another initialization differs, and whether their removal helps the new spatial tasks remain unresolved. The bias-removed new-task run was then cancelled. The user explicitly authorized a separate restart with original biases from intact attention 8400; no extra component ablation is authorized by the old result. [Paired evidence](experiments/18-unbiased-attention-spatial-battery.md).


The [original-bias replacement](experiments/19-biased-spatial-battery.md) is now the active acquisition question. Its cancelled counterpart is not equally trained and cannot support a clean bias-effect comparison. The new run is training at its pinned exposure; no completed outcome is yet available.


## After completed motion-only continuation

[Entry20](experiments/20-single-task-motion.md) finds no acquisition gain after17,600 additional motion episodes, even atD0. The unresolved computation precedes or extends beyond inserted-blank retention. Current evidence does not distinguish learning target selection, extracting each patch's motion, aggregating direction duration or mapping evidence to outputs. Near-chance AUC and class collapse describe deployed outputs; no new state probe or GPU experiment is authorized by this note. Cloud joint training continues unchanged.


## Shutdown boundary

All project training is now stopped or complete; the cloud pod is deleted and the monitor paused. Questions above do not authorize new runs. Continue from [current status](CURRENT_STATUS.md) and saved artifacts only until the user authorizes further experiments.
