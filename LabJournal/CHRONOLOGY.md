# How we got here

[Index](README.md) · [Current status](CURRENT_STATUS.md)

The chronology follows the post-reset development sequence. Run-directory dates show the main sensory work on September 12,2026 and memory/attention development through September 13. A run may begin one local date and finish another UTC date; ordering below is based on ancestry and saved reports, not an invented minute-by-minute history.

## 1. Establish useful sensory encoding before choosing a complete system

The user reset the repository because an inherited architecture and its justification had displaced component-wise research. The new scaffold was Guided Search 6.0, with PAV first and other components later. The first binary dot-displacement design was corrected to cardinal random-dot motion with exactly two ordered frames. Its stopped artifacts are historical diagnostics, not final evidence.

[Five encoders](experiments/01-pav-encoder-screen.md) then received the same seven-task exposure. ConvNeXt learned broad sensory distinctions and motion best in that screen, while other candidates were stronger at contour grouping. The next question was therefore whether useful computations could be combined, not whether one model should be permanently mandated.

## 2. Combination ideas did not substitute for acquisition

The [CPU ensemble audit](experiments/02-saved-ensemble-audit.md) found complementary contour errors but a motion cost from averaging. A [three-arm hybrid comparison](experiments/03-hybrid-continuation.md) tested a Gabor branch and late-SE against equally trained continuation. Neither met the contour target; continuation itself improved markedly.

The decisive [allocation experiment](experiments/04-contour-task-allocation.md) changed only which existing task received more updates. Contour improved 67.9%→96.2%, while all seven point accuracies stayed above 95%. This is the project's first clear demonstration that weak endpoint behavior need not require a new architecture. It also exposed a small contrast cost that remains part of the record.

## 3. Replace simultaneous pair access with causal sensory state

The user asked whether motion should be computed from successive frames and an accumulator. The [KDA/ConvGRU/opponent comparison](experiments/05-causal-temporal-accumulators.md) froze the trained encoder and learned new temporal/readout components. Opponent fast/slow traces with fixed quadrature-energy computations reached 98.44–100% across the two-step battery. Reset and reversed-frame tests supported use of history and order.

The user selected this neuroscience-inspired winner for further work. High two-frame accuracy was never evidence of long memory; that became the next measured question.

## 4. A broad sequence battery failed to acquire several new rules

The [sequence battery](experiments/06-sequence-battery.md) added instructions, cues, integration, delays and retrospective probes while unfreezing learned weights. Many rules stayed weak even at minimal delays. Longer-delay chance behavior therefore did not cleanly identify a retention limit. The correction was to focus acquisition and compare explicit recurrent memory, not to claim the sensory model had a measured universal memory capacity.

## 5. Focused recurrent memories acquire tasks; investigate E/I's motion gap

The [LSTM/EI comparison](experiments/07-lstm-versus-ei.md) learned focused motion-duration and orientation tasks after 40,000 episodes per arm. LSTM did better on longer motion integration. A [matched-order diagnostic](experiments/08-recency-diagnostic.md) showed greater E/I benefit from late winning evidence.

An appealing immediate story would have been excessive leak. The [state-accessibility diagnostic](experiments/09-state-accessibility.md) instead found early count information still available in final firing rates and a linear duration-readout rescue. The project consequently deferred slow synapses and [refit the existing outputs](experiments/10-readout-refit.md), improving standard-task motion 70.31%→79.30% without changing upstream computation.

## 6. Retaining a decision and comparing a delayed visual feature separate

[Retention training](experiments/11-retention-learning.md) made motion accuracy approximately flat through 24 blanks. Its remaining 20% error was already present without delay. Orientation comparison, whose answer requires a new probe, remained near chance at 24 blanks.

The [orientation diagnostic](experiments/12-orientation-accessibility.md) found accurate pre-probe angle information despite the deployed comparison failure. A decoder trained at sample time failed later, while a late-trained decoder succeeded. A label-only diagnostic comparator rescued performance. “Blanks erase the feature” was therefore too strong: accessibility and use had to be separated.

## 7. Spatial memory solves a binding task but introduces a major trade-off

The user's proposed 13 × 13 × 64 memory became a [spatial competitor](experiments/13-spatial-memory.md) against dense memory, both with explicit old-memory/current-input comparison. Spatial binding improved strongly, including new location centers. Native single-item D24 did not improve, and motion deteriorated. Binding and native orientation differ in task construction, so high binding is not evidence of higher two-item capacity.

This was a useful component with a known cost, not an accepted universal replacement. The earlier motion-competent model remains preserved.

## 8. Keep teaching fixed while testing learned maintenance

The user rejected changing the target to an angle-supervised delayed report. The implemented [additive controller](experiments/14-selective-maintenance.md) therefore used the same tasks and losses. It produced only a narrow delayed-orientation gain; interrupting its blank-period current did not remove the benefit.

The alternative [pre-update attention](experiments/15-preupdate-attention.md) lets old memory query both current sensory and old-memory values. It improved single D24 to 79.30% versus 58.40% continuation, but hurt immediate motion. A [phase-specific intervention](experiments/15a-attention-mechanism.md) showed that orientation depends on memory-source routing during blanks. Active refresh versus rejection of blank input is still unresolved.

## 9. Preserve the gain and investigate the regression

The [motion audit](experiments/15b-motion-audit.md) found severe class bias, weak task allocation and a motion deficit predating attention. Calibration partially rescued choices but did not solve delayed performance. The user explicitly requested investigation instead of accepting progress that sacrifices another domain.

[Current Stage1](experiments/16-training-exposure.md) compares unchanged 10% motion continuation against 50% motion at equal total additional updates, starting the same attention 8400 checkpoint. Local control starts first and the additional arm uses a concurrent authorized RunPod. Fresh selection checks all trained cells, preventing an aggregate improvement from hiding task regressions. A proposed temporal residual remains Stage2 design only, contingent on these findings.

## Decisions that remain intentionally unresolved

No universal architecture winner, biological validation, formal multi-item capacity or general attentional selection ability has been established. Slow synapses were deferred after recoverability evidence; new angle teaching was rejected; a residual has not been trained. These are scientific decisions and user constraints, not missing implementations to quietly complete.

## Prospective-query intervention launched (2026-09-15T01:04:25.1474724+00:00)

After the five-task original-bias run and task-only motion continuation remained weak on cued motion, the user authorized one architecture-only test from the intact attention8400 parent. [Experiment 21](experiments/21-prospective-query.md) adds a zero-initialized learned scalar current-sensory residual to pre-update queries while preserving keys, values, biases, E/I memory, tasks, losses, optimizer settings, exposure and evaluation. Exact-zero equivalence and nonzero-gradient checks passed.

Fresh RunPod pod `dqi13o2x3qkvos` uses one A100-SXM4-80GB. Setup exposed CRLF and omitted-fixture packaging defects before training; both were corrected under the same deadline with no training exposure or scientific change. Profiling then passed and production advanced beyond step8400. Frozen local diagnostics found cue-dependent routing/output changes but near-chance task behavior, while a split linear probe of local final-moving-frame sensory features remained near chance. This separates detectable cue effects from successful duration-rule use and supplies no evidence for the probed sensory summary. Final training claims await scheduled results; historical control matching ends at step11600.

## Prospective-query run stopped after its first validation (2026-09-15T02:04:33.9212457+00:00)

Step-9200 validation improved orientation and spatial binding but did not improve
the primary cued motion-duration failure; recognition was mixed and worse at
load 24. The user stopped the run at logged step9430 /41,200 episodes. Durable
checkpoint9216 contains32,640 episodes. All29 indexed artifacts were verified,
the A100 pod was deleted, and account spend returned to zero. This is a
user-stopped negative primary result, not a completed 160,000-episode
comparison or final held-out evaluation.

## Spatial-priority readout launched (2026-09-15T03:31Z)

The user authorized an independent attention8400 branch that changes only the
terminal decision readout. Final sensory, E/I-rate and comparator fields stay
aligned at13×13 through a shared1×1/3×3 convolutional stack; each task learns
spatial selection and local class-evidence maps. Original attention and its
source/locality biases remain intact, and prospective gamma is absent.

[Experiment22](experiments/22-spatial-priority-readout.md) passed exact
inheritance/Adam, shape, normalization and gradient checks. A fresh community
RTX4090 RunPod began the fixed160,000-episode run. Profiling projected the
work within its eight-hour deadline; production was verified beyond step8400
with finite metrics and a live45-second monitor. No validation result existed
at this entry, so no performance conclusion is recorded.

## Spatial-priority lineage corrected and relaunched from scratch (2026-09-15T03:53Z)

The user corrected the experiment before its first validation: the complete
trainable architecture, not merely its new readout, must start from scratch.
The inherited attempt was immediately stopped at 320 updates /12,800 episodes;
its RTX4090 pod was deleted and confirmed absent. It produced no validation
and is not architecture evidence.

The corrected `spatial_priority_readout_scratch_v2` bundle contains no parent
checkpoint and loads zero model or Adam tensors. It freshly constructs the
opponent sensory path, original biased `JointAttention`, spatial E/I memory,
comparator and priority-map readout from documented seeds. Scratch
determinism, component identity, gamma/global-pooling absence, normalized maps
and finite gradients passed locally and on the pinned cloud runtime.

A new community RTX3090 pod `ce00y2ooosl7wc` launched the same five-task,
4,000-update /160,000-episode protocol from step0. The three-update profile
averaged3.4266seconds and projected the complete bounded work at about
5.51hours/$1.21. Training was verified active with finite metrics; no
validation existed at this chronology entry. Historical model comparisons are
not initialization-controlled ablations of this scratch run.

## Independent dual-attention priority arm launched (2026-09-15T04:36Z)

[Experiment 23](experiments/23-dual-attention-priority.md) separates
pre-update integration from post-update spatial selection. One width-64 head
produces raw context and the E/I drive; after the unchanged E/I update, another
width-64 head lets updated rates query `C/H/R`. Only terminal post-attention
field `P_T` enters the spatial-priority decoder.

The complete 549,344-parameter model and Adam optimizer start from scratch.
Local and pinned cloud checks verified the requested shapes, finite
forward/backward, nonzero output-loss gradients through both Q/K/V paths,
normalized selection, no comparator/gamma/global bypass, and exact equality
for 119 unchanged scratch-control tensors. Independent RTX 3090 pod
`f7y0zw02f4fzum` began the fixed 4,000-update /160,000-episode run with finite
metrics. The control and concurrently launched comparator pods remain
separate and untouched. No validation conclusion is available at launch.

## September 13 local evening update: cloud allocation arm completes

The 50% motion arm completed and selected 11600. Fresh-test motion improves over parent, but single D12 falls 2.93 pp. Cloud retrieval and deletion are complete. The local 10% control still trains, so the schedule comparison remains incomplete. The user also authorized queued frozen latent-state visualization after the local GPU becomes free. See the dated [current status](CURRENT_STATUS.md) for receipt links and the distinction between earlier validation and new test scores.

## User correction: inspect attention maps first (2026-09-14T00:45:26.831440+00:00)

The user paused further model experiments and the broad latent/probe investigation, requesting condition-averaged attention maps over immediate and remembered-scene references instead. LatentDynamics supervisor 4644 was stopped before extraction; no compute budget was started. The researcher now prepares frozen forward-only attention capture after the existing local run's training/evaluation exits. Both source banks will be shown within each of the two heads; neither head is source-exclusive. The earlier latent proposal remains documented as paused. See [current status](CURRENT_STATUS.md) and the [pause receipt](../WorkingMemory/LatentDynamics/queue_receipt.json).

## Training and requested visualization complete (2026-09-14T00:56:21.960683+00:00)

Both Stage1 arms completed 32,000 additional episodes. Local control selectedterminal 12400 with motion D0/D24 41.02%/41.99% and single D12/D24 96.88%/91.60%; focused selected 11600 has stronger motion but weaker orientation and a D12 regression below parent. No all-domain winner is established. Full parent/selected/terminal tables and uncertainty are in [the final Stage1 entry](experiments/16-training-exposure.md).

After local training/evaluation exited, the narrower [AttentionMaps capture](experiments/17-attention-maps.md) completed 192 paired base episodes /384 presentations in 30.45 seconds using unchanged attention 8400. The viewer now shows both sources for each head, phase/condition means, receiving-query views and labelled scene references. Source-mass differences do not establish spatial focus or remembered content. Broad latent/probe work remains paused; the cloud is deleted and owned workers are absent. No subsequent model experiment has been launched.

## Presentation mismatch corrected rather than hidden (2026-09-14T01:02:13.120024+00:00)

The user rejected the first promoted single-query figure and the phase/time-averaged overview as answers to the requested full-grid temporal view. The initial capture was complete, but its presentation did not meet that request. Cached matrices confirmed strong locality (about 94% joint mass at each query's coordinate, distance penalty about 4.03); the single bright cell therefore did not demonstrate absent attention. Researchers are preparing individual trials at every timestep, retaining earlier averages as explicitly historical summaries and staying within the same capture deadline. See [entry 17](experiments/17-attention-maps.md).

## Every-timestep correction completed (2026-09-14T01:05:05.388626+00:00)

The corrected [AttentionMaps viewer](../WorkingMemory/AttentionMaps/index.html) now displays individual trials at exact frames, all receiving-grid cells for each head/source, and timeline playback without temporal or trial averaging. Frozen capture added 48 movies /912 frame observations in 6.76 seconds within the original deadline, with no added budget or model change. The earlier phase summaries are preserved as historical averages. [Entry 17](experiments/17-attention-maps.md) records both the initial mismatch and the completed correction rather than treating the first delivery as sufficient.


## Explicit attention priors become the next controlled change (2026-09-14T01:40:21.630790+00:00)

Inspecting the actual maps exposed strong local routing under the learned distance penalty. The user authorized removing explicit source and distance biases while preserving trained attention projections, then both continuing old tasks and teaching a spatial battery. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) separates those questions: the old arm uses the prior 32,000-episode control recipe and reused paired evaluation, while the new five-task arm studies acquisition with fresh heads and task streams. A single L40S pod is provisioned under an eight-hour cap; no production training or performance result is yet verified at this snapshot. The protocol preserves the corrected consecutive-image recognition timeline, signed orientation glyph, retrocued binding and one explicitly adapted Arcizet–Krauzlis recipe. Broader latent analyses remain paused.


## Cloud setup replacement without budget renewal (2026-09-14T01:50:22.617728+00:00)

The first L40S host failed CUDA initialization under two runtimes and one restart; it trained zero updates. Failure evidence was retrieved and verified, then the pod stopped/deleted. A replacement L40 48 GB was created at 01:48:02.15 UTC with the original 09:36:25.001 UTC deadline, unchanged source/model/tasks and no allowance renewal. It is in setup, not yet verified training. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) links both receipts.


## Production launch verified (2026-09-14T01:57:21.565752+00:00)

Both profiles passed on the replacement L40 with Torch 1.13.1+cu117, and the full 32,000 old-task / 160,000 new-task episode allocations were pinned. The launch receipt records old-arm global 8605, +205 updates / 1,640 fresh episodes and finite loss; the five-task arm is queued after it. Supervisor and incremental retrieval watcher continue under the original deadline. Profile durations are estimates, not completed work. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md).


## Old-task bias-removal result is negative (2026-09-14T02:40:33.400826+00:00)

Both old-task continuations completed 32,000 episodes. Joint source/locality-bias removal drives terminal binding D24 from control 99.41% to 49.80% and single D24 from 91.60% to 57.62%, while small motion changes have paired intervals including zero. All five validation looks failed; automatic selection falls back to the still-biased parent 8400. This does not erase the negative trained-terminal result. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) preserves all 14 paired cells, uncertainty and the warm-start/reused-test limits. The separately authorized five-task arm continues unchanged toward 160,000 episodes under the original deadline; no additional experiment is implied by the old-arm failure.


## User cancels bias-removed acquisition, then restores the original model (2026-09-14T02:44:04.427099+00:00)

The five-task worker/supervisor stopped at logged global 8550 (+150 updates / 6,000 episodes), with durable checkpoint 8448 and GPU processes empty. The user subsequently asked to restore original biases and launch again. Researchers prepare a separate acquisition run from intact attention 8400, keeping tasks, fresh heads/streams, optimization and the original deadline. No damaged or partial model is the new parent. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) preserves the cancellation; [entry 19](experiments/19-biased-spatial-battery.md) records the new authorization without claiming launch.


## Original-bias acquisition launch verified (2026-09-14T02:48:44.540830+00:00)

The separate BiasedTraining launch receipt verifies global 8448 (+48 updates / 1,920 episodes), finite loss 0.85955 and full 4,000-update / 160,000-episode exposure. It retains intact attention 8400 source/locality terms and 123 compatible Adam states, with fresh heads identical to the cancelled initialization. The healthy L40 and original deadline are unchanged. [Entry 19](experiments/19-biased-spatial-battery.md) links launch evidence; the bias-removed acquisition stays cancelled and no old-task rerun occurs.


## Uneven joint acquisition motivates one local task-focused branch (2026-09-14T04:42:55.701953+00:00)

The original-bias joint model's validation checkpoint 10000 solves binding but remains near chance on cued motion duration; the cloud worker continues beyond this evaluated checkpoint. The user authorizes a local motion-only continuation from 10000, preserving model/cues/optimizer policy and changing the gradient-producing task allocation. Target 32,000 episodes under a new 7,200-second local cap is pending profiling. This asks whether focused training can acquire the task; it does not predeclare interference. [Entry 20](experiments/20-single-task-motion.md) records the current authorization and provisional validation evidence.


## Motion-only local launch verified (2026-09-14T04:48:12.203271+00:00)

Profiling reduced the proposed 32,000-episode local target before production to 17,600 episodes / 2,200 updates within the unchanged 7,200-second cap. The launch receipt records global 10065, +520 motion episodes, finite loss and restored model/133 Adam states/RNG/family stream/delay scheduler. Full motion CE uses inherited learning rates without compensation; this is focused acquisition, not isolated interference measurement. Cloud joint training remains untouched. [Entry 20](experiments/20-single-task-motion.md) records deadline, evaluation splits and launch evidence.


## Task-focused duration continuation does not acquire the task (2026-09-14T05:45:20.729828+00:00)

The local motion-only run completed all2,200updates/17,600episodes. Selected11760 and terminal12200 remain near25% held-out accuracy acrossD0/4/12/24; all paired gains includezero and AUC is near.5. Finite losses settle nearlog4 with restricted class choices. Failure already atD0 rules out a solely inserted-blank explanation, but does not identify a bug, interference or a capacity limit. All97 artifacts were verified, localworkers exited and the cloud joint run was left unchanged. [Entry20](experiments/20-single-task-motion.md) preserves the negative result and exposure/split distinctions.


## User-requested shutdown completed (2026-09-14 05:52:23 UTC)

The partial original-bias cloud run stopped at logged11923/140,920episodes, with durable11776/135,040episodes; no final held-out evaluation was run. All73 artifacts were retrieved and verified, then root confirmed stopEXITED, deleteHTTP204 and an empty pod list. The completion monitor is paused and no project model workers remain. Last completed validation11600 is preserved as validation, while selected-so-far10000 and durable11776 remain distinct. The local motion-only experiment had already completed. [Entry19 closure](experiments/19-biased-spatial-battery.md) and [cleanup receipt](../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json). No automatic restart or additional experiment is authorized.

## Frozen terminal fields do not yield a spatial-probe motion rescue (2026-09-15T03:58Z)

While the independent scratch cloud run continued untouched, a bounded local
[post-hoc diagnostic](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
froze motion-only step12200 and trained pooled and spatial-priority probes on
the exact same cached final sensory/rate/comparator tensors. The probes were
capacity matched within70 parameters and used grouped, independently split
base movies across D0/4/12/24.

Held-out BA was23.83% pooled and24.32% spatial, paired difference+0.49pp
(95% CI -1.76 to+2.64), with AUCs0.4805 and0.4832. No delay-level advantage
was reliable. The priority map localized the target region above uniform
overall but did not decode the direction winner. This is evidence against a
simple frozen-final-field readout rescue under the tested probes, not evidence
that relevant information is absent at other times or that end-to-end spatial
readout training must fail.

## 2026-09-15T22:01-07:00 — AV-context v2 local launch

Pre-flight on the local scratch motion-only v1 step-3200 checkpoint measured
1.8e-17 attention mass beyond 3 cells per head and chance-level linear
decoding of the cued direction from H_T, R_T, C_T and H_9. The combined
five-change arm (A opponent channels, B mixed pooling, C wide neutral head 1,
D non-zero selection init, E single learning rate) passed the bit-identity
gate against v1 and started training locally; 6.51 s/update; stopping rule
at validation 2,400. See [experiment 24](experiments/24-av-context-v2.md).

## 2026-09-15T22:35-07:00 — user-requested shutdown of both cloud pods

At the user's instruction both RunPod pods were stopped and deleted: the v1
AV-context comparator resume pod `6mopzgoioemdp2` (logged step ≈6,400, mid
validation 6400) and the dual-attention resume pod `629g1utqkk8non` (logged
step ≈5,464). Stop returned 200, delete 204, subsequent lookup 404, and the
account pod list is empty. **The pre-deletion retrieval failed on both pods**
(the remote inventory probe exited non-zero after the remote workers were
killed; cause not reproducible because the pods are gone), and the shutdown
script deleted anyway. Lost: all remote checkpoints and the full
`metrics.csv` of both arms. Kept locally from the 45-second watchers:
validation summaries, predictions and priority maps through 5600 (AV-context)
and 4800 (dual), the last 256 metric rows of each, and the last aggregate
snapshots. Receipts:
[AV-context](../WorkingMemory/AttentionContextComparator/runs/resume_20260915_183128/user_stop_receipt.json),
[dual](../WorkingMemory/SpatialPriorityReadout/DualAttention/runs/dual_resume_20260915_183345/user_stop_receipt.json).
`WorkingMemory/cloud_shutdown.py` now refuses to delete a pod whose retrieval
failed unless `--force` is passed.

## 2026-09-16T08:15-07:00 — v2 overnight pod stopped as a negative result

Pod `vqpgk21cpi53b6` trained the five-change v2 model unattended overnight to
step 10,621 (424,840 episodes) with all tasks at chance and decaying
gradients; the user called it a failure and had it stopped. Retrieval of all
49 checkpoints and logs was verified before deletion (stop 200, delete 204,
lookup 404). Overnight, all local processes had died at ≈22:40; the local v2
run was resumed in place from checkpoint 256. See
[experiment 24](experiments/24-av-context-v2.md).
