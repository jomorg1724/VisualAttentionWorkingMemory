# Current state of the project

[Next-agent handoff](../HANDOFF.md) ·
[paper-writing research handoff](../PAPER_HANDOFF.md).

[Journal index](README.md) · [Chronology](CHRONOLOGY.md) · [Open questions](OPEN_QUESTIONS.md)

**Updated 2026-09-16T08:20-07:00: the v2 overnight pod `vqpgk21cpi53b6` was
user-stopped at step 10,621 / 424,840 episodes and deleted after complete
verified retrieval (49 checkpoints, all logs, validations 800–10000). Every
task stayed at chance; the user judged the arm a failure. No pod remains on
the account. Key reading: no from-scratch arm on this battery has learned even
binding or recognition, which the warm-started lineage solved, so the
scratch-optimisation regime, not the attention/readout changes, is the
binding constraint. See [experiment 24](experiments/24-av-context-v2.md). The
local v2 run was also killed at the user's request at step ≈490, before any
validation. Nothing is training locally or in the cloud.**

**Earlier 2026-09-15T22:05-07:00: a local five-change AV-context v2 scratch
run ([experiment 24](experiments/24-av-context-v2.md)) is training on the
laptop RTX 3070 in
`WorkingMemory/AttentionContextComparator/V2/runs/v2_local_20260915_220124`
(supervisor PID 34688). Changes: restored opponent sign/magnitude channels,
mixed avg+max pooling, a wide source-neutral attention head, non-zero
priority-selection init, and one learning rate for the scratch arm. The
identity gate (v2 minus C/D/E bit-identical to v1), the attention-locality
measurement (1.8e-17 mass beyond 3 cells on trained v1) and the linear probe
(all fields at chance) preceded launch. A pre-registered stopping rule at
validation 2,400 ends the arm if priority entropy and head-1 locality have
not moved. Profile 6.51 s/update; no validation yet.**

**Updated 2026-09-15T22:40-07:00: at the user's request both cloud pods were
stopped and deleted (`6mopzgoioemdp2` v1 AV-context at ≈6,400 updates,
`629g1utqkk8non` dual attention at ≈5,464 updates; account pod list empty).
Pre-deletion retrieval failed on both, so their checkpoints and full metric
logs are lost; the watcher mirrors keep validation results through 5600 and
4800 respectively (see [chronology](CHRONOLOGY.md)). Neither arm produced a
held-out test. A single replacement pod is being provisioned for a long
overnight run of the v2 model; see [experiment 24](experiments/24-av-context-v2.md).
The cloud statuses in the older blocks below are historical.**

**Earlier 2026-09-15T04:36Z: a third independent scratch architecture run is
live on pod `f7y0zw02f4fzum`. It uses width-64 pre-update attention, width-64
post-update attention over `C/H/R`, and a terminal `P_T`-only spatial-priority
decoder. Required remote construction/gradient/lineage checks and profiling
passed; production has finite metrics. Its own 45-second watcher and hard-stop
lifecycle do not touch control pod `ce00y2ooosl7wc` or comparator pod
`h1ygacz4zcsfj3`. No validation result exists yet.**

**Earlier 2026-09-15T03:53Z status: the inherited spatial-priority-readout attempt was
cancelled immediately after a lineage correction. Its local monitors were
stopped and Palladio RunPod `mbi39b005jp884` was stopped, deleted, and confirmed
absent. Last known display step was 8720 (320 updates; 12,800 fresh episodes);
no validation completed. The trace is not an architecture result. A separate
fully scratch replacement, pod `ce00y2ooosl7wc`, was provisioned at 03:47Z.
Remote scratch-construction checks and a three-update profile passed. At the
verified launch snapshot it had reached update 46 / 1,840 episodes; no
validation had completed. The older prospective-query run remains
user-stopped, retrieved and deleted.**

## Fully scratch replacement: early-training snapshot

[Pinned scratch model](../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/model.py) ·
[pinned protocol](../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/protocol.py) ·
[lineage receipt](../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/lineage_correction_receipt.json) ·
[launch receipt](../WorkingMemory/SpatialPriorityReadout/launch_receipt.json) ·
[live status](../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_status.json)

Version `spatial_priority_readout_scratch_v2` constructs every model tensor from
documented seeds and starts with an empty optimizer state. It loads zero parent
tensors and zero inherited Adam states. The unchanged target is 4,000 updates /
160,000 five-task episodes. The provider-ready artifact records a requested
RTX 3090 at listed $0.22/hour and deadline `2026-09-15T11:47:20Z`. The
three-update profile averaged3.4266seconds/update with1.014GB peak allocation,
projecting5.51hours/$1.21 including margin and evaluation/retrieval reserve.
At the launch receipt snapshot, worker PID332, lifecycle PID36128 and watcher
PID36812 were active. Latest batch: loss0.82499, aggregate accuracy0.550,
pre-clip gradient norm0.59707. Checkpoint0 was indexed and no
validation had completed. This is live operational evidence, not a result; the
scratch run is not initialization-matched to the prior trained models.

### Complementary local frozen-core result

The [analysis-only local diagnostic](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
completed without touching the cloud run. From frozen motion-only step12200,
SHA256 `7cec4c48c3da81b4d4935c65098b58974a38a8ff3a825dfe12e4d8d16e9a0e26`,
a74,303-parameter pooled probe scored23.83% BA/0.4805 AUC and a74,373-parameter
spatial-priority probe24.32%/0.4832 on the same1,024 held-out delay
presentations (256 independent base movies). The paired BA gain was+0.49pp
(95% grouped bootstrap CI -1.76 to+2.64); no delay showed a reliable gain.
Both remain near25% chance.

The spatial map placed13.13% mass in an evaluation-only target region versus
5.33% uniform expectation, but this localization did not decode the motion
winner. This post-hoc negative decoding result neither substitutes for nor
predicts the end-to-end scratch experiment. Frozen checkpoint/state hashes
were unchanged; total local wall time was738.5seconds.

[Held-out predictions](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/predictions_test.jsonl) ·
[priority maps](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/priority_maps_test.npz) ·
[results](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/results.json) ·
[run manifest](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/run_manifest.json) ·
[completion receipt](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/completion_receipt.json)

## Cancelled spatial-priority-readout attempt

[Experiment 22](experiments/22-spatial-priority-readout.md) ·
[implementation](../WorkingMemory/SpatialPriorityReadout/README.md) ·
[historical run artifacts](../WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/) ·
[superseding cleanup receipt](../WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/lineage_correction_cleanup_receipt.json)

The launch inherited attention8400 model and Adam state, contrary to the
corrected requirement to initialize the complete trainable model and optimizer
from scratch. The cleanup receipt supersedes its “live” wording and snapshots.
No validation was completed before cancellation.

The sole forward change is the terminal decision readout. Final sensory
`H_T`, firing-rate `R_T` and comparator `C_T` fields remain aligned at 13×13,
are concatenated, and pass through 1×1 then 3×3 convolutions. Each task has a
spatial selection map and local class-evidence maps; a spatial softmax
weighted sum produces logits. Original attention queries and trained
source/locality biases are unchanged, and prospective gamma is absent.

That cancelled migration retained compatible tensors and 111 Adam states.
Those facts describe why it was invalidated; none of those inherited tensors
or states enter the replacement run.

**Historical immutable launch snapshot, not validation:** global step8512 /4,480 fresh
episodes, finite aggregate loss0.8246, accuracy0.45 and gradient norm0.4724.
Three profile updates averaged2.9833 seconds on the RTX4090, with1.032GB peak
allocated. Conservative training plus evaluation/retrieval projection is
about4.89hours/$1.66. The fixed target is step12400 /160,000fresh episodes;
the absolute deadline is `2026-09-15T10:56:06.615367Z`. No validation had
completed at this snapshot.

Detached lifecycle PID38436 and watcher PID30480 previously recorded rolling
aggregate metrics and GPU state. Both were stopped; the pod was deleted and
provider absence was confirmed by HTTP 404. No validation artifact was
produced.

## User-stopped prospective-query experiment

[Experiment 21](experiments/21-prospective-query.md) · [Completion report](../WorkingMemory/ProspectiveQuery/completion_report.md) · [Completion receipt](../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/completion_receipt.json) · [Cleanup receipt](../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/cleanup_receipt.json)

The sole architectural change is a learned scalar current-sensory residual in the pre-update query, initialized at zero:

`Q_t = W_Q(LN(R_{t-1}) + P + e_m + gamma * LN(H_t))`.

Focused tests show exact original behavior at `gamma=0`, a nonzero gamma gradient, query/routing sensitivity when enabled, strict original-attention8400 lineage, and unchanged source/locality terms and five-task protocol. The run stopped at logged step 9430 / 41,200 fresh episodes; durable checkpoint 9216 contains 32,640 episodes. The 214 later logged updates are not checkpointed.

The first setup/profile attempt consumed no training exposure. Its CRLF script and missing BSDS500 runtime payload were corrected without changing scientific settings, the source-hash ledger or the original deadline. The immutable source bundle is `a747fe…afa9`; the separately transferred exact dataset payload is `990e2b…883e`. Retrieval verified 29 artifacts. Pod deletion was confirmed by HTTP 404 and an empty pod list; `currentSpendPerHr` is zero.

Historical control comparisons are valid only at matched validation steps 9200/10000/10800/11600. The prior biased control was stopped before 12400/final held-out tests, so prospective terminal and final held-out scores will be unmatched exploratory results.

**Live validation 9200 (2026-09-15T01:54:44.3853582+00:00):**
[full matched table](../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/validation_9200_report.md).
The 1,836 prospective and historical-control examples match exactly.
Prospective improves orientation normalized BA by 0.1719 and binding by 0.3203,
but recognition falls by 0.0799 and motion duration by 0.0260; Krauzlis remains
at chance-normalized zero. Mean task AUC rank rises from 0.6186 to 0.6422 and
the minimum normalized BA from -0.1016 to -0.0260. This is the first planned
validation snapshot, not checkpoint selection or final held-out evidence.

[Local diagnostic summary](../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/diagnostic_summary.json): changing only valid cue rings on 32 paired identical-evidence movies changed the frozen baseline prediction in 46.88% (Wilson 95% interval 30.87–63.55%) and changed attention routing. However, accuracy and correct-class probability remained near chance, so this is cue sensitivity rather than successful duration-rule use. A split local pre-pooling linear probe scored 25.98% overall versus 25% chance (23.44% target, 26.82% foil), providing no evidence that this particular final-frame local summary exposes longest-duration direction. It does not prove absence from other times or representations.

## Final shutdown and saved cloud results

[Cleanup receipt](../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json) verifies stop `EXITED`, delete HTTP204 and an empty pod list. [Retrieval receipt](../WorkingMemory/SpatialTaskBattery/BiasedTraining/retrieval_receipt.json) verifies all73 saved artifacts. The local motion-only experiment is complete. No training, extraction or follow-up GPU experiment should resume without new authorization.

The cloud run is **user-stopped and partial**: logged global11923 /140,920new episodes; durable checkpoint11776 /135,040new episodes;147 logged updates were not checkpointed. Validation-selected so far is10000; last completed validation is11600. **No final held-out evaluation was performed.** These are distinct endpoints.

[Last completed validation11600](../WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md): binding100% across delays; recognition N4 96.88%,N12 82.81%,N24 71.88–73.44%; signed orientation42.19–53.12%; motion duration23.44–25%; Krauzlis BA50%. These partial validation scores do not describe the durable11776 checkpoint and must not be presented as final held-out results. [Experiment19](experiments/19-biased-spatial-battery.md) preserves the full table and history.

## Completed local branch: cued motion duration remains unlearned

[Experiment 20](experiments/20-single-task-motion.md) · [Final report](../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md) · [Completion receipt](../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_receipt.json)

All 2,200 added updates / 17,600 motion episodes finished. Selected checkpoint11760 scores 24.80% / 25.78% / 25.00% / 25.00% at D0/D4/D12/D24; terminal12200 scores 25.20% / 25.00% / 25.78% / 25.00%, versus parent 25% at all delays. All paired gain intervals include zero. Terminal AUC is about 0.51–0.52 with severe restriction to a few output classes. This continuation did not acquire the task; failure at D0 cannot be explained solely by blank-period retention.

Losses/gradients stayed finite; last200-update CE was1.3877, close to log(4). The receipt verifies97 immutable artifacts and no remaining local model workers. Training/evaluation took56.4minutes, final analysis59.2minutes, within the120-minute cap. The branch retains current architecture, cues, optimizer and learning rates while training full motion CE. Its failure does not establish a capacity limit, bug or interference mechanism.

The earlier cloud five-task run was stopped at the user’s request and its pod
was subsequently deleted. Its completed motion validation through11600 is also
near chance; [exposure-aligned trajectories](../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/trajectory_comparison.md)
use different validation draws and must not be treated as paired comparisons.
No extra local GPU training is running or implied by this result; the later
prospective-query and spatial-priority pods have also been deleted.

## Historical context: bias removal and restored-bias launch (now stopped)

[Completed/cancelled experiment 18](experiments/18-unbiased-attention-spatial-battery.md) · [Replacement experiment 19](experiments/19-biased-spatial-battery.md) · [Architecture/training implementation](../WorkingMemory/UnbiasedAttention/README.md) · [Five-task protocol](../WorkingMemory/SpatialTaskBattery/PROTOCOL.md) · [Rendered task previews](../WorkingMemory/SpatialTaskBattery/previews/index.html)

In the preceding bias-removal experiment, both arms independently started from attention 8400 and removed explicit source and distance penalties while preserving learned projections and compatible training state. **The completed old-task comparison is negative:** after equal 32,000-episode continuations, single-orientation D24 falls from 91.60% biased control to 57.62% with biases removed; binding D24 falls from 99.41% to 49.80%. Motion D0/D24 becomes 42.38%/42.38%, compared with 41.02%/41.99%; both change intervals include zero.

All five validation looks failed the preservation screen. The selected checkpoint is therefore the untouched **biased parent 8400**, not a trained bias-removed success. The terminal 12400 comparison reports the actual learned competitor. See [full 14-cell paired results](../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/comparison.md) and [findings](../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/findings.json). This is joint source/locality removal under one warm start and fixed budget; it cannot separate the two terms or establish a universal attention rule. Reused tests and cross-GPU training make it exploratory.

The user cancelled the five-task bias-removed arm at logged global 8550 (+150 updates / 6,000 episodes); the last durable checkpoint is 8448 (+48 updates / 1,920 episodes). [Cancellation receipt](../WorkingMemory/UnbiasedAttention/Cloud/cancellation_receipt.json) verifies worker/supervisor stopped and no GPU compute processes remained. Logged uncheckpointed updates are not a saved endpoint, and this is not a completed acquisition result.

The subsequent original-bias five-task run started from intact attention8400, preserving learned bias terms and compatible Adam state, with fresh task heads. Its launch receipt is historical evidence, not current process status. It was later stopped before its planned160,000 episodes, retrieved and deleted as recorded above. Broader latent/probe analysis remains paused.

## Corrected attention viewer: every timestep (2026-09-14T01:05:05.388626+00:00)

[Open the individual-frame viewer](../WorkingMemory/AttentionMaps/index.html) · [Journal correction history](experiments/17-attention-maps.md) · [Per-frame receipt](../WorkingMemory/AttentionMaps/perframe_receipt.json)

The first promoted figure showed one query and the overview averaged phases; those did not meet the user's request. The correction is now complete: all 13×13 receiving sites appear for **each of two heads and each of two source banks**, with cell values, a fixed 0–1 scale, actual scene/reference images, trial/task/delay selectors and every-timestep slider/playback. **No temporal or trial averaging** is used in the primary view. The optional source-key view averages queries within the selected frame only.

Capture covers 48 movies: eight paired base trials per task family × three families ×D0/D24, totaling 24 base trials and 912 frame observations. It took 6.76 seconds within the original AttentionMaps deadline; no extra budget, model update or intervention occurred. It uses frozen attention 8400, not the newly continued checkpoints. Broader latent/probe work remains paused.

[Cached locality evidence](../WorkingMemory/AttentionMaps/locality_summary.json) explains why the original one-query map looked like one bright cell: representative orientation frames put about 94% joint attention at the query's own spatial coordinate, under a trained distance penalty about 4.03. This is local source routing, not evidence that the network failed to attend. The earlier phase averages below are historical aggregates, not instantaneous maps.

## Initial attention maps (historical summary)

[Open the interactive viewer](../WorkingMemory/AttentionMaps/index.html) · [Journal entry](experiments/17-attention-maps.md) · [Original report](../WorkingMemory/AttentionMaps/report.md)

This is frozen **attention 8400**, the parent of the latest training comparison. Each of its two heads can read both current-sensory and old-memory sources; the viewer separates both banks within each head, shows condition means, and supports representative receiving-query inspection. A remembered-scene overlay is spatial context, not a reconstruction of memory.

Capture used 192 independent base episodes paired acrossD 0/D24, giving 384 presentations and 208 condition groups. It completed in 30.45 seconds with the checkpoint unchanged. Head 2 memory mass during blanks averages 77.75% for single orientation,84.71% for binding and 85.19% for motion; head 1 is around 12%. These totals show source allocation, not spatial focus or proof of retained task content. See [completion evidence](../WorkingMemory/AttentionMaps/completion_receipt.json).

## Completed training-allocation comparison

[Stage1 journal](experiments/16-training-exposure.md) · [Final report](../WorkingMemory/TrainingExposure/report.md) · [Full analysis](../WorkingMemory/TrainingExposure/analysis.json)

Both arms start the same attention 8400 checkpoint and complete 4,000 additional updates /32,000 episodes. Control retains 10% motion training; focused allocates 50%. Architecture, tasks, losses and inherited learning-rate policy are unchanged. Control selects terminal 12400; focused selects 11600 but also reports terminal 12400.

| Fresh held-out cell | Parent8400 | Control10% selected/terminal12400 | Focused50% selected11600 | Focused50% terminal12400 |
|---|---:|---:|---:|---:|
| Motion D0 |38.28%|41.02%|65.62%|49.80%|
| Motion D24 |34.57%|41.99%|57.62%|55.08%|
| Single orientation D12 |92.97%|96.88%|90.04%|90.23%|
| Single orientation D24 |80.47%|91.60%|83.98%|84.38%|

Ordinary continuation improves delayed orientation strongly while recovering some motion. Increased motion allocation recovers substantially more motion but gives weaker orientation outcomes. Focused single D12 falls 2.93 pp below parent (paired 95% CI−5.47 to−0.59); control has no observed point declines across 14 cells. Neither result establishes simultaneous all-cell preservation against the 2 pp margin. Preserve both checkpoints rather than naming an all-task winner.

Both have equal terminal exposure; their **selected** exposures differ:32,000 new episodes for control versus 25,600 for focused. The selected-model comparison includes the declared checkpoint selection; the terminal columns provide equal-endpoint exposure. Different local/cloud hardware remains a qualification.

The focused terminal model also develops severe up-choice bias: on the same 512 D0 test trials, down predictions fall 113→2 and up predictions 123→308 from selected 11600 to terminal 12400. AUC also decreases. [The dated instability note](experiments/16-training-exposure.md) records the evidence; shared-task interference, representation/head mismatch and noisy constant-step continuation are still hypotheses rather than established mechanisms.

## Completed capabilities and their boundaries

| Capability | Evidence | Boundary |
|---|---|---|
| Two-step causal sensory discrimination | Opponent temporal model98.44–100% across7tasks | Does not establish long memory |
| Useful early motion in E/I state | Frozen-state linear readout gains9.47pp | Accessibility can exceed deployed decision quality |
| Motion judgment retained through24blanks | Earlier Retention14800 about79–80% across delays | This is a preserved earlier model/task evaluation, not the new attention model's score |
| Spatial feature-location binding | Spatial4400 bindingD24 93.36%; later models near99% | Full swaps can be detected using one remembered location; no two-item capacity claim |
| Improved native delayed orientation | Attention8400 79.30% versus58.40%continuation; latest control91.60%on its new test | Different experiments/draws must not be treated as one paired comparison |
| Dependence on blank-period memory routing | Frozen exclusion79.30%→50.00% | Does not separate refresh from rejecting blank drive |

The project has useful sensory, memory and attention components, not a complete Guided Search system or a biologically validated architecture. General search, independent multi-item capacity, broad transfer and seed replication remain unestablished.

## What is paused or unrun

The broader [LatentDynamics investigation](../WorkingMemory/LatentDynamics/RESEARCH.md) was [paused before extraction](../WorkingMemory/LatentDynamics/queue_receipt.json) at the user's request. Its unspent allowance was reused for the narrower attention maps. No UMAP/t-SNE, latent probes, new training, attention interventions or cloud restart were performed for that capture.

The proposed temporal-feature residual remains Stage2 design only. No optimizer sweep is included. The new authorization is specifically the two bias-removal arms above; it does not restart the paused latent/probe investigation.

## Preserved model ancestry

- Contour-focused PAV 2268 → causal opponent 4032 → broad sequence selected 6860.
- Sequence 6860 → LSTM 5000 and dense E/I 5000.
- E/I 5000 → existing-output refit 9840 → retention 14800.
- Retention 14800 → dense-comparator 4400 and spatial 4400, with a newly initialized spatial core.
- Spatial 4400 → ordinary continuation 8400, additive-feedback 8400 and pre-update-attention 8400.
- Attention 8400 → control 10% selected/terminal 12400 and focused 50% selected 11600/terminal 12400.

Step numbers are experiment-specific and do not imply equal cumulative lineage exposure. [Architecture math and tensor shapes](ARCHITECTURE.md) describe the unchanged attention architecture.

## Previous experiment execution closure

Local TrainingExposure [exit receipt](../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/exit.json): completed without error in 4,546.23 seconds. Cloud [retrieval](../WorkingMemory/TrainingExposure/Cloud/retrieval_receipt.json) verified 141 files; [cleanup](../WorkingMemory/TrainingExposure/Cloud/cleanup_receipt.json) deleted the pod at2026-09-14T00:21:33.450146Z, estimated cost$0.165 rather than an invoice. [AttentionMaps completion](../WorkingMemory/AttentionMaps/completion_receipt.json) records 30.45 seconds and no model changes. No compute allowance is automatically renewed.
