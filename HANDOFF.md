# Agent handoff — Visual Attention and Working Memory

Prepared 2026-09-14 from the implementation, experiment reports and cleanup receipts. This is a handoff, not authorization to restart experiments.

## Update 2026-09-15T22:05-07:00

The user authorized and launched a local five-task scratch run of the
five-change AV-context v2 arm. See
[experiment 24](LabJournal/experiments/24-av-context-v2.md) and the
[V2 README](WorkingMemory/AttentionContextComparator/V2/README.md). Monitor
with `python WorkingMemory/AttentionContextComparator/V2/status.py`. The
supervisor applies a pre-registered stopping rule at validation 2,400 on its
own. The material below predates this run.

## Start here

We have a functioning sensory / spatial recurrent-memory / pre-update attention model, but the newest spatial task battery exposes major acquisition failures. It learns spatial binding and image-set recognition. It has not learned the new cued signed-orientation and motion tasks. A completed motion-only continuation also stayed at chance, including with zero inserted blank frames.

**All requested training is stopped or complete. The latest prospective-query cloud pod was deleted after verified retrieval, and no monitor remains active. Do not relaunch anything from an old plan or launcher.** The latest user action was to stop the prospective-query run after its first matched validation failed to improve the primary cued motion-duration result.

Workspace: `C:/Users/jomor/Documents/VisualAttentionWorkingMemory`; Windows / PowerShell. Start with this file, then [current journal status](LabJournal/CURRENT_STATUS.md), [cloud partial report](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md), and [local final report](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md).

## User intent and working rules

- Build components incrementally, using Jeremy Wolfe's Guided Search 6.0 as a functional scaffold. Omit the diffuser; treating activated long-term memory as synaptic weights is the user's approximation, not Wolfe's claim.
- PAV is only the sensory component, not the entire project. The repository was deliberately reset; do not recover deleted pre-reset architectures or plans.
- The user wants **study, learn, justify, build**. They reject repeatedly training a failing design without investigating its failure. Explain neural-network computations directly, with equations and tensor shapes when useful.
- Researchers implement and execute experiments. Use existing researcher agents where available. Root coordinates, reads results and handles cloud lifecycle. Maintain the wiki-style `LabJournal/` alongside experiment reports.
- Use primary ML and neuroscience sources for new designs. Distinguish biological inspiration from measured performance and biological validation.
- Do not silently change stimuli, cues, losses, curricula or supervision to make a model succeed. The user explicitly objected to replacing known failures with an easier teaching task.
- Avoid repeated approval gates and broad validation campaigns. Use proportionate checks and finite compute. Historical budgets are closed, not reusable allowances.
- Current authorization is documentation/handoff. Ordinary read-only investigation is fine; no new training, cloud provisioning, budget renewal or paused latent extraction is implied.

## Operational closure

Latest closure: prospective-query pod `dqi13o2x3qkvos` was user-stopped at
logged step9430 /41,200 episodes; durable checkpoint9216 contains32,640
episodes. All29 artifacts were verified locally, the provider reported
`EXITED`, deletion succeeded, subsequent lookup returned404, the pod list was
empty and account spend was $0/hour. See
[completion](WorkingMemory/ProspectiveQuery/completion_report.md) and
[cleanup](WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/cleanup_receipt.json).

| Item | Verified state / evidence |
|---|---|
| Cloud pod | `ep8bmcjxz9k66l`, NVIDIA L40 48 GB, Palladio account; **deleted** |
| Cleanup | Stop returned `EXITED`; delete returned success / HTTP 204; subsequent pod listing was empty |
| Cleanup timestamp | 2026-09-14 05:52:23 UTC; this is a recorded shutdown check, not a claim of a fresh account check on every read |
| Cloud retrieval | All **73** frozen result files, including saved checkpoints, checksum-verified locally |
| Local motion run | Completed; all **97** immutable run artifacts verified; model workers exited |
| Monitor | `finish-attention-runpod-experiment`, name `Completed training cleanup`, **PAUSED** |

Evidence: [cloud retrieval](WorkingMemory/SpatialTaskBattery/BiasedTraining/retrieval_receipt.json), [API cleanup](WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json), [user stop](WorkingMemory/SpatialTaskBattery/BiasedTraining/user_stop_receipt.json), [local completion](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_receipt.json).

Some launch-time READMEs/protocols still describe a running job. Their historical tense is not operational authority; cleanup receipts and the user's stop instruction supersede them. Do not reconnect to the deleted pod or restart a watcher to satisfy stale text.

## Architecture currently being tested

Implementation entry points:

- [BiasedTraining/model.py](WorkingMemory/SpatialTaskBattery/BiasedTraining/model.py): current five-task wrapper and migration; version `spatial_five_task_original_attention_biases_v1`.
- [PreUpdateAttention/model.py](WorkingMemory/PreUpdateAttention/model.py): attention and per-frame computation.
- [SpatialComparison/model.py](WorkingMemory/SpatialComparison/model.py): spatial E/I dynamics, normalization and comparator.
- [TemporalIntegration/accumulators.py](PreAttentiveVision/TemporalIntegration/accumulators.py): sensory temporal computations.

The model processes RGB frames sequentially. Its learned convolutional sensory encoder and opponent temporal computations produce a field `H_t: [B,64,13,13]`. Opponent trace state persists within the trial. Do not confuse this with the obsolete vector-memory bottleneck: the current memory is spatial.

Previous firing rates `R_(t-1)` and adaptation `A_(t-1)` each have shape `[B,64,13,13]`. Memory and sensory traces reset at the beginning of each episode. There is no growing frame cache.

### Pre-update joint attention

Flatten space into 169 tokens while keeping 64 features per token. Previous memory supplies 169 queries. Concatenate 169 current sensory tokens and 169 previous-memory tokens to supply 338 keys and values. Two heads each have width 32.

With learned position `P` and source embeddings `e_v,e_m`:

\[
Q=W_Q(\operatorname{LN}_q(R)+P+e_m),\qquad
K=W_K(\operatorname{LN}_k([H;R])+[P+e_v;P+e_m]),\qquad
V=W_V[H;R].
\]

For head h, query i and source token j:

\[
L_{hij}=\frac{Q_{hi}\cdot K_{hj}}{\sqrt{32}}
+b_{h,\operatorname{source}(j)}
-\operatorname{softplus}(\ell_h)\|p_i-p_j\|^2,
\qquad U=W_O\operatorname{concat}_h(\operatorname{softmax}_j(L_h)V_h).
\]

Attention weights have shape `[B,2,169,338]`; reshape U to `[B,64,13,13]`. **Both heads can read both sources**; they are not a dedicated visual head and a dedicated memory head. Source bias and locality penalty are currently present and trainable. Learned source/position embeddings are separate parameters.

### Spatial E/I update and decision

Apply channel LayerNorm independently at each location to U. The E/I core has a learned 1×1 input projection, signed 3×3 recurrent convolution, 51 excitatory and 13 inhibitory channels, ReLU firing rates and adaptation. Synapse signs are fixed by presynaptic channel; magnitudes are learned through softplus.

\[
R_t=(1-\alpha)R_{t-1}+\alpha\operatorname{ReLU}(W_{in}z_t+K*R_{t-1}-gA_{t-1}+b),
\]
\[
A_t=(1-\beta)A_{t-1}+\beta R_{t-1}.
\]

The implementation uses **previous** rates in the adaptation update. Per-channel learned bounded time constants give `alpha=1-exp(-1/tau_r)` and `beta=1-exp(-1/tau_a)`; `tau_r` is bounded 1–32 frames, `tau_a` 4–128, and adaptation gain 0–0.5.

In parallel, a comparator processes concatenated **old memory and current H** through 1×1 and 3×3 convolutions. At the final frame, spatial mean/max pooled sensory, updated-memory and comparator branches each produce a 128-dimensional contribution; these are summed before the task-specific linear head. Attention supplies the external drive into memory; the separate sensory/comparator decision paths still exist.

The five-task model has **557,676 parameters**: 556,128 inherited parameters plus 1,548 parameters for five new heads. All learned components used by the task train; unused legacy heads receive no gradients. Fixed Gabor kernels, prescribed sensory retention coefficients and energy equations remain fixed. This is not a fully learned replacement of those predefined computations, nor a biologically validated circuit.

## Initialization and shared training — avoid ambiguity

The cloud five-task run was **continued training**, not random initialization. It started independently from original intact **attention8400**, with compatible learned tensors and 123 Adam parameter-state entries retained. Five new semantic heads were initialized fresh. It did not inherit the cancelled bias-free run's degraded weights.

All five tasks share encoder, temporal processing, attention, memory and comparator weights. Only the small output heads differ. Each cloud update processes five microbatches of eight episodes, averages their five mean cross-entropies, clips the combined gradient at 1, then performs one Adam update. Full BPTT, fp32; inherited sensory LR `3e-5`, memory/attention/comparator/new-head LR `3e-4`, Adam epsilon `1e-10`. Task identity selects the output head; this is not unconstrained inference of which task to perform.

## Current five-task battery

Canonical specification: [PROTOCOL.md](WorkingMemory/SpatialTaskBattery/PROTOCOL.md), [stimuli.py](WorkingMemory/SpatialTaskBattery/stimuli.py), [SOURCES.md](WorkingMemory/SpatialTaskBattery/SOURCES.md). [Actual rendered previews](WorkingMemory/SpatialTaskBattery/previews/index.html) are useful before interpreting failures.

| Task | What the model must do |
|---|---|
| `orientation_cued` | Four Gabors. A local plus/minus cue identifies both the relevant location and relevant rotation sign. Report whether its sample-to-probe rotation matches that sign; negatives include no rotation and opposite rotation. Rotation magnitudes 15/30/45°. |
| `motion_duration_cued` | Four independent dot patches. A ring selects one patch. Across eight cardinal-direction transitions, report that patch's direction with greatest total duration. Four classes, chance 25%. Cue stays present during evidence. |
| `krauzlis_cued_motion` | Two dot patches, brief target precue, then detect a small mean-direction change in the target while ignoring foil changes. Based on Arcizet & Krauzlis 2018; disclosed 100×100 and compressed-timing adaptations, not a replication. |
| `spatial_binding` | Four orientations are shown; after the delay a retrocue selects a location. Report whether a swap involved that location. Every trial contains one swap, so total global change does not reveal the answer. |
| `image_recognition` | Show 0/4/12/24 consecutive natural images, one frame each, then **three blanks total**, then a query repeated identically for 3/4/5 frames. Report exact membership. Positive raster is identical to a study image. |

Orientation, duration and binding use D=0/4/12/24 inserted blanks. Four-region centers are `(27,27),(73,27),(27,73),(73,73)`. Recognition uses 500 BSDS images with official identity-separated splits; empty lists are negative-only controls, excluded from BA/AUC averages. Krauzlis uses 12/20/28 baseline transitions and target/foil/catch proportions 57/29/14%; always saying target-change gives 57% ordinary accuracy but only 50% balanced accuracy.

These are **new tasks**, not the old single-field motion / same-versus-different orientation tasks on which related checkpoints succeeded. Spatial cue use, multiple competing patches, rotation-sign rules, small direction changes and image lists add demands. Do not interpret a cross-battery score change as an isolated architecture effect.

## Most recent results

### Cloud five-task run — stopped early by the user

Planned: 4,000 added updates / 160,000 episodes. Logged at stop: **3,523 updates / 140,920 episodes**, global11923. Latest saved checkpoint: **11776**, representing **3,376 updates / 135,040 episodes**. The last 147 logged updates / 5,880 episodes were not checkpointed.

**No final held-out test was performed.** Last completed validation was **11600**; selection-so-far chose **10000** under the minimum chance-normalized family BA / mean AUC rule. These are three different saved/evaluated endpoints; never attach validation11600 scores to durable11776.

| Last validation11600 | Balanced accuracy |
|---|---:|
| Spatial binding, all delays | 100% |
| Recognition, list4, all query holds | 96.88% |
| Recognition, list12, all query holds | 82.81% |
| Recognition, list24, holds3/4/5 | 71.88 / 71.88 / 73.44% |
| Orientation, D0/4/12/24 | 42.19 / 53.12 / 45.31 / 46.88% |
| Motion duration, D0/4/12/24 | 25.00 / 23.44 / 25.00 / 25.00% |
| Krauzlis, all three baseline lengths | 50% |

Non-Krauzlis validation cells contain 64 examples; Krauzlis cells contain 100. Empty-list recognition accuracy was 100%, with BA undefined. Report per-condition results rather than hiding weak tasks in an aggregate.

### Local motion-only run — complete, negative result

Branched from the cloud's **10000** checkpoint, which had already seen 12,800 new-battery motion episodes. Restored model, 133 Adam states, RNG, motion stream and delay scheduler. Trained **only** motion duration for 2,200 updates / **17,600 additional episodes**, batch8. All used learned components remained trainable. Profiling reduced the initial target to fit a new two-hour cap; training/evaluation took 56.4 minutes and final analysis 59.2 minutes. No worker remains.

| Blanks | Parent10000 | Selected11760 | Terminal12200 |
|---|---:|---:|---:|
| 0 | 25.00% | 24.80% | 25.20% |
| 4 | 25.00% | 25.78% | 25.00% |
| 12 | 25.00% | 25.00% | 25.78% |
| 24 | 25.00% | 25.00% | 25.00% |

Final tests use 512 independent base episodes paired across delays and models, not 2,048 independent histories. All paired gain intervals include zero. Output AUC is roughly 0.50–0.52. Strong class collapse remains; selected D12/D24 outputs one class for every example. Last-200-update mean CE=1.3877, near `log(4)=1.3863`; losses/gradients were finite.

This amount of task-only continuation **did not teach the task**. Failure already at D0 cannot be explained only by losing information during inserted blanks. It does not by itself establish an architecture capacity limit, a generator bug, or a specific optimization mechanism. Cloud/local validation draws differ; their trajectory comparison is unpaired. Local full motion CE also differs from the cloud's five-loss average, so this is not a pure gradient-conflict intervention.

## Checkpoint and evidence map

Paths below are relative to the repository. Step numbers are not globally unique: identify the run and version as well as the step.

| Purpose | Location |
|---|---|
| Original attention8400 | `WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/checkpoint_008400.pt` |
| Cloud retrieved result root | `WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/retrieved/biased_results/` |
| Cloud current saved weights | Under that root: `spatial_biased/training/checkpoint_011776.pt` |
| Cloud selected-so-far / local branch parent | Same directory: `checkpoint_010000.pt` |
| Cloud last evaluated checkpoint | Same directory: `checkpoint_011600.pt` |
| Local run root | `WorkingMemory/SpatialTaskBattery/SingleTaskMotion/runs/motion_20260913_214605/` |
| Local selected / terminal | Under local root: `training/checkpoint_011760.pt`, `training/checkpoint_012200.pt` |
| Local detailed findings | `WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_findings.json` |
| Cloud actual config/lineage/logs | Retrieved result root plus `BiasedTraining/construction_checks.json`, launch/retrieval/stop receipts |

Recorded SHA256:

- Original attention8400: `e37602aa20ccfc400ea8fe9d98c11f29c508069388897c55803b97f2ccdf1bc9`.
- Cloud10000 / local parent: `35281f264131e01678ab5725a5b66816d47302583f78b9751f775f18adf22942`.
- Cloud durable11776: `8014155821a1df1456c580be6dcdc7856a9fd969e4458444b627d0329cabea5c`.

Use checkpoint config/version and source receipts to reconstruct models; do not indiscriminately run a migration function on an already migrated checkpoint. Existing local worker restoration code is a reference, not a command to launch another training run.

## How we reached this point

1. PAV compared five lightweight encoders on seven two-frame tasks. More contour allocation improved the selected convolutional reference. The causal opponent temporal model subsequently reached 98.44–100% across those seven tasks. This does not establish long-sequence or multi-patch ability.
2. LSTM versus E/I recurrent-memory experiments exposed ordering sensitivity and readout limitations. A frozen E/I-state diagnostic found accessible early motion evidence; output-only refitting improved deployed long-motion decisions to about79%.
3. Delayed orientation remained weak in an earlier vector E/I model. A frozen diagnostic nevertheless decoded sample orientation at D24 with about4.25° error; an analysis-only comparator reached73.44% versus deployed50%. This distinguished accessible information from successful use, for that earlier model/task.
4. Spatial64×13×13 E/I memory improved binding but did not alone resolve delayed orientation. Pre-update joint sensory/memory attention subsequently improved delayed orientation. Acute memory-source exclusion during blanks harmed it, but did not uniquely distinguish refreshing memory from rejecting blank drive.
5. Old-task training allocation tests recovered some motion at the cost of weaker orientation and unstable terminal direction predictions. There was no clear all-task winner. See [experiment16](LabJournal/experiments/16-training-exposure.md).
6. Attention maps showed a nearly constant first-head source preference and strong locality. Removing both explicit source bias and locality together hurt old-task D24 orientation (91.60→57.62%) and binding (99.41→49.80%), with no reliable motion gain. This one continuation cannot separate the two terms or show unbiased attention universally fails. See [experiment18](LabJournal/experiments/18-unbiased-attention-spatial-battery.md).
7. The new five-task bias-free attempt was cancelled early. The user restored the original bias terms and launched the independent intact-parent cloud run described above. It was stopped for the night, while the local task-only branch completed without acquisition.

## Visualization and paused work

`WorkingMemory/AttentionMaps/index.html` is the corrected **per-timestep** viewer: 48 movies, 912 frames, old attention8400, three old task families, D0/D24. Four full13×13 grids show each head × source bank. Do not present it as the current five-task model or as evidence from cloud11776.

The user rejected single-query one-pixel maps and time-averaged maps. Preserve the corrected viewer; `frame_view.py` is the relevant renderer. Do not blindly run older `render.py`. Source allocation is not image-pixel attribution; remembered-scene overlays are spatial references, not reconstructions. Constant routing weights do not imply constant value vectors or outputs.

Broader `WorkingMemory/LatentDynamics` probes / PCA / UMAP / t-SNE remain paused. No authorization to restart them is inherited from this handoff.

## Unresolved questions and a useful next conversation

The immediate weakness is **acquiring the relevant motion / signed-change decision**, not just retaining an already correct decision through blanks. Binding success suggests some location-cue use, but is not a causal demonstration of general attention. The motion-only failure means competing task updates are not a sufficient explanation for the failure under this continuation; inherited weights/Adam, feature availability, optimization and task difficulty remain possible factors.

Possible next investigations to discuss, **not launched or pre-authorized**:

- Inspect actual movies, labels and saved prediction confusions to separate sensory difficulty from cue-dependent selection/rule use.
- Pair identical multi-patch evidence with different valid cues that change the required answer. This would test cue influence more directly than comparing unrelated trials.
- Compare available evidence before memory with the final decision, using analysis-only probes with appropriate splits if authorized. Probe failure alone would not establish erasure.
- If training a diagnostic simplification is requested, change one factor explicitly: single versus multiple patches, cue selection, duration integration, or delay. Do not silently rewrite the main battery or introduce privileged supervision.

Start by telling the user that the local focused run did not solve acquisition and that the cloud results are partial validation. Bring a concrete hypothesis and the smallest discriminating experiment rather than another unexplained long run. Preserve the models that already work on the earlier task battery.
