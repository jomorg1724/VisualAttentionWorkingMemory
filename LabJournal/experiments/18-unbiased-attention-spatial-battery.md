# 18 — Remove explicit attention biases; test spatial selection and recognition

[Index](../README.md) · [Current status](../CURRENT_STATUS.md) · [Previous: attention maps](17-attention-maps.md)

**Status as of 2026-09-14T02:44:04.427099+00:00: old-task comparison complete and negative; the five-task bias-removed arm was cancelled by the user. A separately authorized original-bias restart now trains from intact attention 8400; its launch evidence is in entry 19.**

## User cancellation, followed by a separately authorized restart (2026-09-14T02:44:04.427099+00:00)

The [cancellation receipt](../../WorkingMemory/UnbiasedAttention/Cloud/cancellation_receipt.json) records the five-task bias-removed worker/supervisor stopped at 2026-09-14T02:41:38.123993+00:00, with no remaining GPU compute processes. Last logged global step was 8550: +150 updates / 6,000 episodes. The last durable checkpoint is 8448: +48 updates / 1,920 episodes. The remaining 102 logged updates were not checkpointed. Do not treat the logged endpoint as a saved model or this cancelled arm as a completed task-acquisition result.

The user then explicitly requested restoring the original bias terms and launching again. This is a **separate restart from intact attention 8400**, not continuation of the degraded old-task or partially trained five-task bias-removed weights. The same healthy L40 pod is retained. It preserves the five-task protocol, fresh head/stream initialization policy, five batch-8 family microbatches per update and inherited learning rates. Target remains 4,000 updates / 160,000 episodes within the original 09:36:25.001 UTC deadline. Researchers are preparing `WorkingMemory/SpatialTaskBattery/BiasedTraining`; no production launch is verified at this snapshot. See [the replacement experiment](19-biased-spatial-battery.md). The completed old-task paired comparison below remains unchanged.

## Completed old-task result (2026-09-14T02:40:33.400826+00:00)

Both the saved biased control and trained bias-removed model reached terminal 12400 after 4,000 additional updates / 32,000 episodes from the same parent. The paired results show a large, delay-dependent loss in binding and single-orientation accuracy. Motion differences are small and their intervals include zero.

| Task / inserted blanks | Biased control | Biases removed | Change (paired 95% interval) |
|---|---:|---:|---:|
| motion_D0 | 41.02% | 42.38% | +1.37pp [-0.98, +3.52] |
| motion_D24 | 41.99% | 42.38% | +0.39pp [-2.54, +3.52] |
| single_D0 | 99.80% | 98.24% | -1.56pp [-2.93, -0.39] |
| single_D4 | 99.80% | 99.22% | -0.59pp [-1.56, +0.20] |
| single_D12 | 96.88% | 82.81% | -14.06pp [-17.58, -10.74] |
| single_D24 | 91.60% | 57.62% | -33.98pp [-38.67, -29.10] |
| binding_D0 | 99.41% | 97.85% | -1.56pp [-2.93, -0.39] |
| binding_D4 | 99.41% | 93.36% | -6.05pp [-8.20, -3.71] |
| binding_D12 | 99.61% | 58.98% | -40.62pp [-43.95, -37.50] |
| binding_D24 | 99.41% | 49.80% | -49.61pp [-52.73, -46.29] |
| binding_D0_locations | 99.61% | 98.24% | -1.37pp [-2.73, -0.20] |
| binding_D4_locations | 99.61% | 94.34% | -5.27pp [-7.23, -3.32] |
| binding_D12_locations | 99.61% | 59.38% | -40.23pp [-43.55, -37.30] |
| binding_D24_locations | 99.61% | 53.91% | -45.70pp [-48.83, -42.38] |

**All five unbiased validation looks failed the preservation screen.** The automatic selected model is the untouched, still-biased parent 8400. It is not a successfully trained unbiased checkpoint. The table deliberately compares the actual terminal bias-removed model with the equally exposed terminal biased control; reporting parent fallback as the learned competitor would conceal this negative result.

The comparison covers 14 cells × 512 scored presentations = 7,168 presentations. There are 2,048 distinct family/mode base IDs, with repeated evidence paired across delays rather than counted as extra independent episodes. Paired metadata matched exactly. Intervals use 2,000 paired bootstrap replicates, preserving complete four-case counterbalance blocks for binding and class-stratified episode resampling otherwise. They describe test-sample uncertainty, not seed-to-seed training variability. The reused test and hardware qualifications above still apply.

This supports the narrow conclusion that **joint removal of explicit source and locality terms was harmful in this warm start and fixed training allocation**. It does not identify which term mattered, prove that unforced attention cannot learn, or establish a universal biological requirement. Immediate binding remains 97.85% while D24 is 49.80%, so a single aggregate would hide the most important change. No model was recalibrated or retrained to obtain this report.

[Completed paired report](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/comparison.md) · [Machine-readable findings](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/findings.json) · [Full metrics/confusions](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/report.md) · [Paired comparison data](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/paired_terminal_comparison.json).

The researcher reports retrieval/hash verification of the 73 old-arm files and partial archive. The separate five-task acquisition arm continues unchanged on the same pod: latest researcher-observed global 8502, +102 updates / 4,080 episodes, finite loss 0.8673. This progress is provisional and is not a new-task performance result. Target remains 4,000 updates / 160,000 episodes and deadline 2026-09-14 09:36:25.001 UTC. No additional GPU evaluation or follow-up experiment was launched for this documentation update.

## Why this follows from the observations

The corrected individual-frame maps made a strong spatial prior visible. Representative frames route about 94% of joint attention to the query's own coordinate, with learned locality coefficients 4.0316 and 4.0276. This does not prove failed attention: a local query can still choose sensory versus remembered values. But explicit distance and source biases constrain what learned content can select, motivating the user's request to remove those biases and train.

The request has two distinct parts. An old-task continuation isolates the migration as far as the saved comparison permits. A new spatial battery teaches genuinely different spatial-selection and recognition demands. Its scores cannot be treated as paired with old-task scores.

## Architectural change and preserved parent

Both arms independently migrate the same trained attention 8400 checkpoint. Remove the two-head `source_bias` and `softplus(raw_locality) × distance` terms from attention logits, plus the unused distance buffer. Preserve learned query, key, value and output projections, source/position embeddings, normalization, opponent sensory computations, spatial E/I recurrence and comparator. Do not reinitialize Q/K.

For each of two heads, 169 old-memory queries attend jointly to 169 current-sensory and 169 old-memory tokens:

\[
A_h=\operatorname{softmax}_{338}\!\left(Q_hK_h^\top/\sqrt{32}\right),\qquad U_h=A_hV_h.
\]

Here `Q_h:[B,169,32]`, `K_h,V_h:[B,338,32]`, `A_h:[B,169,338]`; the existing output projection returns a 64-channel field on 13 × 13. This removes explicit scalar priors, not all possible learned positional/source preferences. The removal changes the function immediately; it is a warm migration, not an initially equivalent model.

The implementation records six fewer parameters (556,122 remaining) and 121 compatible Adam states copied by name. Five fresh linear task heads add 1,548 parameters in the new battery. Those counts are implementation records, not evidence of successful training. [Model and optimizer details](../../WorkingMemory/UnbiasedAttention/README.md) · [Model code](../../WorkingMemory/UnbiasedAttention/model.py).

## Arm A: unchanged old tasks

Continue for exactly 4,000 updates / 32,000 fresh episodes, batch 8, under the inherited 80-update 10%-motion schedule. Preserve compatible Adam, RNG and task-local streams and the existing learning rates. Compare with the saved equally exposed biased TrainingExposure control, which starts the same attention 8400 parent. No new biased-control training is included.

Use the existing validation seed 53973001 at five 800-update looks, 128 examples per cell, and final seed 54973001 with 512 per cell including held-out binding positions. Selection retains the prior 2 pp trained-cell preservation screen against parent, then minimum motion BA and mean motion AUC; parent fallback and terminal unbiased scores remain explicit. Those evaluation draws have already been examined in TrainingExposure, so this is an **exploratory reused-test comparison**, with a local/cloud hardware difference, not a pristine confirmatory test.

## Arm B: five newly taught tasks

All inputs remain 100 × 100 RGB, and only rendered frames enter the model. Targets, event indices, source IDs and phases are supervision/analysis metadata, not privileged network inputs. [Full executable protocol](../../WorkingMemory/SpatialTaskBattery/PROTOCOL.md) · [Primary sources and adaptations](../../WorkingMemory/SpatialTaskBattery/SOURCES.md) · [Actual stimulus previews](../../WorkingMemory/SpatialTaskBattery/previews/index.html).

| Family | Required judgment and controls | Sequence length |
|---|---|---|
| Cued signed orientation | Four Gabors; existing ± glyph precues a location and relevant rotation sign. Report whether its single sample-to-probe rotation matches that sign. Foil rotation histograms are balanced across labels. | D + 4; D = 0/4/12/24 |
| Cued motion duration | Four independent moving-dot patches; ring selects the patch. Report the cardinal direction occupying most of its eight transitions. No arbitrary sign filter. | D + 11 |
| Spatial orientation binding | Four distinct orientations; target ring appears after the delay. Every probe exchanges exactly one pair: target-involving versus foil-only. Both labels have two changed positions and the same orientation inventory. | D + 5 |
| Exact image-set recognition | L consecutive one-frame study photographs, then three blank frames total, then a probe repeated K times. Decide whether it was in the study set. | L + 4 + K; L = 0/4/12/24, K = 3/4/5 |
| Krauzlis target/foil change | Flashed ring selects one of two motion patches. Report a target direction change; ignore a foil change and reject catch trials. | Npre + 17; Npre = 12/20/28 |

Recognition uses source-disjoint BSDS500 splits (200/100/200) and canonical 100 × 100 rasters. Positive probes are bit-identical to a study raster; negative source IDs are absent from that episode's set. Load 0 is always negative, evaluated as specificity/false-positive rate only. It cannot provide defined balanced accuracy or AUC and is excluded from checkpoint ranking. This is a learned exact-set membership benchmark, not natural-image semantic recognition or a human capacity estimate.

The Krauzlis recipe is specifically Arcizet and Krauzlis (2018), with target/foil/catch proportions 57/29/14, 16-degree dot-direction dispersion, 10-frame lifetime and 15 degrees/second speed. At a declared 100 Hz and 2.5 pixels/degree, displacement is 0.375 pixels/frame. The 100 × 100 implementation uses 16 dots per patch, shorter cue/baseline/post-event intervals and an end-of-trial classifier. It does not reproduce the primary joystick reaction-time task. Separate hit, foil false-alarm and catch false-positive results are necessary. The final sampler preserves 57/29/14 exactly per 100 trials; target-side assignment reverses across successive cycles.

## Exposure, validation and outcome recording

Each new-battery update processes five batch-8 family microbatches, averages their mean losses, backpropagates full sequences, clips once at norm 1, then takes one Adam step. Target exposure is 4,000 updates × 40 episodes = 160,000 fresh episodes. This is equal family loss weight rather than simultaneous same-length concatenation. Existing LR groups remain 3e-4 or 3e-5, new heads use 3e-4; fp32, no new schedule. Profile and pin any lower new-battery exposure before production in complete 200-update increments. The old arm's 32,000 episodes stay fixed.

Validation/final evaluation contains 27 conditions: 12 task-delay cells, 12 recognition load/repeat cells and three Krauzlis baseline lengths. Use 64 validation / 256 final examples per non-Krauzlis condition and 100 / 400 per Krauzlis condition. Rank checkpoints by the minimum chance-normalized family BA, then mean family AUC, averaging conditions equally within family and excluding the three load-0 recognition cells. Chance is 25% for motion direction and 50% for the other binary tasks. Train/schedule/validation/test seeds are 61973001/62973001/63973001/64973001.

No success criterion may conceal one family's collapse. Report all task/condition confusions, selected and terminal checkpoints, actual exposure and raw attention summaries. The new battery has no simultaneously trained biased counterpart: success would show acquisition by this competitor, not that removing biases uniquely caused success.

## Execution receipt and limits (launch history)

The first L40S pod `bil7lpuah5o3x9`, created 2026-09-14 01:36:25.001 UTC, failed CUDA initialization (`cuInit 999`) under both Torch 1.13/cu117 and base Torch 2.0/cu118; one restart did not resolve it. It performed **zero production updates**. Its failure archive was retrieved and SHA-256 verified; the pod was stopped and deleted (HTTP 204, subsequent pod list empty). [Failure and cleanup receipt](../../WorkingMemory/UnbiasedAttention/Cloud/failed_pod_cleanup_receipt.json).

The [replacement provisioning receipt](../../WorkingMemory/UnbiasedAttention/Cloud/cloud_provisioning.json) records `ep8bmcjxz9k66l`, one NVIDIA L40 48 GB on the same Palladio account, created **2026-09-14 01:48:02.15 UTC**. It retains the **original hard deadline of 09:36:25.001 UTC**; the eight-hour allowance does not restart. Its GPU rate is $0.69/hour, approximately $0.695/hour including recorded storage, rather than the initial host's $0.79/hour GPU rate. These are estimates, not invoices.

The [launch receipt](../../WorkingMemory/UnbiasedAttention/Cloud/launch_receipt.json) verifies production advancing under pinned Torch 1.13.1+cu117, Python 3.10.12. Both profiles passed and the full exposures were pinned: 4,000 old-task updates / 32,000 episodes and 4,000 five-task updates / 160,000 episodes. At 2026-09-14T01:57:21.565752+00:00, the old arm reached global 8605, adding 205 updates / 1,640 episodes; the latest recorded loss was finite (0.55895). This is launch/progress evidence, not a performance conclusion. The five-task arm is queued after the old arm.

Remote supervisor 2036 and incremental artifact watcher 39256 continue beyond the interactive turn. Profiles estimate about 63 minutes old-arm training, 344 minutes new-arm training, plus 18 minutes combined evaluation; these are planning estimates, not completion promises. Source, architecture and tasks are unchanged. All work shares the original 09:36:25.001 UTC deadline. [Saved launch snapshot](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/launch_snapshot.json). Retrieve and verify artifacts, stop at completion/cap and delete after retrieval. Broader LatentDynamics and the temporal residual remain paused/unrun. No GPU work was performed by the documentation agent.

## Protocol corrections preserved

Before production, the recognition timeline was corrected from an earlier interleaved-blank proposal to **consecutive study frames, then three blanks total**. Orientation uses one signed sample-to-probe change, not the old cumulative threshold rule; its glyph persists through the two sample frames. Motion uses only a location ring, present through motion; binding receives a retrocue. The Krauzlis event sampler was refined from a proposed unconstrained 200-trial shuffle to exact per-100 event proportions. These are recorded implementation decisions, not post hoc adjustments to observed outcomes.

[Stimulus source](../../WorkingMemory/SpatialTaskBattery/stimuli.py) · [Construction receipt](../../WorkingMemory/SpatialTaskBattery/construction_checks.json) · [Recognition manifest](../../WorkingMemory/SpatialTaskBattery/recognition_manifest.json) · [Training implementation](../../WorkingMemory/UnbiasedAttention/worker.py).
