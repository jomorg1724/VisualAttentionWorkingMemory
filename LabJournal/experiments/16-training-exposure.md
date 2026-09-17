# Training exposure: recover motion while preserving orientation

[Index](../README.md) · [Current status](../CURRENT_STATUS.md) · [Preceding motion audit](15b-motion-audit.md)

Status: **both arms and final reports completed**. Initial journal entry 2026-09-14T00:21:31.576956+00:00. This page will be updated from completion receipts and combined analysis; a completed validation look is not a final result.

## Question and design

Can the selected pre-update attention architecture recover motion through more training, and does allocating more of those updates to motion help? The ordinary 10% arm tests continuation under the existing schedule. The 50% arm changes allocation while retaining exactly the same architecture, tasks, cues, labels, losses and learning-rate policy. Both start attention global 8400, preserve compatible Adam, RNG and family-local stream progress, and have fixed 4,000 additional updates × batch 8 =32,000 fresh episodes each.

The parent is [attention 8400](../../WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/checkpoint_008400.pt), recorded SHA 256 `e37602aa20ccfc400ea8fe9d98c11f29c508069388897c55803b97f2ccdf1bc9` in launch evidence. No diagnostic class-offset calibration is installed.

| Arm | Placement | Updates per80-cycle: each of8primary cells | Each of2motion delays | New motion episodes per delay | Each primary cell |
|---|---|---:|---:|---:|---:|
|control_10|Local RTX3070Laptop|9|4|1,600|3,600|
|focused_50|Palladio RunPod RTX3090|5|20|8,000|2,000|

The existing sampler is **family-local**, not delay-cell-local. Common family evidence prefixes match, but schedule changes can attach different delays to particular examples. Equal total updates produce different task exposure and frame workload. Cross-platform differences remain.

## Selection and final evidence plan

Fresh validation seed 53973001,128 examples/cell; looks at global 9200,10000,10800,11600,12400. Candidate eligibility requires no BA decline greater than 2 pp versus freshly evaluated parent in any of 10 trained cells. Among eligible candidates, maximize minimum motion D0/D24 BA, then mean motion AUC, with earlier ties. Parent fallback is allowed. Healthy training still reaches the fixed endpoint.

Fresh final testseed 54973001,512 examples/cell:10 trained conditions plus 4 held-out binding-center conditions. Report parent, selected and terminal, paired per-task results, class recalls/confusion, exposure curves and simultaneous one-sided preservation bounds. A 2 pp validation screen is not a statistical equivalence guarantee. Report terminal regressions even if selection falls back to the parent.

## Execution and budget (planned at launch)

Local control started first in [the local run](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/). Its separate 14,400 s cap ends2026-09-14T03:35:44.401Z. Focused cloud pod `sm1kbctuhqfpo5` was created2026-09-13T23:37:38.206Z and has a four-hour cap ending 03:37:38.206 Z. Recorded rate is approximately $0.225/hour total, not an invoice. The cloud arm must be retrieved and stopped/deleted promptly on completion, independently of local progress.

No exposure extension, extra architecture arm or residual training is included. The current worker/protocol uses fp 32, fullBPTT, gradient clipping 1, Adam epsilon 1e-10, inherited sensory LR 3e-5 and new-module LR 3e-4. Profile states do not become production initialization.

## Dated progress and completed cloud result

Updated 2026-09-14T00:24:12.640157+00:00. The focused cloud arm completed 4,000 additional updates /32,000 episodes. Validation selected global 11600 after 25,600 additional episodes; terminalglobal 12400 was evaluated separately. Its 141 retrieved files were hash-verified, and the pod was stopped/deleted2026-09-14T00:21:33.450146Z. Receipt estimates $0.165 lifetime cost, not a final invoice. The local control continues; [CURRENT_STATUS.md](../CURRENT_STATUS.md) records its dated progress and validation separately.

### Fresh held-out cloud results

The following table is transcribed from the partial cloud-only combined-report artifact. Parent, selected and terminal share the same fresh test examples. Parent/selected paired intervals use 2,000 resamples; binding uses 128 four-case blocks and other cells 512 class-stratified base episodes. Simultaneous lower bounds account for the 14 reported cells within this contrast, conditional on the trained models.

| Cell | Parent BA% | Selected BA% | Terminal BA% | Selected−parent pp [95%CI] | Selected simultaneous lower95 pp |
|---|---:|---:|---:|---:|---:|
| single_D0 | 99.61 | 99.41 | 99.22 | -0.20 [-1.17,+0.59] | -5.08 |
| single_D4 | 99.22 | 98.63 | 98.83 | -0.59 [-1.37,+0.00] | -5.47 |
| single_D12 | 92.97 | 90.04 | 90.23 | -2.93 [-5.47,-0.59] | -7.81 |
| single_D24 | 80.47 | 83.98 | 84.38 | +3.52 [+0.00,+6.84] | -1.37 |
| binding_D0 | 98.83 | 99.41 | 99.41 | +0.59 [-0.20,+1.56] | -4.30 |
| binding_D4 | 99.22 | 99.41 | 99.02 | +0.20 [-0.59,+0.98] | -4.69 |
| binding_D12 | 99.02 | 99.02 | 99.22 | +0.00 [-0.59,+0.59] | -4.88 |
| binding_D24 | 98.44 | 99.22 | 99.22 | +0.78 [+0.00,+1.76] | -4.10 |
| binding_D0_locations | 99.41 | 99.61 | 99.80 | +0.20 [+0.00,+0.59] | -4.69 |
| binding_D4_locations | 99.41 | 99.80 | 99.41 | +0.39 [+0.00,+0.98] | -4.49 |
| binding_D12_locations | 99.41 | 99.80 | 99.61 | +0.39 [+0.00,+0.98] | -4.49 |
| binding_D24_locations | 99.22 | 99.41 | 99.61 | +0.20 [-0.39,+0.78] | -4.69 |
| motion_D0 | 38.28 | 65.62 | 49.80 | +27.34 [+22.65,+32.23] | +22.46 |
| motion_D24 | 34.57 | 57.62 | 55.08 | +23.05 [+17.97,+28.12] | +18.16 |


### Historical interpretation at the cloud-only stage

The focused arm clearly recovers useful motion accuracy relative to its untouched parent. It does **not** demonstrate a general upgrade: single D12 falls 2.93 pp, with paired 95% CI[−5.47,−0.59], and all-cell preservation remains unresolved against the 2 pp margin. The selected checkpoint also outperforms terminal 12400 on immediate motion, so longer training is not monotonically better at every look.

The earlier cloud 74.22% motion D0,65.62% motion D24 and 86.72% single D24 were validation at 11600. The 65.62%,57.62%,83.98% figures here are fresh held-out test. A validation pass does not guarantee held-out preservation. Retain both selected and terminal outcomes instead of hiding the regression behind selection.

The local 10% arm is still needed at equal total exposure before attributing a gain specifically to increased motion allocation. This cloud-versus-parent comparison combines ordinary extra training with the new allocation. No final two-arm conclusion or Stage2 residual launch is made.

## Dated diagnostic note: unstable motion decisions (2026-09-14T00:39:06.263187+00:00)

The final 800 updates substantially changed which direction the focused model chooses. On the **same 512 balanced held-out D0 movies** (128 per direction), selected 11600 → terminal 12400 motion BA falls 65.625% → 49.8047%, while macro OVR-AUC falls .88834 → .86650. This is a large change in class decisions with a smaller observed decline in ranking, not evidence that every direction representation disappeared.

| Direction | Selected recall | Terminal recall | Selected prediction count | Terminal prediction count |
|---|---:|---:|---:|---:|
| Right |61.72%|53.12%|100|93|
| Up |67.19%|98.44%|123|308|
| Left |75.78%|46.09%|176|109|
| Down |57.81%|1.56%|113|2|

The terminal model chooses up on 308/512 trials and down only twice. This identifies a marked **up-choice bias and near-disappearance of down choices**. It does not identify whether the cause lies in attention routing, changing sensory/memory representations, readout fitting, or their interaction. Up and down one-versus-rest ranking remain above chance despite the biased argmax output. The comparison uses saved summaries and confusions; no new inference was run.

The validation D0 trajectory also repeatedly swings: 60.16%,28.125%,46.875%,74.22%,50.78% at global 9200/10000/10800/11600/12400. Those 128-example validation scores are different evidence from the 512-example held-out comparison above.

In the final 800 updates, all 200 motion D0 updates and all 200 motion D24 updates hit the gradient-norm clip threshold 1. Mean raw norms are 72.85 and 59.25; maxima 217.58 and 300.00 respectively. These are **pre-clipping gradients**, not parameter-update magnitudes or evidence of exploding hidden states. The inherited policy uses LR 3e-4 for recurrent/attention/comparator parameters and 3e-5 for sensory parameters and the ordinary motion head, with batch 8 and no learning-rate annealing.

Plausible explanations to test are shared-task gradient interference, a learned representation moving faster than its slower head can track, and noisy constant-step continuation. **None is identified by these logs.** Task-gradient cosines, actual parameter-update magnitudes and checkpoint-to-checkpoint decision-branch drift have not been measured here. Clipping controls a norm; it cannot ensure that the direction of an update improves every task. Preserve the selected checkpoint and report terminal behavior rather than treating additional updates as automatically better.

Evidence: [selected test summary](../../WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieved/remote_results/selected_test/summary.json), [terminal test summary](../../WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieved/remote_results/terminal_test/summary.json), [cloud training metrics](../../WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieved/remote_results/focused_50/metrics.csv), [aggregate and validation looks](../../WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieved/remote_results/aggregate.json), [unchanged optimization policy](../../WorkingMemory/TrainingExposure/README.md).

This note authorizes no new run or parameter change. Subsequent user steering paused broad latent-state analysis in favor of frozen attention-map visualization; see the dated update below.

## Final local completion and complete Stage1 result (2026-09-14T00:56:21.960683+00:00)

The local 10% motion arm completed its fixed 4,000 additional updates /32,000 episodes and selected terminalglobal 12400. The [local exit receipt](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/exit.json) records completion without error in 4,546.23 seconds, within its original finite allowance. The combined [report](../../WorkingMemory/TrainingExposure/report.md) is now marked completed. Root coordination verified that owned TrainingExposure workers were absent; cloud retrieval/deletion was already complete.

### Local control: fresh paired held-out results

The table below is copied from the final report. The parent is evaluated on the same fresh test draw as both arms. All 14 local control point estimates are at least as high as parent, but simultaneous lower bounds still leave all-cell preservation unresolved against the 2 pp margin. “No observed point declines” is not an equivalence claim.

| Cell | Parent BA% | Selected BA% | Terminal BA% | Selected−parent pp [95%CI] | Selected simultaneous lower95 pp |
|---|---:|---:|---:|---:|---:|
| single_D0 | 99.61 | 99.80 | 99.80 | +0.20 [-0.39,+0.78] | -3.32 |
| single_D4 | 99.22 | 99.80 | 99.80 | +0.59 [+0.00,+1.37] | -2.93 |
| single_D12 | 92.97 | 96.88 | 96.88 | +3.91 [+1.95,+6.05] | +0.39 |
| single_D24 | 80.47 | 91.60 | 91.60 | +11.13 [+7.81,+14.65] | +7.62 |
| binding_D0 | 98.83 | 99.41 | 99.41 | +0.59 [+0.00,+1.18] | -2.93 |
| binding_D4 | 99.22 | 99.41 | 99.41 | +0.20 [-0.59,+0.98] | -3.32 |
| binding_D12 | 99.02 | 99.61 | 99.61 | +0.59 [-0.20,+1.56] | -2.93 |
| binding_D24 | 98.44 | 99.41 | 99.41 | +0.98 [+0.20,+1.95] | -2.54 |
| binding_D0_locations | 99.41 | 99.61 | 99.61 | +0.20 [+0.00,+0.59] | -3.32 |
| binding_D4_locations | 99.41 | 99.61 | 99.61 | +0.20 [+0.00,+0.59] | -3.32 |
| binding_D12_locations | 99.41 | 99.61 | 99.61 | +0.20 [-0.39,+0.78] | -3.32 |
| binding_D24_locations | 99.22 | 99.61 | 99.61 | +0.39 [-0.39,+1.17] | -3.12 |
| motion_D0 | 38.28 | 41.02 | 41.02 | +2.73 [+0.00,+5.66] | -0.78 |
| motion_D24 | 34.57 | 41.99 | 41.99 | +7.42 [+4.10,+10.55] | +3.91 |


### What the completed comparison tells us

| Held-out cell | Parent8400 | Local10% selected12400 | Focused50% selected11600 | Focused50% terminal12400 |
|---|---:|---:|---:|---:|
| Motion D0 |38.28%|41.02%|65.62%|49.80%|
| Motion D24 |34.57%|41.99%|57.62%|55.08%|
| Single orientation D12 |92.97%|96.88%|90.04%|90.23%|
| Single orientation D24 |80.47%|91.60%|83.98%|84.38%|

Both arms received equal **terminal** exposure of 32,000 additional episodes. Their selected checkpoints differ: control 12400 includes all 32,000, while focused 11600 includes 25,600. The selected-model comparison reflects the predeclared selection procedure; it is not equal exposure at selection. Focused terminal 12400 is reported explicitly for the equal-terminal-exposure view.

Ordinary continuation strongly improves delayed orientation and modestly improves motion. Increased motion allocation produces substantially stronger motion decisions, but gives less orientation practice and weaker orientation outcomes; its single D12 score falls below parent as well. The result supports a training-allocation trade-off in these fitted systems, not one universally superior model. Both schedules preserve high binding performance. Hardware differs across arms, so this does not isolate scheduling from every platform effect.

The earlier cloud-vs-parent result alone could not distinguish extra training from task allocation. The completed control now shows that ordinary continuation does **not** reach the focused arm's motion performance at this fixed endpoint. However, focused motion also swings between checkpoints, and the final 800 updates nearly eliminate down choices. The diagnostic note above remains relevant: mechanism hypotheses are not established by these endpoint results.

Preserve both useful checkpoints and all terminal artifacts. No residual, learning-rate sweep or further model experiment has been launched. At the user's request, broader latent/probe analysis remains paused. The narrower [attention-map visualization](17-attention-maps.md) has now completed on the frozen **attention 8400** parent, not either new continuation checkpoint.

## Evidence

- [Protocol README](../../WorkingMemory/TrainingExposure/README.md)
- [Research design](../../WorkingMemory/Research/training_exposure_and_temporal_residual.md)
- [Launch receipt](../../WorkingMemory/TrainingExposure/launch_receipt.json)
- [Local aggregate](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/aggregate.json)
- [Local metrics](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/control_10/metrics.csv)
- [Local validation 10000](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/validation_010000/summary.json)
- [Cloud provisioning / lifecycle coordinates](../../WorkingMemory/TrainingExposure/Cloud/cloud_provisioning.json)
- [Cloud launch receipt](../../WorkingMemory/TrainingExposure/Cloud/launch_receipt.json)

- [Partial cloud report](../../WorkingMemory/TrainingExposure/report.md)
- [Cloud paired analysis](../../WorkingMemory/TrainingExposure/analysis.json)
- [Verified retrieval](../../WorkingMemory/TrainingExposure/Cloud/retrieval_receipt.json)
- [Pod cleanup and cost](../../WorkingMemory/TrainingExposure/Cloud/cleanup_receipt.json)
