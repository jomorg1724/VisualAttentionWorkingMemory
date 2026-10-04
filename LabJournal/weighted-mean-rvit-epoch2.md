# Weighted-mean CNN–RViT: two epochs per trial set

Question: does reducing repeated exposure to one sample pool improve held-out change detection? The previous weighted model lowered training CE near the end of a pool, but validation1000 remained50% balanced accuracy /meanAUC0.496260. This motivates testing replay frequency; it does not establish the cause of failure. Earlier fresh-per-update runs also failed.

User approved the recommended epoch-only change. Same causal rawRGB mean .5Xt+.4Xt-1+.1Xt-2, fresh residualCNN256×13×13,169×256tokens, original shared recurrent visual block and256→16/flatten/FFN decoder. **2 epochs per1,000-trial pool**,66updates/2,000presentations, then fresh samples. The focused fake-stream test verifies exact once-per-epoch coverage and refresh at update67. Architecture, Adam1e-4/no clipping, FP32/mathSDPA/TF32off, full BPTT, batch32/micro4, stimuli/labels57/43, seeds/namespaces and regularization remain the baseline settings. No ×10 input multiplier in this mean variant. Fullmodel/Adam/RNG/streams start fresh; no checkpoint input. Identical initialization and initial samples permit a matched schedule comparison, while later sample exposure differs by design.

Target **2,310updates /70,000trial presentations /35,000unique trials**,35complete two-epoch pools. Same total presentations as the earlier seven10-epoch pools; fivefold unique exposure. Validation100/250/500/1000/1500/2000/2310,100trials perB12/B20/B28; selectionmeanAUC thenBA. Final paired200/cell.

Actual training on existing A40pod`7q3pqnv51lil36`; no new rental or cap extension. Original8h/$5 cap retains science deadline 1791089334.23731 and hard deadline 1791089934.23731 (October3,9:48:54PM and9:58:54PM PDT). Prior native GPU timings reused because model/precision/batches are unchanged; prospectively pinned full target fits remaining allowance. Existing authenticated deadline guard stays armed. Atomic same-directory handoff prevents a guard race, new mirror retrieves metadata and overwriteslatest, verifies best/latest and deletes the pod after completed retrieval. Only best/latest retained perexperiment.

Old cloud model stopped at **1304updates /39544trial presentations**; best100/latest1304 retrieved andCPUverified, oldrun archived at `/workspace/vawm_weighted_mean_rvit_01/cancelled_epoch10_run`. Cancelled run has no new final held-out test. Current local input×10 experiment continues independently.

**Production verified:** saved update3 /96presentations, all92Adam states at3 and all92learned tensors changed from the direct fresh constructor. Parent downloaded andCPUreloaded the checkpoint; native proof records replay_epochs2/updates_per_pool66. Snapshot liveupdate5, no validation yet. Remote active run remains `/workspace/vawm_weighted_mean_rvit_01/run`, with immutable new sources in`repo_epoch2`, owner10843. Source/protocol namespace`SecondPass.WeightedMeanRViTEpoch2.cloud_worker`.

[Production proof](../SecondPass/WeightedMeanRViTEpoch2/CloudRuntime/production_verified.json) | [Downloaded checkpoint proof](../SecondPass/WeightedMeanRViTEpoch2/CloudRuntime/downloaded_checkpoint_verified.json) | [Same-pod handoff](../SecondPass/WeightedMeanRViTEpoch2/CloudRuntime/same_pod_handoff_verified.json) | [Prior-run cancellation](../SecondPass/WeightedMeanRViT/CloudRuntime/epoch2_replacement_cancelled.json)

October3,23:14UTC live snapshot: epoch2 weighted-cloud202/2310 updates /6128presentations, pool_index3/epoch0 (fourth fresh1000-trial set), last100CE0.685434. Validation100: BA50% allcells, meanAUC0.508908 (B12 .516116/B20 .480620/B28 .529988), all-change decisions. Local input×10:200/330 /6064presentations, pool0/epoch6, last100CE0.686339; validation100 BA50% allcells, meanAUC0.476404 (B12 .416157/B20 .555692/B28 .457364), all-change decisions. Both healthy; next validation250. Preliminary validation shows no acquisition yet.

**October3,7:09PM PDT snapshot:** localinput×10 completed330/10000, freshfinalBA50%/meanAUC.489996 (all-change), best/latest330 verified/guardexited. Cloud epoch2weightedCNNRViT2099/2310 /63640presentations, last100loss.683817; validation2000BA50%/meanAUC.516252/all-change. Cloudcontinuesunderoriginalcap9:58PM, localstaysfinished.


**Completed and poddeleted:**2310/70000/35000unique; selected2000 finalBA50%/meanAUC0.503969; terminalBA50%/AUC0.509520. Best/latest verified/retained, originalcapunchanged.
