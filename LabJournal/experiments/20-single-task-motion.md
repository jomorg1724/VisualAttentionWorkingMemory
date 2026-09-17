# 20 — Can task-focused continuation learn cued motion duration?

[Index](../README.md) · [Current status](../CURRENT_STATUS.md) · [Parent five-task experiment](19-biased-spatial-battery.md)

**Completed 2026-09-14T05:45:20.729828+00:00: all 2,200 added updates / 17,600 motion episodes finished. Held-out direction accuracy remains near 25% at every delay, including D0. The local model worker/supervisor have exited; cloud joint training continues unchanged.**

## Completed acquisition result

| Blanks | Parent BA | Selected BA | Terminal BA | Terminal AUC | Terminal prediction counts (classes0/1/2/3) |
|---|---:|---:|---:|---:|---|
| D0 | 25.00% | 24.80% | 25.20% | 0.521 | [87, 365, 0, 60] |
| D4 | 25.00% | 25.78% | 25.00% | 0.521 | [1, 2, 0, 509] |
| D12 | 25.00% | 25.00% | 25.78% | 0.514 | [0, 43, 21, 448] |
| D24 | 25.00% | 25.00% | 25.00% | 0.518 | [0, 256, 256, 0] |

Selected checkpoint **11760** used 14,080 additional motion episodes; terminal **12200** used all 17,600. Parent checkpoint 10000 had 12,800 prior motion episodes. All selected-minus-parent and terminal-minus-parent paired gain intervals include zero. For terminal minus parent, D0 is +0.20 pp [−3.71, +3.71], D4 0.00 pp [−2.15, +2.15], D12 +0.78 pp [−2.73, +4.30], and D24 0.00 pp [−4.10, +4.30]. These use 2,000 class-stratified paired bootstrap replicates on the same 512 base episodes per delay. Generation metadata and labels match across evaluated models.

Output choices collapse to a subset of directions; selected D12/D24 predicts one class throughout. Terminal AUC is only about 0.51–0.52, so this is not merely a good ranking with an unfortunate final threshold. No state probe was performed here, so weak outputs do not establish erased or absent internal directional information.

Training had finite losses and gradients. First/last 200-update mean CE was 1.3916/1.3877, close to log(4) = 1.3863; 23.8% of updates were clipped. Training/evaluation took 3,384.04 seconds (56.4 minutes), and final analysis finished by 3,554.76 seconds (59.2 minutes), within the 7,200-second cap. The completion receipt verifies 97 immutable artifacts and no remaining local model workers. No new GPU job follows automatically.

**Conclusion:** this task-focused continuation failed to acquire cued duration at its pinned exposure. Failure already at D0 cannot be attributed solely to forgetting during inserted blanks. It does not prove a renderer bug, an architecture capacity limit or task interference. The cloud validation remains near chance through 11600, but different validation draws and the full-motion versus five-loss objective prevent a paired or uniquely causal interference comparison.

[Completed report](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md) · [Full findings/confusions](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_findings.json) · [Completion receipt](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_receipt.json) · [Exposure-aligned validation trajectories](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/trajectory_comparison.md).

## Original rationale and pinned execution


The current five-task model learns binding strongly but remains near four-way chance on cued motion duration at validation checkpoint 10000. The user authorized training only one weak task locally. The chosen task is the existing cued four-patch motion-duration judgment: identify the direction occupying most of the selected patch's eight motion transitions, then report after D = 0/4/12/24 blanks.

## Parent and one changed training factor

Branch from the **original-bias five-task checkpoint 10000**, retaining the complete current architecture, learned source/locality terms, stimuli/cues, delays, labels, learning-rate policy and compatible Adam state. This parent has 1,600 joint updates / 64,000 new five-task episodes beyond attention 8400. It is not the cancelled bias-removed model or the completed old-task terminal model.

Only the motion-duration loss contributes training gradients in the local branch. The remaining task heads/components remain part of the current architecture; no new module, different cue, simplified motion, extra supervision or optimizer sweep is introduced. All learned shared components still receive gradients through the motion computation. The [implementation](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/README.md) restores every model tensor, all 133 compatible Adam states, global RNG, the motion-family stream/pending counterbalance cases and the original motion-delay scheduler. Parent SHA-256 is `35281f264131e01678ab5725a5b66816d47302583f78b9751f775f18adf22942`; it had already seen 12,800 motion episodes. Unused heads receive no gradients or optimizer updates.

The proposed target was **4,000 updates × batch 8 = 32,000 episodes**. Conservative profiling showed that would not fit, so production was pinned beforehand to **2,200 updates / 17,600 additional episodes** under the new finite **7,200-second local cap**, including baseline, profiles, training, final evaluation and reporting. The local absolute deadline is **2026-09-14 06:46:05.956011 UTC**. No allowance was extended. One local GPU worker is authorized. The existing joint cloud worker is unaffected and retains its original 09:36:25.001 UTC deadline. This is not another cloud arm.

## Question and limits

The question is whether task-focused continuation of this model can acquire the existing cued duration judgment. Success would show a useful response to concentrated optimization. It would **not by itself prove multi-task gradient interference**: exclusive training changes task exposure, loss weighting and optimizer updates together. Failure would also not establish an architectural impossibility or absent motion representation. The old uncued duration scores are a different task and must not be substituted for this cued four-patch result.

Motion performance is reported separately at every delay for parent, selected and terminal checkpoints with actual exposure. No other-task preservation evaluation was included in this motion-only run. Do not use one favorable average to declare all-task progress. Five scheduled validation looks use 128 examples per delay, seed 71973001. Selection maximizes minimum delay BA, then mean delay AUC, earlier ties and parent fallback. Final evaluation uses seed 72973001 with 512 base episodes paired across all four delays and parent/selected/terminal models: 2,048 presentations, not 2,048 independent histories. Class-stratified paired bootstrap intervals and all class confusions are planned. Local/cloud validation seeds differ, so available trajectory comparisons are unpaired and must use added motion exposure rather than total five-task episodes.

## Observation motivating launch: live validation only

Root's latest remote inspection reports joint cloud step 10613, worker 48265 active; the latest **completed validation** is checkpoint 10000, not 10613. At 10000 the equal-condition family averages were binding 100%, signed orientation 54.2969%, motion duration 25.3906%, Krauzlis 50%, and recognition 69.7917% excluding load 0. These are provisional validation observations, not final held-out scores or terminal model performance.

Motion BA at D0/D4/D12/D24 was 28.125% / 23.4375% / 25% / 25%. Recognition accuracy/BA at probe holds 3/4/5 was L4: 92.1875% / 90.625% / 90.625%; L12: 60.9375% / 64.0625% / 65.625%; L24: 53.125% / 56.25% / 54.6875%. Load-0 rejection is excluded from BA/AUC ranking. The checkpoint can learn some new tasks, but task acquisition is uneven within this still-running experiment.

[Unchanged task specification](../../WorkingMemory/SpatialTaskBattery/PROTOCOL.md) · [Joint model and training](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/README.md) · [Parent experiment history](19-biased-spatial-battery.md). [Launch receipt](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/launch_receipt.json) verifies the parent, pinned exposure and advancing updates. Local worker 43072 / supervisor 41812 run `runs/motion_20260913_214605`; CPU report helper 31520 waits for completion and reads available cloud curves. Cloud watcher 43720 is untouched. No GPU/model work was performed by the journal agent.


The local objective is the **full mean motion cross-entropy**, whereas motion contributes one fifth of the cloud's five-family mean loss before other gradients are added. Inherited learning rates are unchanged; there is no compensating LR adjustment. This is part of the task-focused policy change and further limits any claim to have isolated gradient interference. The shared next motion stream permits exposure matching, but does not make differing validation examples paired.
