# Additive controller feedback: unchanged teaching

The added controller produced a narrow improvement on single-item orientation at24 blank frames: **64.26% versus58.40%** for ordinary continuation, a paired gain of**5.86 percentage points (95%CI2.15–9.77)**. It was not an overall upgrade: single-itemD12 fell by3.32pp, motionD24 fell by6.64pp, and trained-center bindingD24 fell by0.78pp despite both models remaining near ceiling on binding. Both arms improved several tasks over the unchanged parent, so further training explains part of the progress.

The planned interruption does **not support a benefit that requires ongoing feedback currents into spatial memory during the blank interval**: silencing those currents changed single-itemD24 accuracy by−0.20pp (95%CI−1.17–0.78), with no changed decisions on binding or motion. This does not rule out feedback during encoding/query/probe processing, training-induced changes, or content retained in the controller and returned after the interruption. The controller itself kept updating during the intervention. The result supports a modest task-specific effect of the added package, not a general attention-as-maintenance mechanism or a solution to the motion failure.

Both arms completed 32,000 additional episodes from the same spatial4400 parent. All images, cues, objectives, task heads, comparator and90/10 allocation were unchanged. Checkpoints were selected using the predeclared eight-primary-cell validation AUC.

| Condition | Parent BA% | Continued BA% | Feedback BA% | Feedback−continued pp [95%CI] |
|---|---:|---:|---:|---:|
|single_D0|96.68|99.80|100.00|+0.20 [+0.00,+0.59]|
|single_D4|96.48|99.02|99.02|+0.00 [-0.39,+0.59]|
|single_D12|79.69|90.04|86.72|-3.32 [-5.08,-1.56]|
|single_D24|50.00|58.40|64.26|+5.86 [+2.15,+9.77]|
|binding_D0|96.68|99.22|99.41|+0.20 [-0.39,+0.79]|
|binding_D4|96.09|99.80|99.61|-0.20 [-0.59,+0.00]|
|binding_D12|94.73|99.61|99.61|+0.00 [-0.59,+0.59]|
|binding_D24|93.16|99.41|98.63|-0.78 [-1.56,-0.20]|
|binding_D0_locations|96.68|99.41|99.61|+0.20 [-0.39,+0.98]|
|binding_D4_locations|95.90|99.61|99.80|+0.20 [-0.39,+0.79]|
|binding_D12_locations|93.95|99.61|99.61|+0.00 [-0.78,+0.78]|
|binding_D24_locations|93.55|99.02|98.63|-0.39 [-1.37,+0.59]|
|motion_D0|46.09|50.78|45.31|-5.47 [-8.60,-2.34]|
|motion_D24|25.00|31.64|25.00|-6.64 [-8.79,-4.69]|

[Retention curves](retention_curves.png) · [Full BA/AUC/confusions and paired intervals](analysis.json) · [Source, checkpoints and validation curves](results.json)

## Acute feedback interruption

Only the24 inserted blank frames were interrupted, on exactly paired examples and unchanged selected weights. This is an out-of-distribution intervention, not the ordinary forward path.

- single_D24: feedback-off minus normal BA -0.20pp [95%CI -1.17,+0.78].
- binding_D24: feedback-off minus normal BA +0.00pp [95%CI +0.00,+0.00].
- motion_D24: feedback-off minus normal BA +0.00pp [95%CI +0.00,+0.00].

Single-itemD24 AUC slightly increased when blank-interval feedback was silenced: +0.00235 (95%CI0.00084–0.00389). Thus the intervention was not literally prediction-identical, even though its accuracy effect was small. Feedback motionD24 AUC was0.734 despite25% argmax accuracy, so the decision failure does not establish information erasure. Full paired score-level intervals are retained in analysis.json.

## Exposure and limits

- continuation: selected global update8400 (32,000 added episodes), terminal32,000 added episodes and505,600 logical frames; train3396.2s, clipping on62.6% of updates. Per-cell selected exposure and gradient/coupling trajectories are saved in analysis.json.
- controller_feedback: selected global update8400 (32,000 added episodes), terminal32,000 added episodes and505,600 logical frames; train3523.6s, clipping on61.7% of updates. Per-cell selected exposure and gradient/coupling trajectories are saved in analysis.json.

The feedback arm adds45,537 parameters and64 controller rate/adaptation state scalars. The controller receives the full sensory/memory summaries and may itself retain content, although it has no direct classifier output. This compares the added controller-feedback package, not attention isolated from extra capacity. Both arms preserve the same learned parent and optimizer histories; equal added exposure does not establish a sufficient acquisition horizon. No new cue, teaching target, angle supervision or sampling intervention was introduced.

Binding full swaps can be detected by remembering one location; they do not establish two-item capacity. Binding and native single-item renderings/change sizes differ. Held-out centers measure spatial interpolation. Four-case block uncertainty and per-model seed limitations remain. Motion keeps the prior10% allocation and is excluded from checkpoint selection. Acute feedback interruption can indicate functional dependence but is not a complete causal decomposition of training benefits. This completed local experiment used no cloud compute; separately authorized experiments are outside its ledger.

## Runtime recovery

A Windows sharing violation interrupted only the original supervisor while replacing results.json after it had launched the third feedback validation. The24,000-episode training checkpoint was already saved. That validation completed independently; its saved result and absent process were adopted by a replacement supervisor. Its original exit handle was unavailable and is explicitly marked as such in the event record. No training steps or profiles were repeated. The replacement added retries to status-file publication and resumed the remaining fixed jobs with the original model/trainer/config/source hashes and16:44:44PDT deadline. The failed status snapshots, error log and recovery_receipt.json remain under the run directory. All planned computations subsequently completed.

Final closure: 8527.7s (142.1min) elapsed within the original14,400s allowance. All owned local workers exited, pinned model/trainer/stimulus hashes remain unchanged, and the allowance is closed. Selected checkpoint identities, the status-recovery history and final interpretation review are preserved with the completion receipt.
