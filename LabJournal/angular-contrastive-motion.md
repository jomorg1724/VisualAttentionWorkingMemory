# Angular contrastive motion — local queue

User requested a fresh CNN trained with contrastive force proportional to angular difference, queued for the next model. Status QUEUED, training not launched. Three ordered100×100 RGB frames → fresh residualCNN32/64/128/256 → global spatial mean →128-dimensional normalized vector, shared across clips. 4,738,528 trainable parameters/64 tensors. Loss0.5(distance−shortest_angle/π)², giving linearly graded outward force at fixed separation. Independent dot layout/count/speed per clip, consistent direction throughout each three-frame clip. One-third same, one-third26/28°, one-third45–180°. No inheritedweights/KL/reconstruction/classifier arm.

CPUengineeringfixture passed all gradients/Adam states, force scaling and circular boundary. Target25024updates, batch32pairs/micro4, 1000freshpairs×2epochs thenrefresh, new8h local ceiling begins only on launch. Onlybest/latest. Validation/angular geometry plus fresh continuous before/after change detection at.375/1/2speed and26/28°; threshold fitted on validation, independent final tests. See [implementation](../SecondPass/AngularContrastiveMotion/README.md) and LocalRuntime/queue.json. No active local/cloud training, no launch budget started.


## Actual local launch, October4

After user questioned the idle queue, parent activated the ready contrastive worker on the free MPS accelerator. Fresh whole model/all64Adam states independently verified in the saved update1 checkpoint. Current worker65848, 149 updates/4672 pair presentations; deadline guard armed, hard stop2026-10-04T16:37:56.095541-07:00. Target25,024updates or8h. No previous model was interrupted: the frozen predictive-encoder FFN completed10,240planned updates at8:05AM PDT and both head checkpoints remain, as do predictorbest50,000/latest50,240. This active contrastive model inherits no weights. SeeLocalRuntime/production_verified.json.


## Completed October4, 12:06PM PDT — simplified change task solved

Fresh whole CNN completed25,024updates/782,000pair presentations/391,000unique training pairs in3.48hours (about3h29m), planned_complete within8h cap. Onlybest19,000/latest25,024 retained; both independently CPUchecked for64 finite learned-weight/Adam states at their respective steps. Guard/worker normal exit; nothing active.

Final independent continuous before/after comparison:3072trials,512per cell,100%balanced accuracy andAUC1.000 in allsix cells. Threshold0.08698047697544098 fitted exclusively on validation for the selected19,000checkpoint; test uses fresh indices24,000,000 onward. Distance alone yields the decision; no FFN was trained.

| Speed(px/frame) | Change(deg) | Test trials | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|
| 0.375 | 26 | 512 | 100% | 1.000 |
| 0.375 | 28 | 512 | 100% | 1.000 |
| 1.0 | 26 | 512 | 100% | 1.000 |
| 1.0 | 28 | 512 | 100% | 1.000 |
| 2.0 | 26 | 512 | 100% | 1.000 |
| 2.0 | 28 | 512 | 100% | 1.000 |

Independent angular test384pairs: loss0.00136432 versus collapsed-distance baseline0.07994250; distance/target correlation0.98783, same-direction mean distance0.00583070. Last100trainingmeanloss0.00128867. These results show that a fresh CNN can learn motion-direction similarity with explicit angular supervision and solve this simplified comparison. They do not establish performance on the original cue/distractor/native dot-lifetime Krauzlis movies. Frozen predictive-encoder FFN failure used a different learning objective/model; this is not a controlled isolation of one causal factor. No additional training or cloud launch is authorized automatically. Evidence: LocalRuntime/completion_verified.json andrun/report.json.
