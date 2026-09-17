# Five-task training with retained attention biases

Status: stopped_by_user.

The original trained source and locality terms were retained, together with every compatible parent weight and Adam state. Five new semantic heads use the same initialization as the cancelled bias-free arm. That arm was cancelled before a comparable exposure or evaluation and is not an equal-training comparator.

Stopped at the user's request after 3,523 logged updates / 140,920 episodes (global step 11923). The latest durable checkpoint is step 11776, preserving 3,376 updates / 135,040 episodes. The last 147 logged updates were not checkpointed. No final held-out evaluation was run.

Validation-selected so far: global step 10000. The table below is the last completed validation, not a final held-out result.

| Last validation condition | BA | Accuracy | AUC |
|---|---:|---:|---:|
| image_recognition_N0_H3 | undefined | 100.00% | None |
| image_recognition_N0_H4 | undefined | 100.00% | None |
| image_recognition_N0_H5 | undefined | 100.00% | None |
| image_recognition_N12_H3 | 82.81% | 82.81% | 0.94140625 |
| image_recognition_N12_H4 | 82.81% | 82.81% | 0.947265625 |
| image_recognition_N12_H5 | 82.81% | 82.81% | 0.94140625 |
| image_recognition_N24_H3 | 71.88% | 71.88% | 0.7958984375 |
| image_recognition_N24_H4 | 71.88% | 71.88% | 0.8193359375 |
| image_recognition_N24_H5 | 73.44% | 73.44% | 0.8271484375 |
| image_recognition_N4_H3 | 96.88% | 96.88% | 0.99609375 |
| image_recognition_N4_H4 | 96.88% | 96.88% | 0.998046875 |
| image_recognition_N4_H5 | 96.88% | 96.88% | 0.998046875 |
| krauzlis_B12 | 50.00% | 57.00% | 0.41819665442676457 |
| krauzlis_B20 | 50.00% | 57.00% | 0.4500203998368013 |
| krauzlis_B28 | 50.00% | 57.00% | 0.5214198286413709 |
| motion_duration_cued_D0 | 25.00% | 25.00% | 0.51953125 |
| motion_duration_cued_D12 | 25.00% | 25.00% | 0.486328125 |
| motion_duration_cued_D24 | 25.00% | 25.00% | 0.49674479166666674 |
| motion_duration_cued_D4 | 23.44% | 23.44% | 0.5198567708333333 |
| orientation_cued_D0 | 42.19% | 42.19% | 0.34375 |
| orientation_cued_D12 | 45.31% | 45.31% | 0.521484375 |
| orientation_cued_D24 | 46.88% | 46.88% | 0.4921875 |
| orientation_cued_D4 | 53.12% | 53.12% | 0.451171875 |
| spatial_binding_D0 | 100.00% | 100.00% | 1.0 |
| spatial_binding_D12 | 100.00% | 100.00% | 1.0 |
| spatial_binding_D24 | 100.00% | 100.00% | 1.0 |
| spatial_binding_D4 | 100.00% | 100.00% | 1.0 |

Recognition load0 is a specificity/accuracy control with no positive class; BA/AUC are undefined and excluded from selection. All class confusions, predictions, validation trajectories, gradient logs and per-head attention summaries are preserved. One training continuation does not establish biological plausibility or population-level model superiority.