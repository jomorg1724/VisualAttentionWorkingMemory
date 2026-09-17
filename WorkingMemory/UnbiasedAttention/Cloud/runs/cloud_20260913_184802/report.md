# Attention without explicit source/distance priors

Run status: cancelled_by_user. Source/task definitions and original deadline were fixed before production.

## old_unbiased
Additional episodes: 32000; selected global step: 8400; parent fallback: True.

### selected_test

| Condition | BA | Accuracy | AUC |
|---|---:|---:|---:|
| motion_D0 | 38.28% | 38.28% | 0.7717030843098958 |
| motion_D24 | 34.57% | 34.57% | 0.7108662923177084 |
| single_D0 | 99.61% | 99.61% | 0.9999542236328125 |
| single_D12 | 92.97% | 92.97% | 0.979034423828125 |
| single_D24 | 80.47% | 80.47% | 0.9060211181640625 |
| single_D4 | 99.22% | 99.22% | 0.9995880126953125 |
| binding_D0 | 98.83% | 98.83% | 0.9997711181640625 |
| binding_D0_locations | 99.41% | 99.41% | 0.99993896484375 |
| binding_D12 | 99.02% | 99.02% | 0.9997711181640625 |
| binding_D12_locations | 99.41% | 99.41% | 0.9998626708984375 |
| binding_D24 | 98.44% | 98.44% | 0.99969482421875 |
| binding_D24_locations | 99.22% | 99.22% | 0.99969482421875 |
| binding_D4 | 99.22% | 99.22% | 0.9997100830078125 |
| binding_D4_locations | 99.41% | 99.41% | 0.9998931884765625 |

### terminal_test

| Condition | BA | Accuracy | AUC |
|---|---:|---:|---:|
| motion_D0 | 42.38% | 42.38% | 0.8114013671874999 |
| motion_D24 | 42.38% | 42.38% | 0.7726999918619791 |
| single_D0 | 98.24% | 98.24% | 0.9993896484375 |
| single_D12 | 82.81% | 82.81% | 0.919677734375 |
| single_D24 | 57.62% | 57.62% | 0.6320953369140625 |
| single_D4 | 99.22% | 99.22% | 0.9995269775390625 |
| binding_D0 | 97.85% | 97.85% | 0.9994354248046875 |
| binding_D0_locations | 98.24% | 98.24% | 0.99908447265625 |
| binding_D12 | 58.98% | 58.98% | 0.7656707763671875 |
| binding_D12_locations | 59.38% | 59.38% | 0.798614501953125 |
| binding_D24 | 49.80% | 49.80% | 0.549957275390625 |
| binding_D24_locations | 53.91% | 53.91% | 0.58465576171875 |
| binding_D4 | 93.36% | 93.36% | 0.9988250732421875 |
| binding_D4_locations | 94.34% | 94.34% | 0.998626708984375 |

### Equal-exposure terminal comparison

Unbiased terminal minus saved biased control terminal, using identical test episodes. The paired bootstrap uses complete four-case blocks for binding and class-stratified episodes for other tasks (2,000 replicates). This reused test is exploratory; hardware differs; intervals are conditional on these trained models.

| Condition | BA change | Paired95% interval |
|---|---:|---:|
| binding_D0 | -1.56pp | [-2.93, -0.39] |
| binding_D0_locations | -1.37pp | [-2.73, -0.20] |
| binding_D12 | -40.62pp | [-43.95, -37.50] |
| binding_D12_locations | -40.23pp | [-43.55, -37.30] |
| binding_D24 | -49.61pp | [-52.73, -46.29] |
| binding_D24_locations | -45.70pp | [-48.83, -42.38] |
| binding_D4 | -6.05pp | [-8.20, -3.71] |
| binding_D4_locations | -5.27pp | [-7.23, -3.32] |
| motion_D0 | +1.37pp | [-0.98, +3.52] |
| motion_D24 | +0.39pp | [-2.54, +3.52] |
| single_D0 | -1.56pp | [-2.93, -0.39] |
| single_D12 | -14.06pp | [-17.58, -10.74] |
| single_D24 | -33.98pp | [-38.67, -29.10] |
| single_D4 | -0.59pp | [-1.56, +0.20] |
## spatial_unbiased
Additional episodes: 6000; selected global step: None; parent fallback: False.


All per-class confusion matrices, probabilities and per-head attention summaries remain in the unchanged retrieved artifacts. New-task recognition load0 has specificity/accuracy only. The five-task battery has no equally trained biased control and does not isolate an architecture effect.