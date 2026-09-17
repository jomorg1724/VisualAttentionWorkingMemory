# Prospective-query validation at step 9200

The prospective and historical-control predictions contain the same 1,836 ordered examples, with task, condition, label, paired/trial IDs and metadata matching exactly. This is the first planned validation look, not selected or final held-out performance.

## Task summary

| Task | Prospective normalized BA | Control | Delta | Prospective mean AUC | Control | Delta |
|---|---:|---:|---:|---:|---:|---:|
| image_recognition | 0.2014 | 0.2812 | -0.0799 | 0.6746 | 0.7205 | -0.0459 |
| krauzlis_cued_motion | 0.0000 | 0.0000 | +0.0000 | 0.4964 | 0.5172 | -0.0208 |
| motion_duration_cued | -0.0260 | 0.0000 | -0.0260 | 0.4946 | 0.4955 | -0.0009 |
| orientation_cued | 0.0703 | -0.1016 | +0.1719 | 0.5603 | 0.4875 | +0.0728 |
| spatial_binding | 0.8047 | 0.4844 | +0.3203 | 0.9849 | 0.8723 | +0.1125 |

Selection rank `[minimum normalized task BA, mean task AUC]`: prospective `[-0.026041666666666668, 0.6421557248635761]` versus historical control `[-0.1015625, 0.6186005306879421]`.

## Per-condition summary

| Condition | Metric | Prospective | Control | Delta pp | Prospective AUC | Control AUC | AUC delta |
|---|---|---:|---:|---:|---:|---:|---:|
| image_recognition_N0_H3 | accuracy | 100.00% | 100.00% | +0.00 | N/A | N/A | N/A |
| image_recognition_N0_H4 | accuracy | 100.00% | 100.00% | +0.00 | N/A | N/A | N/A |
| image_recognition_N0_H5 | accuracy | 100.00% | 100.00% | +0.00 | N/A | N/A | N/A |
| image_recognition_N12_H3 | BA | 56.25% | 67.19% | -10.94 | 69.82% | 69.53% | +0.0029 |
| image_recognition_N12_H4 | BA | 56.25% | 67.19% | -10.94 | 72.85% | 71.58% | +0.0127 |
| image_recognition_N12_H5 | BA | 60.94% | 64.06% | -3.12 | 73.14% | 72.17% | +0.0098 |
| image_recognition_N24_H3 | BA | 46.88% | 56.25% | -9.38 | 49.12% | 60.06% | -0.1094 |
| image_recognition_N24_H4 | BA | 45.31% | 54.69% | -9.38 | 48.83% | 58.40% | -0.0957 |
| image_recognition_N24_H5 | BA | 43.75% | 56.25% | -12.50 | 48.24% | 54.88% | -0.0664 |
| image_recognition_N4_H3 | BA | 78.12% | 71.88% | +6.25 | 83.01% | 87.89% | -0.0488 |
| image_recognition_N4_H4 | BA | 79.69% | 71.88% | +7.81 | 82.52% | 86.91% | -0.0439 |
| image_recognition_N4_H5 | BA | 73.44% | 67.19% | +6.25 | 79.59% | 87.01% | -0.0742 |
| krauzlis_B12 | BA | 50.00% | 50.00% | +0.00 | 42.64% | 42.71% | -0.0007 |
| krauzlis_B20 | BA | 50.00% | 50.00% | +0.00 | 52.18% | 57.77% | -0.0559 |
| krauzlis_B28 | BA | 50.00% | 50.00% | +0.00 | 54.10% | 54.67% | -0.0057 |
| motion_duration_cued_D0 | BA | 25.00% | 25.00% | +0.00 | 45.90% | 48.08% | -0.0218 |
| motion_duration_cued_D12 | BA | 25.00% | 25.00% | +0.00 | 51.73% | 49.53% | +0.0220 |
| motion_duration_cued_D24 | BA | 25.00% | 25.00% | +0.00 | 52.67% | 50.68% | +0.0199 |
| motion_duration_cued_D4 | BA | 17.19% | 25.00% | -7.81 | 47.56% | 49.90% | -0.0234 |
| orientation_cued_D0 | BA | 57.81% | 45.31% | +12.50 | 55.18% | 50.59% | +0.0459 |
| orientation_cued_D12 | BA | 54.69% | 48.44% | +6.25 | 57.03% | 49.02% | +0.0801 |
| orientation_cued_D24 | BA | 48.44% | 45.31% | +3.12 | 57.81% | 47.27% | +0.1055 |
| orientation_cued_D4 | BA | 53.12% | 40.62% | +12.50 | 54.10% | 48.14% | +0.0596 |
| spatial_binding_D0 | BA | 84.38% | 78.12% | +6.25 | 99.02% | 93.95% | +0.0508 |
| spatial_binding_D12 | BA | 92.19% | 68.75% | +23.44 | 98.44% | 83.30% | +0.1514 |
| spatial_binding_D24 | BA | 90.62% | 73.44% | +17.19 | 96.88% | 82.62% | +0.1426 |
| spatial_binding_D4 | BA | 93.75% | 76.56% | +17.19 | 99.61% | 89.06% | +0.1055 |

## Interpretation

At this first matched look, prospective queries improve orientation and spatial binding relative to the historical control, while recognition is mixed and degrades at load 24. Motion duration and Krauzlis motion remain near chance. The prospective rank improves because its worst normalized task BA is less negative and its mean task AUC is higher; this does not establish a final winner.
