# Pre-update attention: unchanged-teaching competitor

Completed 32,000 additional episodes from spatial4400; selected global checkpoint8400. Tasks, cues, labels, losses, comparator, heads and90/10schedule remain unchanged. The new arm ran on a single cloud RTX3090; controls ran locally. Package versions match, but this is not bitwise cross-platform equivalence.

| Condition | Parent BA% | Continuation BA% | Feedback BA% | Attention BA% | Attention−continuation pp [95%CI] |
|---|---:|---:|---:|---:|---:|
| single_D0 | 96.68 | 99.80 | 100.00 | 100.00 | +0.20 [+0.00, +0.59] |
| single_D4 | 96.48 | 99.02 | 99.02 | 99.22 | +0.20 [-0.39, +0.98] |
| single_D12 | 79.69 | 90.04 | 86.72 | 91.99 | +1.95 [-0.20, +4.10] |
| single_D24 | 50.00 | 58.40 | 64.26 | 79.30 | +20.90 [+15.82, +25.98] |
| binding_D0 | 96.68 | 99.22 | 99.41 | 99.22 | +0.00 [-0.78, +0.78] |
| binding_D4 | 96.09 | 99.80 | 99.61 | 99.41 | -0.39 [-0.98, +0.00] |
| binding_D12 | 94.73 | 99.61 | 99.61 | 99.41 | -0.20 [-0.98, +0.39] |
| binding_D24 | 93.16 | 99.41 | 98.63 | 99.22 | -0.20 [-0.98, +0.39] |
| binding_D0_locations | 96.68 | 99.41 | 99.61 | 99.80 | +0.39 [-0.39, +1.17] |
| binding_D4_locations | 95.90 | 99.61 | 99.80 | 99.61 | +0.00 [-0.78, +0.78] |
| binding_D12_locations | 93.95 | 99.61 | 99.61 | 99.61 | +0.00 [-0.59, +0.59] |
| binding_D24_locations | 93.55 | 99.02 | 98.63 | 99.61 | +0.59 [+0.00, +1.37] |
| motion_D0 | 46.09 | 50.78 | 45.31 | 39.65 | -11.13 [-14.65, -7.62] |
| motion_D24 | 25.00 | 31.64 | 25.00 | 35.35 | +3.71 [+0.59, +6.84] |

The attention package adds27,590 parameters and no persistent state. Both heads jointly select current sensory and old-memory tokens before the existing E/I memory input drive. Existing comparator and sensory decision routes remain. Attention weights alone do not establish retention or a uniquely biological mechanism. Motion has only10% training allocation and does not participate in checkpoint selection. Binding swaps can be solved using one location and therefore do not establish two-item memory capacity. Unequal task renderings/change sizes preclude direct capacity ranking between binding and native single-item orientation.

1000 paired resamples; binding resamples128 four-case blocks; other tasks512 class-stratified episodes; intervals conditional on these trained seeds; repeated models/delays do not add independent examples.

[Full AUC, confusion matrices, intervals and numerical diagnostics](analysis.json)
