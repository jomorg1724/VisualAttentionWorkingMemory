# Training exposure: all capabilities remain visible

Status: completed. No architecture, cue, objective, LR or calibration changes. Selection uses the predeclared ten-cell validation screen; terminal results remain reported even if rejected.

## control_10

Trained 32,000 additional episodes; selected global12400.

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

Selected preservation across all14cells: unresolved against the predeclared2pp margin; do not label a universal upgrade.
Observed selected declines: []. Terminal declines>2pp: [].

## focused_50

Trained 32,000 additional episodes; selected global11600.

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

Selected preservation across all14cells: unresolved against the predeclared2pp margin; do not label a universal upgrade.
Observed selected declines: ['single_D0', 'single_D4', 'single_D12']. Terminal declines>2pp: ['single_D12'].

2000 paired resamples. Binding uses128 independent four-case blocks per location split; other tasks512 class-stratified base episodes. Within each report contrast, one-sided95% simultaneous lower bounds use the bootstrap maximum centered estimation error over all14cells. Intervals are conditional on trained seeds; this is not equivalence proof or a cross-hardware causal isolation.

[Full BA/AUC, class recalls, paired differences and validation exposure](analysis.json). Equal added episodes/updates are not equal logical frames. Family evidence prefixes match; per-delay assignments can differ. Hardware differences prevent a strictly hardware-controlled causal interpretation. A flat finite curve does not establish a capacity ceiling. No follow-on residual run is automatic.
