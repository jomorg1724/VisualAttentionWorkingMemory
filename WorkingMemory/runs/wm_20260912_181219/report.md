# Learned opponent model: sequence-memory battery

The warm-started opponent model completed 39,200 fresh training episodes. The selected checkpoint is update 6,860 (27,440 episodes); training continued to the fixed endpoint of 9,800 updates. Task-specific held-out results below separate sensory/cue acquisition, accumulation, decision retention and retrospective probes.

Held-out mean OVR-AUC was 0.9964 for sensory anchors, 0.5154 for integration/decision-retention cells, and 0.5184 for retrospective recall cells (chance ranking: 0.5). These descriptive protocol means average the declared cells equally; the minimal-delay results below determine how much a harder-condition deficit can be attributed to retention rather than incomplete rule acquisition.

The new sequence rules were not reliably acquired across the minimal-delay bridge cells. Consequently, the longer-delay/load scores do not provide a clean estimate of memory capacity in this run. Near-chance decisions must also be separated from ranking: the motion minimal retrospective probe has BA 25.00% but OVR-AUC 0.8517; biased class decisions therefore coexist with some useful score ordering. No calibration was fitted on these held-out results.

The trace update remains fixed at 0.25/0.75. All 408,728 learned encoder, projection, temporal-output and readout parameters were allowed to adapt. No separate working-memory module, stored prior prediction or hidden sample cache was added.

## Sensory and cue bridge at the selected checkpoint

Held-out balanced accuracy. Anchors retain the original two-frame tasks. Integration and retrospective columns use the new cue/rule formats; previous two-frame scores are not a paired baseline for those tasks.

| Family | Sensory anchor | Minimal integration, visible cue | Minimal integration, flashed cue | Minimal retrospective probe |
|---|---:|---:|---:|---:|
| motion_direction | 72.53% | 35.94% | 42.19% | 25.00% |
| orientation | 100.00% | 50.00% | 50.00% | 50.00% |
| contrast | 97.10% | 50.00% | 50.00% | 45.31% |
| spatial_frequency | 97.22% | 57.03% | 54.69% | 51.56% |
| chromatic_increment | 87.68% | 50.00% | 50.00% | 47.66% |
| contour | 83.06% | 49.22% | 50.78% | 51.56% |
| natural_spectrum | 100.00% | 49.22% | 51.56% | 46.88% |

## Integration length and cue use

| Family | integrate_L8 | integrate_L16 | integrate_L32 | integrate_transient |
|---|---:|---:|---:|---:|
| motion_direction | 25.00% | 25.00% | 25.00% | 25.00% |
| orientation | 50.00% | 49.22% | 50.00% | 50.00% |
| contrast | 50.00% | 50.00% | 50.00% | 50.00% |
| spatial_frequency | 45.31% | 55.47% | 52.34% | 47.66% |
| chromatic_increment | 51.56% | 57.03% | 55.47% | 48.44% |
| contour | 57.81% | 52.34% | 53.12% | 42.97% |
| natural_spectrum | 55.47% | 46.88% | 53.12% | 49.22% |

## Decision retention and interference

| Family | integrate_L8 | decision_D2 | decision_D4 | decision_D8 | decision_D16 | decision_distractor |
|---|---:|---:|---:|---:|---:|---:|
| motion_direction | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% |
| orientation | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| contrast | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| spatial_frequency | 45.31% | 50.00% | 50.00% | 50.00% | 50.00% | 50.78% |
| chromatic_increment | 51.56% | 39.84% | 50.00% | 50.00% | 50.00% | 43.75% |
| contour | 57.81% | 52.34% | 50.78% | 50.00% | 50.00% | 50.78% |
| natural_spectrum | 55.47% | 49.22% | 49.22% | 50.00% | 50.00% | 52.34% |

## Retrospective probe delay

| Family | recall_D0 | recall_D2 | recall_N1 | recall_D8 | recall_D16 |
|---|---:|---:|---:|---:|---:|
| motion_direction | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% |
| orientation | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| contrast | 48.44% | 50.00% | 48.44% | 47.66% | 50.78% |
| spatial_frequency | 53.91% | 49.22% | 56.25% | 53.91% | 45.31% |
| chromatic_increment | 48.44% | 52.34% | 46.88% | 50.00% | 41.41% |
| contour | 53.12% | 53.91% | 46.88% | 46.88% | 50.00% |
| natural_spectrum | 57.03% | 50.00% | 50.78% | 50.00% | 45.31% |

## Retrospective load, binding conditions and timing controls

| Family | recall_N1 | recall_N2 | recall_N4 | recall_matched_one | recall_distractor | recall_precue | recall_postcue |
|---|---:|---:|---:|---:|---:|---:|---:|
| motion_direction | 25.00% | 25.00% | 25.00% | 25.00% | 26.56% | 25.00% | 25.00% |
| orientation | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% |
| contrast | 48.44% | 48.44% | 49.22% | 48.44% | 50.00% | 50.78% | 49.22% |
| spatial_frequency | 56.25% | 46.09% | 56.25% | 50.78% | 47.66% | 53.91% | 53.12% |
| chromatic_increment | 46.88% | 61.72% | 55.47% | 46.09% | 47.66% | 49.22% | 46.88% |
| contour | 46.88% | 48.44% | 50.78% | 44.53% | 55.47% | 53.91% | 56.25% |
| natural_spectrum | 50.78% | 45.31% | 47.66% | 47.66% | 49.22% | 53.12% | 51.56% |

The load comparison changes target age unless controlled. Read recall_N4 alongside the matched-duration one-item condition and the recorded target-age/serial-position strata. Precues, retrocues and postcues differ in what can be selected when; a score difference alone does not isolate an internal selection mechanism. Decision retention can store an already-computed category. The retrospective binary probe cannot be answered before the new probe appears.

## Cell uncertainty and score ranking

Each cell has 128 fresh held-out episodes. Intervals are 500-replicate percentile bootstraps, conditional on this fitted model. Natural cells cluster by the queried target photo; reuse of uncued photos is not separately clustered. The source pools are the existing held-out BSDS photos, not newly independent source images. A perfect empirical interval can collapse at 100% and is not a guarantee of population certainty.

| Family / condition | BA | BA 95% interval | OVR-AUC | AUC 95% interval | Source groups |
|---|---:|---:|---:|---:|---:|
| chromatic_increment/anchor | 87.68% | [82.32,92.46] | 1.0000 | [1.0000,1.0000] | 128 |
| chromatic_increment/bridge_integrate_transient | 50.00% | [50.00,50.00] | 0.3923 | [0.2919,0.4896] | 128 |
| chromatic_increment/bridge_integrate_visible | 50.00% | [50.00,50.00] | 0.4512 | [0.3540,0.5604] | 128 |
| chromatic_increment/bridge_recall | 47.66% | [40.01,55.83] | 0.5366 | [0.4336,0.6243] | 128 |
| chromatic_increment/decision_D16 | 50.00% | [50.00,50.00] | 0.3862 | [0.2968,0.4842] | 128 |
| chromatic_increment/decision_D2 | 39.84% | [31.72,47.57] | 0.4551 | [0.3596,0.5533] | 128 |
| chromatic_increment/decision_D4 | 50.00% | [50.00,50.00] | 0.4883 | [0.3973,0.5914] | 128 |
| chromatic_increment/decision_D8 | 50.00% | [50.00,50.00] | 0.5444 | [0.4495,0.6471] | 128 |
| chromatic_increment/decision_distractor | 43.75% | [35.16,52.44] | 0.4248 | [0.3219,0.5266] | 128 |
| chromatic_increment/integrate_L16 | 57.03% | [49.33,65.69] | 0.5776 | [0.4868,0.6778] | 128 |
| chromatic_increment/integrate_L32 | 55.47% | [46.74,64.13] | 0.5369 | [0.4379,0.6405] | 128 |
| chromatic_increment/integrate_L8 | 51.56% | [42.69,60.12] | 0.5051 | [0.4044,0.5998] | 128 |
| chromatic_increment/integrate_transient | 48.44% | [39.10,56.49] | 0.4785 | [0.3768,0.5718] | 128 |
| chromatic_increment/recall_D0 | 48.44% | [40.64,56.20] | 0.5259 | [0.4183,0.6244] | 128 |
| chromatic_increment/recall_D16 | 41.41% | [32.73,50.25] | 0.3793 | [0.2912,0.4770] | 128 |
| chromatic_increment/recall_D2 | 52.34% | [43.18,61.79] | 0.5925 | [0.4833,0.6851] | 128 |
| chromatic_increment/recall_D8 | 50.00% | [41.25,58.60] | 0.5420 | [0.4391,0.6411] | 128 |
| chromatic_increment/recall_N1 | 46.88% | [38.24,55.47] | 0.5161 | [0.4180,0.6199] | 128 |
| chromatic_increment/recall_N2 | 61.72% | [53.58,69.11] | 0.6465 | [0.5526,0.7318] | 128 |
| chromatic_increment/recall_N4 | 55.47% | [46.91,64.06] | 0.5864 | [0.4823,0.6825] | 128 |
| chromatic_increment/recall_distractor | 47.66% | [39.39,57.32] | 0.4680 | [0.3766,0.5721] | 128 |
| chromatic_increment/recall_matched_one | 46.09% | [38.24,54.84] | 0.4109 | [0.3091,0.5051] | 128 |
| chromatic_increment/recall_postcue | 46.88% | [38.58,55.54] | 0.4731 | [0.3699,0.5785] | 128 |
| chromatic_increment/recall_precue | 49.22% | [40.70,58.35] | 0.4834 | [0.3793,0.5913] | 128 |
| contour/anchor | 83.06% | [77.14,88.78] | 0.9929 | [0.9825,0.9993] | 128 |
| contour/bridge_integrate_transient | 50.78% | [50.00,52.57] | 0.5557 | [0.4518,0.6420] | 128 |
| contour/bridge_integrate_visible | 49.22% | [47.12,50.00] | 0.5286 | [0.4383,0.6275] | 128 |
| contour/bridge_recall | 51.56% | [43.00,60.37] | 0.4973 | [0.3979,0.5930] | 128 |
| contour/decision_D16 | 50.00% | [50.00,50.00] | 0.4797 | [0.3767,0.5717] | 128 |
| contour/decision_D2 | 52.34% | [47.48,57.85] | 0.5249 | [0.4220,0.6271] | 128 |
| contour/decision_D4 | 50.78% | [46.78,54.84] | 0.5314 | [0.4207,0.6312] | 128 |
| contour/decision_D8 | 50.00% | [50.00,50.00] | 0.5713 | [0.4679,0.6616] | 128 |
| contour/decision_distractor | 50.78% | [42.85,59.33] | 0.5005 | [0.4092,0.5991] | 128 |
| contour/integrate_L16 | 52.34% | [45.07,60.04] | 0.4929 | [0.3995,0.5899] | 128 |
| contour/integrate_L32 | 53.12% | [44.98,60.76] | 0.5535 | [0.4455,0.6523] | 128 |
| contour/integrate_L8 | 57.81% | [50.30,65.33] | 0.5454 | [0.4527,0.6463] | 128 |
| contour/integrate_transient | 42.97% | [36.21,50.15] | 0.3555 | [0.2617,0.4592] | 128 |
| contour/recall_D0 | 53.12% | [44.13,61.85] | 0.5144 | [0.4134,0.6148] | 128 |
| contour/recall_D16 | 50.00% | [43.25,57.65] | 0.5085 | [0.4012,0.6096] | 128 |
| contour/recall_D2 | 53.91% | [45.62,62.01] | 0.5378 | [0.4430,0.6361] | 128 |
| contour/recall_D8 | 46.88% | [38.70,55.48] | 0.4692 | [0.3601,0.5665] | 128 |
| contour/recall_N1 | 46.88% | [39.70,55.28] | 0.4084 | [0.3182,0.5029] | 128 |
| contour/recall_N2 | 48.44% | [39.63,57.19] | 0.4937 | [0.3929,0.6097] | 128 |
| contour/recall_N4 | 50.78% | [43.28,59.14] | 0.5103 | [0.4196,0.6133] | 128 |
| contour/recall_distractor | 55.47% | [47.77,63.66] | 0.6338 | [0.5302,0.7414] | 128 |
| contour/recall_matched_one | 44.53% | [36.76,53.54] | 0.4824 | [0.3842,0.5833] | 128 |
| contour/recall_postcue | 56.25% | [47.39,64.24] | 0.5764 | [0.4738,0.6714] | 128 |
| contour/recall_precue | 53.91% | [46.28,61.91] | 0.5879 | [0.4936,0.6783] | 128 |
| contrast/anchor | 97.10% | [94.20,99.35] | 0.9995 | [0.9975,1.0000] | 128 |
| contrast/bridge_integrate_transient | 50.00% | [50.00,50.00] | 0.4866 | [0.3974,0.5909] | 128 |
| contrast/bridge_integrate_visible | 50.00% | [50.00,50.00] | 0.5076 | [0.4021,0.6135] | 128 |
| contrast/bridge_recall | 45.31% | [40.63,50.31] | 0.5183 | [0.4161,0.6238] | 128 |
| contrast/decision_D16 | 50.00% | [50.00,50.00] | 0.5266 | [0.4293,0.6312] | 128 |
| contrast/decision_D2 | 50.00% | [50.00,50.00] | 0.6213 | [0.5286,0.7125] | 128 |
| contrast/decision_D4 | 50.00% | [50.00,50.00] | 0.5461 | [0.4461,0.6502] | 128 |
| contrast/decision_D8 | 50.00% | [50.00,50.00] | 0.5283 | [0.4334,0.6202] | 128 |
| contrast/decision_distractor | 50.00% | [50.00,50.00] | 0.4441 | [0.3502,0.5316] | 128 |
| contrast/integrate_L16 | 50.00% | [50.00,50.00] | 0.4482 | [0.3660,0.5333] | 128 |
| contrast/integrate_L32 | 50.00% | [50.00,50.00] | 0.5986 | [0.4901,0.6916] | 128 |
| contrast/integrate_L8 | 50.00% | [50.00,50.00] | 0.4905 | [0.3979,0.5994] | 128 |
| contrast/integrate_transient | 50.00% | [50.00,50.00] | 0.5498 | [0.4503,0.6475] | 128 |
| contrast/recall_D0 | 48.44% | [45.45,50.93] | 0.5261 | [0.4288,0.6274] | 128 |
| contrast/recall_D16 | 50.78% | [48.20,53.94] | 0.4458 | [0.3518,0.5359] | 128 |
| contrast/recall_D2 | 50.00% | [47.96,52.02] | 0.5125 | [0.4182,0.6110] | 128 |
| contrast/recall_D8 | 47.66% | [44.14,50.66] | 0.3806 | [0.2832,0.4767] | 128 |
| contrast/recall_N1 | 48.44% | [45.31,51.39] | 0.5059 | [0.4110,0.6065] | 128 |
| contrast/recall_N2 | 48.44% | [44.74,51.94] | 0.4648 | [0.3797,0.5723] | 128 |
| contrast/recall_N4 | 49.22% | [45.85,52.21] | 0.4478 | [0.3382,0.5440] | 128 |
| contrast/recall_distractor | 50.00% | [44.80,54.85] | 0.3911 | [0.2932,0.4884] | 128 |
| contrast/recall_matched_one | 48.44% | [45.24,51.11] | 0.5183 | [0.4177,0.6154] | 128 |
| contrast/recall_postcue | 49.22% | [46.07,52.46] | 0.5640 | [0.4723,0.6614] | 128 |
| contrast/recall_precue | 50.78% | [48.20,53.28] | 0.5164 | [0.4168,0.6215] | 128 |
| motion_direction/anchor | 72.53% | [65.34,79.54] | 0.9824 | [0.9706,0.9910] | 128 |
| motion_direction/bridge_integrate_transient | 42.19% | [37.69,45.90] | 0.7908 | [0.7461,0.8291] | 128 |
| motion_direction/bridge_integrate_visible | 35.94% | [31.67,40.25] | 0.7650 | [0.7243,0.8066] | 128 |
| motion_direction/bridge_recall | 25.00% | [25.00,25.00] | 0.8517 | [0.8192,0.8794] | 128 |
| motion_direction/decision_D16 | 25.00% | [25.00,25.00] | 0.5433 | [0.4849,0.6062] | 128 |
| motion_direction/decision_D2 | 25.00% | [25.00,25.00] | 0.5769 | [0.5210,0.6379] | 128 |
| motion_direction/decision_D4 | 25.00% | [25.00,25.00] | 0.5114 | [0.4450,0.5802] | 128 |
| motion_direction/decision_D8 | 25.00% | [25.00,25.00] | 0.5119 | [0.4477,0.5789] | 128 |
| motion_direction/decision_distractor | 25.00% | [25.00,25.00] | 0.4744 | [0.4063,0.5433] | 128 |
| motion_direction/integrate_L16 | 25.00% | [25.00,25.00] | 0.5127 | [0.4483,0.5728] | 128 |
| motion_direction/integrate_L32 | 25.00% | [25.00,25.00] | 0.4910 | [0.4152,0.5604] | 128 |
| motion_direction/integrate_L8 | 25.00% | [25.00,25.00] | 0.5083 | [0.4459,0.5692] | 128 |
| motion_direction/integrate_transient | 25.00% | [25.00,25.00] | 0.5173 | [0.4530,0.5841] | 128 |
| motion_direction/recall_D0 | 25.00% | [25.00,25.00] | 0.7150 | [0.6613,0.7786] | 128 |
| motion_direction/recall_D16 | 25.00% | [25.00,25.00] | 0.5325 | [0.4662,0.6033] | 128 |
| motion_direction/recall_D2 | 25.00% | [25.00,25.00] | 0.6237 | [0.5620,0.6790] | 128 |
| motion_direction/recall_D8 | 25.00% | [25.00,25.00] | 0.5017 | [0.4402,0.5727] | 128 |
| motion_direction/recall_N1 | 25.00% | [25.00,25.00] | 0.6140 | [0.5436,0.6910] | 128 |
| motion_direction/recall_N2 | 25.00% | [25.00,25.00] | 0.4787 | [0.4072,0.5511] | 128 |
| motion_direction/recall_N4 | 25.00% | [25.00,25.00] | 0.4530 | [0.3855,0.5100] | 128 |
| motion_direction/recall_distractor | 26.56% | [20.93,31.64] | 0.5052 | [0.4439,0.5703] | 128 |
| motion_direction/recall_matched_one | 25.00% | [25.00,25.00] | 0.5514 | [0.4837,0.6170] | 128 |
| motion_direction/recall_postcue | 25.00% | [25.00,25.00] | 0.5564 | [0.4920,0.6243] | 128 |
| motion_direction/recall_precue | 25.00% | [25.00,25.00] | 0.5651 | [0.4954,0.6325] | 128 |
| natural_spectrum/anchor | 100.00% | [100.00,100.00] | 1.0000 | [1.0000,1.0000] | 98 |
| natural_spectrum/bridge_integrate_transient | 51.56% | [45.60,57.28] | 0.5017 | [0.4166,0.5941] | 95 |
| natural_spectrum/bridge_integrate_visible | 49.22% | [42.11,55.88] | 0.5518 | [0.4600,0.6499] | 96 |
| natural_spectrum/bridge_recall | 46.88% | [39.14,55.03] | 0.5193 | [0.4262,0.6168] | 95 |
| natural_spectrum/decision_D16 | 50.00% | [50.00,50.00] | 0.5122 | [0.4098,0.6140] | 96 |
| natural_spectrum/decision_D2 | 49.22% | [47.54,50.00] | 0.5481 | [0.4383,0.6466] | 99 |
| natural_spectrum/decision_D4 | 49.22% | [44.37,54.27] | 0.5266 | [0.4296,0.6206] | 97 |
| natural_spectrum/decision_D8 | 50.00% | [50.00,50.00] | 0.4580 | [0.3549,0.5705] | 96 |
| natural_spectrum/decision_distractor | 52.34% | [47.60,56.86] | 0.5386 | [0.4346,0.6410] | 102 |
| natural_spectrum/integrate_L16 | 46.88% | [39.15,54.58] | 0.4604 | [0.3643,0.5672] | 94 |
| natural_spectrum/integrate_L32 | 53.12% | [45.80,60.55] | 0.5439 | [0.4419,0.6465] | 90 |
| natural_spectrum/integrate_L8 | 55.47% | [50.58,60.23] | 0.5139 | [0.4175,0.6094] | 100 |
| natural_spectrum/integrate_transient | 49.22% | [43.55,54.83] | 0.5051 | [0.4126,0.6019] | 94 |
| natural_spectrum/recall_D0 | 57.03% | [48.62,64.54] | 0.5830 | [0.4755,0.6843] | 102 |
| natural_spectrum/recall_D16 | 45.31% | [37.83,53.42] | 0.4355 | [0.3582,0.5215] | 89 |
| natural_spectrum/recall_D2 | 50.00% | [42.48,58.74] | 0.5225 | [0.4341,0.6313] | 104 |
| natural_spectrum/recall_D8 | 50.00% | [41.58,58.85] | 0.5242 | [0.4203,0.6137] | 100 |
| natural_spectrum/recall_N1 | 50.78% | [42.93,58.69] | 0.5005 | [0.4049,0.5862] | 95 |
| natural_spectrum/recall_N2 | 45.31% | [37.85,53.14] | 0.4280 | [0.3385,0.5247] | 90 |
| natural_spectrum/recall_N4 | 47.66% | [38.06,57.07] | 0.4883 | [0.3799,0.5948] | 93 |
| natural_spectrum/recall_distractor | 49.22% | [39.91,58.82] | 0.4575 | [0.3521,0.5686] | 97 |
| natural_spectrum/recall_matched_one | 47.66% | [38.57,55.78] | 0.5010 | [0.3898,0.6090] | 96 |
| natural_spectrum/recall_postcue | 51.56% | [44.63,59.10] | 0.5190 | [0.4280,0.6218] | 93 |
| natural_spectrum/recall_precue | 53.12% | [45.50,61.62] | 0.4983 | [0.4010,0.5874] | 94 |
| orientation/anchor | 100.00% | [100.00,100.00] | 1.0000 | [1.0000,1.0000] | 128 |
| orientation/bridge_integrate_transient | 50.00% | [50.00,50.00] | 0.4807 | [0.3749,0.5835] | 128 |
| orientation/bridge_integrate_visible | 50.00% | [50.00,50.00] | 0.5886 | [0.4773,0.6874] | 128 |
| orientation/bridge_recall | 50.00% | [50.00,50.00] | 0.5635 | [0.4649,0.6551] | 128 |
| orientation/decision_D16 | 50.00% | [50.00,50.00] | 0.4363 | [0.3385,0.5303] | 128 |
| orientation/decision_D2 | 50.00% | [50.00,50.00] | 0.4912 | [0.4038,0.6037] | 128 |
| orientation/decision_D4 | 50.00% | [50.00,50.00] | 0.4229 | [0.3312,0.5319] | 128 |
| orientation/decision_D8 | 50.00% | [50.00,50.00] | 0.5017 | [0.4003,0.6101] | 128 |
| orientation/decision_distractor | 50.00% | [50.00,50.00] | 0.4824 | [0.3818,0.5809] | 128 |
| orientation/integrate_L16 | 49.22% | [47.58,50.00] | 0.4985 | [0.3940,0.6071] | 128 |
| orientation/integrate_L32 | 50.00% | [47.56,52.21] | 0.5686 | [0.4619,0.6767] | 128 |
| orientation/integrate_L8 | 50.00% | [50.00,50.00] | 0.4912 | [0.4024,0.5888] | 128 |
| orientation/integrate_transient | 50.00% | [50.00,50.00] | 0.5957 | [0.4860,0.6907] | 128 |
| orientation/recall_D0 | 50.00% | [50.00,50.00] | 0.4749 | [0.3795,0.5718] | 128 |
| orientation/recall_D16 | 50.00% | [50.00,50.00] | 0.4558 | [0.3467,0.5576] | 128 |
| orientation/recall_D2 | 50.00% | [50.00,50.00] | 0.5818 | [0.4867,0.6846] | 128 |
| orientation/recall_D8 | 50.00% | [50.00,50.00] | 0.4702 | [0.3687,0.5729] | 128 |
| orientation/recall_N1 | 50.00% | [50.00,50.00] | 0.4648 | [0.3671,0.5674] | 128 |
| orientation/recall_N2 | 50.00% | [50.00,50.00] | 0.4690 | [0.3764,0.5739] | 128 |
| orientation/recall_N4 | 50.00% | [50.00,50.00] | 0.4487 | [0.3521,0.5509] | 128 |
| orientation/recall_distractor | 50.00% | [50.00,50.00] | 0.5220 | [0.4209,0.6209] | 128 |
| orientation/recall_matched_one | 50.00% | [50.00,50.00] | 0.6045 | [0.5085,0.7022] | 128 |
| orientation/recall_postcue | 50.00% | [50.00,50.00] | 0.5244 | [0.4283,0.6383] | 128 |
| orientation/recall_precue | 50.00% | [50.00,50.00] | 0.4751 | [0.3638,0.5725] | 128 |
| spatial_frequency/anchor | 97.22% | [94.52,99.35] | 0.9998 | [0.9985,1.0000] | 128 |
| spatial_frequency/bridge_integrate_transient | 54.69% | [47.31,62.50] | 0.6201 | [0.5270,0.7200] | 128 |
| spatial_frequency/bridge_integrate_visible | 57.03% | [49.85,64.87] | 0.6108 | [0.5138,0.7034] | 128 |
| spatial_frequency/bridge_recall | 51.56% | [45.30,57.60] | 0.6069 | [0.5088,0.7016] | 128 |
| spatial_frequency/decision_D16 | 50.00% | [50.00,50.00] | 0.4702 | [0.3510,0.5763] | 128 |
| spatial_frequency/decision_D2 | 50.00% | [50.00,50.00] | 0.4325 | [0.3387,0.5304] | 128 |
| spatial_frequency/decision_D4 | 50.00% | [50.00,50.00] | 0.5237 | [0.4221,0.6342] | 128 |
| spatial_frequency/decision_D8 | 50.00% | [50.00,50.00] | 0.5444 | [0.4519,0.6405] | 128 |
| spatial_frequency/decision_distractor | 50.78% | [44.12,57.92] | 0.4934 | [0.3931,0.5943] | 128 |
| spatial_frequency/integrate_L16 | 55.47% | [48.31,62.61] | 0.5383 | [0.4334,0.6363] | 128 |
| spatial_frequency/integrate_L32 | 52.34% | [46.09,58.06] | 0.4722 | [0.3655,0.5694] | 128 |
| spatial_frequency/integrate_L8 | 45.31% | [38.48,51.54] | 0.4482 | [0.3459,0.5479] | 128 |
| spatial_frequency/integrate_transient | 47.66% | [40.46,55.09] | 0.4866 | [0.3824,0.5874] | 128 |
| spatial_frequency/recall_D0 | 53.91% | [46.15,60.73] | 0.6169 | [0.5139,0.7039] | 128 |
| spatial_frequency/recall_D16 | 45.31% | [36.76,53.01] | 0.4792 | [0.3780,0.5825] | 128 |
| spatial_frequency/recall_D2 | 49.22% | [42.62,55.91] | 0.5251 | [0.4202,0.6335] | 128 |
| spatial_frequency/recall_D8 | 53.91% | [47.20,61.57] | 0.4780 | [0.3773,0.5872] | 128 |
| spatial_frequency/recall_N1 | 56.25% | [48.74,63.93] | 0.5315 | [0.4405,0.6307] | 128 |
| spatial_frequency/recall_N2 | 46.09% | [39.07,53.10] | 0.4709 | [0.3747,0.5809] | 128 |
| spatial_frequency/recall_N4 | 56.25% | [49.30,63.11] | 0.4836 | [0.3828,0.5833] | 128 |
| spatial_frequency/recall_distractor | 47.66% | [42.02,53.50] | 0.4399 | [0.3348,0.5489] | 128 |
| spatial_frequency/recall_matched_one | 50.78% | [43.92,57.42] | 0.5996 | [0.4916,0.6970] | 128 |
| spatial_frequency/recall_postcue | 53.12% | [46.47,59.61] | 0.5806 | [0.4720,0.6613] | 128 |
| spatial_frequency/recall_precue | 53.91% | [47.02,60.98] | 0.5789 | [0.4739,0.6764] | 128 |

Confusion matrices, binary hit/miss/false-alarm/correct-rejection counts, sensitivity/criterion, class recall intervals and condition-specific strata are in results.json. Binary and four-class chance accuracy differ; no pooled raw accuracy is a capacity claim. High ranking with biased argmax decisions is not evidence of absent information.

## Acquisition trajectory and selection

The bridge was trained first and its compatible weights/Adam state continued into the joint mixture. There was no accuracy gate or restart. Bridge validation is reported separately and never competes directly with joint checkpoint selection. Joint selection uses mean task OVR-AUC: equal family/protocol groups, with conditions averaged inside each group. Worst-cell BA is descriptive.

| Stage | Update | Episodes | Validation mean AUC | Validation worst-cell BA |
|---|---:|---:|---:|---:|
| bridge | 980 | 3,920 | 0.7406 | 37.50% |
| joint | 3920 | 15,680 | 0.6783 | 25.00% |
| joint | 6860 | 27,440 | 0.6834 | 25.00% |
| joint | 9800 | 39,200 | 0.6769 | 15.62% |

Weak minimal-delay sensory/cue/probe performance makes stronger memory-capacity interpretations inconclusive. A finite exploratory exposure is not a sufficient acquisition horizon. Conversely, good trained-condition performance does not establish arbitrary-length retention or general working-memory competence.

## Exposure and execution

Batch 4; 3,920 bridge episodes and 35,280 joint episodes. The joint mixture is 10% sensory anchors, 45% integration/decision-retention, 45% retrospective probes, with families balanced within groups. Every episode contributes one terminal cross-entropy loss; blank/cue/distractor frames still update the state. There is no padding or temporal gradient truncation.

Executed scalar/recall renderers use nine discrete levels. Orientation spacing is 7.5°; contrast spans 0.08–0.36 in log modulation amplitude; spatial frequency spans 4–10 cycles/image in log spacing; the chromatic-axis coordinate spans −0.12…+0.12; natural beta spans +0.6…−0.6. New contour paths span bend −9…9 pixels with 2° jitter; the old sensory anchors retain their original difficulty sampling. Thus mismatch values in the raw strata are level differences, not a continuous human precision measurement.

Terminal encoder/frame presentations: 539,784; selected-checkpoint presentations: 365,316. These counts include informative frames and repeated/blank/cue observations, which are not independent episodes.

| Stage / family / condition | Selected-checkpoint episodes | Terminal episodes |
|---|---:|---:|
| bridge/motion_direction/anchor | 140 | 140 |
| bridge/orientation/anchor | 140 | 140 |
| bridge/contrast/anchor | 140 | 140 |
| bridge/spatial_frequency/anchor | 140 | 140 |
| bridge/chromatic_increment/anchor | 140 | 140 |
| bridge/contour/anchor | 140 | 140 |
| bridge/natural_spectrum/anchor | 140 | 140 |
| bridge/motion_direction/bridge_integrate_visible | 140 | 140 |
| bridge/orientation/bridge_integrate_visible | 140 | 140 |
| bridge/contrast/bridge_integrate_visible | 140 | 140 |
| bridge/spatial_frequency/bridge_integrate_visible | 140 | 140 |
| bridge/chromatic_increment/bridge_integrate_visible | 140 | 140 |
| bridge/contour/bridge_integrate_visible | 140 | 140 |
| bridge/natural_spectrum/bridge_integrate_visible | 140 | 140 |
| bridge/motion_direction/bridge_integrate_transient | 140 | 140 |
| bridge/orientation/bridge_integrate_transient | 140 | 140 |
| bridge/contrast/bridge_integrate_transient | 140 | 140 |
| bridge/spatial_frequency/bridge_integrate_transient | 140 | 140 |
| bridge/chromatic_increment/bridge_integrate_transient | 140 | 140 |
| bridge/contour/bridge_integrate_transient | 140 | 140 |
| bridge/natural_spectrum/bridge_integrate_transient | 140 | 140 |
| bridge/motion_direction/bridge_recall | 140 | 140 |
| bridge/orientation/bridge_recall | 140 | 140 |
| bridge/contrast/bridge_recall | 140 | 140 |
| bridge/spatial_frequency/bridge_recall | 140 | 140 |
| bridge/chromatic_increment/bridge_recall | 140 | 140 |
| bridge/contour/bridge_recall | 140 | 140 |
| bridge/natural_spectrum/bridge_recall | 140 | 140 |
| joint/motion_direction/anchor | 336 | 504 |
| joint/motion_direction/integrate_L8 | 168 | 252 |
| joint/motion_direction/recall_N1 | 140 | 208 |
| joint/orientation/integrate_L8 | 168 | 252 |
| joint/orientation/recall_N1 | 140 | 208 |
| joint/contrast/integrate_L8 | 168 | 252 |
| joint/contrast/recall_N1 | 140 | 208 |
| joint/spatial_frequency/integrate_L8 | 168 | 252 |
| joint/spatial_frequency/recall_N1 | 140 | 208 |
| joint/chromatic_increment/integrate_L8 | 168 | 252 |
| joint/orientation/anchor | 336 | 504 |
| joint/chromatic_increment/recall_N1 | 140 | 208 |
| joint/contour/integrate_L8 | 168 | 252 |
| joint/contour/recall_N1 | 140 | 208 |
| joint/natural_spectrum/integrate_L8 | 168 | 252 |
| joint/natural_spectrum/recall_N1 | 140 | 208 |
| joint/motion_direction/integrate_L16 | 168 | 252 |
| joint/motion_direction/recall_N2 | 140 | 208 |
| joint/orientation/integrate_L16 | 168 | 252 |
| joint/orientation/recall_N2 | 140 | 208 |
| joint/contrast/anchor | 336 | 504 |
| joint/contrast/integrate_L16 | 168 | 252 |
| joint/contrast/recall_N2 | 140 | 208 |
| joint/spatial_frequency/integrate_L16 | 168 | 252 |
| joint/spatial_frequency/recall_N2 | 140 | 208 |
| joint/chromatic_increment/integrate_L16 | 168 | 252 |
| joint/chromatic_increment/recall_N2 | 140 | 208 |
| joint/contour/integrate_L16 | 168 | 252 |
| joint/contour/recall_N2 | 140 | 208 |
| joint/natural_spectrum/integrate_L16 | 168 | 252 |
| joint/spatial_frequency/anchor | 336 | 504 |
| joint/natural_spectrum/recall_N2 | 140 | 208 |
| joint/motion_direction/integrate_L32 | 168 | 252 |
| joint/motion_direction/recall_N4 | 140 | 208 |
| joint/orientation/integrate_L32 | 168 | 252 |
| joint/orientation/recall_N4 | 140 | 208 |
| joint/contrast/integrate_L32 | 168 | 252 |
| joint/contrast/recall_N4 | 140 | 208 |
| joint/spatial_frequency/integrate_L32 | 168 | 252 |
| joint/spatial_frequency/recall_N4 | 140 | 208 |
| joint/chromatic_increment/anchor | 336 | 504 |
| joint/chromatic_increment/integrate_L32 | 168 | 252 |
| joint/chromatic_increment/recall_N4 | 140 | 208 |
| joint/contour/integrate_L32 | 168 | 252 |
| joint/contour/recall_N4 | 140 | 208 |
| joint/natural_spectrum/integrate_L32 | 168 | 252 |
| joint/natural_spectrum/recall_N4 | 140 | 208 |
| joint/motion_direction/decision_D2 | 168 | 252 |
| joint/motion_direction/recall_D0 | 140 | 208 |
| joint/orientation/decision_D2 | 168 | 252 |
| joint/contour/anchor | 336 | 504 |
| joint/orientation/recall_D0 | 140 | 208 |
| joint/contrast/decision_D2 | 168 | 252 |
| joint/contrast/recall_D0 | 140 | 208 |
| joint/spatial_frequency/decision_D2 | 168 | 252 |
| joint/spatial_frequency/recall_D0 | 140 | 208 |
| joint/chromatic_increment/decision_D2 | 168 | 252 |
| joint/chromatic_increment/recall_D0 | 140 | 208 |
| joint/contour/decision_D2 | 168 | 252 |
| joint/contour/recall_D0 | 140 | 208 |
| joint/natural_spectrum/anchor | 336 | 504 |
| joint/natural_spectrum/decision_D2 | 168 | 252 |
| joint/natural_spectrum/recall_D0 | 140 | 208 |
| joint/motion_direction/decision_D4 | 168 | 252 |
| joint/motion_direction/recall_D2 | 136 | 208 |
| joint/orientation/decision_D4 | 168 | 252 |
| joint/orientation/recall_D2 | 136 | 208 |
| joint/contrast/decision_D4 | 168 | 252 |
| joint/contrast/recall_D2 | 136 | 208 |
| joint/spatial_frequency/decision_D4 | 168 | 252 |
| joint/spatial_frequency/recall_D2 | 136 | 208 |
| joint/chromatic_increment/decision_D4 | 168 | 252 |
| joint/chromatic_increment/recall_D2 | 136 | 208 |
| joint/contour/decision_D4 | 168 | 252 |
| joint/contour/recall_D2 | 136 | 208 |
| joint/natural_spectrum/decision_D4 | 168 | 252 |
| joint/natural_spectrum/recall_D2 | 136 | 208 |
| joint/motion_direction/decision_D8 | 168 | 252 |
| joint/motion_direction/recall_D8 | 136 | 208 |
| joint/orientation/decision_D8 | 168 | 252 |
| joint/orientation/recall_D8 | 136 | 208 |
| joint/contrast/decision_D8 | 168 | 252 |
| joint/contrast/recall_D8 | 136 | 208 |
| joint/spatial_frequency/decision_D8 | 168 | 252 |
| joint/spatial_frequency/recall_D8 | 136 | 208 |
| joint/chromatic_increment/decision_D8 | 168 | 252 |
| joint/chromatic_increment/recall_D8 | 136 | 208 |
| joint/contour/decision_D8 | 168 | 252 |
| joint/contour/recall_D8 | 136 | 208 |
| joint/natural_spectrum/decision_D8 | 168 | 252 |
| joint/natural_spectrum/recall_D8 | 136 | 208 |
| joint/motion_direction/decision_D16 | 168 | 252 |
| joint/motion_direction/recall_D16 | 136 | 204 |
| joint/orientation/decision_D16 | 168 | 252 |
| joint/orientation/recall_D16 | 136 | 204 |
| joint/contrast/decision_D16 | 168 | 252 |
| joint/contrast/recall_D16 | 136 | 204 |
| joint/spatial_frequency/decision_D16 | 168 | 252 |
| joint/spatial_frequency/recall_D16 | 136 | 204 |
| joint/chromatic_increment/decision_D16 | 168 | 252 |
| joint/chromatic_increment/recall_D16 | 136 | 204 |
| joint/contour/decision_D16 | 168 | 252 |
| joint/contour/recall_D16 | 136 | 204 |
| joint/natural_spectrum/decision_D16 | 168 | 252 |
| joint/natural_spectrum/recall_D16 | 136 | 204 |
| joint/motion_direction/integrate_transient | 168 | 252 |
| joint/motion_direction/recall_distractor | 136 | 204 |
| joint/orientation/integrate_transient | 168 | 252 |
| joint/orientation/recall_distractor | 136 | 204 |
| joint/contrast/integrate_transient | 168 | 252 |
| joint/contrast/recall_distractor | 136 | 204 |
| joint/spatial_frequency/integrate_transient | 168 | 252 |
| joint/spatial_frequency/recall_distractor | 136 | 204 |
| joint/chromatic_increment/integrate_transient | 168 | 252 |
| joint/chromatic_increment/recall_distractor | 136 | 204 |
| joint/contour/integrate_transient | 168 | 252 |
| joint/contour/recall_distractor | 136 | 204 |
| joint/natural_spectrum/integrate_transient | 168 | 252 |
| joint/natural_spectrum/recall_distractor | 136 | 204 |
| joint/motion_direction/decision_distractor | 168 | 252 |
| joint/motion_direction/recall_precue | 136 | 204 |
| joint/orientation/decision_distractor | 168 | 252 |
| joint/orientation/recall_precue | 136 | 204 |
| joint/contrast/decision_distractor | 168 | 252 |
| joint/contrast/recall_precue | 136 | 204 |
| joint/spatial_frequency/decision_distractor | 168 | 252 |
| joint/spatial_frequency/recall_precue | 136 | 204 |
| joint/chromatic_increment/decision_distractor | 168 | 252 |
| joint/chromatic_increment/recall_precue | 136 | 204 |
| joint/contour/decision_distractor | 168 | 252 |
| joint/contour/recall_precue | 136 | 204 |
| joint/natural_spectrum/decision_distractor | 168 | 252 |
| joint/natural_spectrum/recall_precue | 136 | 204 |
| joint/motion_direction/recall_postcue | 136 | 204 |
| joint/orientation/recall_postcue | 136 | 204 |
| joint/contrast/recall_postcue | 136 | 204 |
| joint/spatial_frequency/recall_postcue | 136 | 204 |
| joint/chromatic_increment/recall_postcue | 136 | 204 |
| joint/contour/recall_postcue | 136 | 204 |
| joint/natural_spectrum/recall_postcue | 136 | 204 |
| joint/motion_direction/recall_matched_one | 136 | 204 |
| joint/orientation/recall_matched_one | 136 | 204 |
| joint/contrast/recall_matched_one | 136 | 204 |
| joint/spatial_frequency/recall_matched_one | 136 | 204 |
| joint/chromatic_increment/recall_matched_one | 136 | 204 |
| joint/contour/recall_matched_one | 136 | 204 |
| joint/natural_spectrum/recall_matched_one | 136 | 204 |

Detailed event/cue/probe/blank/distractor counts: `training.metadata_counts` in the aggregate. These are overlapping event tags (for example, a probe can also be the report frame), not an additive partition of total images. Logical observed-frame updates exclude exact backward activation recomputation; recomputation costs time but supplies no new observation. Profiles were separate preserved fits, never production initialization; their training/evaluation presentations are not added to production exposure.

The supervisor completed in 5444.22 seconds of the new 14,400-second allowance. One GPU worker ran at a time, fp32, no AMP/TF32, two PyTorch CPU threads. Exact non-reentrant encoder/projection activation checkpointing preserved RNG and full temporal gradients. The focused CPU check matched old two-frame logits, showed nonzero earliest-frame/encoder gradients and zero checkpointed-versus-uncheckpointed parameter-gradient difference.

## Reproduction and interpretation limits

Run: `C:\Users\jomor\Documents\VisualAttentionWorkingMemory\WorkingMemory\runs\wm_20260912_181219`. Source/data manifests, fixed_config.json, budget.json, supervisor/worker logs, metrics.csv, immutable full-state checkpoints and byte indices, raw held-out predictions and exit.json are preserved. Initial trained parent, exact seeds, all condition dictionaries and fixed exposure are in fixed_config.json. The new optimizer and sampler are versioned; no strict continuation of the old temporal optimizer is claimed.

The held-out set contains the declared trained-condition slices. Longer 64-transition, delay 32 and load 6 extrapolations were not included in this fixed allocation. The generator has a verified matched-motion construction; a separate model counterfactual sweep was not silently added. Standard motion-duration results alone therefore cannot rule out every correlated summary heuristic.

For identical future image suffixes, earlier-history trace differences decay by 0.25^D/0.75^D. Trainable per-frame encoding can use a simultaneously visible cue but has no history feedback or learned retention coefficient. This structure predicts a useful stress axis; small residuals/readout amplification and finite precision prevent a universal accuracy theorem. Any proposed memory addition must address an observed, acquired-task deficit and compete with changes to the existing computation. This experiment does not force the conclusion that a separately named working-memory module is necessary.
