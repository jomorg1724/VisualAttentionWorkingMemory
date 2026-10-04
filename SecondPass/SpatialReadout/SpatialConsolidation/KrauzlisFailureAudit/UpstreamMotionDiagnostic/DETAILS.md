# Supplemental scoring and execution audit

## Event hits and catch false alarms
| Observer | Hits /258 | Catch false positives /42 |
|---|---:|---:|
| pixel/full | 233 | 6 |
| pixel/endpoint | 217 | 24 |
| cnn25/phases | 258 | 42 |
| cnn25/final | 233 | 39 |
| kda25/phases | 258 | 42 |
| kda25/final | 236 | 35 |
| kda7/phases | 258 | 42 |
| kda7/final | 215 | 38 |
| memory/phases | 258 | 42 |
| memory/final | 258 | 42 |

The pixel any-event thresholds are16° full and18° endpoint. Target-report thresholds are13° full and20° endpoint; the target is decoded from the cue pixels, never supplied from truth. All phase neural event probes and both memory access probes declare event on all258 events AND all42 catches. Their degenerate BA bootstrap [.5,.5] is not proof of no information or exact population performance; inspect AUC uncertainty.

## Actual target reports
| Pixel access | Target hits /171 | Foil false reports /87 | Catch false reports /42 |
|---|---:|---:|---:|
| full | 158 | 16 | 6 |
| endpoint | 125 | 22 | 13 |

## Group-shuffled controls
One deterministic shuffled-label fit per task/access, not a permutation-test null distribution. Validation selection uses the same candidate grid and untouched validation truth.
| Access | True /shuffled side BA | True /shuffled event BA | True /shuffled changed-patch error |
|---|---:|---:|---:|
| cnn25/phases | 0.496 /0.477 | 0.500 /0.558 | 27.00° /26.99° |
| cnn25/final | 0.504 /0.477 | 0.487 /0.491 | 27.29° /27.32° |
| kda25/phases | 0.496 /0.505 | 0.500 /0.500 | 27.41° /26.88° |
| kda25/final | 0.504 /0.474 | 0.541 /0.495 | 27.14° /26.34° |
| kda7/phases | 0.489 /0.527 | 0.500 /0.500 | 27.18° /26.75° |
| kda7/final | 0.546 /0.562 | 0.464 /0.496 | 27.41° /26.31° |
| memory/phases | 0.477 /0.496 | 0.500 /0.500 | 26.86° /26.99° |
| memory/final | 0.449 /0.539 | 0.500 /0.500 | 27.34° /27.09° |

## Matched access uncertainty
| Phase minus final | Side BA gain [95% CI] |
|---|---:|
| cnn25 | -0.008 [-0.089,+0.069] |
| kda25 | -0.008 [-0.090,+0.083] |
| kda7 | -0.058 [-0.147,+0.027] |
| memory | +0.027 [-0.054,+0.112] |

Pixel full minus endpoint side BA: +0.206 [+0.153,+0.264]. Unlike the neural access pairs, full pixels explicitly contain more raw samples; this is a temporal-evidence benchmark, not a capacity-matched architecture test.

## Pixel support and timing
{
  "full": {
    "zero_flow_patch_phases": 1,
    "total_patch_phases": 1200,
    "mean_accepted_flow_vectors_by_phase_patch": [
      [
        36.82333333333333,
        36.666666666666664
      ],
      [
        14.163333333333334,
        13.636666666666667
      ]
    ]
  },
  "endpoint": {
    "zero_flow_patch_phases": 91,
    "total_patch_phases": 1200,
    "mean_accepted_flow_vectors_by_phase_patch": [
      [
        3.506666666666667,
        3.7666666666666666
      ],
      [
        3.61,
        3.296666666666667
      ]
    ]
  }
}
All episodes retained. A phase with no accepted flow receives angle0 by the unchanged method; no ground-truth-dependent filtering. Endpoint vectors sample only the last two transitions while targets summarize all movement angles in the phase. This target/window mismatch and dot resets contribute to endpoint error. No learned neural pooling-loss measurement is made.
Image cue decoding300/300 has a Wilson95% accuracy bound [0.9874,1], not population certainty.

## Scope of temporal-access comparison
Each neural assay reads TWO fixed32-coordinate random projections (with the same slot-specific coordinate rolls), then all quadratic products. It does not read all raw pre/post coordinates or an explicit uncompressed feature difference. True labels/directions only supervise fits and score predictions. Prior direction decoders use full1152/576/3136 inputs and are not capacity-equivalent to the2144 polynomial-feature direct comparator. These restrictions can hide information; a failure is not neural erasure. The physical-side probe is conditional on a real event only for fitting/scoring; truth is not input to its predictor and it emits scores on catches too.

## Failure, recovery and verification
The initial process failed at import with ModuleNotFoundError: threadpoolctl, before main(), fitting, selection or test generation. failed_import.log and failure_recovery.json preserve it. The optional dependency was removed; all numerical environment caps were set before imports and PyTorch used2 intra-op/1 inter-op threads. No install, budget reset, duplicate scientific worker or renewed allowance. The recovered run used the SAME persisted deadline.
The process tool subsequently returned premature exited/exit_code:null notices. OS ps verified the sole worker77584 still running while run.log advanced; no duplicate was launched. Completion was accepted only after completion.json, final log and OS zombie/no-running-worker state. Saved-array audit ran afterward, serially.
Independent audit:48 scalers exactly reconstructed from their true fit groups; maximum ridge normal-equation relative residual3.05e-11. All new and reused direction predictions replay with maximum absolute error0.0. Legacy full-history pixel method agrees within3.55e-15 degrees on the three saved B12/B20/B28 examples. Frozen fit/selection hashes unchanged; renderer/hook/source/checkpoint checks pass. Unit tests cover input-only final access/NaN removal and circular wrap/pixel window contract.

All uncertainty is episode-sampling uncertainty conditional on this one checkpoint and fixed probe fits; validation was reused and no multiplicity correction applied. No fitted target-label neural rescue or training remedy was chosen.
