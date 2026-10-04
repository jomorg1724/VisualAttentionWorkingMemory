# Upstream motion diagnostic — frozen selected2297

**Fresh-test diagnostic, not deployed training or a causal lesion test.**

## Decision and interpretation

**Go for the conclusion that usable native motion-change evidence is present in these exact pixels; no go for a rescue by these fixed neural comparators. Inconclusive about the unique neural bottleneck or any training remedy.** On the same300 fresh episodes, full-history pixel event-side BA is0.965 [0.940,0.985], versus0.496/0.496/0.489 for stored pre/post CNN25/KDA25/KDA7 and0.477 for stored pre/post memory. The two-transition endpoint pixel observer still reaches0.760 [0.699,0.810]: its advantage over CNN25 phases is+0.264 [0.188,0.335]. Thus full-history access alone does not explain the entire gap. The handcrafted optical-flow observer has important geometric, timing, speed and computation privileges; it is not a capacity-matched neural decoder or a deployable replacement.

**Weak precision versus failed comparison:** full-history pixels estimate pre/post directions within roughly4°/6° and signed change within7.4° on changed patches. Full-memory direction decoding remains coarse (roughly36° pre/40–41° post), while subtracting those separately decoded phases gives54.4° changed-patch error. Direct circular-change probes largely shrink toward zero: changed-patch errors26.9–27.4°, essentially the always-zero27.16° baseline. Their apparently modest all-patch MAE is not successful event recovery. The capacity-matched external pre/post probes do not reliably outperform final-only access (memory side gain+0.027 [−0.054,+0.112]). This fails to support a *pure final-state temporal-access explanation under these assays*, but cannot separate fine encoding, retention, comparator nonlinearity, projection or sample limitations causally.

**A small positive result must not be hidden:** subtracting the prior uncompressed memory direction estimates gives side BA0.566 [0.511,0.619]. It is a weak exploratory signal, not competent localization or evidence of no information: one shuffled KDA7 final side control also reaches0.562 [0.504,0.622], and these intervals are unadjusted across comparisons. Absolute direction error alone never proves comparison failure; correlated errors and robust magnitude ordering can matter. In particular, endpoint pixels localize reasonably despite26.65° changed-patch mean error. No neural erasure, pooling-induced information loss, causal lesion, architecture change or training remedy is established.

For the actual target-report task, image-decoded cue plus full pixel motion yields BA0.877 [0.837,0.915], AUC0.952; endpoint pixels yield BA0.730 [0.680,0.781], AUC0.774. These scores use predicted/image-derived inputs only, with cue decoded300/300. The native selected2297 head remains all-positive: BA0.500, AUC0.475 [0.407,0.542]. This is not a cue-validity, inhibition, microstimulation or reaction-time experiment.

## Same-trial physical comparison
Native originals: 300 independent groups; 258 events, 42 catches. Each has a separate cue-swapped extraction; the table counts groups once.

| Observer/access | Event BA [95% CI] | Event AUC | Event-side BA [95% CI] | Side AUC |
|---|---:|---:|---:|---:|
| pixel/full | 0.880 [0.821,0.932] | 0.944 | 0.965 [0.940,0.985] | 0.992 |
| pixel/endpoint | 0.635 [0.557,0.710] | 0.684 | 0.760 [0.699,0.810] | 0.774 |
| cnn25/phases | 0.500 [0.500,0.500] | 0.507 | 0.496 [0.433,0.549] | 0.497 |
| cnn25/final | 0.487 [0.447,0.531] | 0.496 | 0.504 [0.442,0.558] | 0.489 |
| kda25/phases | 0.500 [0.500,0.500] | 0.482 | 0.496 [0.434,0.566] | 0.508 |
| kda25/final | 0.541 [0.492,0.604] | 0.455 | 0.504 [0.436,0.568] | 0.481 |
| kda7/phases | 0.500 [0.500,0.500] | 0.491 | 0.489 [0.427,0.547] | 0.472 |
| kda7/final | 0.464 [0.418,0.516] | 0.507 | 0.546 [0.484,0.610] | 0.529 |
| memory/phases | 0.500 [0.500,0.500] | 0.450 | 0.477 [0.413,0.542] | 0.455 |
| memory/final | 0.500 [0.500,0.500] | 0.444 | 0.449 [0.389,0.507] | 0.448 |

## Direct signed circular change
MAE in degrees against actual per-dot phase-mean change; not latent event magnitude. No-change predictions are a strong imbalanced baseline.
| Access | Left / right MAE | Changed-patch MAE, events only | Catch MAE |
|---|---:|---:|---:|
| pixel/full | 7.24 / 7.49 | 7.41 | 6.36 |
| pixel/endpoint | 25.12 / 24.31 | 26.65 | 20.80 |
| cnn25/phases | 14.59 / 14.56 | 27.00 | 4.37 |
| cnn25/final | 14.86 / 14.43 | 27.29 | 4.65 |
| kda25/phases | 14.97 / 14.36 | 27.41 | 4.98 |
| kda25/final | 14.49 / 14.64 | 27.14 | 4.97 |
| kda7/phases | 14.51 / 14.64 | 27.18 | 4.64 |
| kda7/final | 14.63 / 14.66 | 27.41 | 5.27 |
| memory/phases | 14.32 / 14.74 | 26.86 | 4.58 |
| memory/final | 14.44 / 14.83 | 27.34 | 4.59 |
| Always zero | 13.57 / 13.38 | 27.16 | 3.16 |

## Direction precision versus temporal comparison
Frozen prior full-dimensional direction fits, never reselected using this test.
| Site | Pre direction MAE L/R | Post direction MAE L/R | Subtracted directions: changed-patch MAE | Subtracted direction side BA |
|---|---:|---:|---:|---:|
| cnn25 | 56.0/56.1 | 61.7/59.5 | 75.7 | 0.527 [0.470,0.589] |
| kda25 | 63.6/62.7 | 55.5/60.5 | 73.6 | 0.453 [0.394,0.515] |
| kda7 | 48.5/48.3 | 52.5/55.9 | 69.0 | 0.438 [0.372,0.501] |
| memory | 36.2/36.4 | 40.2/41.1 | 54.4 | 0.566 [0.511,0.619] |

## Event/catch counts, controls and audit

Full-history pixels detect233/258 events with6/42 catch false positives; endpoint pixels detect217/258 with24/42 catch false positives. For target reporting, full pixels produce158/171 target hits,16/87 foil false reports and6/42 catch false reports. Thus event-side accuracy is not being substituted for any-event or correct-target performance. Full-history side BA exceeds endpoint side BA by+0.206 [0.153,0.264]; extra temporal evidence helps this privileged observer.

[DETAILS.md](DETAILS.md) contains all event/catch counts, true-versus-shuffled results, matched gain intervals, image-derived flow coverage, scope of each input, and exact failure/recovery history. `independent_audit.json` verifies all48 fitted scalers, ridge equations (maximum relative residual3.05e-11), every new/reused direction prediction (replay error0), and the unchanged frozen selections. The original pixel implementation and this summary agree within3.55e-15 degrees on saved examples. Two contract tests passed. One failed optional-dependency import was recovered before fitting without resetting the budget; no scientific run was duplicated. The final report/journal completion time and hashes are in `final_receipt.json`.

## Protocol and limits
- 425 reused train /100 reused validation groups; 300 newly generated test groups, B12/B20/B28 interleaved. Validation reuse and earlier hypothesis generation are exploratory; new test was generated only after the saved hash freeze. No old test used for fitting/selection.
- 48 compact fits: four sites ×two access modes ×three targets ×true/shuffled supervision. Fixed random projection32 per slot →64 coordinates plus2080 quadratic products =2144 features. Same train-only scaling, coefficient counts, groups and alpha grid1/100/10000. Native input dimensions CNN1152, KDA576, memory3136; only425 independent training groups despite850 cue-variant rows. Side fits use event groups only. This limits power and neural information-loss claims.
- Physical pre/post endpoints B+7/B+15 versus final B+16. All sites read both fixed patch neighborhoods; true cue never selects features. Final uses two fixed permutations of the SAME final array. External pre/post storage is unavailable to the deployed head. KDA emitted fields include recurrent history; CNN endpoint sees stack3, so pixel last-two-transition access is the direct instantaneous-window comparison, not equal information or capacity.
- Pixel observer privileges: exact fixed patch geometry, bilinear dot mass/speed, reference/event timing from sequence length, visible image-decoded cue, full externally stored history or designated endpoint windows. It matches isolated centroids, not true dot identity; replacements/overlap reduce accepted matches. It is diagnostic, not deployable. Full and endpoint event/report thresholds selected only on exact re-rendered validation movies (hash verified).
- Native renderer untouched, captured movement angles before each update. Actual circular targets include per-dot offsets, not reset displacement jumps. Means start90°apart: all-site fits cannot prove independent local coding. Signed change differs slightly from nominal26/28° because phase-averaged dot offsets evolve.
- 500 resamples of independent test groups, paired across methods; event-only side and event/catch scored separately. Intervals conditional on fixed fits, no multiple-comparison correction or checkpoint replication. Shuffled controls and paired gain intervals are in metrics.json. Failed probes do not establish neural erasure. Pooling attenuation is NOT a measurement made here.
- Exact selected2297 checkpoint/source hashes checked against prior receipts and archived runtime. Renderer pixels/labels/metadata/final RNG and hooked/unhooked logits are exact at all three B values. No model training, task/architecture changes, cloud work, or optimizer creation.
- Prediction replay error0; pixel flows replayed from serialized image-derived sums; saved example rasters replay exactly; final-only predictions pass earlier-frame removal and NaN poisoning. Fresh noncue movie hashes are disjoint from ALL prior splits.
- Elapsed through initial report: 603.2s under immutable1200s budget, CPU threads2/inter-op1, one worker.

Artifacts: protocol/budget/identity/test_freeze/selection/metrics/verification JSON; fitted_models, projections, fresh test_features/dot_angles/predictions, pixel_validation/pixel_test, bootstrap_groups, example_frames NPZ; test_metadata; upstream.py/run_upstream.py/test_upstream.py. Original artifacts untouched.
