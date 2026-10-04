# Completed local joint-suite KDA training — 2026-09-22

## Outcome

Completed the pinned 715 updates / 22,880 episodes, 1,760 per task, with unchanged tasks and all learned parameters trainable. All 35 cells were tested for both checkpoints. No run failure or hard-cap termination was recorded.

Finished 2026-09-22T10:04:27.871654+00:00; elapsed 3.275 hours including setup/profiles/handover/evaluation; optimizer work 1.968 hours. Original cap: four hours. No further run has been launched.

The terminal model acquired several sensory discriminations but did not establish motion or spatial/sequence-task competence. This is not a converged full-suite learner or an architectural-capacity result.

## Checkpoint distinction

**Validation-selected: step 117. Terminal: step 715.** The prespecified lexicographic rule prioritizes the worst chance-normalized task BA, then mean task AUC. Step 117 retained selection because its worst validation task was less below chance, even though terminal mean validation AUC improved. A floor dominated by a weak task can favor an almost-untrained model over real gains elsewhere; sampling noise is a plausible contributor, not a demonstrated sole cause. Do not retrospectively change selection after reading tests. Both artifacts are preserved.

| Validation step | Minimum chance-normalized task BA | Equal-task mean AUC |
|---:|---:|---:|
| 39 | -0.03125 | 0.49603 |
| 78 | -0.15625 | 0.50464 |
| 117 | -0.03125 | 0.52033 |
| 715 | -0.09375 | 0.62963 |

## Final task results

Balanced accuracy, averaged equally across eligible conditions within each task. Chance is 25% for the two four-direction tasks and 50% for all others. Empty scene sets are excluded here and reported separately. Terminal results are a planned endpoint, not a replacement for the validation-selected model.

| Task | Selected 117 BA | Terminal 715 BA | Terminal AUC |
|---|---:|---:|---:|
| Motion direction | 25.00% | 26.56% | 0.4657 |
| Signed orientation | 50.00% | 56.25% | 0.5613 |
| Contrast | 50.00% | 100.00% | 1.0000 |
| Spatial frequency | 50.00% | 63.28% | 0.7129 |
| Chromatic increment | 68.75% | 99.22% | 1.0000 |
| Contour grouping | 50.00% | 52.34% | 0.5037 |
| Natural-image spectral detail | 50.00% | 92.97% | 0.9839 |
| Ring-cued orientation | 49.22% | 46.09% | 0.4351 |
| Spatially cued orientation | 50.20% | 50.59% | 0.4841 |
| Cued motion duration | 25.00% | 24.41% | 0.5052 |
| Krauzlis target/foil change | 50.00% | 50.00% | 0.5263 |
| Spatial orientation binding | 49.41% | 51.76% | 0.5198 |
| Scene-set recognition (nonempty) | 50.00% | 50.00% | 0.5115 |

## Every primary condition

128 test episodes per ordinary condition; 200 per Krauzlis baseline. Selected and terminal use matching final-only draws (namespace 94292763); official photo test source identities remain separate from training. Conditions at different delays were generated independently, not paired across delays.

| Task / condition | n per checkpoint | Selected BA | Terminal BA | Selected AUC | Terminal AUC |
|---|---:|---:|---:|---:|---:|
| motion_direction / mixed | 128 | 25.00% | 26.56% | 0.5378 | 0.4657 |
| orientation / mixed | 128 | 50.00% | 56.25% | 0.4751 | 0.5613 |
| contrast / mixed | 128 | 50.00% | 100.00% | 0.5654 | 1.0000 |
| spatial_frequency / mixed | 128 | 50.00% | 63.28% | 0.5042 | 0.7129 |
| chromatic_increment / mixed | 128 | 68.75% | 99.22% | 0.7295 | 1.0000 |
| contour / mixed | 128 | 50.00% | 52.34% | 0.4524 | 0.5037 |
| natural_spectrum / mixed | 128 | 50.00% | 92.97% | 0.5427 | 0.9839 |
| orientation_ring / D0 | 128 | 49.22% | 46.09% | 0.4470 | 0.4351 |
| orientation_cued / D0 | 128 | 50.78% | 52.34% | 0.5298 | 0.4458 |
| orientation_cued / D4 | 128 | 50.00% | 49.22% | 0.4773 | 0.5078 |
| orientation_cued / D12 | 128 | 50.00% | 50.00% | 0.5078 | 0.5029 |
| orientation_cued / D24 | 128 | 50.00% | 50.78% | 0.4456 | 0.4800 |
| motion_duration_cued / D0 | 128 | 25.00% | 22.66% | 0.5163 | 0.5122 |
| motion_duration_cued / D4 | 128 | 25.00% | 25.00% | 0.5033 | 0.4968 |
| motion_duration_cued / D12 | 128 | 25.00% | 25.00% | 0.5375 | 0.4969 |
| motion_duration_cued / D24 | 128 | 25.00% | 25.00% | 0.5125 | 0.5150 |
| krauzlis_cued_motion / B12 | 200 | 50.00% | 50.00% | 0.4997 | 0.5189 |
| krauzlis_cued_motion / B20 | 200 | 50.00% | 50.00% | 0.5310 | 0.5148 |
| krauzlis_cued_motion / B28 | 200 | 50.00% | 50.00% | 0.4668 | 0.5453 |
| spatial_binding / D0 | 128 | 47.66% | 52.34% | 0.5100 | 0.5254 |
| spatial_binding / D4 | 128 | 50.00% | 53.12% | 0.5400 | 0.5583 |
| spatial_binding / D12 | 128 | 50.00% | 50.78% | 0.4727 | 0.5076 |
| spatial_binding / D24 | 128 | 50.00% | 50.78% | 0.5007 | 0.4880 |
| image_recognition / N0_H3 | 128 | — | — | — | — |
| image_recognition / N0_H4 | 128 | — | — | — | — |
| image_recognition / N0_H5 | 128 | — | — | — | — |
| image_recognition / N4_H3 | 128 | 50.00% | 50.00% | 0.5649 | 0.5703 |
| image_recognition / N4_H4 | 128 | 50.00% | 50.00% | 0.5061 | 0.5090 |
| image_recognition / N4_H5 | 128 | 50.00% | 50.00% | 0.4602 | 0.4670 |
| image_recognition / N12_H3 | 128 | 50.00% | 50.00% | 0.5166 | 0.5256 |
| image_recognition / N12_H4 | 128 | 50.00% | 50.00% | 0.4348 | 0.4895 |
| image_recognition / N12_H5 | 128 | 50.00% | 50.00% | 0.5261 | 0.4426 |
| image_recognition / N24_H3 | 128 | 50.00% | 50.00% | 0.4585 | 0.5422 |
| image_recognition / N24_H4 | 128 | 50.00% | 50.00% | 0.5071 | 0.4150 |
| image_recognition / N24_H5 | 128 | 50.00% | 50.00% | 0.5557 | 0.6418 |

## Empty recognition and Krauzlis event controls

| Checkpoint | Empty-set condition | Specificity | False-positive rate | n |
|---|---|---:|---:|---:|
| Selected 117 | N0_H3 | 100.00% | 0.00% | 128 |
| Selected 117 | N0_H4 | 100.00% | 0.00% | 128 |
| Selected 117 | N0_H5 | 100.00% | 0.00% | 128 |
| Terminal 715 | N0_H3 | 100.00% | 0.00% | 128 |
| Terminal 715 | N0_H4 | 100.00% | 0.00% | 128 |
| Terminal 715 | N0_H5 | 100.00% | 0.00% | 128 |

Empty-set specificity is not evidence of nonempty recognition: both checkpoints have 50% nonempty-set BA.

| Checkpoint | Baseline | Target hit rate (n) | Foil false-alarm rate (n) | Catch false-positive rate (n) |
|---|---|---:|---:|---:|
| Selected 117 | B12 | 100.00% (114) | 100.00% (58) | 100.00% (28) |
| Selected 117 | B20 | 100.00% (114) | 100.00% (58) | 100.00% (28) |
| Selected 117 | B28 | 100.00% (114) | 100.00% (58) | 100.00% (28) |
| Terminal 715 | B12 | 100.00% (114) | 100.00% (58) | 100.00% (28) |
| Terminal 715 | B20 | 100.00% (114) | 100.00% (58) | 100.00% (28) |
| Terminal 715 | B28 | 100.00% (114) | 100.00% (58) | 100.00% (28) |

Both checkpoints respond positive to every sampled Krauzlis event. The 100% target hit rate is therefore not successful target detection: foil and catch false positives are also 100%. Side counts are retained in the JSON artifacts.

## Next decision — no automatic additional compute

Preserve terminal 715 as the latest optimizer-progress checkpoint and selected 117 as the official selected artifact. Any further acquisition should preserve terminal state rather than discard learned sensory progress. Before a newly authorized longer run, specify a learning-oriented validation/selection rule prospectively, retaining explicit per-task results and avoiding a noisy worst-task floor as the sole priority. Do not change stimuli, introduce a curriculum or alter architecture based solely on these early scores.

The current result establishes selective sensory learning within limited exposure, not full-suite acquisition. One seed, 55 task-specific updates each, limited condition exposure, and no source-independent confidence intervals limit negative conclusions. Cueing benefits, memory mechanisms and causal perturbation signatures were not tested.

## Evidence

- `runs/fresh_kda_joint_01_continuation_v2/report.json`
- `runs/fresh_kda_joint_01_continuation_v2/test_selected.json`
- `runs/fresh_kda_joint_01_continuation_v2/test_terminal.json`
- `runs/fresh_kda_joint_01_continuation_v2/progress.jsonl`
- `AMENDMENT_V2.md`

Parent independently SHA-256 verified and CPU-loaded terminal 715: `f8f308e64c97c99a3ce7c7b1b68ed5386bbc91c35df2197aef1a02cbdbf00048`; 66 Adam parameter states present. No new accelerator work was used for this summary.
