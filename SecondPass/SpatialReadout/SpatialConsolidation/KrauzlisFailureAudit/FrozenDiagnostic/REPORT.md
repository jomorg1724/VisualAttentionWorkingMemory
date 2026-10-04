# Frozen diagnostic: native-angle Krauzlis selected2297

**Measured, exploratory, one frozen trained observer.** No deployed optimizer updates or cloud access. Native task: report a change only in the ring-cued patch; foil and catch reports are errors. This is not a cue-validity experiment.

## Decision and measured diagnosis
**No go for a claim that a simple replacement readout rescues this checkpoint. Inconclusive about the unique failing mechanism.** The early cue is not wholly erased: full ConvGRU memory decodes it perfectly after the five-frame gap and above chance at report. Coarse physical direction is also accessible, with mean angular errors larger than the26/28° changes; correlated errors could still cancel in a comparator, so this is not proof that event information is absent. Physical changed-patch decoding is weak, and neither final-only nor capacity-matched stored-phase label probes establish correct cue-conditioned report accessibility. This favors investigating fine-motion comparison/binding rather than asserting complete cue loss; it does not identify a causal lesion or justify altering task teaching.

## Behavioral result
Fresh native-original held-out episodes: n=175, BA=0.500, AUC=0.622. Paired variants are native-valid cue counterfactuals with altered event frequencies, not independent episodes.
```json
{
  "n_groups": 175,
  "changed_label_groups": 149,
  "mean_absolute_margin_shift": 1.1895384659510455e-06,
  "mean_task_aligned_margin_shift_changed": 5.940462918889602e-08,
  "argmax_flips": 0,
  "both_members_correct": 0,
  "both_members_correct_changed": 0
}
```


Native head still reports positive on every original target/foil/catch trial ({'target': 97, 'foil': 52, 'catch': 26}); natural-frequency AUC95% group-bootstrap CI [0.538,0.709]. The higher AUC on this fresh small sample than the historical selected test does not establish task acquisition: paired-cue AUC is 0.527, no paired action flips, and task-aligned margin shift is 5.94e-08 [-2.02e-07,3.3e-07]. These tiny float32 differences do not establish meaningful cue use. Zero flips has a Wilson95% upper bound 0.021, not proof of exactly zero future effect.

## Cue accessibility through time
| Site | Cue | Gap | Baseline | Post | Final |
|---|---:|---:|---:|---:|---:|
| cnn25 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| kda25 | 1.000 | 1.000 | 0.643 | 0.543 | 0.557 |
| kda13 | 1.000 | 1.000 | 0.714 | 0.609 | 0.577 |
| kda7 | 1.000 | 1.000 | 0.769 | 0.617 | 0.609 |
| memory | 1.000 | 1.000 | 0.831 | 0.709 | 0.663 |

Values are held-out balanced accuracy; detailed group-bootstrap intervals and all validation choices in metrics.json. Cue at frame1 is still visible; gap at6 has no cue in stack3. Later values test retained accessibility, not attention allocation.


Full-memory final cue BA 0.663,95% grouped CI [0.626,0.694]. By baseline duration: B12=0.763 (59groups); B20=0.672 (58groups); B28=0.552 (58groups). These are different independent trials; B manipulates motion duration, not a pure blank-delay experiment.

## Physical change and correct-label accessibility
| Access / probe | Changed patch BA (left/right/catch) | Correct label BA | Label AUC |
|---|---:|---:|---:|
| memory__final | 0.361 | 0.513 | 0.490 |
| terminal__final | 0.361 | 0.504 | 0.488 |
| access_final_linear | 0.348 | 0.494 | 0.497 |
| access_phases_linear | 0.401 | 0.488 | 0.494 |
| access_final_quadratic | 0.361 | 0.475 | 0.485 |
| access_phases_quadratic | 0.413 | 0.501 | 0.519 |


Final-memory label BA95% CI [0.486,0.544]. Phase-quadratic changed-patch BA95% CI [0.344,0.497], three-class chance reference1/3; this modest exploratory result is not correct target binding.

## Actual per-dot circular direction
Mean absolute circular error in degrees; each pair is left / right. Uniform-angle reference is90°, and native change magnitude26/28°.
| Site | Baseline direction at baseline | Post direction at post | Post direction at final |
|---|---:|---:|---:|
| cnn25 | 57.7 / 58.0 | 65.9 / 62.4 | 63.1 / 59.6 |
| kda25 | 63.6 / 62.0 | 57.9 / 56.3 | 61.2 / 58.7 |
| kda13 | 53.2 / 53.0 | 54.3 / 49.7 | 55.1 / 54.7 |
| kda7 | 49.0 / 48.6 | 50.6 / 52.2 | 49.8 / 54.9 |
| memory | 41.0 / 41.2 | 41.3 / 43.1 | 43.6 / 46.3 |

The native baseline directions are90° apart and probe inputs include both patches: successful per-patch decoding does not establish independent local encoding of both patches. Early feature ROIs are deliberately fixed, not selected by the true cue.

Targets are circular means over the actual16 per-dot movement angles over each complete baseline/post phase, captured before native .375px updates. They are not latent categorical proxies or patch means without dot offsets. Replacement jumps are not treated as motion vectors; offsets of reset dots enter their subsequent native movements. Dot-angle schedules are saved.

## Temporal-access controls
| Matched capacity | Final label BA | Phase label BA | Phase−final BA,95% grouped CI | Shuffled final / phase BA |
|---|---:|---:|---|---|
| linear | 0.494 | 0.488 | -0.006 [-0.041,+0.029] | 0.514 / 0.511 |
| quadratic | 0.475 | 0.501 | +0.026 [-0.031,+0.092] | 0.516 / 0.529 |

## Protocol
Grouped counts pinned after profiling: {'train': 425, 'val': 100, 'test': 175}; two cue variants per group, all phases in same split. Native cycle proportions57/29/14 are retained for original trials; incomplete cycles may deviate. B12/B20/B28 are interleaved, not separate resampled copies. Train/val/test seeds: {'train': 71003001, 'val': 71004001, 'test': 71005001}.
Profile paired extraction seconds: [0.4472348690032959, 0.5752367973327637, 0.7320570945739746]. One nonrenewable1800s cap covers profile through final report; completed in approximately435.4s (see receipt for final elapsed). CPU only, torch threads2/inter-op1, numerical libraries capped2, one process.
Exact production sources matched archived runtime files and recorded hashes. Checkpoint2297 had73504 training episodes; parent training completed4595updates/147040episodes. Terminal4595 was not used. identity.json records full hashes and provenance.
Native forward is untouched: centered stack3 → strided CNN → KDAs at25/13/7 → spatial projection/ConvGRU64×7×7 → terminal49-token spatial transformer → dense head. Observational hooks matched unhooked logits bit-for-bit for B12/B20/B28; instrumented renderer matched native pixels, labels and final RNG state. All weights eval/frozen; file and tensor hashes unchanged.
Early CNN/KDA inputs are fixed3×3 neighborhoods around BOTH patch centers (stride4/8/16); no true cue chooses a patch. Feature dimensions: CNN1152, KDA576, full memory/terminal3136. Receptive fields and GroupNorm mix space; these sites do not prove localized coding. Early restricted sampling versus full memory prevents strong layerwise information-loss conclusions.
Cue/gap/baseline/post/final endpoints are1/6/B+7/B+15/B+16. Final stack contains two post frames plus fixation, not cue/baseline. Mean directions aggregate full phase; endpoint decoding is an imperfect measurement of that aggregate.
Ridge with intercept and train-only per-coordinate scaling; regularization1/100/10000 selected on validation BA (angle: minimum circular error), first candidate wins ties. No refit on validation, no held-out tuning. All models and choices frozen/hash-saved before generating test. Fits/predictions replayed exactly. Final-only representations passed actual removal and NaN poisoning of every earlier representation.
Temporal comparisons: fixed random projections →384 dimensions for linear ridge;96 dimensions plus all upper-triangle quadratic products =4752 for quadratic ridge. Final condition uses3 fixed coordinate permutations of final memory; phase condition uses stored cue/baseline/post memory with identical projection/permutation rule. Same train groups, target supervision, candidate grid and learned coefficient counts. Storage of earlier states is external diagnostic memory, not available to deployed head. Fixed quadratic terms permit interactions but are not a supplied true-cue comparator. Shuffled controls permute entire training episode-pairs; val/test truth remains intact.
Intervals:500 resamples of independent held-out groups, keeping paired variants together. One seed, modest catch counts, multiple exploratory probes, no multiplicity-adjusted claims. Exact margins retained in test_predictions.npz/native_logits.

## Interpretation and limits
Probe success establishes accessibility to that analysis-only probe, not deployed learning, causal use, or biological attention. Failure does not establish erasure: weak supervision, sample size, feature extraction, endpoint choice and ridge capacity can limit decoding. Compare actual angular precision with26/28° events before claiming useful fine-motion encoding. Even successful phase access would not show a new final head suffices.
Classical validity effects, allocation maps, inhibition, microstimulation, causal localization and response times are not measured in this bounded diagnostic. No task teaching or architecture change was made.

## Artifacts and reproducibility
`diagnostic.py`, `finalize_audit.py`, `test_diagnostic.py`; `verification.json` independently replays stored metrics and verifies disjoint noncue raster hashes; identity/allocation/budget/test_freeze/selection/receipt JSON; train/val/test metadata, features and per-dot schedules; fitted_probes/projections/test_predictions NPZ; metrics.json. Group IDs and movie/noncue hashes are included. Large local artifacts remain in this directory.
Run with `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python diagnostic.py --out NEW_AUTHORIZED_EMPTY_DIRECTORY` from this directory. Existing budget refuses renewal. Re-execution requires its own explicit authorization; saved features and fits permit audit without retraining.

Final audit reads only saved artifacts, uses the same absolute budget deadline, and performs no refits or test-based selections. Native CPU inference process completed; a process-manager premature null exit notification was contradicted by OS PID/log inspection and was not used as evidence of completion.
