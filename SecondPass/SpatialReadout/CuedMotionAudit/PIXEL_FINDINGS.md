# CPU pixel-solvability audit: cued duration and Krauzlis motion

## Measured answer

Both tasks contain usable information in the native rendered pixels. A fixed, image-only observer solved **128/128 duration D0 trials (BA 1.000)** and **179/200 held-out Krauzlis B20 trials (BA 0.887903, AUC 0.956956)**. These are evidence-existence results, not proof that the fresh ConvGRU has learned the required operations or that either task is noiseless on every possible trial.

No orientation trials, model construction/inference, optimization, checkpoint loads, cloud/pod actions, or network calls were performed. Numerical computation used CPU, one numerical thread, PyTorch 2.8.0 in `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python`. The main measured audit completed in 5.47 seconds after imports; this is not the total implementation wall time.

## Design, privilege, and leakage boundary

- Native `SpatialBatteryStream` renders independent base episodes using audit seeds **92026092901** (duration), **92026092902** (Krauzlis calibration), and **92026092903** (Krauzlis test). Calibration/test have different seeds and disjoint raster hashes. Both use the native procedural `test` split; their separation is the new seed namespace, not the string-valued split.
- **Diagnostic privileges:** known fixed patch geometry, task identity, known reference/motion windows, known Krauzlis event boundary and native dot mass/speed, and external access to the entire input history. No learned cue/timing acquisition or finite recurrent-memory bottleneck is imposed.
- The observer functions accept **pixels only**, plus the fixed Krauzlis baseline length. They never receive labels, true target, directions, event type, true angle, or per-trial metadata. Metadata enters only subsequent scoring and independent label checks.
- Cue location is decoded from the **actual first-frame ring pixels**. Duration uses four centers (27,27), (73,27), (27,73), (73,73), radius 14; Krauzlis uses (20,50), (80,50), radius 9.925. Ring-template mean intensity determines the target.
- Native laws are unchanged: duration uses 32 dots per patch, 16 forced replacements plus aperture exits per transition, and .8/1.2/1.6-pixel steps. Krauzlis uses 16 bilinear dots per patch, .375-pixel steps, Gaussian 16-degree direction offsets, lifetime/boundary resets, and 26/28-degree events.

### Duration estimator

Mask out every static ring pixel; use the dot-only circular ROI of radius 12.3. For each of all four patches and each of the eight reference-to-motion transitions (frames 1–9), compute shifted spatial overlap at 1- and 2-pixel displacement in each cardinal direction. Select the larger displacement score within each direction, then the winning cardinal direction. Thus **true speed is not supplied**; the integer candidate bank accommodates the rasterized native fractional steps. Count the eight estimated directions at the pixel-decoded target.

True schedules have a unique count winner: `_motion_schedule` requires at least a one-transition margin at length eight, and `duration_oracle` rejects ties. Observer count ties would select the largest sum of within-transition normalized evidence among tied classes, then lowest class ID. No observed count tie occurred.

### Krauzlis estimator

Sparse optical flow is obtained by matching isolated bilinear-dot component centroids, not by accessing latent dot positions or identities. Eight-connected positive components with mass .475–.485 identify single, unclipped dots. Mutual nearest-centroid correspondences are retained only if displacement is .375 ± .015 pixels. This known-speed filter rejects most dot births/deaths/resets; merged/occluded components are discarded. Average accepted displacement vectors separately over the **20 baseline and eight postevent transitions**, estimate both patches' angles, and score absolute angular change at the pixel-decoded target.

The event boundary is explicitly **oracle timing**. Among integer thresholds 1–40 degrees, select maximum calibration BA, resolving equal maxima toward the smallest threshold. The independent 100-trial calibration chose **score > 13 degrees** before drawing/scoring the 200 test episodes. No observer or threshold was changed after test results were seen.

## Duration results

| Measurement | Result |
|---|---:|
| Target cue decoding | 128/128 |
| Cued-patch transition direction | **1,024/1,024** |
| All-patch transition direction | **4,096/4,096** |
| Count-winner report | **128/128**, BA **1.000** |
| Per-class recall | 32/32 for each of right/up/left/down |
| Estimated winner ties | 0/128 |
| Independent label recomputation | 128/128 |
| Same-seed D0/D24 evidence and labels identical | **128/128 paired trials** |

The paired check compares all ten cue/reference/motion frames and the report frame exactly; D24 merely adds 24 ignore frames. It does **not** claim the ordinary suite's independently seeded D0/D24 cells are paired, and it does not measure a learner retaining the answer through those blanks. Re-running this externally stored pixel observer on duplicate D24 evidence would add no independent information.

| Native speed | Trials correct | Cued transitions correct | Winner BA |
|---|---:|---:|---:|
| .8 px | 34/34 | 272/272 | 1.000 |
| 1.2 px | 35/35 | 280/280 | 1.000 |
| 1.6 px | 59/59 | 472/472 | 1.000 |

| True count margin | Trials correct | Cued transitions correct | Class coverage |
|---|---:|---:|---:|
| 1 | 75/75 | 600/600 | 4 classes |
| 2 | 39/39 | 312/312 | 4 classes |
| 3 | 11/11 | 88/88 | 4 classes |
| 4 | 2/2 | 16/16 | 1 class |
| 5 | 1/1 | 8/8 | 1 class |

BA is 1.000 for the four-class margin-1/2/3 strata. The saved generic metric's margin-4/5 `balanced_accuracy=1` is only macro recall over represented classes, **not** a valid four-class BA estimate. No transition-level confidence interval treats correlated transitions as independent trials. The trial-accuracy Wilson 95% interval for 128/128 is **[0.970863, 1.000000]**.

No-refitting controls: reporting only the last estimated target direction gives **26/128, BA .203125**; always counting patch zero gives **53/128, BA .414063**. The latter is not a chance-null (patch zero is sometimes the actual target). Correct spatial selection and integration across the history matter.

## Krauzlis B20 results

Calibration: **88/100**, BA **.880457**, AUC **.943288** at the frozen 13-degree threshold. Test:

| Measurement | Result |
|---|---:|
| Cue decoding | 200/200 test; 100/100 calibration |
| Correct report | **179/200 (89.5%)** |
| Balanced accuracy | **.887903** |
| AUC | **.956956** |
| Confusion, rows true [0,1], columns prediction [0,1] | **[[72,14],[7,107]]** |
| Target-event hit rate | **107/114 (.938596)** |
| Foil-only false-report rate | **10/58 (.172414)** |
| Catch false-positive rate | **4/28 (.142857)** |
| Target side 0 | 88/100; BA .874745 |
| Target side 1 | 91/100; BA .901061 |
| Baseline angle MAE, both patches | **4.002450 degrees** |
| Postevent angle MAE, both patches | **6.708640 degrees** |
| Independent label recomputation | 300/300 calibration + test |

Trial-accuracy Wilson 95% interval: **[.844820, .930293]**. These are trial-draw uncertainties, not checkpoint or training-seed uncertainties. The 114/58/28 event distribution and 100/100 target-side distribution preserve the native exact per-100/per-200 schedules.

Flow coverage is deliberately sparse: mean **1.783393 accepted matches per patch-transition**, **1,828/11,200** patch-transitions have no accepted match. Evidence is aggregated across each full phase. This limited observer discards much of the raster and is not an optimal/Bayes observer; its errors do not imply irreducible task ambiguity. Other baseline lengths B12/B28 were not tested.

No-refitting controls using the same calibrated threshold: selecting the **wrong cue location** yields **43/200**, BA **.227152**; declaring a change in **either patch** yields **132/200**, BA **.614647**. These support target-specific pixel selection, rather than generic any-change detection. They are diagnostic algorithm controls, not measured neural attention, inhibition, or microstimulation effects.

## Pixel-scale signal and why a fresh ConvGRU may struggle

Measured temporal raster changes (one grayscale channel; RGB channels are redundant):

| Measure | Duration D0, 128 trials | Krauzlis B20, 200 test trials |
|---|---:|---:|
| Mean changed pixels per transition, of 10,000 | 223.247070 | 139.360714 |
| Full-frame mean absolute intensity difference | .009376377 | .001143951 |
| Temporal difference RMS, 96×96 crop | .065352319 | .013494821 |
| After fixed 4×4 average pooling | .013607912 | .003338698 |
| After fixed 16×16 average pooling | .002563214 | .000692406 |

The averaging control leaves 3.92% / 5.13% of the initial temporal-difference RMS at factor 16. **It is not the trained CNN and does not establish its information loss.** These temporal changes include ordinary movement and resets; they are not an isolated causal event-minus-no-event raster contrast. The successful held-out target-change observer supplies the relevant event-discrimination evidence.

Source inspection establishes four strided convolution stages (stride two each), spatial maps 50→25→13→7, and three recurrent encoder scales 25/13/7, followed by a 7×7 ConvGRU with final-state-only report. The raw input stack is three frames. Consequences are hypotheses/requirements, not measured model failures:

1. **Fine motion must be encoded before spatial reduction.** Krauzlis .375-pixel shifts are only .0234375 of the deepest map's 16-pixel lattice spacing; duration shifts are .05/.075/.10 of that spacing. Learned channels can preserve subcell information, so this is not a proof that downsampling destroys it. But a decoder that merely tracks coarse spatial peak displacement cannot recover these signals reliably.
2. **Duration D0 already requires history integration.** At the final report, the raw three-frame stack contains frames 8,9,10: only the last motion transition plus report, not all eight transitions. The network must form direction-specific counts, select the ring-cued patch, and carry the decision; it is not simply an easy two-frame motion task plus optional blanks. The last-direction control fails despite perfect transition decoding.
3. **Krauzlis needs early cue retention and a relational comparison.** Its cue is present only in frames 0–1, followed by five fixation frames; all dots arrive later. At report, stack-3 contains only late postevent evidence and fixation, not the baseline. The network must retain selected location, estimate baseline motion, detect a small angular change across noisy dot identities, and reject foil changes.
4. **No hidden event signal is available to the learner.** This audit explicitly receives fixed event-boundary timing and stores full histories. The recurrent learner must acquire the useful computation from task supervision. Pixel solvability does not establish learner acquisition, retention, attention allocation, or the specific bottleneck responsible for its current score.

No architecture changes or retraining are proposed or performed here. Current model/checkpoint evidence is a separate analysis; this audit rules out a gross missing-signal/label-mismatch explanation for the audited draws, not all renderer defects or optimization explanations.

## Verification and reproducibility

- Exact adapter/native equality: pixels, labels, and all native metadata for **4 trials each** of duration D0, duration D24, Krauzlis B20.
- Labels independently computed without calling `duration_oracle`: cardinal counts at the true target; circular pre/post mean-angle difference at the true Krauzlis target. All **428** scored labels agree.
- All **428** saved predictions replay from saved correlation scores or optical-flow vectors; **6** initial native rasters/predictions regenerated exactly from saved seeds/hashes.
- All within-split raster hashes unique; calibration and test hash sets disjoint. Renderer, imported cue/schedule helpers, vector definitions, adapter/catalog, inspected architecture sources and SOP hashes remained unchanged.
- Synthetic translation and subpixel-flow unit tests were each observed to fail for missing implementation, then passed. Synthetic fixtures are not counted in reported native results.

Artifacts in this directory:

- `pixel_observer.py`: complete CPU audit and image-only estimators; refuses to append a second run into an existing results namespace.
- `pixel_test.py`: cardinal translation and fractional-flow unit tests.
- `pixel_verify.py`: durable-count, score/prediction, source-hash, native-replay checks and fixed controls.
- `pixel_results_duration.jsonl`, `pixel_results_k_calibration.jsonl`, `pixel_results_k_test.jsonl`: all trial IDs, labels, predictions, cue scores, motion scores/vectors, metadata for scoring, raster hashes and raster statistics.
- `pixel_results_config.json`: seeds, thresholds/procedures, environment, privileges, source hashes, observer hash.
- `pixel_results_threshold.json`: complete calibration-only threshold curve.
- `pixel_results_summary.json`, `pixel_results_verification.json`, `pixel_results_replay.json`: aggregate results and verification receipts.

From repository root, with the existing results retained, rerun verification only:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python SecondPass/SpatialReadout/CuedMotionAudit/pixel_test.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python SecondPass/SpatialReadout/CuedMotionAudit/pixel_verify.py
```

For a fresh full reproduction use a clean copy of the code/source tree without these results, then invoke `pixel_observer.py` with the same interpreter/thread environment. Do not delete or overwrite the recorded result set.

### Source references (current source hashes saved in config)

- `WorkingMemory/SpatialTaskBattery/stimuli.py:34–43,97–111`: local ring, duration dots, rendering and labels.
- Same file `:121–142`: Krauzlis geometry, bilinear rasterization, timeline, direction offsets, resets, event labels.
- `WorkingMemory/stimuli.py:58–89,98–111`: visual cue helper, unique-winner oracle, count-margin schedule.
- `PreAttentiveVision/neuroscience_stimuli.py:21–22`: image-coordinate cardinal vectors.
- `SecondPass/TaskSuite/suite.py:69–99` and `catalog.json` duration/Krauzlis entries: seed mapping and unmodified native adapter.
- `WorkingMemory/PlainBaseline/accum.py:40–68`; `SecondPass/SpatialReadout/model.py:29–51`: frame stack, strided encoder, spatial recurrence, final report. Read only; no model inference.
- `ANALYSIS_SOP.md:20,37–49,98–104,156–165`: target-report semantics, provenance, stack-aware retention, denominators and interpretation limits. Cue-validity curves, neural attention allocation, inhibition, microstimulation, and reaction times remain **unmeasured**.
