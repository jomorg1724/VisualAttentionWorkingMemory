# Current joint-suite architecture and saved-metric audit

## Frozen evidence boundary

Snapshot began **2026-09-23T06:48:14.424088+00:00** and finished **2026-09-23T06:48:15.453628+00:00**. `metrics_snapshot.json` is read-only and embeds six completed evaluations, the original file paths/SHA-256 digests, all confusion matrices and strata, catalog, config/lineage receipts, per-task/cell exposure, all parameter tensor shapes, and source/checkpoint verification. This is a saved-artifact audit: **zero model forward calls, zero new evaluations, zero optimizer steps, no accelerator calls**. Only fresh CPU module construction and CPU checkpoint deserialization were used; all analysis threads were one.

The captured live status (not a completed endpoint) was **step 2647, 84,704 episodes, 800,512 frames**, training; best completed validation **2314**. Target **2860** and fourth look **2860** remained pending, as did both v3 final tests. Deadline remained **2026-09-23 09:08:50.062308 UTC**. Do not imply that the current training step has the measured performance of step 2314. This immutable snapshot intentionally does not incorporate later results.

## Exact architecture, not the historical frozen-PAV wrapper

Authoritative implementation: `WorkingMemory/PlainBaseline/accum.py:40–75`, `PreAttentiveVision/TemporalIntegration/accumulators.py:19–66`; actual instantiation `SecondPass/JointTraining/worker.py:33–40` and `continuation_v3.py:199–205`. The imported `SpatialKDA` is reused, **not** `StreamingPAVClassifier` and not its frozen encoder or 50/25/13 maps.

Input float32 `[B,T,3,100,100]` is centered by subtracting 0.5 (not divided by 0.5). Causal stacking concatenates `[x(t−2),x(t−1),x(t)]` into 9 channels. Left padding is zero *after centering*, equivalent to neutral gray, not an extra observation. At every presented frame:

| Stage | Operation | Output excluding batch | Parameters |
|---|---|---|---:|
| Block 0 | Conv 9→32, 5×5, stride 2, pad 2; GN(8); ReLU | 32×50×50 | 7,296 |
| Block 1 | Conv 32→64, 3×3, stride 2, pad 1; GN(8); ReLU | 64×25×25 | 18,624 |
| Projection / KDA 0 | 1×1 64→32; KDA→32; concatenate with block | 96×25×25 | 2,080 + 24,754 |
| Block 2 | Conv 96→96, 3×3, stride 2, pad 1; GN(8); ReLU | 96×13×13 | 83,232 |
| Projection / KDA 1 | 1×1 96→32; KDA→32; concatenate with block | 128×13×13 | 3,104 + 24,754 |
| Block 3 | Conv 128→128, 3×3, stride 2, pad 1; GN(8); ReLU | 128×7×7 | 147,840 |
| Projection / KDA 2 | 1×1 128→32; KDA→32; concatenate with block | 160×7×7 | 4,128 + 24,754 |
| Feature bottleneck | Flatten 7,840→256 affine, ReLU; identity norm | 256/frame | 2,007,296 |
| Global GRU | One unidirectional layer, input=hidden=256, two biases | 256 final hidden | 394,752 |
| Heads | Externally selected affine 256→classes, 13 heads | 2 or 4 logits | 7,710 |

**Total: 2,750,324 learned/trainable parameters.** All are trainable; no LayerNorm, dropout, fixed encoder, positional embedding, explicit spatial-softmax priority map, or GRU-to-KDA feedback is installed. GRU input weights and recurrent weights are each `[768,256]`; its two biases are `[768]`. The code first collects the sequence of per-frame features, then runs the GRU, so no top-down GRU state can enter the spatial updates. Classification occurs only at the final hidden state, not as wait/declare actions.

The block total is 256,992; projections 9,312; KDA modules 74,262; feature affine 2,007,296; GRU 394,752; heads 7,710. Each KDA has 23,616 input-convolution weights plus 82 biases (32→82, 3×3), and 1,024 output weights plus 32 biases (32→32, 1×1), total 24,754. Each four-class head (`motion_direction`, `motion_duration_cued`) has 1,028 parameters; each of the other eleven heads has 514. This is one shared learner with external task-head selection, not learned task inference. The flatten/256 bottleneck is a global, position-specific readout of the deepest map; it does not expose every stored KDA value directly to the classifier.

## KDA dynamics, locality and retained state

For each scale, site and head, let `S` be 8×16, `Dα=diag(α)` and column vectors `q,k∈R⁸`, `v∈R¹⁶`. Packed channels per head are **q8, k8, v16, α8, β1** (41); there are two heads. q and k use L2 normalization with epsilon 1e−6, hence norms at most one, not a spatial softmax. Values are unconstrained signed linear features. The exact update/readout is

```text
Sbar_t = Dα_t S_(t−1)
e_t    = v_t − Sbar_t^T k_t
S_t    = Sbar_t + β_t k_t e_t^T
       = (I − β_t k_t k_t^T) Dα_t S_(t−1) + β_t k_t v_t^T
r_t    = S_t^T q_t
O_t    = Conv1x1(concatenate the two 16-component r_t vectors)
```

Readout is from the **updated** state, after both decay and correction. At initialization only gate rows of the input-convolution weights are zero: α biases are log(9), β biases zero, giving **α=0.9 and β=0.5 initially**. These gate weights/biases remain learned and become input-dependent; 0.9/0.5 are not fixed trained retentions. No trained gate activity was measured in this audit.

For fixed realized q/k/v/gates, `A_t=(I−β_t k_t k_t^T)Dα_t`. Since `0<β<1` and `||k||≤1`, the first factor has eigenvalues 1 off the key direction and `1−β||k||²` along it. Thus `||A_t||₂ ≤ max_i α_ti ≤ 1`; in exact arithmetic finite sigmoid logits give `<1`. Float32 saturation can reach an endpoint. With unit k at initialization the key-direction retention is 0.45 and orthogonal retention 0.9. The influence of an earlier additive state perturbation is bounded by a product of these realized maximum gates **when subsequent forcing/gates are held fixed**. It is not a fixed exponential time constant, a learned-state measurement, or a contraction proof for the full nonlinear stacked model. Input-dependent gates, lower-scale state feedback into deeper current features, and normalization prevent that broader inference. Even the frozen-forcing bound includes continuing writes: `||S_t||F ≤ αmax ||S_(t−1)||F + β||v_t||₂`.

Unrolling under fixed forcing gives signed temporal associative coefficients `β_τ q_t^T A_t … A_(τ+1) k_τ` on earlier `v_τ`. These are not nonnegative, sum-to-one spatial attention probabilities. Local persistent matrices are not convex mixtures over the four task locations.

**Locality qualification:** each KDA directly carries only its own site's previous 8×16 state; there is no state convolution across same-scale sites. Its input 3×3 convolution does mix neighboring **current** projections. Convolutions, cross-scale concatenation, overlapping receptive fields and especially **GroupNorm's group statistics over spatial positions** mean the complete network is not strictly spatially isolated. GN supplies global statistical coupling, not content-addressed competitive spatial softmax. The final affine/GRU mixes the entire deepest map globally but sends no feedback to the KDA stack.

| State map | Exact state shape | Scalars/example | fp32 bytes/example |
|---|---|---:|---:|
| 25×25 | B×25×25×2×8×16 | 160,000 | 640,000 |
| 13×13 | B×13×13×2×8×16 | 43,264 | 173,056 |
| 7×7 | B×7×7×2×8×16 | 12,544 | 50,176 |

KDA total **215,808 scalars / 863,232 bytes per example**; add 256 GRU hidden scalars for **216,064 recurrent scalars**. These are activations, not model parameters, and exclude full-BPTT saved intermediates, feature sequence and optimizer storage. Two preceding raw RGB frames correspond to another **60,000 scalars/example** of causal sensory history (implemented by stacking, not an extra learned recurrent module). KDA states reset to zeros and GRU state is default zero for each sequence.

The first two nominal blank updates can still carry sample frames in the raw stack; only the third has no preceding sample raster in its three-frame stack. Some blanks still contain task/phase glyphs, so “no sample evidence” does not imply an entirely uniform input. Ring orientation has **D0 only, four frames** (cue, sample, sample, probe), and its final stack still contains both sample frames. Ring performance cannot establish blank-delay retention. Binding D0 similarly retains its last sample through query+probe. The recognition renderer includes three blanks before repeated probes; repeated probe frames are the same raster, not independent evidence or reaction-time decisions.

## Lineage, optimization and exposure

One fresh production seed **94182763**, scheduler seed **94182991**; disposable profiling weights/optimizer were discarded. v1→v2 preserved step 117 state under the original four-hour allowance. v2 ended at **715**, but its original minimum-task-first validation rule selected **117**. v3 resumed **terminal715**, not selected117, under the new explicit eight-hour allowance. Source terminal SHA-256 is `f8f308e64c97c99a3ce7c7b1b68ed5386bbc91c35df2197aef1a02cbdbf00048`. Saved migration receipts verify exact model, Adam, scheduler/condition queues, native streams and RNG preservation. This audit separately checked checkpoint digests and tensor shapes on CPU, not a new migration replay.

Recipe unchanged: fp32/full BPTT; Adam lr=1e−4, β=(.9,.999), eps=1e−8, weight decay 0, one group, **no clipping**; batch32 via microbatch4, one task/cell per update, mean cross-entropy. Shuffled equal 13-task cycles; independently balanced condition queues. Thus equal task updates do not mean equal cell exposure or frame exposure. All13 heads have saved Adam states (66 parameter-tensor states total); inactive task heads do not update on a different head's loss.

| Cumulative step | Episodes | Presented frames | Optimizer seconds |
|---:|---:|---:|---:|
| 117 | 3,744 | 35,712 | 1181.267 |
| 715 | 22,880 | 215,712 | 7083.761 |
| 1248 | 39,936 | 377,856 | 12349.922 |
| 1781 | 56,992 | 540,480 | 17619.592 |
| 2314 | 74,048 | 700,384 | 22800.855 |
| 2647 | 84,704 | 800,512 | 26076.022 |

At step715 each task had 55 updates / 1,760 episodes; at step2314, 178 updates / 5,696 episodes. Planned—not yet attained at the snapshot—step2860 is 220 updates / 7,040 episodes per task (91,520 total). The live cutoff need not finish a task cycle. Actual per-task frames and all cell exposures at each completed look are embedded in the snapshot.

## Completed v2 final tests — separate evidence panel

Both are complete **13-task/35-cell** tests: 128 episodes in each ordinary cell and 200 in each Krauzlis cell, **4,696 episodes per checkpoint**. Same v2 final-only draw namespace **94292763** for selected117 and terminal715. These within-v2 checkpoints share draws; stored aggregate summaries do not provide trial-paired error differences. Neither v2 test should be paired with v3 validation or new v3 tests.

BA/AUC below are equal eligible-cell means within task. N0 recognition controls are excluded from BA/AUC, not omitted from the 35-cell inventory.

| Task | Selected117 BA | Selected117 AUC | Terminal715 BA | Terminal715 AUC |
|---|---:|---:|---:|---:|
| motion_direction | 0.250000 | 0.537760 | 0.265625 | 0.465739 |
| orientation | 0.500000 | 0.475098 | 0.562500 | 0.561279 |
| contrast | 0.500000 | 0.565430 | 1.000000 | 1.000000 |
| spatial_frequency | 0.500000 | 0.504150 | 0.632812 | 0.712891 |
| chromatic_increment | 0.687500 | 0.729492 | 0.992188 | 1.000000 |
| contour | 0.500000 | 0.452393 | 0.523438 | 0.503662 |
| natural_spectrum | 0.500000 | 0.542725 | 0.929688 | 0.983887 |
| orientation_ring | 0.492188 | 0.447021 | 0.460938 | 0.435059 |
| orientation_cued | 0.501953 | 0.490112 | 0.505859 | 0.484131 |
| motion_duration_cued | 0.250000 | 0.517415 | 0.244141 | 0.505249 |
| krauzlis_cued_motion | 0.500000 | 0.499167 | 0.500000 | 0.526316 |
| spatial_binding | 0.494141 | 0.505859 | 0.517578 | 0.519836 |
| image_recognition | 0.500000 | 0.503337 | 0.500000 | 0.511475 |

## v3 fixed-draw validation — preliminary, selection-reused

Baseline715 and looks1248/1781/2314 use the same validation namespace **732001**, 64 ordinary episodes and 100 per Krauzlis cell: **2,348 episodes/look**, 13 tasks and 35 cells. V3 selects lexicographically by equal-task mean AUC, then mean chance-normalized BA, earlier ties. Historical v2 selected117 is not silently reselected. Three N0 cells remain excluded from selection (32 eligible cells); each eligible condition is averaged equally within task, then tasks equally. BA chance is 0.25 for the two four-class tasks, 0.5 otherwise. AUC chance is 0.5 even for the macro one-vs-rest four-class AUC. Chance-normalized BA is `(BA−1/K)/(1−1/K)`.

| Step | Equal-task mean AUC | Mean chance-normalized BA |
|---:|---:|---:|
| 715 | 0.629628923 | 0.221754808 |
| 1248 | 0.653915064 | 0.272970085 |
| 1781 | 0.679238896 | 0.287126068 |
| 2314 | 0.693738531 | 0.289797009 |

The snapshot's recomputed mean keys on v2 *test* entries are descriptive arithmetic checks only, never used for selection. The baseline artifact still contains its historical minimum-task key; `v3_rule_key_recomputed` makes the prospective v3 reinterpretation explicit.

| Task | Baseline715 BA/AUC | Val1248 BA/AUC | Val1781 BA/AUC | Val2314 BA/AUC |
|---|---:|---:|---:|---:|
| motion_direction | 0.296875/0.553060 | 0.296875/0.595703 | 0.250000/0.579753 | 0.312500/0.722005 |
| orientation | 0.625000/0.629883 | 0.546875/0.598633 | 0.546875/0.650391 | 0.500000/0.749023 |
| contrast | 1.000000/1.000000 | 0.984375/1.000000 | 1.000000/1.000000 | 1.000000/1.000000 |
| spatial_frequency | 0.484375/0.559570 | 0.750000/0.828125 | 0.859375/0.969727 | 0.906250/0.981445 |
| chromatic_increment | 0.984375/1.000000 | 1.000000/1.000000 | 0.984375/1.000000 | 1.000000/1.000000 |
| contour | 0.484375/0.514648 | 0.578125/0.579102 | 0.500000/0.601562 | 0.515625/0.586914 |
| natural_spectrum | 0.890625/0.954102 | 0.921875/0.944336 | 0.953125/0.980469 | 0.968750/0.997070 |
| orientation_ring | 0.453125/0.412109 | 0.531250/0.505859 | 0.546875/0.492188 | 0.437500/0.519531 |
| orientation_cued | 0.511719/0.542725 | 0.468750/0.484375 | 0.500000/0.510010 | 0.500000/0.497559 |
| motion_duration_cued | 0.238281/0.502930 | 0.246094/0.501261 | 0.242188/0.513997 | 0.246094/0.516439 |
| krauzlis_cued_motion | 0.500000/0.508364 | 0.500000/0.490140 | 0.500000/0.515300 | 0.500000/0.441452 |
| spatial_binding | 0.484375/0.486084 | 0.460938/0.465332 | 0.484375/0.498047 | 0.476562/0.483398 |
| image_recognition | 0.500000/0.521701 | 0.503472/0.508030 | 0.496528/0.518663 | 0.539931/0.523763 |

Contrast and chromatic increments have strong saved evidence of learning. Frequency and natural spectral detail also improve on repeated validation. This does not establish the spatial battery: at2314 spatial-group mean AUC **0.492522**, mean chance-normalized BA **0.005556**. The raw spatial-group BA mixes four- and two-class tasks and should not be compared with a universal 0.5 chance level. Ring orientation is near chance despite only four frames/no inserted blanks. Negative spatial results are limited-exposure, single-seed observations, not a capacity theorem.

### Confusion and criterion caveats (latest completed validation2314)

All matrices below use **rows=true, columns=argmax predicted**, label order from catalog. Binary argmax corresponds to the probability-0.5 decision rule, not an independently calibrated threshold. AUC is copied from saved probabilities' ranking score; the underlying trial probabilities were not retained in these aggregate JSONs.

- `orientation/mixed`, n64: **[[0,32],[0,32]]**, class recalls **[0,1]**, BA **0.5**, AUC **0.7490234375**. Every sampled trial is called positive rotation despite above-chance score ranking. This is a discriminability/decision-boundary dissociation, not successful 75% classification or proof of good calibration. No threshold was fitted and no d′/criterion claim is estimated here; absent boundary correction, z-scores of extreme rates are undefined.
- `motion_direction/mixed`, n64: **[[6,0,2,8],[0,3,5,8],[0,1,0,15],[0,1,4,11]]**; recalls **[.375,.1875,0,.6875]**, BA **.3125**, macro OvR AUC **.7220052083**. High macro AUC does not imply accurate four-way choice.
- `orientation_ring/D0`, n64: **[[10,22],[14,18]]**, recalls **[.3125,.5625]**, BA **.4375**, AUC **.51953125**. This is rotation-sign classification, not a no-change detector.
- Every Krauzlis B12/B20/B28 cell, n100: **[[0,43],[0,57]]**, accuracy **.57** but BA **.5**. Target hits **57/57**, foil false reports **29/29**, catch false positives **14/14**. Per-cell target-side counts: target **29/28**, foil **14/15**, catch **7/7**. All-positive responding is not selective target detection. The three cell AUCs are **.4753161975/.5112199102/.3378212974**, equal-cell mean **.4414524684**. The 100-trial event mixture is exact, but odd event counts cannot also have exact within-event side balance; final n200 permits it.
- Recognition N0_H3/H4/H5 each: **[[64,0],[0,0]]**, specificity1/FPR0, BA/AUC null. N4_H3/H4/H5 each: **[[32,0],[32,0]]**, BA.5; AUCs .368164/.568359/.449219. Therefore perfect empty-list rejection is not evidence of learned recognition. Nonempty aggregate BA .539931 remains weak, with load/hold-dependent response bias (all per-cell confusions retained).

## Exact 35-condition inventory and denominators

Counts in the last columns are **training episodes**, not evaluation n or updates. All cells have explicit validation/test denominators below; N0 cells are included but not BA/AUC eligible. Frame counts derive from the actual `launch.frames`→renderer `frame_count` code, with catalog kwargs. Evaluation values, all strata and confusions for every cell at every saved look are in the JSON rather than suppressed behind task averages.

| Task | Cell | Frames/episode | Val n | V2 final n | Train@715 | Train@2314 | Train@2647 |
|---|---|---:|---:|---:|---:|---:|---:|
| motion_direction | mixed | 2 | 64 | 128 | 1760 | 5696 | 6496 |
| orientation | mixed | 2 | 64 | 128 | 1760 | 5696 | 6528 |
| contrast | mixed | 2 | 64 | 128 | 1760 | 5696 | 6528 |
| spatial_frequency | mixed | 2 | 64 | 128 | 1760 | 5696 | 6496 |
| chromatic_increment | mixed | 2 | 64 | 128 | 1760 | 5696 | 6528 |
| contour | mixed | 2 | 64 | 128 | 1760 | 5696 | 6528 |
| natural_spectrum | mixed | 2 | 64 | 128 | 1760 | 5696 | 6528 |
| orientation_ring | D0 | 4 | 64 | 128 | 1760 | 5696 | 6528 |
| orientation_cued | D0 | 4 | 64 | 128 | 448 | 1440 | 1600 |
| orientation_cued | D4 | 8 | 64 | 128 | 448 | 1408 | 1632 |
| orientation_cued | D12 | 16 | 64 | 128 | 416 | 1408 | 1632 |
| orientation_cued | D24 | 28 | 64 | 128 | 448 | 1440 | 1632 |
| motion_duration_cued | D0 | 11 | 64 | 128 | 448 | 1408 | 1632 |
| motion_duration_cued | D4 | 15 | 64 | 128 | 416 | 1440 | 1632 |
| motion_duration_cued | D12 | 23 | 64 | 128 | 448 | 1408 | 1600 |
| motion_duration_cued | D24 | 35 | 64 | 128 | 448 | 1440 | 1632 |
| krauzlis_cued_motion | B12 | 29 | 100 | 200 | 608 | 1920 | 2176 |
| krauzlis_cued_motion | B20 | 37 | 100 | 200 | 576 | 1888 | 2176 |
| krauzlis_cued_motion | B28 | 45 | 100 | 200 | 576 | 1888 | 2144 |
| spatial_binding | D0 | 5 | 64 | 128 | 448 | 1440 | 1632 |
| spatial_binding | D4 | 9 | 64 | 128 | 448 | 1440 | 1632 |
| spatial_binding | D12 | 17 | 64 | 128 | 448 | 1408 | 1632 |
| spatial_binding | D24 | 29 | 64 | 128 | 416 | 1408 | 1632 |
| image_recognition | N0_H3 | 7 | 64 | 128 | 160 | 480 | 544 |
| image_recognition | N0_H4 | 8 | 64 | 128 | 160 | 480 | 544 |
| image_recognition | N0_H5 | 9 | 64 | 128 | 160 | 480 | 544 |
| image_recognition | N4_H3 | 11 | 64 | 128 | 128 | 480 | 544 |
| image_recognition | N4_H4 | 12 | 64 | 128 | 160 | 448 | 544 |
| image_recognition | N4_H5 | 13 | 64 | 128 | 128 | 480 | 544 |
| image_recognition | N12_H3 | 19 | 64 | 128 | 128 | 448 | 544 |
| image_recognition | N12_H4 | 20 | 64 | 128 | 128 | 480 | 544 |
| image_recognition | N12_H5 | 21 | 64 | 128 | 160 | 480 | 544 |
| image_recognition | N24_H3 | 31 | 64 | 128 | 160 | 480 | 544 |
| image_recognition | N24_H4 | 32 | 64 | 128 | 160 | 480 | 544 |
| image_recognition | N24_H5 | 33 | 64 | 128 | 128 | 480 | 544 |

Semantic distinctions from `TaskSuite/README.md`, `catalog.json`, `suite.py` and native renderers: seven sensory tasks each have exactly two presented frames; motion direction is cardinal displacement, not the eight-transition duration count winner. Orientation ring is signed target rotation; `orientation_cued` instead asks alignment with a location/sign glyph, with unchanged/opposite targets negative. Binding is a retrocue on an exactly-one-pair orientation exchange, and its negatives are foil-only exchanges, not no-change trials. Krauzlis asks target-only change versus foil/catch; B is baseline transitions, not blank delay. Recognition is exact raster membership, with nonempty N and repeated-probe H factorial conditions; positive probes are byte-identical to a study item. D values are frame counts, not milliseconds; only Krauzlis defines a 100-Hz clock. No probabilistic cue-validity manipulation or variable reaction-time action is evaluated by this suite.

## Interpretation boundaries and reproducibility

- Repeated validation is preliminary and selected on; no new v3 final-test artifact was complete/present at the freeze. Old test/new validation are different split/source populations and never a paired contrast. The eventual v3 final-only namespace is **94392763**, different from v2. New draws do not create new official BSDS500 test source identities; previous tests were seen, so continuation remains exploratory.
- BSDS500 source identity partition is train200/val100/test200, shared consistently by the two photographic tasks. Synthetic splits use separate seeds, not held-out positions. There are no source-independent uncertainty intervals or replicate training seeds here.
- Delay streams are independent, not matched same-scene interventions. Differences across delay cells confound draw variation. Local stimulation/inhibition, anticipatory allocation, cue validity benefits, sensitivity/criterion changes and reaction-time effects remain **unmeasured** in this joint lineage.
- No fixed gate or retention claims, no historical orientation-curriculum competence imported into this fresh joint lineage, no softmax-normalized map inferred from signed KDA coefficients. The 2025 comparison must distinguish target-specific report from legitimate uncued-change detection.
- The audit verifies all13 task names, unique35 cells, complete flags/counts, confusion sums, every stratum denominator, N0 null rules, reconstructed task/group summaries, exposure totals, parameter totals, CPU tensor shapes and all13 active pinned source hashes. All five checkpoint tensor-shape audits were CPU-only; raw checkpoint SHA-256s are recorded. Existing source/config/run files were read only, never modified.

Recheck the frozen snapshot without evaluating a model:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/tmp/vawm-task-suite-venv/bin/python \
SecondPass/JointTraining/TechnicalReport/audit_saved_metrics.py --verify
```

The script refuses to overwrite `metrics_snapshot.json`. Active logs were read once as complete-line prefixes; their captured prefix digests and exposure row timestamps are stored. Status and logs are not atomically synchronized, so exposure is explicitly cut at the captured status step (2647). `.partial.json` artifacts were inventoried by name, not consumed as complete evaluations. All original raw JSON source paths and hashes are available under `evaluations[].source`, `metadata`, `catalog`, and `live_status`.
