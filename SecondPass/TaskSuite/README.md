# Unified visual task suite — preparation, not training

**Authorization update (2026-09-22):** the user subsequently requested starting
one fresh local KDA model across this suite. The suite below remains unchanged;
the separate [joint-training contract](../../LabJournal/joint-suite-training.md)
supersedes its earlier preparation-only authorization. Training code and run
artifacts live separately under `SecondPass/JointTraining`.

Status: **suite assembly only**. The user requested all tasks discussed in the current-model conversation, with a later training run starting from fresh weights. No training, warm start, GPU/cloud allocation, or model comparison is authorized by this suite. Existing model/checkpoint artifacts are untouched.

The suite contains **13 distinct supervised tasks in 35 primary task/condition cells**: seven two-frame sensory tasks, the ring-cued orientation precursor, and the five spatial/sequence tasks. This is not a claim that the current KDA checkpoint has acquired them; it was trained only on ring-cued and signed spatial orientation, using a delay ladder.

## Files and interface

- [`catalog.json`](catalog.json): executable task inventory, label order, renderer source, conditions, scoring eligibility, fresh split seeds and reporting strata.
- [`suite.py`](suite.py): `SuiteStream(split).batch(n, task, cell)` returns `(images, labels, analysis_metadata)`. `task_classes()` supplies the complete head dictionary to the existing KDA or plain model constructor.
- [`verify.py`](verify.py): CPU-only rendering, native-renderer equivalence, stream replay, optional fresh-KDA forward checks. No optimizer, backward pass, checkpoint load/save, or training launch.
- [`test_suite.py`](test_suite.py): contract and validation tests.
- [`checks/verification.json`](checks/verification.json): actual verification output, not model performance.

Images are float32 `[B,T,3,100,100]` in `[0,1]`; labels are int64 `[B]`. Each batch contains **one task and one condition**, so it has a real uniform sequence length. Do not concatenate unlike sequence lengths by feeding extra padding as if it were evidence. Only images enter the sensory/recurrent computation; the task ID selects the supervised head. That head selection is external task identity, not a learned task-inference capability. Metadata, label semantics, cue positions, actual angles, source IDs, and frame-phase indices must never become model inputs.

All renderers and objectives are reused without changes. The adapter adds namespaced analysis metadata and independent streams; it does not recolor cues, change dot motion, remap labels, insert frames, or normalize the images. Input centering and causal stack-3 preprocessing remain operations of the existing model. The sensory tasks still contain exactly two **presented** frames; stack padding is not a third observation.

## 1. Two-frame sensory tasks

Each has one primary `mixed` condition that preserves the renderer's original difficulty mixture. Report both the task score and the listed difficulty strata; a single mixed average does not replace those strata. These tasks use no new cue or delay.

| Task key | Question and output labels | Existing signal range / reporting strata |
|---|---|---|
| `motion_direction` | What cardinal direction links the dot frames? `0=right, 1=up, 2=left, 3=down`. | 1/2/3-pixel displacement; half of 256 domain dot identities survive. Single circular aperture; not the four-patch duration task. |
| `orientation` | What is the sign of the grating's axial-angle change? `0=negative, 1=positive`. | 4/10/22 degrees; random base orientation and independent phases. |
| `contrast` | Which frame contains greater grating contrast? `0=first, 1=second`. | Increments .025/.06/.13; pedestals .08/.18/.30. Report both. |
| `spatial_frequency` | Which frame has higher spatial frequency? `0=first, 1=second`. | Increments .08/.18/.35 octaves. |
| `chromatic_increment` | Which frame has a positive increment along the defined linear-RGB chromatic axis? `0=first, 1=second`. | Increments .018/.045/.1; numerically luminance-matched axis, not calibrated human isoluminance. |
| `contour` | Which frame contains the aligned Gabor contour? `0=first, 1=second`. | Seven path elements among 32; alignment jitter 2/8/16 degrees; orientation multiset matched. |
| `natural_spectrum` | Which frame has greater relative high-frequency detail (smaller spectral beta)? `0=first, 1=second`. | Native BSDS500 luminance crops; beta differences .15/.30/.60, phase preserved, common mean and RMS. This is not semantic image recognition. |

Renderer: [`TaskStream`](../../PreAttentiveVision/neuroscience_stimuli.py), with [`NaturalSpectrum`](../../PreAttentiveVision/natural_stimuli.py) for the photograph task. Do **not** substitute the obsolete `PreAttentiveVision.stimuli.PairStream` change/no-change battery or its single-dot displacement objective.

Angle sign is recorded exactly as the renderer defines it. Image y coordinates increase downward; legacy docstrings use inconsistent clockwise/counterclockwise language across renderers. Keep the numeric sign definitions rather than silently changing labels to reconcile that wording.

## 2. Ring-cued orientation task

**`orientation_ring`**, two classes, cell `D0` only.

Four Gabor patches appear at the existing four locations. A ring marks the target in the cue and sample frames. Compare its sample and probe orientations: `0=negative axial-angle change`, `1=positive`. Foils rotate with independently sampled signs. Magnitudes are 15/30/45 degrees.

Timeline: **one cue frame → two sample frames → one probe/report frame**. No inserted retention blanks. This was the current model's acquisition precursor; here it is simply a separately named suite task. Its presence does not impose a ring-first curriculum or warm start on the future joint run.

Renderer: [`VariantStream`](../../WorkingMemory/PlainBaseline/variants.py). The single-location `orientation_single` and `orientation_sign` diagnostic rungs are not in this suite: they were not part of the requested task list or the current KDA checkpoint's training sequence.

## 3. Spatial selection and sequence-memory tasks

These reuse the exact audited [`SpatialBatteryStream`](../../WorkingMemory/SpatialTaskBattery/stimuli.py) definitions. See the historical [protocol](../../WorkingMemory/SpatialTaskBattery/PROTOCOL.md) for stimulus details, **not its obsolete training authorization/optimizer recipe**.

| Task key | Required decision | Primary conditions | Timeline / total frames |
|---|---|---|---|
| `orientation_cued` | Did the selected Gabor rotate in the sign requested by its spatial +/− glyph? `1=aligned`, `0=unchanged or opposite`. | `D0/D4/D12/D24`; 15/30/45-degree changes. | Cue 1 → sample 2 → blanks D → probe 1; **D+4**. Sign/location glyph visible in cue and both samples. |
| `motion_duration_cued` | Which direction occupied the largest total number of the eight transitions in the ring-cued patch? `right/up/left/down=0/1/2/3`. | `D0/D4/D12/D24`; .8/1.2/1.6-pixel steps. | Cue 1 → reference 1 → moving frames 8 → blanks D → report 1; **D+11**. Ring remains during reference/motion. |
| `krauzlis_cued_motion` | Did mean direction change in the precued patch? `1=target event`, `0=foil event or catch`. | `B12/B20/B28`: baseline-transition counts, **not memory delays**. | Cue 2 → fixation 5 → reference 1 → baseline B → postevent 8 → report 1; **B+17**. |
| `spatial_binding` | Did the item at the retrospectively queried location participate in the orientation swap? `1=target exchange`, `0=foil-only exchange`. | `D0/D4/D12/D24`. | Instruction 1 → sample 2 → blanks D → ring query 1 → probe 1; **D+5**. No target precue. |
| `image_recognition` | Is the probe raster an exact member of the preceding image list? `1=member`, `0=nonmember`. | Every combination of `N0/N4/N12/N24` and `H3/H4/H5`; e.g. `N12_H4`. | Instruction 1 → N consecutive study images → blanks 3 → H identical probe repeats; **N+4+H**. |

### Distinctions that must survive integration

- Duration means the unique **count winner**, not final direction, average displacement, or a winner pooled across patches. Each of four patches has an independent schedule. Keep 32 dots/patch and the existing replacement law.
- Signed orientation includes unchanged negatives; the ring precursor does not. Keep separate heads even though both have two classes. Full-task foils preserve the signed-change multiset before target assignment.
- Binding always exchanges exactly one pair. Negative examples swap two foils; they are not no-change trials. This prevents a global-change detector from solving the task.
- Krauzlis is the documented compressed Arcizet–Krauzlis 2018 adaptation: two patches, subpixel motion, 57 target / 29 foil / 14 catch events per 100 trials. It is not the two-frame cardinal task and not a biological reaction-time model. Preserve its unequal event mixture.
- Recognition positives are byte-identical image rasters from the study list; each repeated probe is the same observation, not a new episode. Empty lists are always negative.
- D measures inserted blank frames, not milliseconds. Only the Krauzlis renderer defines a 100 Hz frame clock.

## Streams, splits and resumption

`catalog.json` defines new base seeds for train/validation/test. Each task has a stable explicit numeric stream ID; each cell has a separate native stream derived from split, task ID and cell index. Sampling one cell never advances another. This changes sample order relative to historical runs, but not the task laws. It is **not** an old-run replay or a matched historical evaluation.

`state_dict()` / `load_state_dict()` preserve the entire catalog, split and all materialized native stream states. Restore rejects catalog/split mismatches and validates before replacing current state. A later scheduler must save its own ordering/RNG progress alongside these streams. Parallel workers would also need explicit disjoint stream namespaces; this v1 adapter intentionally does not pretend to provide a multiprocess training loader.

Once a photograph task has been sampled, suite snapshots also record the
BSDS500 manifest SHA256. Restore rejects a different manifest before replacing
stream progress, covering recognition as well as natural-spectrum sampling.

Delay cells are independent streams, **not paired copies of the same evidence**. If a future experiment needs paired delay interventions, build and verify a separate paired-evaluation plan rather than describing these draws as paired.

The two photograph tasks share the official BSDS500 **train 200 / validation 100 / test 200** identity partition. A photograph used for training on either task must not enter another task's validation/test split. Sharing training photos across these two tasks is intentional. Synthetic procedural tasks use distinct split seeds, not held-out source identities or held-out spatial centers. Training and testing on the same four centers does not demonstrate novel-location generalization.

## Reporting contract for later training

This assembly defines task semantics, not a silently inherited optimizer or sampling allocation.

1. Keep all 13 tasks and all 35 cells visible. Report confusion, class recall, accuracy, balanced accuracy (BA), and binary or multiclass one-vs-rest AUC where defined; include actual denominators and difficulty strata. Binary chance is .5 and cardinal chance .25.
2. For Krauzlis, separately report target hit rate, foil false-alarm rate and catch false-positive rate, event counts and target-side counts. Use multiples of 100 from a cell's cycle boundary for exact event proportions; 200 balances side within each odd-sized event category. Tiny smoke batches need not match these proportions.
3. For recognition N0, report specificity and false-positive rate only. BA/AUC are undefined in a single-class condition: record null/not-applicable, and exclude N0 from checkpoint-selection BA/AUC. `selection_eligible=false` marks these three cells.
4. Within each task, summarize conditions equally before any cross-task average, excluding N0 as above. Also show sensory, orientation-auxiliary and spatial groups separately: seven easy sensory tasks must not conceal failures on the five sequence tasks. Any chance-normalized score must use each task's actual class count.
5. Do not select a checkpoint using the final test split. Freeze allocation, validation policy, exposure, and test plan before the later authorized run. Fresh weights do not make repeatedly inspected historical test examples fresh; this suite uses new seeds.
6. Count episodes, frames and task exposures separately; longer sequences cost more. Repeated probe frames are not extra training examples. Source-aware uncertainty must account for BSDS500's finite photograph pool.

No executable performance scorer or checkpoint-selection policy is added here; the future training harness must implement and test this reporting contract before launch. Current verification scores renderer/interface correctness only.

## Fresh-weight training boundary

For the later run:

- Construct the whole chosen model from a recorded seed with `task_classes()`; create a new optimizer with empty state. Load **no** sensory, KDA, GRU, head, Adam, or stream checkpoint from the old experiments.
- Keep every learned parameter trainable. Do not inherit the old sensory/new-component learning-rate split or freeze a sensory stage.
- Train across the full declared suite. The task groups are reporting categories, not a prescribed curriculum. No automatic ring-first sequence or staged delay curriculum is installed.
- Decide the task/condition allocation, effective batch, loss weighting, validation selection and finite compute budget explicitly before launch. The old five-task microbatch/clip recipe is not carried over by importing its renderer.
- Preserve a plain-model comparison as the research baseline; this document does not authorize a second training arm.

## Run the CPU checks

Use a project environment with PyTorch, NumPy, SciPy, Pillow and pytest. Supplemental dependencies are in [`requirements.txt`](requirements.txt). Older PyTorch builds such as this machine's 2.2.2 require NumPy below 2 for NumPy-tensor conversion; use an isolated environment rather than changing global packages.

From the repository root:

```bash
python -m pip install -r SecondPass/TaskSuite/requirements.txt
python -m PreAttentiveVision.natural_stimuli --prepare
python -m pytest SecondPass/TaskSuite/test_suite.py -q
python -m SecondPass.TaskSuite.verify --model-smoke
```

Dataset preparation is an explicit network operation. Importing or sampling the suite never downloads data. Missing BSDS500 blocks its two tasks rather than substituting synthetic photographs; the verifier lists each blocked cell and exits with code 2. A corrupt/incomplete manifest or failed renderer assertion is an error, not a skipped success. The model smoke test uses freshly initialized weights on CPU, inference only, and verifies unchanged state afterwards. It establishes interface compatibility, **not learnability or trained accuracy**.

## Deferred, not silently added

Smaller orientation thresholds, 48-frame delays, cue weakening/jitter, and distractor sweeps remain diagnostic/evaluation variants outside the primary training suite. The older broad sequence-memory battery, discarded single-dot change objective, and separate single/sign orientation rungs are not restored. Training infrastructure, sampling/loss policy, performance aggregation and any new task are separate next steps.
