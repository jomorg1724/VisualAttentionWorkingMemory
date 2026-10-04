# Task-renderer and report-rule failure audit

## Finding

**The failure is already present without an inserted delay. It is not evidence, by itself, of a deficient memory architecture.** On the frozen terminal test, ring orientation D0 has 53.91% balanced accuracy (BA), signed cued orientation D0 45.31%, spatial binding D0 46.88%, and cued motion duration D0 24.22% (four-class chance 25%). Their AUCs are also near chance. The selected checkpoint does not rescue these cells. Long-delay forgetting cannot be the sole explanation for these acquisition failures.

**No operative label inversion, missing cue, erased target, recognition-membership mismatch, or suite/native pixel mismatch was found in the bounded checks.** All 35 cells matched their native renderers exactly. A nonlearned pixel decoder correctly solved 32/32 fresh D0 episodes in each of ring orientation, signed cued orientation, and binding, including reading the location/sign cue from the actual pixels. This establishes that these sampled stimuli contain usable information; it does not establish that this trained network learned to extract or combine it.

The concrete learnability differences from the successful sensory tasks are smaller/high-frequency multi-item signals, target selection, sign-conditioned reporting, nuisance changes between repeated samples, retrospectively selected comparisons, temporal count integration, and multi-scene list membership. The report below separates these observations from untested network explanations.

## Scope, sources, and execution

Read `ANALYSIS_SOP.md`, the task-suite README/catalog/adapter, the native renderers and their imported cue/schedule helpers, the orientation-variant definitions and historical program, the sensory renderers, and the current model's input stacking/final report. No model was instantiated, no checkpoint was loaded, no neural forward or training update was run, and no task/model/training source was changed. Repository writes were confined to artifacts in this analysis directory; reusable audit lessons were also appended to the existing task-suite skill.

Executed from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /tmp/vawm-task-suite-venv/bin/python -B \
-m SecondPass.SpatialReadout.FailureAnalysis.task_native_checks
```

Actual checks: CPU threads 1; 13 tasks / 35 unique cells; 1,624 new native episodes (32/cell, except 200 per Krauzlis cell), each compared against an independently constructed native stream with the same seed and batch boundaries. These are renderer/metadata checks on fresh **validation-stream** stimuli, not additional trained-model evaluations and not the frozen test draws. Every cell's dtype, range, shape, pixel equality, label equality, native metadata preservation, and class count passed; task-specific label invariants passed. The three simple D0 pixel oracles have only 32 examples each and use known renderer geometry/timing, not a learned general-purpose vision system.

Artifacts:

- `task_native_checks.py`: executable bounded checks, pixel-cue/template decoder and gradient structure-tensor orientation decoder.
- `task_native_checks.json`: per-cell results, class/location/sign counts, cue/phase measurements, oracle outcomes, source hashes and source parity.
- `task_results_extract.json`: source paths/hashes plus saved selected/terminal cell records, without recomputing network outputs.
- `task_results_cells.csv`: both complete 35-cell score/confusion tables.

The frozen results are `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/test_selected.json` and `test_terminal.json`, each complete at 35 unique cells and 4,696 trials. Every confusion matrix sums to its cell denominator. SHA256:

- selected: `40e8438255f0aaed78f09a31dde47f0b6f68552a525f953fa5b7c86f17f7c0d8`
- terminal: `3886e16406733498bdede64f1d28741a973e7a29d3f431815c86367d7a54fa43`

Both record final-test namespace `94792763`; do not treat these as independent random replications. The current journal identifies selected 5486 and terminal 6760 (`LabJournal/CURRENT_STATUS.md:3`). Nine inspected operative source files match the live runtime repository byte-for-byte; eight also exist in the continuation's locked-source directory and match. **Provenance limitation:** imported `WorkingMemory/stimuli.py` is not in that locked-source snapshot. Its current runtime/repository copies match, but this audit cannot prove its historical bytes solely from that snapshot. This is a dependency-freezing gap, not evidence that it changed.

All source references below are repository-relative, with line numbers from the inspected files. Abbreviations: **S** = `WorkingMemory/SpatialTaskBattery/stimuli.py`; **V** = `WorkingMemory/PlainBaseline/variants.py`; **W** = `WorkingMemory/stimuli.py`; **P** = `PreAttentiveVision/neuroscience_stimuli.py`; **N** = `PreAttentiveVision/natural_stimuli.py`.

## 1. What actually fails, and what D0 means

Frozen saved results; n=128/cell except Krauzlis n=200. Values are percentages for BA, raw units for AUC. These small, single-fit cell estimates are descriptive, not proof of exact chance performance or below-chance learning.

| Cell | Selected BA / AUC | Terminal BA / AUC | Meaning |
|---|---:|---:|---|
| motion_direction / mixed | 63.28 / .9389 | 100.00 / 1.0000 | Basic one-transition cardinal direction succeeds terminally. |
| orientation / mixed | 99.22 / 1.0000 | 92.19 / 1.0000 | Strong rank discrimination; terminal criterion errors remain. |
| orientation_ring / D0 | 54.69 / .5925 | 53.91 / .5261 | Failure before adding signed instructions or retention blanks. |
| orientation_cued / D0 | 50.78 / .5415 | 45.31 / .5010 | Signed spatial report not acquired. |
| spatial_binding / D0 | 46.09 / .4556 | 46.88 / .4365 | Retrocue/binding comparison not acquired even at shortest condition. |
| motion_duration_cued / D0 | 22.66 / .5039 | 24.22 / .5154 | No delay, but still eight-transition integration. |
| krauzlis / B12 | 51.24 / .5097 | 50.00 / .5030 | B is baseline motion duration, not a retention delay. |
| recognition / N4_H3 | 53.91 / .6665 | 64.06 / .6333 | Some evidence in this shortest-load/hold cell, not universal total failure. |
| recognition / N0_H3, H4, H5 | specificity 100 each | specificity 100 each | Empty-set rejection succeeds; no item memory is required. |

At terminal, cued orientation D24 predicts positive on all 128 trials (BA .5; AUC .5474), duration D12 and D24 always predict class 1 (BA .25), and all three Krauzlis cells always predict positive. Krauzlis therefore has target hit rate 1 **and** foil/catch false-positive rate 1, not successful target detection. Its .57 raw accuracy is the majority-class rule, not above-chance balanced discrimination. These are score/criterion collapses at the operating point; near-chance AUC in most failing cells means threshold adjustment alone is not an established solution. Conversely basic orientation's AUC 1.0 with BA .921875 is not absence of sensory discrimination.

There is no recognition D0 cell and no Krauzlis D0 cell. Recognition always inserts three gray frames; Krauzlis always has a five-frame fixation-only interval after its cue. Do not invent an absent zero-delay control for either task.

### The raw three-frame stack matters

The actual `SpatialReadout` inherits stack=3, centered input (`SecondPass/SpatialReadout/model.py:29–46`); the inherited stack concatenates the two preceding raw frames and current frame, with zero padding after centering (`WorkingMemory/PlainBaseline/accum.py:54–58`). The head is applied only to final recurrent state (`SecondPass/SpatialReadout/model.py:49–51`).

- At a sensory task's second/final frame, both presented observations are directly present in the channel stack. Success need not demonstrate memory maintained across a stimulus-free interval, or reliance on recurrent state rather than a learned two-frame comparator.
- At ring/signed orientation D0's final frame, the stack contains both cued sample frames plus probe. Relevant cue and comparison evidence are concurrently available in raw channels.
- At binding D0's final frame, the stack contains last sample, retrocue, and probe. A cue-conditioned feedforward comparison is in principle possible without long retention. The successful pixel oracle confirms usable native evidence, not that a specific neural path exploits it.
- At duration D0's report, only the final two moving frames plus report remain in the raw stack. Earlier transitions needed for the count winner are outside it. D0 therefore does **not** remove temporal integration.
- At recognition's first probe, study images have cleared the raw stack. By the final H3/H4/H5 report, the raw stack contains only identical probe repeats. Successful membership must depend on earlier history; merely detecting repeated probes cannot distinguish labels.

Thus D0 failure localizes the problem earlier than **additional** blank retention in the orientation/binding families, but does not tell us whether the culprit is cue extraction, local feature acquisition, combination of information, optimization, interference, or readout.

## 2. Cue precision and phase signatures: present, deterministic, small

**Spatial placement and cue visibility are not probabilistic validity.** The four fixed centers are (27,27), (73,27), (27,73), (73,73) (S:15). The cue defines the scored target. These tasks do not manipulate classical 25/50/75/100% cue validity; uncued-only events are negatives where the report rule specifies a target, not hits as in an any-change task (`ANALYSIS_SOP.md:20,39–44`).

S:34–43 copies the legacy glyph or draws a ring:

- The signed cue occupies a 10×10 box above the selected center, x from center−5 through center+4 and y from center−23 through center−14. For upper-left target, the box is x22–31, y4–13. Its dark strokes are .1 on .5 gray. **Minus has 12 dark spatial pixels, plus 20; sign is carried by only 8 differing spatial pixels.** RGB channels repeat these pixels; this is not 24 independent informative locations. The glyph's sign/location is shown in cue and both sample frames, then removed (S:85–91).
- The ring is bright .95, radius 14, radial tolerance .8; it changes **128 spatial pixels** relative to the cue-free frame. It is outside the Gabor's radius-12 support, so it does not overwrite the Gabor. The signed glyph also lies outside that support (S:40–42,79–84). No cue/target erasure was found.
- Ring orientation displays its cue in frames 0–2 (V:51–58). Duration retains the ring through all reference/moving frames 0–9 (S:105–111). Binding gives **no target precue**; its ring appears only on the blank query immediately before probe (S:92–96). These are different selection operations.
- Krauzlis uses two locations (20,50)/(80,50), a ring only for frames 0–1, then five fixation-only frames before reference; no corner glyphs mark its event/report phases (S:124–142). The report is identical to fixation-only frames. The cue is not available again during evidence.

**Phase signals do exist for orientation, binding, and duration, but are tiny corner pictograms, not numerical inputs or software phase gates.** W:48,58–83 defines three-bit codes: sample=4, ignore=5, query=6, report=7. The fixed top-left protocol marker and top-right phase marker occupy reserved 10×10 corners; the bottom corners are cleared even if unused. On a gray recall frame, measured non-gray spatial pixel counts are sample36, ignore40, query40, report44. Relative to sample, ignore and query each differ on just **4 pixels**, report on **8**. Nominal delay blanks in these tasks therefore retain protocol/ignore glyphs; they are not completely uniform gray (S:89,94,110).

Recognition instead uses one initial recall/sample instruction, raw study images, three truly gray blanks, and raw repeated probes, without overlays (S:119–120). The blank gap and sequence position delimit study from probe; there is no explicit report glyph on photographs. Preserving this distinction is essential for any later “phase decoding” or memory-reset diagnostic.

**Interpretation:** exact cue templates identify location/sign in the sampled native frames, so there is no missing cue. Small strokes and small phase-code differences are plausible acquisition burdens under learned filtering/downsampling, but their disappearance in the trained network has **not** been measured. Ring D0 also fails despite its much larger cue, so the eight-pixel sign contrast is not a sufficient explanation of the entire failure pattern.

## 3. Local orientation: clear native information, harder comparison than the sensory task

P:118,142–168 renders the sensory grating with Gaussian sigma19, frequency5–10 cycles/image, a single random base angle, two independently sampled carrier phases, and a **shared** frequency/contrast across its two frames. The rotation is 4/10/22 degrees and label is the numeric sign.

By contrast S:79–89 and V:34–55 render four small Gabors with sigma4.5, radius12, wavelength5–7 pixels. Each renderer call independently redraws wavelength, amplitude (.28–.36), Gaussian noise (.008 SD), and each patch's carrier phase. Even the two nominal “sample” frames share **angles, not rasters**. Four checked sample pairs had MSE .00212–.00411 and none was pixel-identical. A raw pixel-difference detector therefore sees changes even at an unchanged target. The large sensory task already randomizes phase, so phase variation is not new; smaller support, higher carrier frequency, additional cross-frame frequency/amplitude nuisance, multiple locations, and cue-conditioned reporting are new jointly.

- **Ring orientation:** label is the sign at the ring-selected target; all patches rotate ±15/30/45°, foil signs independent (V:42–58). No unchanged-target negatives. This simpler spatial conjunction already fails.
- **Signed orientation:** label1 iff target rotation × cue sign >0; label0 means unchanged **or** opposite sign. Each scene includes aligned, unchanged, and opposite rotations; a sign-relative multiset is generated before target/label assignment (S:85–91). No global amount-of-change rule solves the intended label. The numeric assertion at S:90 agrees with returned labels; all checked episodes satisfied it. This is not the same binary head semantics as sensory/ring rotation sign.
- **Binding:** four distinct orientations separated by 45° modulo180 are shuffled over the locations. Exactly one pair swaps on every trial. Positive swaps involve the postcued target; negatives swap two foils while preserving target. Global orientation inventory and number of changed locations are identical (S:92–96). No-change/global-change detection is insufficient. The late cue makes all locations initially eligible.

### Pixel-level information check, not learned-model performance

For each of these three D0 tasks, the audit decoded the cue from native cue-frame pixels with fixed native templates, estimated sample/probe local orientation using gradient structure tensors, and applied the actual report rule. It used no label or metadata target as its input decision; it used known geometry/frame timing and then checked its decisions against labels. Each scored **32/32**. Maximum measured orientation-estimation error was .707° for ring and .367° for signed orientation. Binding's swap decision also scored32/32. These checks directly argue against raster ambiguity, complete cue destruction, or wrong labels in these sampled D0 episodes. They are not a neural learnability guarantee and are not a universal ideal-observer audit.

**Minor documentation defect, not operative label inversion:** V:6–12,58 calls positive delta “counter-clockwise”; P:110–112 uses different legacy clockwise wording. In downward-y image coordinates those verbal names are inconsistent. The executable positive/negative deltas, native labels, and catalog numeric sign conventions agree. The suite README explicitly warns against renaming numeric labels to reconcile the prose (`SecondPass/TaskSuite/README.md:41`). No label patch was applied. V:58's generic metadata prose is additionally inaccurate for the out-of-suite `orientation_sign` rung, whose actual label is sign-relative; that unused rung cannot explain this suite result.

## 4. Motion duration and Krauzlis are not the successful two-frame motion task

### Duration report

The sensory task uses one diameter85 aperture, 256 domain dots with half surviving, 1/2/3-pixel displacement, bilinear splatting plus Gaussian smoothing, and .2 background (P:23–29,49,61–101). Duration uses four radius11.5 apertures with32 dots each, .8/1.2/1.6-pixel steps, nearest-pixel rasterization, .5 background and .92 dots (S:97–111). Sixteen dots per patch are randomly replaced each transition, with additional boundary resets. Consequently visible displacement/contrast/spatial scale are not pixel-equivalent to the successful sensory renderer.

The correct label is the unique largest **count over eight transitions in the cued patch**. Foil patches have independently selected winners and schedules (S:104–111). W:98–111 draws first/last directions independently of the desired winner, then rejection-samples middle evidence until a unique winner is obtained. There is no off-by-one: one reference frame plus eight moving frames represents exactly eight transitions, followed by report. D0 totals11 frames, not a two-frame task.

In the fresh32-episode D0 check, 18 had a count margin of only1 transition,12 a margin2, and2 a margin3. Last direction equaled the correct winner in only6/32. These are descriptive draws, not a theoretical class chance estimate; they show why remembering only the end or reading one good motion transition is inadequate. Metadata schedules/counts/labels agree in every audited delay cell. **No pixel-level motion-duration oracle was run**, so the audit does not quantify attainable accuracy under dot replacement/rasterization. Sparse one-pixel dots and small steps are plausible sensory burdens in addition to counting/selection.

### Krauzlis report

S:124–142 implements16 dots/patch, .375-pixel steps with bilinear splatting,16° directional dispersion, finite dot lifetime/boundary replacement, and a26°/28° event. This is a subtle change of mean direction relative to baseline, not cardinal direction classification. The two patches initially have means90° apart; event can occur at target, foil, or nowhere. Dot offsets persist through the event unless dots reset. Baseline B12/B20/B28 counts motion transitions; postevent evidence always lasts8 transitions. There is no corner phase/event marker and no reaction-time choice—the only trained report is terminal.

Every100-case queue enforces57 target,29 foil,14 catch; the next cycle flips the odd subgroup's side assignment (S:69–73). Our200/cell checks returned114/58/28, labels86negative/114positive. Terminal always-positive behavior yields100% hits and100% false alarms on both negative subgroups. That pattern provides no evidence that the model distinguishes the event, let alone allocates detection to the precued side. A matched uncued event is intentionally a negative here; it must not be relabeled a detection hit.

## 5. Scene recognition: identity membership, not spectral discrimination

S:45–57 canonically center-crops/downsamples source RGB images to100×100 via bicubic interpolation and divides byte values by255. These are display-encoded RGB values; unlike natural spectrum, recognition does **not** decode to linear luminance, normalize mean/RMS, or use two variants of the same native crop. Source identities remain partitioned by official BSDS split.

S:112–120 samples N unique source scenes without replacement, then produces either an exact copy of a uniformly chosen studied scene or a different source scene. Positives are truly byte-identical rendered images, not semantic matches under transformed views. The probe is repeated H=3/4/5 times without change; these repetitions are one observation, not H independent episodes or H positive matches. Three blanks separate list from probe; no study/probe corner overlays spoil equality.

All sampled positive probe/study MSEs were exactly0, and the smallest negative nearest-study MSE across checked nonempty cells was .01601. Every probe repeat was equal. Membership truth and hashes agreed. This argues against erroneous nonmember/positive labels or accidental raster augmentation as the cause of failure.

Nonetheless the task requires storing identity information from **4,12,or24 distinct scenes**, each presented for one frame, and excluding probe self-repetition from the stored study set. There is no N1 acquisition rung and no zero-blank-delay recognition cell. The easiest nonempty task already has multiple images, a true blank gap, and repeated-probe processing. Finite training-photo reuse permits identity-specific memorization, whereas held-out source identities require generalizable matching; this audit does not measure which strategy was learned.

Natural spectrum instead presents two filtered versions of one native luminance crop with shared phase, common mean and RMS, and asks which has relatively greater high-frequency detail (N:136–191). Its success demonstrates neither semantic recognition nor list membership nor source-identity retention. Recognition's differences in resizing, color, photometry and sequence structure make a direct “it sees photographs already” inference invalid. These are deliberate renderer distinctions, not a discovered photometric implementation error.

All N0 labels are forced negative (S:116); correctly rejecting them establishes empty-set specificity, not memory. In nonempty cells labels balance exactly. If recognition cells are sampled equally, the three N0 conditions induce a task-wide negative prior of62.5%; that prior is not evidence of within-cell mislabeling. Frozen terminal N4_H3 has modest above-chance evidence, while other hold/load cells mostly lie near chance. These are independently seeded cells—not matched repeats of the same list—so H-dependent score differences cannot isolate overwriting, extra compute, or forgetting from these scores alone.

## 6. Adapter, balancing, order, and audit limits

`SecondPass/TaskSuite/suite.py:69–81` creates a distinct split/task/cell seed and chooses the appropriate native class; `:83–99` passes conditions unchanged, returns images/labels unchanged, and only appends analysis metadata. Exact native comparisons passed all35 cells in this audit, including all photograph cells. There is no adapter recoloring, reshaping, label remap, cue deletion, inserted padding, or altered frame order.

The suite has separate heads for sensory interval-index, numeric orientation sign, signed target alignment and membership (`catalog.json:24–518`; `suite.py:17–19`). External task ID chooses the head; it does not supply location, cue sign, phase or condition to the sensory input. Exact adapter equivalence alone would not expose a bug shared with its native renderer; that is why the label invariants, cue masks, actual membership checks and three pixel decoders were also run.

Label queues are concrete and not globally random imbalance: sensory labels balance within batches of class-count multiples (P:78–79,267–268); ring balances class×target over8 cases (V:29–32); signed orientation balances class×target×sign over16 cases (S:74–76); duration/binding balance class×target (S:77–78); Krauzlis has its intentional57/43 binary prior. All sampled cycles matched those laws. Frozen tests likewise have64/64 binary rows or32/class cardinal rows, except recognitionN0 and Krauzlis. Random difficulty/negative subtype mixtures need not balance in each small batch.

Historical `WorkingMemory/PlainBaseline/program.py:19–24,52–65` gives ring orientation a100,000-episode stage and gates progression at.90 before signed D0 and delay ladders. That program is not the current suite scheduler. It supplies a useful acquisition decomposition but does not justify assuming this jointly trained branch received or mastered those stages, and does not authorize rerunning them. This report makes no cross-recipe architectural superiority claim.

Remaining limits: no trained cue/phase/feature decoding, no causal lesion, no condition-matched delay test, no fitted psychometric curve, no multi-seed trained replication, no exhaustive photometric shortcut search, and no native-pixel motion/Krauzlis observer. We do not equate source-readable task phases with phases actually encoded by the network. We do not conclude “KDA cannot bind,” “ConvGRU cannot remember,” or “more training must fix it.”

## 7. Minimal decisive followups—not executed here

1. **Use the native D0 orientation/binding evidence as the first localization gate.** If new checkpoint diagnostics are authorized, on one fixed matched D0 batch distinguish (a) cue location/sign available, (b) local sample/probe orientation available, (c) target-conditioned report correct. Include ring and full signed task, because ring already fails and does not require the tiny sign difference. Test actual cue/feature use with controlled matched interventions rather than inferring it from a decoder alone. Only extend to genuinely stimulus-free delays after the shortest-condition computation is characterized.
2. **Separate small-patch motion evidence from count/selection.** First run a bounded nonneural native-pixel transition observer and aggregate its predicted directions under the existing count rule, with target versus foil and count-margin strata. For Krauzlis separately measure baseline/postevent direction estimates and target/foil/catch separation. This would test whether rendering noise is itself limiting without changing stimuli or training another architecture.
3. **For recognition, stratify saved trial predictions by studied-item position and load/hold if such trial records exist.** If only aggregate strata exist, an authorized fixed-checkpoint replay would be required. Compare matched probe histories/decision times to distinguish failure to encode a list from probe-period overwrite; do not infer that mechanism from independently seeded H cells. N1 or shorter blank controls would be new task conditions and require explicit approval.

The current evidence supports a narrower, actionable conclusion: **successful two-frame sensory comparison has not transferred to cue-conditioned spatial comparison and longer sequence report rules.** The native renderers supply coherent labels and visible evidence in the audited examples; the location of the learned computation's failure remains to be measured.
