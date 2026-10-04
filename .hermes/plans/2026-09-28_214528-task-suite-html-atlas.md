# Visual Task Suite Atlas — Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task if available; otherwise use the available delegation tools for independent work. Do not assume an unavailable skill exists. One Astra agent owns integration and verification. Do not commit or push unless the user asks.

**Goal:** Build a visually compelling, scientifically faithful HTML atlas of every task and native condition in the current visual attention / working-memory suite, with real animated GIFs, controllable frame playback, and optional complete explanations of stimuli, rules, objectives, implementation, neuroscience motivation, and neural-network demands.

**Architecture:** Export reproducible examples from the existing Python renderers on CPU, without modifying their behavior. Build a local-first static HTML application from a manifest containing task explanations, trial metadata, lossless frame assets, GIF previews, and citations. Keep the observer's actual input separate from explanatory graphics and answer annotations.

**Tech stack:** Existing Python/NumPy/Pillow/Torch CPU renderer dependencies; plain HTML/CSS/JavaScript; locally bundled KaTeX if needed. No frontend framework or cloud service is necessary. Confirm dependency versions before use; `SecondPass/TaskSuite/requirements.txt` and the renderer imports are authoritative.

**Status:** PLAN ONLY. This file does not implement the atlas, run rendering, or authorize any model experiment. Prepared from repository inspection on September 28, 2026 PDT. The live training run must remain untouched.

---

## 1. Handoff and scope

### User intent

The user wants another Astra agent to make an HTML demo while training continues. They explicitly want visually appealing GIFs of **all tasks and variants**, potentially running simultaneously, and a choice of full descriptions covering every property, rule and objective: how the task is implemented and what the model must do.

Deliver a working explanatory artifact, not a dashboard of training results, an architecture pitch, a slideshow of static thumbnails, or a literature review without demos.

### Non-negotiable boundaries

- **13 task IDs and 35 primary condition IDs**, as enumerated in `SecondPass/TaskSuite/catalog.json`.
- Include within-condition stimulus variants and outcome examples as well as the 35 primary cells; see the coverage contract below.
- Render the current generators. Do not replace tiny native stimuli with attractive but different synthetic illustrations, generative video, stock animation, reconstructed JavaScript stimuli, or smoothed/interpolated motion.
- Do not alter tasks, cues, class definitions, sampling distributions, supervision, model architecture, or training files.
- No checkpoint reads, inference, fitting, evaluation, accelerator use, RunPod calls, restarts, budget changes, or cloud deployment. This is CPU-only explanatory rendering and static-site work.
- Do not access or change runtime folders, training monitors, mirroring, stop guards, credentials, or saved experiment artifacts.
- Use at most two CPU threads for asset generation, one exporter process. Set the thread limits before importing numerical libraries. Keep rendering resumable and bounded; do not run an unbounded rejection sampler for a rare illustration.
- Use independent demo stream objects and a recorded demo-only seed namespace. Do not mutate global suite seeds or restore any live stream state.
- Use **train-split photographs** for public-facing examples, not validation/test photographs. Demo trials are curated illustrations, not evaluation data or representative empirical frequencies.
- No model performance claims, guessed attention maps, fabricated predictions, human capacity claims, or representation diagrams presented as recorded internal state.
- Preserve the dirty working tree. Scope writes to the new atlas subtree. Do not opportunistically repair historical renderer comments or old protocols.

### Design decision

**Primary surface: Explore. Secondary surface: Learn / Inspect.**

Make a scientific visual atlas: moving experimental scenes first, with a strong editorial reading experience beside each scene. Not a SaaS dashboard with decorative KPI cards. The top-level story is:

> What is shown? What is the correct answer? What computation makes the answer possible? Why is that a useful experimental question?

Choose one polished direction rather than asking the user to select among mockups: warm paper, dark ink, restrained blue for selection and muted orange for event annotations outside the stimulus; Newsreader-style serif for essays/headings, a legible sans for controls, monospace only for values/code. Use local font assets with clear fallbacks. Generous scene area; restrained borders; no gradient hero, glowing cards, arbitrary badges, or giant decorative numbers.

## 2. Authoritative context and source map

Read these before implementation. **Executable code wins over stale prose for current behavior.** Preserve any disagreement as a documentation note instead of silently changing the experiment.

| Purpose | Existing repository source |
|---|---|
| Exact task and condition inventory, labels, reporting strata | `SecondPass/TaskSuite/catalog.json` |
| Thin adapter / native stream dispatch | `SecondPass/TaskSuite/suite.py` (`SuiteStream`, `TASKS`) |
| Current suite narrative and caveats | `SecondPass/TaskSuite/README.md` |
| Existing suite test conventions | `SecondPass/TaskSuite/test_suite.py`, `SecondPass/TaskSuite/verify.py` |
| Seven sensory task implementations | `PreAttentiveVision/neuroscience_stimuli.py` (`TaskStream`, `CardinalMotionStream`) |
| Declared sensory numerical values | `PreAttentiveVision/sensory_battery_parameters.md` |
| Sensory literature starting points | `PreAttentiveVision/two_frame_task_research.md`, `cardinal_motion.md`, `krauzlis_stimulus.md` |
| Actual photo spectral transformation | `PreAttentiveVision/natural_stimuli.py` (`NaturalSpectrum.sample`) |
| Historical spectral motivation, not current implementation status | `PreAttentiveVision/natural_image_task.md` |
| Ring-cued auxiliary orientation task | `WorkingMemory/PlainBaseline/variants.py` (`VariantStream`) |
| Five spatial families and timing metadata | `WorkingMemory/SpatialTaskBattery/stimuli.py` (`SpatialBatteryStream`, `frame_count`, `local_cue`) |
| Spatial task specification and source pointers | `WorkingMemory/SpatialTaskBattery/PROTOCOL.md`, `SOURCES.md` |
| Imported visual vocabulary and duration schedule/oracle | Trace imports into `WorkingMemory/stimuli.py` and any further helpers; inspect actual definitions |
| Source image provenance / identities | `PreAttentiveVision/data/bsds500/manifest.json` and its README |
| Interpretation boundaries | `ANALYSIS_SOP.md` |

Inspect any existing HTML/reference design in the repository before styling, but do not let an older demo redefine the tasks. Read `AGENTS.md` and live git status at the start. Historical training authorizations and old scores are not content for this atlas.

### Known traps already identified

1. Some historical prose says tasks are proposed/unimplemented even though renderers now exist. Use current code.
2. Clockwise wording differs across legacy orientation comments. Use **positive/negative axial angle change** until the displayed grating-axis convention has been checked against the actual raster. A carrier wavevector and the visible stripe orientation are not interchangeable labels.
3. The chromatic task's numerical linear-RGB luminance constraint is not observer-calibrated isoluminance.
4. Contrast metadata describes intensity modulation amplitude, not automatically calibrated Michelson contrast.
5. Only the Krauzlis adaptation declares a physical 100 Hz frame clock. Other delays are frame counts, not invented milliseconds.
6. `orientation_ring` has only **D0** in this suite. Do not import other ladder tasks or give it extra delays because `VariantStream` can render them.
7. Independent delay-cell streams are not matched trials. If different examples appear at different delays, label them independent. Do not imply identical evidence unless raster/metadata equality is explicitly verified.
8. Photo spectral detail and photo recognition use different preprocessing and ask different questions. Neither is BSDS segmentation.
9. All frames, including blanks, are model inputs. Explanation overlays and metadata are not.
10. The externally selected task-specific head is supplied by the training harness. The model does not infer an unrestricted verbal task instruction from the website's explanatory text.

## 3. Exact coverage contract

### A. Primary cells — mandatory animated coverage

| Task ID | Primary condition IDs | Count | Native sequence length |
|---|---|---:|---|
| `motion_direction` | `mixed` | 1 | 2 frames |
| `orientation` | `mixed` | 1 | 2 frames |
| `contrast` | `mixed` | 1 | 2 frames |
| `spatial_frequency` | `mixed` | 1 | 2 frames |
| `chromatic_increment` | `mixed` | 1 | 2 frames |
| `contour` | `mixed` | 1 | 2 frames |
| `natural_spectrum` | `mixed` | 1 | 2 frames |
| `orientation_ring` | `D0` | 1 | 4 frames |
| `orientation_cued` | `D0`, `D4`, `D12`, `D24` | 4 | D + 4 |
| `motion_duration_cued` | `D0`, `D4`, `D12`, `D24` | 4 | D + 11 |
| `krauzlis_cued_motion` | `B12`, `B20`, `B28` | 3 | B + 17 |
| `spatial_binding` | `D0`, `D4`, `D12`, `D24` | 4 | D + 5 |
| `image_recognition` | `N0_H3`, `N0_H4`, `N0_H5`, `N4_H3`, `N4_H4`, `N4_H5`, `N12_H3`, `N12_H4`, `N12_H5`, `N24_H3`, `N24_H4`, `N24_H5` | 12 | N + 4 + H |

Generate this inventory directly from the catalog, not a second hand-maintained array. The build must fail on missing, duplicate or extra primary cells.

Each primary cell gets at least one actual downloadable animated GIF and one frame-exact interactive episode. Provide an example bank per cell: both labels where possible, all four answers for four-class tasks, all three event types for Krauzlis, and negatives only for empty recognition. A cell's showcase may cycle through multiple examples with clearly marked trial boundaries **outside** the native input. Do not concatenate episodes into a fictional continuous model trial.

### B. Native variants / explanatory examples — mandatory secondary coverage

“All variants” means every discrete native difficulty value and important rule/outcome category, with continuous property ranges fully documented. It does **not** mean inventing a finite list of every random angle, pixel-noise realization, or factorial combination. The interface must distinguish **Primary conditions**, **Stimulus variants**, and **Example trials**.

Cover each item below in the asset manifest. A single illustration may satisfy several requirements. Do not require an unnecessary full Cartesian product unless explicitly stated.

- Motion direction: right/up/left/down; displacements 1/2/3 pixels.
- Sensory orientation: positive/negative signs; magnitudes 4/10/22 degrees.
- Contrast: all nine increment × pedestal combinations: increments 0.025/0.06/0.13 and pedestals 0.08/0.18/0.30; both interval answers across the bank.
- Spatial frequency: all octave increments 0.08/0.18/0.35; both interval answers.
- Chromatic: increments 0.018/0.045/0.10; both interval answers, varied base colors.
- Contour: jitter SD 2/8/16 degrees; both structured-interval answers.
- Natural spectrum: beta deltas 0.15/0.30/0.60; both interval answers, several distinct source photographs.
- Ring orientation: four target locations; both signs; 15/30/45-degree magnitudes. D0 only.
- Cued orientation: every delay; both cue signs; all target locations; all magnitudes; aligned positive, unchanged negative, and opposite negative exemplars.
- Cued motion duration: every delay; all four answers and target locations; 0.8/1.2/1.6-pixel steps; at least one example whose last direction differs from the duration winner and one with a foil winner differing from the target. The native eight-transition schedule is unchanged.
- Krauzlis: each baseline length; target / foil / catch examples; both sides; 26/28-degree event magnitudes and both signs across event examples. Curated equal-looking examples must be explicitly distinguished from the native 57/29/14 event mixture.
- Binding: every delay; target-involved swap and foil-only swap; all queried locations across examples. Always one pair exchanged; do not add identity/no-swap negatives.
- Recognition: all 12 N×H conditions; positive and negative for nonempty lists; early/middle/late membership examples across the bank; empty lists never positive. Preserve exact raster membership and identical repeated probe frames.

Keep a machine-readable coverage map from requirement → concrete episode/GIF IDs. The final report states actual episode and GIF counts computed from the manifest. Do not advertise “every variant” if any required row is missing.

## 4. Experience and layout

### Landing gallery: see the experiment immediately

- A concise title and two-sentence introduction, followed immediately by **13 substantial moving task tiles** organized under sensory discrimination, spatial selection/comparison, and retention/recognition. These are navigational groups, not a claim that each task isolates one neural function.
- Each tile has a real native scene at a readable integer enlargement, task name, one precise question, current condition, and “Explore task”. Avoid tiny decorative GIFs above a large text block.
- Global controls: **Play all / Pause all**, playback rate, hide/show answers, and “All 35 conditions”. The latter opens the condition wall with a real animation for every cell.
- All visible tiles can run together. Use a shared animation scheduler for PNG playback; lazy-load offscreen assets, suspend when the page is hidden, and obey reduced-motion by starting paused. “Play all” must not secretly play only one selected tile.
- Make a static overview/poster available when animation is paused. No autoplay audio.

### Task detail: one experiment, understood completely

Desktop: large scene/player and timeline on the left, editorial explanation on the right. Below: condition matrix and full-width technical details. Mobile: scene first, then question/controls, then explanation. Permit a dedicated reader view with all descriptions expanded.

Controls:

- Native condition picker, discrete variant/example picker, previous/next example.
- Play/pause, replay from frame zero, previous/next frame, frame slider, speed control, and current `frame / T`.
- Clickable phase timeline driven by actual metadata, not inferred from brightness. Show exactly where the cue is present.
- **Observer view / Explanation view**. Observer view contains only native pixels. Explanation view may add outside-image arrows, locations, phase labels and computed evidence counters.
- **Try the judgment** mode: hide answer/annotation clues, collect a user's response, reveal the generator's answer and reasoning. This is informal exploration, not a calibrated psychophysical test. Do not present response speed as a scientific reaction-time measure.
- **Overview / Full description** control plus independent sections for rules, neuroscience, network demands, implementation/properties, and references. Also “Expand all” and print-friendly full descriptions.
- Download current GIF, PNG frames and trial metadata. Deep links encode task, condition and example; preserve state when switching reading modes.

Explanatory graphics should be task-specific:

- Ordered two-frame tasks: synchronized A/B filmstrip alongside sequential replay; clearly label the side-by-side panel an analysis display, not extra simultaneous model input.
- Cued tasks: mark location/sign and cue visibility in an external schematic without replacing the real glyph.
- Duration: four direction count bars advancing from actual schedule metadata in Explanation view, plus a final-direction versus duration-winner contrast. These are ground truth, not decoded model evidence.
- Krauzlis: an explanatory event line and target/foil labels; event marker appears only with explanations enabled. Native observer pixels have no event announcement.
- Binding: sample/probe correspondence diagram and swapped pair after answer reveal. Before the retrocue, no external target highlight in Observer view.
- Recognition: study filmstrip with serial positions, explicit three-blank segment, repeated probe counter, and matching study item revealed only after the answer. Do not skip the long N24 study list to make a prettier loop.

## 5. Animation and photometry contract

**Use lossless frame playback as the authoritative in-browser experience; also deliver real GIFs.** GIF alone is inadequate for frame stepping, pause-all, photometric comparisons and short repeated frames.

1. Preserve each float32 native frame and metadata at export, with source/raster hashes. Save lossless PNG display frames and deterministic conversion details. Save canonical float arrays in the development artifacts if needed for verification, not necessarily in the lightweight delivery bundle.
2. Native raster remains exactly 100×100 RGB. Enlarge with integer nearest-neighbor scaling by default; optionally provide clearly labeled smooth display enlargement. Never crop, denoise, sharpen, regenerate dots, alter contrast, or rescale patches within the image.
3. Do not insert tween frames, optical flow, transitions or crossfades between actual stimuli. Those create sensory evidence that the task never supplied.
4. Keep titles, answers, timelines and decorative framing outside the 100×100 stimulus rectangle. Store raw stimulus assets separately from annotated showcase GIFs.
5. Two-frame tasks remain two observations. A loop repeats the *trial*, with an external restart indication. Never present the loop seam as a valid additional motion transition. No added blank is a model input.
6. Preserve every repeated frame and its semantic frame index. GIF encoders may merge identical frames into longer durations; maintain a source-frame timing map and test the decoded timeline, not just encoded `n_frames` equality. PNG playback always exposes each original index.
7. Default playback is readable **illustration speed**, clearly labeled as such. Delay frames remain individually represented; optionally allow a labeled navigation jump over delay, but never export a shortened trial as native.
8. For Krauzlis, display both the nominal 10 ms/frame source clock and the chosen demonstration slowdown. Browsers/GIF readers can clamp very short delays; do not promise calibrated 100 Hz presentation. Other tasks have no assigned biological seconds.
9. Use one stable GIF palette across the frames of a trial to avoid palette flicker. Disable dithering for scientific previews unless it is explicitly inspected and justified. Measure uint8 conversion and GIF quantization errors. Prefer lossless PNG viewing for subtle contrast/chromatic judgments; a GIF is an illustration, not an exact photometric reproduction.
10. Declare the display mapping. Raw tensor-value preview should use one consistent mapping without per-frame normalization. If a linear-to-sRGB explanatory mode is added, label it and apply it identically to all frames; never call browser colors calibrated stimuli.
11. Select readable examples from the native distribution and provide the harder levels too. Record any curation/filter. If a required example is not found within a finite sampling limit, emit a coverage failure rather than silently changing the generator or fabricating metadata.
12. Set and report asset size/performance targets after a pilot export. Avoid embedding every frame as a huge base64 HTML document. Ship a folder/ZIP containing relative assets; no remote requests should be needed to explore it.

## 6. Required content for every task

Full description is substantive connected prose, not a few recycled bullets. Write at three levels: a one-sentence question on the tile; a short accessible explanation beside the player; then a complete scientific/technical account. Equations supplement prose rather than replace it.

Every task must contain:

1. **What appears:** all objects, positions, relevant photometry, motion/stimulus statistics, and visual vocabulary.
2. **Sequence:** each phase, exact indices/durations in frames, cue persistence, reference versus transition frames, report time, and repetitions.
3. **What to report:** exact classes, mapping from numeric labels to meaning, verbal rule and mathematical rule when useful.
4. **What to ignore:** distractors, nuisance variation, irrelevant events, and what a seemingly plausible wrong answer would be.
5. **Properties:** condition controls, every discrete value, continuous ranges/distributions, shared versus independently drawn variables, fixed constants and derived values. Each row has name, value/range, units, sampling law, role, and code provenance. Separate model-visible quantities from analysis-only metadata.
6. **Implementation:** actual module/class/helper, preprocessing, deterministic stream behavior, label construction, shortcuts controlled by the generator, and its remaining limitations.
7. **Neuroscience motivation:** the relevant scientific question and primary evidence; identify what is inherited from a paradigm and what is a project-specific adaptation.
8. **Neural-network motivation:** what input information must survive, what comparisons/selection/integration are required, potential computational failure modes, and what alternatives could solve it. Do not prescribe a unique module or claim “this requires a ConvGRU”.
9. **Interpretation boundary:** what success could support and what it would not establish. Distinguish computational demand from measured acquisition and biological mechanism.
10. **Metrics, as definitions only:** class chance levels, BA/AUC where applicable, and necessary subgroup reporting. No fabricated scores or stale run results.
11. **Worked example:** explain the currently selected trial using its actual metadata, including why the alternative answer is wrong.
12. **Sources:** inline references for claims, source notes, adaptation statement, and exact repository provenance.

Trace imported helper defaults as well as top-level task parameters. Do not claim the property table is exhaustive simply because all catalog `report_by` fields were copied: the catalog is not the full generator specification. Include noise, phases, normalization, lifetime/reset behavior, gamut/clipping and source preprocessing where applicable.

## 7. Task-specific editorial brief

These are the required explanatory directions and code-grounded starting facts, not already completed or independently re-verified literature summaries. The implementing agent must verify primary-source claims before publication.

### 7.1 `motion_direction` — ordered motion correspondence

- **Rule:** two frames; choose right/up/left/down in class order 0/1/2/3. Displacement is 1/2/3 pixels. Explain a periodic domain of 256 dots, 128 surviving identities, 128 reborn identities, and a circular aperture; the visible count is not always 256. Surviving dots share the selected direction.
- **Neuroscience discussion:** random-dot direction discrimination and the distinction between local motion evidence, dot correspondence, and coherent-motion manipulations. Start from `cardinal_motion.md` and its primary sources; do not conflate this two-frame adaptation with the later target/foil change task.
- **Network discussion:** order-sensitive spatiotemporal comparison and nuisance tolerance; a swap-symmetric representation cannot retain the signed direction information needed here. Two frames do not establish long-delay working memory.
- **Worked illustration:** show both original frames, the actual answer, and an optional external displacement diagram. Never draw identity-matching lines into the input image.

### 7.2 `orientation` — signed axial comparison

- **Rule:** positive versus negative local axial change; magnitudes 4/10/22 degrees, roving base angle, independent carrier phases. Explain axial periodicity modulo 180 degrees and use the doubled-angle wrapped difference rather than naïve subtraction across the boundary.
- **Neuroscience discussion:** controlled Gabor orientation discrimination; begin with Zhang et al. in `two_frame_task_research.md`. Separate their presentation protocol from this two-frame model input.
- **Network discussion:** retaining orientation despite phase changes, ordered comparison, and the distinction between feature coding and performing the signed decision. No claim of calibrated human acuity.

### 7.3 `contrast` — preserve amplitude information

- **Rule:** select frame 0 or 1 with the greater modulation amplitude; increments 0.025/0.06/0.13 and pedestals 0.08/0.18/0.30. Shared grating pattern within a pair; amplitude differences must not be normalized away.
- **Neuroscience discussion:** pedestal-dependent contrast discrimination / masking; begin with Legge and Foley from the research note. Explain why a pedestal is different from detection against a blank.
- **Network discussion:** normalization can remove a task-relevant amplitude difference; energy is a legitimate solution here. This low-level capability is not contour grouping, recognition, or attention. Label the numerical amplitude convention accurately.

### 7.4 `spatial_frequency` — retain spatial scale

- **Rule:** choose the higher-frequency interval; `f_high = f_low × 2^delta`, delta 0.08/0.18/0.35 octaves. Lower frequency 4–9 cycles/image; envelope unchanged; independent phases.
- **Neuroscience discussion:** frequency-ratio discrimination; begin with Campbell, Nachmias and Jukes in the research note.
- **Network discussion:** spatial resolution, filtering, pooling and aliasing; compare this narrow-band task with broadband natural-spectrum discrimination. Changing carrier frequency is not resizing the entire patch.

### 7.5 `chromatic_increment` — chromatic information beyond grayscale

- **Rule:** identify the positive increment along the declared linear-RGB axis proportional to `(0.7152, -0.2126, 0)`, normalized to unit length; delta 0.018/0.045/0.10. Numerical luminance weights are `(0.2126, 0.7152, 0.0722)`.
- **Neuroscience discussion:** controlled chromatic discrimination and the reason luminance must be considered; begin with Krauskopf and Gegenfurtner. Explicitly state that this is not a cone-isolating stimulus or observer-calibrated isoluminance.
- **Network discussion:** retaining signed chromatic differences rather than discarding color. A color-mean solution is valid, not evidence of color constancy or object recognition. Add the display/GIF quantization caveat beside the scene.

### 7.6 `contour` — relations among local features

- **Rule:** select the interval generated with an aligned seven-element path among 32 Gabor elements. Jitter SD 2/8/16 degrees. The comparison reassigns an orientation multiset among matched positions; explain the actual carrier/path-tangent convention from the code rather than drawing a guessed contour.
- **Neuroscience discussion:** association-field/contour integration paradigm; begin with Field, Hayes and Hess.
- **Network discussion:** sensitivity to relations among orientations and locations beyond isolated feature statistics. Discuss receptive-field extent and grouping as possible computational concerns, not proof recurrence is necessary.
- **Boundary:** random controls can contain accidental alignment and partially retained paths. Label means structured generator, not guaranteed human perceptual visibility. Do not claim grouping is exclusively preattentive.

### 7.7 `natural_spectrum` — broadband detail without semantic recognition

- **Rule:** choose the interval with smaller spectral beta, hence greater relative high-frequency weighting. Same source crop and Fourier phase; beta deltas 0.15/0.30/0.60; common mean and pairwise common RMS.
- **Implementation:** use `NaturalSpectrum.sample`, not the old proposal text: source sRGB is converted to linear luminance; a native 100×100 crop is selected; quarter-turn/reflection can be shared; radial spectral filtering is applied; mean and shared RMS/gamut handling are preserved. Document these separately from recognition's central RGB crop.
- **Neuroscience discussion:** second-order image statistics and scale-dependent contrast; begin with Tadmor and Tolhurst. Tajima and Okada is computational background, not new human experimental evidence.
- **Network discussion:** preserving broadband spatial-scale information under nuisance image content; normalization should not erase the signal. Not scene semantics, BSDS segmentation or item recognition.

### 7.8 `orientation_ring` — selection without the sign rule

- **Rule:** four Gabors, ring indicates the target, report positive/negative target angle change; D0 only; 15/30/45-degree magnitude. Cue appears in instruction/sample frames and is absent at probe; verify actual metadata.
- **Neuroscience discussion:** spatial selection among competing local features. Explain this as a project-designed simplification, not a separately validated biological paradigm.
- **Network discussion:** bind a location cue to the correct local comparison while ignoring other locations. Contrast explicitly with the single central sensory orientation task and the location-plus-sign task below.
- **Boundary:** no clean delayed-retention claim at D0; model frame stacking can expose sample and probe together. Do not import `orientation_single` or `orientation_sign` into the official task count.

### 7.9 `orientation_cued` — location × signed rule × comparison

- **Rule:** a translated plus/minus glyph indicates target location and relevant sign. `y = 1[c × Δθ_target > 0]`; positives align with the cue; negatives are unchanged or opposite. Magnitudes 15/30/45 degrees. Delay D0/4/12/24.
- **Sequence:** cue 1, sample 2, D blanks, probe 1. The location/sign glyph is in cue **and sample** frames, not only a momentary precue.
- **Neuroscience discussion:** task-dependent relevance, selective feature comparison, maintenance of information across a delay. Start with the spatial sources and a verified primary study relevant to rule-dependent selection; distinguish custom sign glyphs from a replicated human/primate protocol.
- **Network discussion:** combine sign instruction, spatial target and a signed comparison. Global change detection is insufficient because target/foil change multiset controls are built into the generator.
- **Worked examples:** same conceptual rule across aligned / unchanged / opposite cases; show actual sampled evidence, not fabricated paired counterfactuals.

### 7.10 `motion_duration_cued` — selective temporal accumulation

- **Rule:** ring identifies one of four dot patches. Answer the unique most frequent direction among its eight motion transitions: `argmax_k sum_t 1[d_t=k]`. No ties. Delay D0/4/12/24.
- **Sequence:** cue 1, reference 1, moving frames 8, D blanks, report 1. Ring remains through the reference and moving frames, not the delay/report.
- **Implementation:** 32 dots/patch, radius 11.5 pixels, step 0.8/1.2/1.6; 16 random replacements plus boundary resets per transition. Do not borrow the Krauzlis lifetime rule.
- **Neuroscience discussion:** evidence integration and selecting a relevant spatial stream; find and verify suitable primary evidence-integration literature. Be explicit that majority-of-discrete-directions is the project's engineered objective, not a claim to reproduce a named accumulator experiment.
- **Network discussion:** duration counts, not final direction, net displacement, or pooled foil votes. Retain selected evidence through blanks; successful performance need not imply an explicit symbolic counter.

### 7.11 `krauzlis_cued_motion` — target change versus irrelevant change

- **Rule:** target direction change is positive; foil-only change and catch are negative. Baseline B12/B20/B28; eight post-event transitions. Native mixture 57% target, 29% foil, 14% catch.
- **Sequence:** 2 cue frames, 5 fixation-only, 1 dot reference, B baseline transitions, 8 post-event, 1 report. Explain nominal 100 Hz and the demonstrator slowdown separately.
- **Implementation:** two opposite patches; means differ by 90 degrees; 16 dots/patch; 16-degree Gaussian direction dispersion, not a coherent-dot percentage; 10-frame lifetimes; 0.375-pixel motion step; signed 26/28-degree events. Inspect exact subpixel rendering and birth/boundary rules.
- **Neuroscience discussion:** Arcizet and Krauzlis, “Covert spatial selection in primate basal ganglia,” DOI `10.1371/journal.pbio.2005930`, methods “Motion-direction CD task” and Figure 1; verify directly. Keep this recipe separate from Lovejoy/Zénon studies referenced elsewhere.
- **Network discussion:** carry brief cue identity into later motion, compare pre/post evidence at the selected patch, and reject an equally real foil event.
- **Boundary:** shortened timing and final binary report are not joystick withholding or reaction-time behavior. Define target hit rate, foil false alarms and catch false positives separately, without displaying invented measurements.

### 7.12 `spatial_binding` — remember which orientation was where

- **Rule:** four orientations from a random axial base plus 0/45/90/135 degrees, assigned to four locations. A query after the delay identifies which location to judge. Positive: target participates in the exchange; negative: two foils exchange while target stays put. Exactly two locations change in both classes; global inventory is unchanged.
- **Sequence:** instruction 1, sample 2, D blanks, query 1, full-array probe 1. No advance target cue.
- **Neuroscience discussion:** feature-location binding and retrospective selection from working memory. Start with Luck and Vogel in `SOURCES.md`; add a directly verified retrocue source rather than citing Luck/Vogel as if it used this exact rule.
- **Network discussion:** location-conditioned correspondence, retention before knowing which site will matter, and retrieval/comparison. Inventory-only memory or a global changed-pixel total is inadequate by design.
- **Boundary:** this assay does not establish four discrete biological slots, a human capacity number, or a uniquely identified binding mechanism.

### 7.13 `image_recognition` — exact set membership over a visual list

- **Rule:** report whether the probe is an exact member of the preceding study-image set. N0/4/12/24 and H3/4/5 give 12 cells. Empty sets always negative.
- **Sequence:** instruction 1, N consecutive study frames, 3 blanks after the entire list, H identical probe frames. No interleaved study blanks; no hidden skipped samples.
- **Implementation:** official BSDS500 source-identity splits 200/100/200; use train only for the demo. Canonical central square crop, bicubic resize to 100×100 RGB. Positive probe is bit-identical to a study item; list has unique source identities; negative probe is outside the list. No glyph overlay on study/probe images.
- **Neuroscience discussion:** recognition, interference and serial-position questions, with a verified primary memory reference. Describe exact-image membership and repetition as this project's computational adaptation, not a direct reproduction of human timing or abstract semantic recognition.
- **Network discussion:** membership across multiple recent inputs, false familiarity, interference, and retaining early versus late items; distinguish repeated probe exposure from additional study items or independent trials.
- **Boundary:** N0 uses specificity/false-positive rate, not BA/AUC. The finite photo pool and curated examples are not a capacity estimate or a generalization result.

## 8. Scientific sourcing standard

The prose above specifies the intended discussion; it does not waive literature checking.

- Build a task-by-claim reference ledger. For each source: authors/title/year, DOI or stable URL, primary versus review/model, exact supporting section/figure/page, what the source supports, and which implementation choices it does **not** support.
- Existing repository source notes are **starting points**. Retrieve and read the primary methods/results before attributing specific neuroscience facts. If access is blocked, qualify what was verified; do not turn a search snippet into a methods claim.
- Give every task at least one relevant primary scientific anchor. Related tasks may share a source when the claim truly matches. Custom rules should say “project-specific computational assay motivated by …”, not pretend they were published protocols.
- Separately source claims about neural-network phenomena if stated as empirical facts. Explain logical information requirements as such; hypothetical bottlenecks are hypotheses, not diagnoses of the current model.
- Link citations inline beside the motivating claim, not only in a giant footer. Add a complete bibliography and a short “adapted here” paragraph per task.
- Reference photo provenance and rights without inventing permission to redistribute. Ship locally for this request; verify asset redistribution terms before any public deployment. No public deployment is requested.
- No assertion that these tasks prove Guided Search 6.0, human-like attention, biological memory capacity, neural localization or a particular recurrent mechanism. If GS6 is mentioned, describe it as a functional scaffold, omit its diffuser, and distinguish the user's activated-LTM-as-weights approximation from Wolfe's claims.

## 9. Proposed file layout and data contracts

All paths below are **new planned artifacts**, not files that already exist.

Root: `SecondPass/TaskSuite/Demo/`

- `README.md` — build/open instructions, scope, provenance, animation/photometry limitations.
- `DESIGN.md` — composition, type/color/spacing tokens and responsive interaction rules.
- `content/tasks.json` — structured explanations, equations, property tables, rule definitions and source IDs for all 13 tasks.
- `content/sources.json` — verified reference/claim/adaptation ledger.
- `export_assets.py` — CPU-only deterministic native renderer wrapper and resumable export.
- `build_site.py` — validate inputs and build portable static output, not a server dependency.
- `validate_assets.py` — coverage, hashes, labels, timeline, GIF/PNG and link validation.
- `tests/test_manifest.py`, `tests/test_export.py`, `tests/test_build.py` — focused CPU tests without checkpoint/model work.
- `web/index.html`, `web/atlas.css`, `web/atlas.js` — authored application/template and player code.
- `artifacts/manifest.json`, `artifacts/coverage.json`, `artifacts/build_receipt.json` — generated reproducibility/coverage evidence.
- `artifacts/raw/` — optional canonical arrays/metadata for development verification; exclude bulky arrays from user ZIP unless needed.
- `dist/index.html`, `dist/atlas.css`, `dist/atlas.js`, `dist/data.js` — built portable site. Use a generated data script rather than mandatory `fetch()` so opening `file://.../index.html` works.
- `dist/assets/frames/`, `dist/assets/gifs/`, `dist/assets/posters/`, `dist/vendor/` — relative local dependencies.
- `verification/` — browser screenshots, contact sheets, validator output and concise final QA report.
- `task-suite-atlas.zip` — only built delivery files plus readable instructions/provenance, no repository or runtime secrets.

Manifest schema must include task ID, condition ID and exact kwargs; example ID, demo seed and native trial ID; source split and photo IDs where relevant; numeric label and label meaning; full native metadata; native frame count and indices; phases/cue visibility; source file hashes and dataset-manifest hash; float/display conversion settings; display frame hashes and asset paths; playback timing and GIF frame mapping; curation reason; covered variant IDs; and content/source references.

Treat metadata as explanation data. The UI must not imply it was supplied to a deployed model. Distinguish renderer values from analyst-derived fields explicitly. If a property cannot be derived safely from recorded metadata, expose the limitation rather than guessing it.

For demo-only seeds, use fresh native streams dispatched using the same catalog/adapter conventions with a documented fixed namespace; do not modify `CATALOG`. Preserve the exact `train` photo split. Test wrapper/native equality for identical seed, task and condition. Using `SuiteStream('train')` alone with its default production seed namespace is not the intended isolation strategy.

## 10. Implementation sequence

Each item has a concrete completion check. Add failing tests before code for the exporter/build contracts; no commits without user instruction. Independent literature drafting and UI styling can proceed in parallel after the inventory is locked, but only one owner writes shared manifests.

### Task 1 — inventory and boundaries

**Files:** create `README.md`; inspect the existing sources in Section 2.

- Record current catalog/version/hash, 13 task IDs, all 35 cell IDs, and the allowed write subtree.
- Trace renderer imports, phase/glyph helpers and dataset paths.
- Produce a property checklist per task, including uncertain sign/photometry wording to resolve.
- Verify no runtime/training code is imported for rendering.

**Pass:** exact catalog match; no unresolved task inventory; no training process touched.

### Task 2 — manifest/coverage tests

**Files:** create `tests/test_manifest.py` and planned coverage validation in `validate_assets.py`.

- First tests reject missing/duplicate/unknown primary cells, invented ring delays, absent outcome examples, and incomplete native difficulty coverage.
- Assert all properties/description sections/reference IDs exist for each task.
- Enumerate required variants programmatically from explicit reviewed requirements.

**Pass:** tests first fail against missing data, then pass with valid fixture manifests; deliberate corrupt fixtures still fail.

### Task 3 — export one representative episode per renderer path

**Files:** create `export_assets.py`, `tests/test_export.py`.

- Start with one sensory grating, motion, ring orientation, a spatial delayed task and a photo task.
- Use fresh CPU streams and record the exact native metadata without modification.
- Export lossless frames, GIF and canonical hashes; verify native equality and deterministic replay.
- Resolve display/sign orientation from actual pixels before writing clockwise descriptions.

**Pass:** dimensions, range, order, labels, native frame counts, repeated-frame semantics and conversion errors verified; no CUDA/MPS or checkpoints.

### Task 4 — bounded complete example bank

**Files:** extend `export_assets.py`; generate `artifacts/manifest.json` and `coverage.json`.

- Build all 35 primary cells and the secondary coverage bank.
- Persist each cell batch; resume by matching source/config hashes, never trust a partial file.
- Use finite sampling limits for curated examples; emit a clear missing-coverage list on failure.
- Generate condition contact sheets plus representative phase strips.

**Pass:** zero missing primary/secondary requirements; actual counts and assets exist; no fabricated frames or labels.

### Task 5 — content and references

**Files:** create `content/tasks.json`, `content/sources.json`.

- Write the 13 accessible introductions and 13 complete descriptions using Section 7.
- Populate all property tables by tracing code and helper defaults.
- Verify primary-source claims and link each adaptation statement.
- Check class definitions and worked-example explanations against native metadata.

**Pass:** no placeholder paragraphs, no unsupported biological equivalence, and no copied generic motivation repeated for unrelated tasks.

### Task 6 — frame player first

**Files:** create `web/atlas.js` plus minimal `web/index.html`.

- Implement actual per-frame playback, slider, stepping, replay, speed, observer/explanation separation and per-example answer reveal.
- Use one scheduler, robust visibility handling and explicit loop boundaries.
- Test repeated recognition frames, D24 blanks, two-frame seams and Krauzlis event position.

**Pass:** every source frame reachable; timeline/index agree with metadata; pause actually freezes all players.

### Task 7 — polished gallery and condition wall

**Files:** create `DESIGN.md`, `web/atlas.css`; extend HTML/JS.

- Build the 13-task overview and all-35-condition view with native animations.
- Build task detail/reading views, full-description choice, native condition/variant controls and deep links.
- Add keyboard focus, reduced-motion, mobile layout, download links and print styling.

**Pass:** all visible scenes can run together; every task/cell reachable; no clipping or hidden detail sections; visual hierarchy emphasizes scenes and scientific explanation.

### Task 8 — portable build and integrity checks

**Files:** create `build_site.py`, `tests/test_build.py`; finish `validate_assets.py`.

- Bundle local data/fonts/KaTeX as required; all relative paths work when moved.
- Validate all manifest URLs, GIF downloads, deep links, citations and local assets.
- Produce source/dataset/build hashes and actual coverage totals.

**Pass:** direct-file and localhost opening both work without an internet connection; build is reproducible apart from explicitly recorded timestamps.

### Task 9 — scientific and visual QA, then delivery

**Files:** generated `verification/`, `artifacts/build_receipt.json`, delivery ZIP; update Demo README only.

- Programmatically validate every asset/cell; visually inspect contact sheets for every task, and animation playback for all 35 conditions.
- Inspect the full range of sensory difficulty, all three Krauzlis events, target/foil binding swaps, and empty/long recognition.
- Browser-test gallery, detail view, all-conditions wall, play-all/pause-all, full descriptions, answer hiding, mobile and keyboard controls. Check console errors and missing resources.
- Inspect screenshots at desktop, tablet and phone widths; check prose length, scene sizes, color contrast and overflowing equations/tables. Fix before delivery.
- Summarize actual tests and limitations; no guessed pass counts.

**Pass:** all acceptance criteria below verified and the site opens from the delivered folder/ZIP.

## 11. Planned verification commands and evidence

These are implementation targets, **not commands already run**. The implementing agent must make these entry points exist as specified, choose an existing compatible Python environment, and report actual output.

From repository root, with CPU thread limits set:

- `python -m pytest SecondPass/TaskSuite/Demo/tests -q`
- `python -m SecondPass.TaskSuite.Demo.export_assets --all --device cpu --threads 2`
- `python -m SecondPass.TaskSuite.Demo.validate_assets --all`
- `python -m SecondPass.TaskSuite.Demo.build_site`
- `python -m http.server 8765 --bind 127.0.0.1 --directory SecondPass/TaskSuite/Demo/dist`

Check whether namespace-package invocation fits repository conventions; add only the minimal Demo-local package marker if required. Do not modify upstream package initializers merely for convenience. Port 8765 is an example: check availability; never terminate an unrelated service.

No model-smoke tests, whole historical test suites that load checkpoints, or fresh inference are required. Existing renderer-specific tests may be selected only after inspecting their bodies for side effects.

### Acceptance checklist

- [ ] 13 unique task descriptions; 35 unique primary condition assets; exact catalog ID match.
- [ ] GIF coverage for every primary cell and every required variant example; actual counts derived from manifest.
- [ ] All original input frames and native sequence lengths preserved, including dot reference frames and repeated probes.
- [ ] Native pixels separate from explanations; neither task labels nor cue/event answers leak into Observer / Try mode.
- [ ] Full property/rule/objective/implementation descriptions available for every task, including nuisance distributions and source helpers.
- [ ] Neuroscience and neural-network motivations are separate, substantive and appropriately sourced/qualified.
- [ ] Numeric sign, contrast convention, display photometry and timing caveats resolved/documented.
- [ ] Empty recognition and Krauzlis subgroup semantics correct.
- [ ] No extra tasks, conditions, biological time conversions, or alleged model abilities invented.
- [ ] Play-all, pause-all, controls, deep links, reader view, downloads and reduced-motion verified in a real browser.
- [ ] Color/contrast-sensitive examples checked against lossless previews; GIF error documented.
- [ ] Train-only source photo usage, deterministic isolated demo streams, source/dataset hashes and coverage receipt saved.
- [ ] Offline/direct-file operation and relocated ZIP verified; no external API requirement.
- [ ] Training/cloud/runtime state untouched; no checkpoint, accelerator or optimization work performed.

## 12. Risks and decisions already made

- **GIF accuracy versus visual appeal:** retain authentic inputs; polish typography, framing, timeline and annotation, not the stimulus. PNG playback is authoritative, GIF is an export/preview.
- **All-at-once cost:** use a shared lossless player, visibility-aware loading and global pause. Measure rather than promising an arbitrary frame rate on all machines. Do not silently drop conditions for performance.
- **Subtle signals:** offer enlargement, A/B inspection and worked explanations; do not amplify native contrast or motion without labeling it as a separate schematic. No altered stimuli are necessary for this deliverable.
- **Long lists:** preserve all frames, show a useful filmstrip and progress cue, provide navigation. Never shrink N24 into a handful of images.
- **Scope explosion:** cover all primary cells and all declared discrete levels/categories; document continuous ranges. No exhaustive nuisance Cartesian product or new scientific experiment.
- **Primary literature access:** preserve a claim-level verification status. Missing full-text access is a sourcing limitation, not permission to invent methods details.
- **Concurrent work:** the training run owns its source/runtime state. Demo agent only reads existing generator code and writes the new subtree; stop and report if another agent is modifying those same generator files.
- **Publication:** local delivery only. Recheck image rights before any later public hosting.

## 13. Final delivery to the user

Return a short report with:

1. The actual HTML entry point and downloadable ZIP.
2. Verified task/primary-condition/variant-example/GIF totals.
3. What can be played together and how to reveal full descriptions.
4. The real test/browser verification results and any remaining limitations.
5. Confirmation that no model/training/cloud changes were made.

Use `MEDIA:` for downloadable files and the app's preview mechanism if suitable. Do not return only a screenshot or instructions to implement later. The implementing agent's finished deliverable is the working, inspected atlas.

### Ready-to-send prompt for the implementing Astra agent

> Implement `.hermes/plans/2026-09-28_214528-task-suite-html-atlas.md` in `/Users/jonathanmorgan/Desktop/VisualAttentionWorkingMemory`. Build the CPU-rendered, local-first HTML atlas in `SecondPass/TaskSuite/Demo/`, with native GIFs and frame-exact players for all 13 tasks / 35 conditions plus the specified native variants, full optional task descriptions, verified scientific motivations, and a polished editorial gallery. Read current code and project instructions before acting; preserve all existing work. Do not touch training, RunPod, checkpoints, runtime infrastructure, model code or stimulus semantics. Do not commit or publish. Finish with real asset/coverage tests and browser inspection, and deliver the working HTML folder/ZIP with concise verification evidence. Treat this as a task-suite explainer, not a report of model competence.
