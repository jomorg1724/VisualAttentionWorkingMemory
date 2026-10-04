# Independent scientific-content and coverage review

## Verdict

**One localized factual correction is required before claiming scientifically correct finished prose. No missing task, primary condition, required coverage family, or outcome-bank requirement was found.** The correction concerns the signed-orientation introduction, not the renderer or exported labels. A separate low-priority reproducibility wording issue is noted below.

Scope: read-only inspection of all 13 task descriptions, their property tables/equations/reference ledgers, native source and imported stimulus helpers, exporter/validator/UI source, manifest, coverage map and saved source-access evidence. Independently recomputed coverage and metadata rules with standard-library code; read existing recognition float-array bytes to check membership and repetition. No renderer was imported or executed; no model, checkpoint, training, runtime infrastructure, browser rendering, evaluation or network retrieval was used. This is not a browser/photometric QA certificate or an independent native replay. Only this report was written.

Paths below are relative to `SecondPass/TaskSuite/Demo/` unless prefixed with `repo:`. The reviewed plan is the supplied `pasted_content_2026-09-29_04-56-28-698_6c7a40.txt`, especially lines 71–109, 153–170 and 243–252.

## Required correction

### SCI-1 — The signed-orientation introduction incorrectly guarantees all three categories among the foils

- **Location:** `content/spatial_tasks.json:268` says: “Other locations include aligned, opposite and unchanged orientations in every trial”.
- **Native fact:** the **entire four-location array**, including the target, contains those categories. The three other locations need not. `repo:WorkingMemory/SpatialTaskBattery/stimuli.py:86–91` constructs `[-1,0,1,r]`, removes the selected target entry, and permutes the remainder onto foils. The longer explanation correctly describes the whole-array control at `content/spatial_tasks.json:283`.
- **Concrete counterexample:** `orientation_cued--D0--00000`, `artifacts/manifest.json:21367–21440`: cue sign `+1`, target location `2`, rotations `[0,0,45,-45]`. Its foils are unchanged/unchanged/opposite: **no aligned foil exists**. Independently checking the bank found that 11 of its 16 signed-orientation episodes lack at least one category among the foils.
- **Fix:** replace the sentence with “Across the whole array, aligned, opposite and unchanged orientations occur in every trial, so merely detecting a change cannot answer the question.” Do not alter generator, metadata or assets. Rebuild propagated content after the editorial correction. This introductory text is actually used by `web/atlas.js:202`, so this is not an unused comment.

## Non-blocking documentation correction

### DOC-1 — “No restoration performed for the atlas” is too broad

`content/sensory_tasks.json:80,301,529,772,1010,1240,1518` and the natural-spectrum implementation paragraph at `:1459` deny atlas stream restoration. However, `export_assets.py:97–105,259–270` explicitly supports restoring **isolated demo** stream state, and `verification/export-report.md:37–40` records round-trip and interrupted-resume tests. This is not evidence of accessing live training state; those are different things. Prefer “No live training stream state is restored; isolated demo progress may be restored for resumable export.” This affects implementation provenance wording, not scientific labels or coverage.

## Independently verified coverage against the plan

The manifest catalog equals the current repository catalog. Its cell IDs/kwargs match exactly: **13 tasks, 35 unique primary cells, 115 unique episodes, 115 unique existing GIF paths, 1,580 native PNG frame indices**. Every referenced PNG exists. No extra ring delay is present. Each cell has both labels where possible, all four answers for four-class tasks, all three Krauzlis events, or negatives only for N0.

All **200 requirement rows** in `artifacts/coverage.json` were reconciled against episode metadata independently of `covered_variants` and independently of importing the exporter/validator. Each row's concrete episode IDs, GIF IDs and diversity count agreed with the recomputation. The 200 count includes primary/outcome requirements; it must not be advertised as 200 factorial stimulus variants. The rule construction at `validate_assets.py:30–75` covers the plan's families, and `:78–125` derives them from metadata rather than trusting tags.

| Task | Cells / episodes | Actual secondary coverage and substantive checks | Content / native anchors |
|---|---:|---|---|
| motion_direction | 1 / 5 | Four answers; 1/2/3-pixel steps; 256 square-domain identities, 128 survivors, wrap/bilinear blur/aperture semantics distinguished from visible dots. 17 property rows. | `content/sensory_tasks.json:3–222`; `repo:PreAttentiveVision/neuroscience_stimuli.py:33–101` |
| orientation | 1 / 3 | Both numeric signs; absolute magnitudes 4/10/22; roving carrier angle, independent phases/noise, peak normalization. No requirement for every sign×magnitude combination. 18 rows. | `content/sensory_tasks.json:225–449`; native `:142–169` |
| contrast | 1 / 9 | All nine increment×pedestal combinations, both interval labels; shared pattern and phase, independent noise, modulation amplitude not automatically Michelson contrast. 19 rows. | `content/sensory_tasks.json:452–692`; native `:171–183`; `artifacts/coverage.json:2486–2656` |
| spatial_frequency | 1 / 3 | 0.08/0.18/0.35 octaves; both intervals; ratio rule, fixed envelope, independently drawn phases and non-RMS normalization caveat. 19 rows. | `content/sensory_tasks.json:695–929`; native `:185–197` |
| chromatic_increment | 1 / 4 | All three increments, both intervals, four distinct base colors; unit RGB axis, ideal-color luminance versus noisy raster, no cone-isolation/calibration claim. 18 rows. | `content/sensory_tasks.json:932–1161`; native `:199–221` |
| contour | 1 / 4 | Jitter SD 2/8/16, both intervals; seven path elements among 32, tangent/carrier convention, pre-rounding placement, permutation fixed points, phase-by-position and accidental-alignment limits. 23 rows. | `content/sensory_tasks.json:1163–1434`; native `:223–260` |
| natural_spectrum | 1 / 3 | Beta deltas 0.15/0.30/0.60, both intervals, three distinct train photographs; actual random native crop, linear luminance, FFT phase, common RMS and no-clipping law. 24 rows. | `content/sensory_tasks.json:1437–1720`; `repo:PreAttentiveVision/natural_stimuli.py:114–192` |
| orientation_ring | 1 / 5 | D0 only; both signs, four locations, 15/30/45; ring in frames 0–2, not probe; repeated angles are not identical sample rasters. 21 rows. | `content/spatial_tasks.json:3–261`; `repo:WorkingMemory/PlainBaseline/variants.py:24–68` |
| orientation_cued | 4 / 16 | Every delay and both labels; aligned/unchanged/opposite at each delay, both cue signs, all locations/magnitudes across bank. Target rule correct; SCI-1 is confined to intro wording. 23 rows. | `content/spatial_tasks.json:264–542`; `repo:WorkingMemory/SpatialTaskBattery/stimuli.py:74–91` |
| motion_duration_cued | 4 / 18 | Every delay and all four answers per cell; all targets and 0.8/1.2/1.6 steps; 12 examples with last direction differing from winner, 17 with a differing foil winner. Eight-transition counts and unique winners recomputed. Plurality versus strict majority is correctly distinguished. 23 rows. | `content/spatial_tasks.json:545–823`; spatial native `:97–111`; `repo:WorkingMemory/stimuli.py:86–111` |
| krauzlis_cued_motion | 3 / 11 | Target/foil/catch in each B cell, both sides; actual event changes include −28/−26/+26/+28. Catch nuisance draws do not count toward event sign/magnitude coverage. 23 rows. | `content/spatial_tasks.json:826–1105`; spatial native `:69–73,121–142` |
| spatial_binding | 4 / 11 | Both swap outcomes at every delay; all queried locations. Exactly one two-location exchange and identical angle inventory independently checked; no no-swap negatives. 24 rows. | `content/spatial_tasks.json:1108–1397`; spatial native `:92–96` |
| image_recognition | 12 / 23 | All N×H cells; both outcomes for N>0, N0 negatives only; first/interior/last positive membership across bank. Existing raw bytes independently confirm membership and identical probe repeats for all 23 episodes. 22 rows. | `content/spatial_tasks.json:1400–1677`; spatial native `:45–57,112–120` |

Property tables are substantive rather than copies of catalog strata: they cover rendering nuisance, shared versus redrawn variables, normalization/gamut, imported cue masks, dot reset/age rules, schedule rejection/fallback, and the different photo preprocessing pipelines. The fixed glyph details were checked against `repo:WorkingMemory/stimuli.py:48–83`.

## Equations, N0 and Krauzlis safeguards

- **Axial conventions:** `content/sensory_tasks.json:233–240,432–435` and `content/spatial_tasks.json:239–242,519–523` use the doubled-angle wrapped difference, not naïve wrapped-angle subtraction. For every ring/cued episode, recomputed differences from recorded radian angles agree with degree rotations and labels. Carrier wavevector versus stripe axis is explicitly separated. Contour's tangent+π/2 carrier rule agrees with native cosine rendering (`content/sensory_tasks.json:1169–1170,1416–1419`). No clockwise relabeling was introduced.
- **Other equations:** independently checked amplitude differences, frequency ratios, chromatic vector increments, smaller-beta label selection, duration counts, swap membership and frame-length formulas. The recognition serial-age expression at `content/spatial_tasks.json:1647–1652` equals final frame index minus the matched study frame index, including repeated probes.
- **Recognition N0:** specificity/FPR only, null/not-applicable BA/AUC and exclusion from those aggregates/selection are correctly stated at `content/spatial_tasks.json:1439–1442,1638–1643`. All three N0 cells contain label 0 only. Every recognition episode retains exactly three postlist blanks and H identical probes. No semantic-recognition, human-capacity or extra-study-item claim is inferred from repetition.
- **Krauzlis:** zero-based cue 0–1, fixation 2–6, reference 7, B baseline transitions, first postevent B+8, eight postevent frames and report B+16 agree across metadata and prose (`content/spatial_tasks.json:836–838,1063–1068`). T=B+17 and the 15×0.01×2.5=0.375 step conversion check. The 57/29/14 cycle and separate target-hit/foil-FA/catch-FP denominators are correct (`:848–849,864–866,1045–1061`), including the 0.57 always-positive accuracy baseline, rather than calling it measured competence. Nominal 100 Hz is not applied to other tasks.
- **Explanation separation, source inspection only:** `web/atlas.js:85–95` suppresses the invisible Krauzlis event boundary outside explanation mode; `:168–175` distinguishes virtual catch timing and N0 metrics. This establishes the code intent, not browser-tested behavior.

## Scientific references and source-access limits

I read the saved retrieved/OCR evidence, checked every ledger quotation against its evidence file with whitespace normalization, and inspected the relevant methods/abstract context. This is **independent reading of already saved retrieval evidence**, not a claim of fresh internet access.

- Sensory anchors are relevant and match their bounded claims: Lovejoy/Krauzlis limited-lifespan motion, Zhang phase-randomized Gabor orientation, Legge/Foley pedestal discrimination, Putzeys higher-frequency interval judgment, Krauskopf/Gegenfurtner controlled chromatic oddity, Field/Hayes/Hess path-in-clutter, and Tadmor/Tolhurst image-statistics discrimination (`content/sensory_sources.json:3–226`). The latter's exact native spectral algorithm is attributed to code, not an inaccessible paper.
- **Campbell:** metadata-only, no verified methods/results (`content/sensory_sources.json:90–109`); the prose acknowledges this and supplies accessible primary Putzeys evidence instead (`content/sensory_tasks.json:719–721`).
- **Tadmor, Griffin/Nobre and Luck/Vogel:** abstract-only access is explicitly disclosed (`content/sensory_sources.json:200–224`; `content/spatial_sources.json:88–135`). The accompanying prose does not promote abstract access into full-method verification or import human capacity values.
- Spatial anchors match the claims: Mante contextual relevance, Roitman/Shadlen temporal evidence versus delayed report, Griffin/Nobre retrospective selection, Brady repeat detection/recognition (`content/spatial_sources.json:35–164`). These do not claim the custom glyph, count-winner, swap or exact-raster rules were published protocols.
- **Arcizet/Krauzlis:** the saved full-text Methods at `verification/spatial_arcizet_fulltext.txt:225–239` directly support the 90° relation, 16° SD, 10-frame/100-ms lifetime, 15°/s, event proportions and timing. The content correctly identifies 26°/28° as original monkey-specific medians, not an exhaustive original distribution; discloses 68-trial cue blocking versus native interleaving; and does not turn final classification into joystick/reaction-time behavior (`content/spatial_tasks.json:851–869`).
- Photo provenance and unverified public-redistribution permission are disclosed (`content/sensory_tasks.json:1474`; `content/spatial_tasks.json:1436`). Computational requirements are presented as logical requirements or possible strategies, not measured model abilities or unique neural mechanisms.

## Review identity and remaining verification boundary

Native source hashes agree with `artifacts/source_inventory_before.json`; no upstream code changes were made by this review. Reviewed SHA256 identities:

- `content/sensory_tasks.json`: `37242215a27c846a26735aa2c68e936fc92a56ced0cca1d1768354b2c4716217`
- `content/spatial_tasks.json`: `4fde572dcefcd42e3dd1b5cebd3d65c954d710c58214e0b2a56dd8235a53bbbd`
- `artifacts/manifest.json`: `10458565580760c10653aa4ffde752e5e30876dd80c83b45599ff082db283711`
- `artifacts/coverage.json`: `9a139c4ca35577514c8edca8ab4c48fc3da8cbbc2ca7b00b6b19e20ad04e35e5`

After SCI-1 is corrected, the examined scientific/coverage scope has no remaining identified blocker. That statement does not certify subsequent content changes, compiled-bundle propagation, visual legibility, actual display photometry, GIF decoded fidelity, offline relocation or interactive answer hiding; those remain the integrating agent's separately executed checks. Full-text access limitations remain disclosed limitations, not fabricated verification.
