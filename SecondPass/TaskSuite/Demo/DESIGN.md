# Visual Task Atlas — design & interaction contract

## Editorial direction

An illustrated field guide, not a monitoring dashboard. **Explore** is the first surface: thirteen substantial native scenes, grouped into sensory discrimination, spatial selection/comparison, and retention/recognition. Group names are navigational—not claims of isolated biological functions. **All conditions** is the same generous gallery with one actual native player per manifest cell. **Learn / Inspect** preserves the exact task/condition/example while opening the complete scientific account.

The large opening line introduces the act of looking rather than invented metrics. Fine rules, open paper margins, readable essays and restrained typographic hierarchy do the visual work. No gradients, decorative statistics, shadows, glass or stimulus alterations.

## Tokens

| Token | Value / role |
|---|---|
| Paper | `#f5f2ea`, main reading surface |
| Sheet | `#fcfaf5`, external explanation / controls |
| Ink | `#222a2a`, text and section rules |
| Muted | `#626965`, contextual labels |
| Rule | `#cecfc5`, divisions |
| Selection blue | `#285c83`, active controls, links, cue schematics |
| Blue wash | `#e5edf1`, active condition / phase |
| Annotation orange | `#a85532`, explanation-only event / correspondence |
| Annotation wash | `#f2e7db`, revealed answer |
| Essay face | Locally bundled `Newsreader`; Iowan / Palatino / Georgia fallback |
| Interface face | Locally bundled `Source Sans 3`; system sans fallback |
| Values | System monospace only for indices / code |
| Main width | 1550 px maximum, 18–64 px responsive side margins |
| Grid | 3 columns desktop; 2 below 1190 px; 1 below 720 px |
| Spacing | 8 / 12 / 16 / 24 / 32 / 40 / 56 px rhythm |

`vendor/fonts.css` is a local, integration-owned stylesheet loaded before `atlas.css`. There are no runtime font/CDN requests. All application data is delivered synchronously through `data.js` and `window.ATLAS`; `fetch()` and ES module loading are not required, including under `file://`.

## Native scenes are not design surfaces

Each displayed input is the exported lossless RGB PNG. Its authored dimensions remain 100 × 100. Scene images are **300 × 300** in the desktop gallery, **400 × 400** in detail (**500 × 500** on wide desktop), and **200 × 200** on phones. A/B analysis uses integer 1× or 2× images; recognition uses 1× study thumbnails. Nearest-neighbor enlargement is explicit. Never crop, recolor, normalize independently, interpolate, blur, crossfade, annotate internally or reconstruct stimuli in JavaScript.

The native rectangle contains an image and nothing else. Titles, answer boxes, sliders, phase labels, indices, count bars, location maps and event lines are outside it. All A/B or study filmstrips are explicitly **analysis displays**, not extra simultaneous model input. The renderer's own glyphs remain untouched.

## Playback contract

- One shared requestAnimationFrame scheduler controls every visible player. Play all is not a selection of one active tile.
- IntersectionObserver suspends offscreen advancement and lazily primes current/next frames. A bounded 256-entry image cache avoids loading the full bank.
- Hidden documents stop advancement and reset clocks on return; no catch-up jump skips frames. Reduced-motion users start paused but can explicitly play.
- Decoder backpressure holds advancement instead of jumping across missing source indices. Slow machines extend illustration timing, not erase repeated probes or blanks.
- Per-player play/pause, replay, previous/next frame, slider and metadata-derived phase jumps are available. Global and local speed controls use 0.25× / 0.5× / 1× / 2× / 4×.
- Every semantic index exists even if successive PNG pixels are identical. Two-frame loop boundaries are explicitly labeled “Trial restart”; no blank or tween is inserted as a model input.
- Frame numbering is zero-based and shows last index T−1. The captions state that default playback is an illustration clock, not biological time. Krauzlis additionally discloses its native 100 Hz / 10 ms source clock.
- Scene pauses retain the actual native raster. A separately downloadable static poster is available.

## Navigation and reading

Deep links are `#task=ID&condition=ID&example=ID`, with `&mode=reader` for full descriptions. `#conditions` is the native condition wall. Unknown IDs report an unavailable atlas address instead of silently substituting another condition. Moving between reading modes preserves exact episode, current frame, play state and rate.

Condition, example and variant controls have different labels. Previous/next example steps through the selected cell's bank. The condition matrix clearly discloses independent streams, never paired delay interventions. Sections use independent native `<details>` elements: opening neuroscience does not close implementation. Full description / Expand all and print expand the full account. Inline citations open and scroll to the local claim ledger without destroying the episode deep link; only explicit literature links leave the offline atlas.

## Answers and judgment

Answers start hidden. Trial-specific analysis, outcome-bearing variant selectors, technical metadata and metadata downloads are omitted from the rendered DOM while hidden—not merely covered with CSS. Example picker labels become neutral ordinals.

Try the judgment restarts the exact native episode, hides all answers, disables explanation and global answer reveal until response/reveal, and withholds downloadable files until reveal. Response choices use the native catalog class order. Reveal records only the selected response in local page state, shows the generator's answer and its external worked reasoning; no timing, score history, network transmission or psychophysical validity is claimed. A new example clears that example's reveal/response. The static source data necessarily contains labels: this is protection from accidental UI spoilers, not an anti-cheating or secure assessment system.

Observer mode never draws explanatory objects inside the scene. In particular, the Krauzlis baseline/postevent phases merge to an unlabeled motion segment in observer mode; no invisible event boundary is announced. Binding correspondence and recognition matching are only shown after answer reveal and with Explanation enabled. The ring precursor's internal `cue_sign=1` metadata must not be depicted as a visible sign glyph.

## Task-specific analysis

- Sensory: original ordered A/B rasters, actual native numerical evidence and task-specific decision equation / displacement schematic. No contour is guessed over the pixels.
- Spatial orientation: external selected-location diagram, actual cue visibility and signed target comparison. Ring-only and signed-glyph rules remain distinct.
- Duration: four directional bars computed from `directions_by_patch[target_location]`, advancing only for recorded `moving_frames`, plus final-direction versus full count-winner contrast.
- Krauzlis: recorded reference/event/report line, selected versus changed patch and target/foil/catch semantics. Event line exists only in revealed explanations.
- Binding: revealed source-site → destination-site mapping from the native swapped pair. Both classes exchange exactly two locations.
- Recognition: all recorded study frames in order, serial positions, all three blanks, repeated-probe counter and exact matching position after reveal. N0 explicitly has no study image and no positive membership claim.

All such diagrams are ground truth/analyst-derived aids, not measured internal representations.

## Accessibility and responsive rules

Semantic headings, landmarks, native labeled selects, button states, independent summaries, keyboard-visible focus and a skip link. Space toggles detail playback; left/right step; Home/End reach first/last frame when focus is not already on a control. The frame slider and phase buttons are keyboard reachable. Announcements avoid a live region firing on every frame. Play/pause status is the restrained live message.

Phone composition remains scene-first, then explanation. Content and equations wrap; technical tables scroll in their own containers rather than expanding the page. Print omits interactive controls, expands descriptions, preserves scene/explanation separation and uses a paper layout.

## Verification

Run from repository root:

```sh
node --test SecondPass/TaskSuite/Demo/tests/ui_*.test.cjs
node --check SecondPass/TaskSuite/Demo/web/atlas.js
node SecondPass/TaskSuite/Demo/tests/ui_browser_smoke.cjs
```

The browser smoke uses installed Google Chrome and Node's native WebSocket/CDP, with no new package dependency. It serves the authored UI against a read-only in-memory snapshot of actual exported trial JSONs and authored task content; it does not render stimuli or write build/content files. The completed smoke exercised all 13 tasks / 35 cells from the 115-episode native snapshot, frame/answer controls, duration schedule counters, full N24 filmstrips, reader-state preservation, 1440/900/390 px layout and reduced-motion initialization with zero uncaught exceptions. It exposed and regression-tested a slider pause/repaint value-reset bug and a hidden phone rate selector; both were corrected. Runtime browser profile files are temporary and removed on exit.

Node tests cover native index preservation, repeated identical rasters, two-frame restarts, all-player scheduling, visibility suspension, reduced-motion initialization, decode backpressure, speed, exact deep links, observer event suppression, Try leak protection, duration schedule arithmetic, actual photometry field names, ring/glyph distinction, catalog-driven gallery/wall counts and local entry/style contracts. Tests were introduced and observed failing before implementing their contracts. Integration owns actual full asset build and real-browser visual/interaction verification, including file://, relocated delivery, all 35 cells and phone/tablet/desktop screenshots.
