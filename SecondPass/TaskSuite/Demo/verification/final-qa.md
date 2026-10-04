# Atlas delivery verification

## Delivered artifact

Entry: `../dist/index.html` (open directly). Portable archive: `../task-suite-atlas.zip`; extract and open `task-suite-atlas/index.html` with its relative assets intact.

Verified inventory: **13 tasks, 35 primary conditions, 115 curated example episodes, 115 native GIFs, 1,580 original PNG frame indices, 274 property rows**. All **200 coverage requirements** are satisfied; this requirement total is not a count of factorial stimulus variants. Each of the 115 examples has lossless playback, a GIF, a frame ZIP and downloadable metadata.

## Executed checks

- **24 Python tests passed**, **17 Node tests passed**, both content validators passed. Final stdout: `final-python-tests.txt`, `final-node-tests.txt`, `final-content-sensory.txt`, `final-content-spatial.txt`.
- Native asset validator passed, including direct native replay of **all 115 episodes**, exact float pixels/labels/metadata, source/dataset identity, PNG hashes and decoded GIF duration-to-source-index timelines. See `export-validation-native.json` and `export-report.md`.
- Actual built site tested in Chrome over HTTP and again from the extracted ZIP through **file:// with network access disabled**. All 35 cells animated and could seek their last frame; keyboard Home/ArrowRight, frame slider, phase navigation, reading-mode frame preservation, downloads, reference targets and complete recognition filmstrips passed.
- All **35 players simultaneously visible and advancing** were checked in one expanded viewport; pause froze all 35. Reduced-motion starts paused. Try mode suppresses outcome-bearing variants, analyses, metadata and download clues until reveal.
- Desktop/tablet/phone checks at **1440/900/390 px** passed, with native scenes integer-enlarged and no horizontal document overflow. Print-media checks passed: descriptions expanded, navigation hidden, 12pt body. The bibliography contains all 23 source records and no invented external links for local source code.
- **Zero uncaught browser exceptions or HTTP 4xx resource errors** in the recorded final runs. Reports: `http-browser-results.json`, `file-offline-browser-results.json`.
- Reviewed screenshots of gallery, mobile/detail, wall and print presentation; inspected contact sheets for every task, all conditions, the contrast bank, full long-recognition and Krauzlis sequences. Authentic subtle stimuli were not enhanced.
- Measured ink/muted/selection/event text contrast against paper exceeded 4.5:1 (`contrast-check.json`).
- ZIP integrity and every relocated member verified byte-for-byte. A repeated build produced an identical archive hash. Final archive identity and exact byte count are in `final-verification.json`; all delivered files are hashed in `../artifacts/build_receipt.json`.
- Six renderer/adapter closure source hashes, the catalog and the dataset manifest remain identical to the starting snapshot. Source check: `final-source-immutability.txt`.

## Independent scientific review and disposition

The read-only review is preserved unchanged in `independent-science-review.md`. It independently reconciled all coverage rows and checked all 23 recognition examples' saved raw membership/repeated-probe bytes, equations, native metadata and saved primary-source evidence.

**SCI-1 fixed:** the short cued-orientation introduction now states that aligned/opposite/unchanged changes occur across the **whole array**, not necessarily among the foils. Native code constructs the multiset before choosing the target. Integration independently confirmed all 16 exported cued-orientation examples contain the three whole-array categories and 11 lack at least one category among foils. The corrected prose propagated to merged JSON, delivered JSON and embedded browser data; no stimulus or metadata changed.

**DOC-1 fixed:** sensory reproducibility prose now distinguishes restoring isolated demo progress from forbidden live training-state restoration. Both the authored JSON and its editorial generation script were corrected. Content validators, the complete tests and both browser modes were rerun after the corrections.

No identified scientific/coverage blocker remains in the reviewed scope. This is a task explainer, not a model-performance or biological-validity certificate.

## Disclosed limits

PNGs preserve deterministic display rasters, not calibrated display luminance. GIFs are quantized; their errors are measured per episode. Illustration playback is slowed, not calibrated psychophysical timing. The sole nominal biological clock is Krauzlis's 10 ms/frame source convention.

Campbell was verified at metadata depth only; Tadmor–Tolhurst, Griffin–Nobre and Luck–Vogel at abstract depth. These restrictions are explicit beside the claims and in the source ledger; none is represented as verified full methods. BSDS photo public-redistribution permission was not established; delivery is local, with no publication or deployment.

No model/checkpoint/inference/optimization/training/cloud operations were performed. No upstream task or generator edits, commits or pushes were made. Raw float development arrays, retrieved papers and environment/test files are excluded from the delivery ZIP.
