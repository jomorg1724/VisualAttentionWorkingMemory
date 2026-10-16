# A shared spatial priority map for visual selection and working memory

An illustrated research explainer with detailed neuroscience, task protocols, a motion tutorial, mathematical architecture documentation, and verified measured results. Updated October 11 with the completed 19-task test, four new measured figures, and an independent-findings chapter relating the local comparison pathway to James Herman's analyses. Earlier 13-task and 16-task VDA results and their limitations remain intact.

The new chapter is `sections/independent.html`; its figures and hashed numerical ledger are in `assets/independent_20261011/`. `research/independent_findings.py` regenerates plots from completed saved measurements, without model inference or training. The source implementation and original results are unchanged. The portable sharing copy is under `output/share_james_2026-10-11/` at the repository root; its `START_HERE.html` embeds the four new plots and two measured GIFs. The complete ZIP preserves the original task atlas and both frozen analysis galleries, including all sixteen existing GIFs. It excludes checkpoint weights and private third-party manuscripts.

## Read

Open `index.html` with its sibling folders intact. The manuscript bundles its fonts, MathJax, scripts, and stimulus rasters; scientific reference links require internet access only when followed. The Codex preview uses a local HTTP server. Browser automation verified HTTP rendering; its security policy does not permit direct `file:` navigation, so direct-file playback was not tested in that browser.

The included `task-atlas/index.html` can also be read separately. In the manuscript, select a task above the embedded atlas; inside it, select a condition or example. Condition explanations and selectors precede the player, including on narrow displays. Recognition defaults to four study images; separate controls expose 0/4/12/24 images and 3/4/5 repetitions of the same probe. Its blank interval is always three frames. Other delayed tasks explicitly label their 0/4/12/24 blank frames, while Krauzlis labels baseline motion transitions. Use **Try the judgment** for one complete presentation followed by a human response, or use the frame controls for inspection. Horizontal scrolling keeps wide mathematical figures legible on small screens.

## Contents

- Introduction extending Morgan, Albanna, and Herman's recurrent-attention hypothesis, with testable causal predictions and explicit differences from their learning protocol.
- Original 13 tasks / 35 conditions with 115 native movies, plus detailed VDA4/9/16 protocols (16 tasks / 44 conditions total) and links to the VDA frame-by-frame analysis atlas.
- Optical-flow tutorial: brightness constancy, aperture ambiguity, Lucas–Kanade, Horn–Schunck, robust/warped and learned methods, correlation, motion energy, and population ambiguity.
- Complete implemented architecture: convolutional appearance, raw statistics, three-lag correlation bank, refinement, pixel transport/error, local comparisons, shared spatial attention, normalized spatial and CLS GRUs, readout, objective, and optimizer.
- Interactive correspondence, temporal-window, shared-attention intervention, and memory-retention demonstrations. These figures show mathematical teaching examples, not trained-model measurements.
- A substantial neuroscience chapter: SC perturbations, cortical/behavioral dissociations, biased competition, normalization, cue validity and value, memory mechanisms, and their specific implications for this architecture.
- Local explanations for all 35 original displayed tutorial/implementation equations, unambiguous RGB versus grayscale notation, worked examples, and a corrected value-weighted VDA loss.
- Measured parent/continuation task performance, primary VDA effects and uncertainty, native difficulty checks, psychometrics, and attention figures. Earlier null benefits remain visible alongside the new result.
- 76 distinct scholarly references, numbered links and searchable bibliography, with per-source verification notes.

## Evidence and editing

`sections/` contains editable HTML; `build.py` assembles `index.html` and the reference ledger. `results_data.py` reads the existing frozen JSON/CSV results and generates tables plus the hashed `research/results_sources.json` evidence ledger. It does no inference or training. `research/implementation_sources.json` records the implementation reviewed. `research/bibliography.json` is the authoritative delivered reference selection; earlier retrieval files are working research records, not the final bibliography. The source notes distinguish metadata/abstract checks from full-text inspection. These do not assert that every cited paper was exhaustively reviewed.

`research/static_audit.json` records asset and source integrity checks. `research/revision_browser_qa.json` records the current revision's browser checks; earlier browser receipts concern earlier versions. The atlas retains its own native-renderer provenance and adaptation notes. Prospective experiments are explicitly separate from completed findings.

The manuscript documents reviewed source snapshots. It does not launch, stop, or alter training or contain model checkpoints. The later 19-task search continuation and Doom remain separate from the completed evidence reported here. Keep the sibling Suite/Analysis and StabilizedAttentionVDA/Analysis folders intact for the linked result figures and data; the explainer is not a single-file portable bundle.

The October 7 trial audit replayed all 115 examples directly from the native generators: every label and all 1,580 displayed PNG frames match. See `research/trial_audit.md`, `research/current_native_audit.json`, and `research/trial_browser_qa.json` for results and numerical/display limitations.
