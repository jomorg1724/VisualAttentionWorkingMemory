# Visual Task Suite Atlas

A local scientific explainer of the existing visual attention / working-memory **stimulus generators**, not a report of model performance. The gallery contains the catalog's 13 tasks and the condition wall exposes its 35 primary cells. A curated example bank covers the explicitly required native difficulty and outcome categories; generated totals and requirement-to-episode links live in `coverage.json` and `build_receipt.json`.

## Open the delivered atlas

Extract `task-suite-atlas.zip`, keep its folder intact, and open `task-suite-atlas/index.html` in a modern browser. No internet connection, package installation, server, API, or account is required. Do not move the HTML away from its sibling files. Fonts, descriptions, data and native frames are bundled locally. Reference links open external scientific sources only when clicked.

**Explore** shows native animated scenes. **All conditions** opens one player per catalog cell. Visible scenes can run together; **Pause all** freezes their native frame indices. Playback is an illustration, not a calibrated experiment. On a reduced-motion system playback starts paused. Open a task for its condition/example selectors, timeline, frame stepping, replay, answer reveal, Observer/Explanation controls and full scientific description. The reader view expands the prose while retaining the selected trial. Current GIF, frame ZIP and trial metadata downloads are provided beside the player.

Try-the-judgment mode is informal exploration. It does not measure psychophysical thresholds or scientific reaction times. Answers and explanatory annotations are recorded generator ground truth—not predictions or decoded model state. Developer tools and downloadable metadata necessarily contain the correct answer; this is not a secure testing platform.

## Scientific boundaries

- All input scenes originate in the existing Python renderers. The native raster is 100 × 100 RGB. Native frame order, blanks, reference frames and repeated probes are retained. No interpolated motion, synthetic replacement scenes or contrast enhancement is inserted.
- Float32 arrays are the canonical development input; browser PNGs use a single raw-tensor-value mapping to uint8 with rounding. There is no per-frame normalization or display-gamma correction. Nearest-neighbor enlargement is the default. Browser colors are not calibrated laboratory stimuli.
- PNG playback is authoritative for the displayed uint8 preview. GIFs use a stable per-trial palette and no dithering but introduce additional quantization. Exported trial records report conversion and GIF errors. Prefer PNGs for subtle contrast/color comparisons. Raw float artifacts stay in `artifacts/raw/` and are excluded from the delivery ZIP.
- Two-frame tasks remain two observations. Replay is a new presentation of the same trial, not a third valid motion transition. Loop boundaries are indicated outside the image; standalone GIF loops cannot provide the same external UI cue.
- Only Krauzlis declares a native 100 Hz clock (10 ms/frame). Its demonstrator is deliberately slowed, and browser/GIF timing is not calibrated. Other tasks' delays are frame counts, not biological milliseconds.
- Every native input frame—including visually quiet frames—is an observation. Glyphs already drawn by the renderer are native input; website annotations, labels, timeline, source IDs and computed counts are not. The training harness chooses the task-specific head externally; the website prose is not a model instruction channel.
- Demo streams have their own fixed seed namespace. Different delay cells are **independent examples**, not matched delay manipulations. Selection is curated to cover native outcomes/discrete properties and is not an estimate of their frequencies. In particular, the Krauzlis generator's native target/foil/catch mixture remains 57/29/14; an illustrative bank is not that mixture.
- Numeric positive/negative axial-angle language is retained. A Gabor's carrier wavevector is perpendicular to the visible stripes. Historical clockwise comments are not used to relabel classes.
- Contrast numbers denote intensity modulation amplitude, not automatically calibrated Michelson contrast. The chromatic axis obeys a numerical linear-RGB luminance constraint, not cone isolation or observer-calibrated isoluminance.
- Recognition is exact image-set membership. N0 has only negatives and is described with specificity/false positives, not BA/AUC. No current or historical model scores are displayed.

## Source photographs and rights

Only the BSDS500 **train** identities are used for both photograph tasks. Spectral-detail trials use the native linear-luminance crop/filter pipeline; recognition uses the canonical central-square RGB crop and bicubic resize. Neither task uses segmentation labels or claims a BSDS segmentation benchmark result. Each photo example records its source identity and the dataset manifest hash.

Official resource page: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html . Dataset citation: P. Arbelaez, M. Maire, C. Fowlkes and J. Malik, *Contour Detection and Hierarchical Image Segmentation*, IEEE TPAMI 33(5), 898–916 (2011). The page was retrieved into the development verification folder and confirms the disjoint source partitions and research resource provenance. It does **not** establish unrestricted public redistribution rights for these photographs. This is a local delivery for the requested research explainer; recheck rights before any public hosting. No public deployment is included.

Newsreader and Source Sans 3 are locally bundled with their SIL Open Font Licenses and a download/hash provenance record in `vendor/`.

## Rebuild from the repository

All new authored/generated work belongs to `SecondPass/TaskSuite/Demo/`. Do not run historical model-smoke tests, import a model, open checkpoints, or launch training to rebuild this site. The inspected renderer import closure contains only:

- `SecondPass/TaskSuite/suite.py`
- `PreAttentiveVision/neuroscience_stimuli.py`
- `PreAttentiveVision/natural_stimuli.py`
- `WorkingMemory/PlainBaseline/variants.py`
- `WorkingMemory/SpatialTaskBattery/stimuli.py`
- `WorkingMemory/stimuli.py`

Set limits **before** importing numerical libraries. Use one exporter process, at most two threads. From the repository root:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=SecondPass/TaskSuite/Demo/.venv/bin/python
$PY -m pytest SecondPass/TaskSuite/Demo/tests -q \
  -o cache_dir=SecondPass/TaskSuite/Demo/.pytest_cache \
  --basetemp=SecondPass/TaskSuite/Demo/.test-tmp
$PY -m SecondPass.TaskSuite.Demo.export_assets --all --device cpu --threads 2
$PY -m SecondPass.TaskSuite.Demo.validate_assets --all
python3 SecondPass/TaskSuite/Demo/inventory.py
python3 -m SecondPass.TaskSuite.Demo.build_site
```

Dependencies follow `SecondPass/TaskSuite/requirements.txt` and actual renderer imports. Older PyTorch's NumPy bridge requires NumPy <2; use the Demo-local environment, not a global package change. Dataset download is not part of the build; absent/corrupt photos must fail rather than become fabricated examples. The source inventory check fails if catalog, renderer dependencies or dataset manifest changed during this build.

The builder merges independently authored content, rejects missing descriptions/references, checks catalog cells and local asset paths, emits synchronous `data.js` (no mandatory `fetch()`), copies local fonts, writes a per-file hash receipt, and constructs a reproducible ZIP with sorted entries/fixed timestamps. Development literature notes, tests, raw arrays and environment are not included in that ZIP.

For optional HTTP viewing, first check that the chosen port is free, then:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory SecondPass/TaskSuite/Demo/dist
```

Never terminate an unrelated service to free that port.

## Measured export limits

The completed bank contains **115 curated native episodes / 115 GIFs and 1,580 original frame indices**. The coverage ledger has **200/200 satisfied requirements**; that number includes primary cells, per-cell labels and discrete coverage checks, not 200 independent stimulus variants. Direct native replay matched pixels, labels and metadata for all 115 episodes. Source photographs comprise 155 distinct train identities across the two photo tasks.

Maximum float-to-uint8 absolute error is **0.0019607871** on the [0,1] scale. GIF quantization adds up to **86/255 in an individual channel** in this bank; the worst episode's mean absolute channel error is **4.79443/255**. The largest errors occur in photograph previews. These are measured export errors, not perceptual thresholds. Contrast and chromatic tasks should be inspected with the lossless PNG player; their worst episode mean GIF errors are 1.5226/255 and 0.953017/255 respectively. Each downloadable trial record has its own error measurements.

Full-text access was obtained for the cited Lovejoy, Zhang, Legge–Foley, Putzeys, Krauskopf–Gegenfurtner, Field–Hayes–Hess, Arcizet–Krauzlis, Mante, Roitman–Shadlen and Brady papers. Tadmor–Tolhurst, Griffin–Nobre and Luck–Vogel are explicitly abstract-only; the Campbell record is metadata-only. The prose limits claims to the retrieved evidence, and the spatial-frequency account also has a verified full-text Putzeys primary anchor. `bibliography.html` presents the complete support/adaptation ledger. None of these citations establishes biological equivalence of the project-specific assays.

## Verification records

`artifacts/source_inventory_before.json` records the original catalog, all cells, allowed write subtree and renderer/dataset hashes. `artifacts/manifest.json` records every native episode, source/raster identity, phase index, cue visibility, asset path, display mapping and curation reason. `artifacts/coverage.json` maps reviewed requirements to concrete episode/GIF IDs. `artifacts/build_receipt.json` records actual bundle totals and every delivered file hash. `verification/` contains actual test output, contact sheets, browser evidence and the final QA report; consult that report for what was actually exercised and remaining limitations.

No model/training/cloud change, checkpoint read, inference, fitting, optimization, deployment, commit or push is part of this atlas work. Existing dirty work is preserved.
