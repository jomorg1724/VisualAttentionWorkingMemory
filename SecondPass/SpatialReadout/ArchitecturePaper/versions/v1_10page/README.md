# SpatialReadout architecture paper

**Delivered:** `SpatialReadout_Architecture.pdf` — 10 pages, built from editable `SpatialReadout_Architecture.tex` using Tectonic. The TikZ architecture diagram is vector content within the TeX/PDF. Bibliography is embedded in the TeX (no separate BibTeX dependency).

## Scope

The exact current CNN + three multiscale spatial KDA modules + final spatial ConvGRU, not the older standalone KDA report or paused NeurosciencePaper. Describes source-defined computation, exact states/equations, task-known head routing, design intent versus plausible rationale, historical compression order, current fresh-origin/unchanged-continuation lineage, limits, and source provenance. No live-status or performance claim is made. Continuation starts at 7,739 and has an authorized additional target of 104,000 / cumulative target 111,739; targets are not presented as completed training.

## Verified counts

| Component | Parameters |
|---|---:|
| Four CNN blocks (including GN) | 256,992 |
| Three KDA input projections | 9,312 |
| Three KDA modules | 74,262 |
| Spatial input projection | 10,304 |
| Final ConvGRU | 221,376 |
| Terminal linear readout | 803,072 |
| Thirteen task heads | 7,710 |
| **Total, all trainable** | **1,383,028** |

Each KDA contains 24,754 parameters. KDA states contain 160,000 / 43,264 / 12,544 scalars per episode at 25 / 13 / 7 spatial sides; final ConvGRU contains 3,136 state scalars. Catalog has 13 tasks / 35 conditions. Constructor audit uses CPU PyTorch 2.8.0, at most two intra/inter-op threads, and analytically derives spatial dimensions from actual convolution attributes.

## Rebuild and audit

From this directory:

```sh
python3 build.py
python3 verify_pdf.py
```

`build.py` uses `TECTONIC`, a PATH Tectonic, or `/Users/jonathanmorgan/.local/bin/tectonic`; it generates `model_hash.tex` from the saved audit. Dependencies for PDF verification: Python with PyMuPDF (`fitz`). The available system `python3` has it. First-time Tectonic runs can download TeX packages; no model or dataset download is involved.

To repeat the constructor-only audit in the existing repository:

```sh
PYTHONDONTWRITEBYTECODE=1 \
/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python verify_counts.py
```

This imports only the model constructors and reads the catalog directly. No forward/backward, checkpoint load, dataset stream, training, accelerator operation, cloud fetch, or commit. Rebuilding changes the PDF hash and intentionally invalidates the previous hash-bound visual receipt until inspected again.

## Receipts and evidence

- `count_verification.json`: total/module/per-tensor counts, head labels, exact constructor attributes, CPU settings, and source hashes.
- `source_manifest.json`: complete SHA-256 hashes of 15 local authority files and repository HEAD. Dirty-worktree hashes, not HEAD alone, identify this source snapshot; cloud byte parity is not asserted.
- `reference_verification.json`: primary-source URLs, checked locations, and supported claims for all five cited external references.
- `pdf_verification.json`: 10-page check, section/page alignment, per-page text bounds, font embedding, replacement glyph check, source-integrity check, warnings and visual receipt.
- `visual_review.json`: hash-bound visual checks and repairs.
- `build_output.txt`, `.log`, `.aux`, `.out`: actual successful compilation artifacts.
- `rendered/`: the final 10 page images and extracted text. All 10 pages were inspected across final layout verification; the diagram, equations, tables and source-map page were explicitly checked.
- `artifact_manifest.json`: SHA-256 hashes and byte sizes for the PDF, editable sources, scripts and delivery receipts.

Final build has **no TeX warnings**, all fonts embedded, no text outside pages, no replacement glyphs, and all 15 source hashes unchanged. The PDF contains 4,656 extracted words including equations, captions, tables and headers. The paper does not claim that constructor inspection is a trained inference test.

## Reusable authoring workflow

Trace inherited model definitions and training lineage separately; retain source hashes. Count real constructed modules on CPU without inference, excluding deleted superclass modules. Separate local implementation equations from method precedents, and preserve exact write/reset and normalization conventions. Build early, check expected section/page alignment as well as page count, fix even tiny box warnings, and inspect rasterized diagrams/equations/tables. Same-endpoint TikZ state loops can collapse without warnings; use distinct boundary points. Bind visual review to the final PDF digest.

Only new files under this ArchitecturePaper directory were authored. Existing model, analyses, training sources and concurrent dirty work were preserved.
