# SpatialReadout architecture paper — teaching revision v2

**Canonical deliverable:** `SpatialReadout_Architecture.pdf`, **15 pages including references**, built from editable `SpatialReadout_Architecture.tex` with Tectonic. The original 10-page delivery is preserved, unchanged, under `versions/v1_10page/`.

## What changed

This is an explanatory rewrite, not five pages of padding. Every section starts with a reader question; the computation is introduced through roles, shapes, operations and takeaways. A dedicated notation page separates learned parameters, generated activations, episode states and derived diagnostics. The vector diagram separates visual cue processing from externally supplied task-head routing.

The overloaded `P` has been removed. The CNN projection explicitly includes learned `W_proj,s` and bias. The KDA transition is derived from the prediction error, with no hidden algebra. `S0`, `S1` and `S2` are expanded before naming the product of intervening transitions `T_{t←τ}`; it is not a learned parameter, and the sequence length has distinct notation. Empty products, order, transposition and signed temporal coefficients are explained explicitly.

A complete hand-chosen 2×2 example shows zero initialization, decay, predicted value, residual, corrective write, new matrix and read. `verify_toy.py` checks it with exact rational arithmetic and checks a third update and the equivalent transport decomposition. These numbers are teaching arithmetic, **not model inference or behavioral results**. The optional norm argument states its assumptions; initial 0.9 retention is not treated as a fixed lifetime. ConvGRU reset/write semantics are explained with a separate scalar fraction example.

## Verified model and scope

| Component | Parameters |
|---|---:|
| Four CNN blocks including GroupNorm | 256,992 |
| Three KDA input projections | 9,312 |
| Three KDA modules | 74,262 |
| Spatial input projection | 10,304 |
| Final ConvGRU | 221,376 |
| Terminal readout | 803,072 |
| Thirteen task heads | 7,710 |
| **Total, all trainable** | **1,383,028** |

The CPU constructor audit was rerun, without a forward pass, training, checkpoint, dataset or accelerator operation. The catalog contains 13 tasks / 35 conditions. All 15 authority-file hashes remain unchanged. KDA state footprints are 160,000 / 43,264 / 12,544 scalars per episode, distinct from learned parameter counts; the final hidden field contains 3,136 scalars.

The selected lineage starts with every learned tensor freshly initialized; authorized continuation restores that same run at 7,739, with 104,000 additional / 111,739 cumulative updates as **targets**, not completion claims. No latest status, performance, feature-semantic or biological-circuit claim is made. Proposed ablations were not run.

## Rebuild and verify

From this directory:

```sh
python3 verify_toy.py
python3 build.py
python3 verify_pdf.py
```

`build.py` uses `TECTONIC`, a PATH executable, or `/Users/jonathanmorgan/.local/bin/tectonic`. PDF verification requires PyMuPDF (`fitz`), available in the system `python3`. The exact CPU constructor command is:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python verify_counts.py
```

Rebuilding changes PDF metadata/hash and invalidates the old hash-bound visual receipt until reviewed again. `verify_pdf.py` checks all original safeguards, now with 15 section/page alignments, plus the 13–15 page budget, title/author metadata, no overloaded P, saved toy pass and the preserved v1 PDF digest.

## Receipts and editable assets

- `REVIEWER_CHECKLIST.md`: symbol/derivation, scope, provenance and delivery checklist.
- `verify_toy.py`, `toy_verification.json`: exact rational arithmetic and recurrence/read reconstruction.
- `count_verification.json`, `source_manifest.json`: CPU constructor counts, all named tensors and 15 authority hashes.
- `reference_verification.json`: previously verified primary references, retained unchanged.
- `pdf_verification.json`: page count, metadata, text bounds, fonts, glyphs, warnings, source integrity and section alignment.
- `visual_review.json`: final-hash-bound review; all 15 pages inspected across build iterations, and final changed pages reinspected.
- `rendered/`: all 15 actual PDF page images and extracted texts.
- `build_output.txt`, `.log`, `.aux`, `.out`: actual build artifacts.
- `artifact_manifest.json`: delivery file hashes and sizes, including preserved v1 identity.
- Adjacent `../SpatialReadout_Architecture_sources.zip`: refreshed portable source/receipt bundle; no copyrighted reference-paper copies.

The final build has no TeX warnings, all fonts embedded, no out-of-page text or replacement glyphs, and 15-page section alignment. The visual review checked equations, toy matrices, diagram loops/routing, tables and source map; an automated bounds pass alone is not claimed as readability proof. Only this paper directory and the adjacent source ZIP were changed. Original scientific model, training and analysis artifacts are unchanged.
