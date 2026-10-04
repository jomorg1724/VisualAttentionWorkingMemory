# Native atlas export — completed evidence

## Results

- 13 tasks, 35 primary cells, **115 native episodes / 115 GIFs**.
- **200/200 explicit machine-readable coverage requirements satisfied**, zero missing. This counts primary cells, per-cell outcomes, native difficulty values, the nine contrast cross-products, and rule/example categories. It is not a 200-variant factorial claim.
- 1,580 original PNG frame indices; 1,079 encoded GIF frames. Decoded GIF durations expand back to all 1,580 source indices exactly (identical frames can merge).
- 115 canonical float32 arrays, 115 frame ZIPs, 115 metadata JSON downloads, 115 native posters, 49 analysis contact sheets.
- 202 total native candidates; 115 selected by bounded coverage curation. This is not an empirical outcome distribution. Native Krauzlis sampling remains 57/29/14.
- All 115 exported arrays, labels and unmodified native metadata matched independently replayed direct native streams across all 35 cells.
- 155 distinct train-split source photographs used across recognition/spectral tasks. Used image bytes and dataset identity verified. No val/test source photographs rendered.

## Entry points and environment

From repository root:

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
SecondPass/TaskSuite/Demo/.venv/bin/python -m SecondPass.TaskSuite.Demo.export_assets --all --device cpu --threads 2
SecondPass/TaskSuite/Demo/.venv/bin/python -m SecondPass.TaskSuite.Demo.validate_assets --all --native
SecondPass/TaskSuite/Demo/.venv/bin/python -m pytest SecondPass/TaskSuite/Demo/tests/test_export.py SecondPass/TaskSuite/Demo/tests/test_manifest.py -q -o cache_dir=SecondPass/TaskSuite/Demo/.pytest_cache --basetemp=SecondPass/TaskSuite/Demo/.test-tmp-export
```

Demo-local system-site-packages venv: Python 3.9.6, NumPy 1.26.4 (installed only into this venv), Torch 2.2.2, Pillow 11.3.0, SciPy 1.13.1. The global NumPy 2.0.2 installation was not changed. No model imports/operations occurred. Numerical thread environment and Torch intra/inter-op limits are both 2. An advisory exclusive lock prevents concurrent bank exporters/native validators.

## Evidence files

- `artifacts/manifest.json` — exact CONTRACT.md core schema, unaltered catalog and native metadata.
- `artifacts/coverage.json` — each reviewed requirement maps to episode IDs and GIF paths.
- `artifacts/raw/*.npy` — canonical little-endian float32 TCHW arrays.
- `dist/assets/{frames,gifs,posters,frame-zips,metadata}/` — local portable display/download assets.
- `verification/export-tests.txt` and `export-tests.xml` — **16 passed in 13.45s**.
- `verification/export-tdd-01..07-{red,green}.txt` — recorded failing assertions before implementation and subsequent green runs.
- `verification/export-validation-native.json` — all 115 direct-native comparisons passed, all bytes/timelines/semantics validated.
- `verification/export-validation.json` — separate static validator run, no native replay (its native_equal_episodes=0 correctly means not requested).
- `verification/export-receipt.json` — computed counts, sizes, 49 contact-sheet paths.
- `verification/export-progress.json` — atomically persisted independent demo stream progress and source/config identity; no live state.
- `verification/export-resume.txt` — completed production export rerun, zero new candidates; total remains 202.

Tests cover missing/duplicate/extra cells, invented ring delays, forged covered-variant claims, absent contrast combinations, duplicate/orphan episodes, missing/corrupt assets, changed source identity, direct-native equality, JSON round-trip stream replay, interrupted/export-resume byte equality, finite sampling failure, repeated probes, exact recognition membership, GIF decoded timing, PNG photometry, ZIP bytes, and forged conversion/quantization measurements. A full native bank is independently created under the Demo-only test directory. No whole historical/model suite was run.

## Display limits and measured size

Authoritative display mapping: `rint(clip(x, 0, 1) * 255)`, no gamma or per-frame normalization. Every lossless image is RGB 100×100. GIFs use one whole-trial median-cut palette, no dithering, 300 ms per source index, no extra/tween frames. Only Krauzlis metadata has a native 10 ms frame clock. Demonstration speed is not calibrated presentation.

Measured display/download assets: **22,412,481 bytes**. Raw arrays: **189,614,720 bytes**, development-only. Post-pilot delivery targets: display/download assets <30 MiB and development raw <200 MiB; both met. Exclude `.venv`, raw arrays, test temporary outputs and verification progress from the portable site ZIP.

Maximum float-to-uint8 absolute error: 0.001960787118649998 in [0,1] units. Maximum GIF channel error: 86/255; worst episode mean absolute error: 4.794432258064516 channel levels, from recognition photographs. Worst contrast mean error: 1.5226; chromatic: 0.9530166666666666. GIFs are illustrative; use PNGs for fine photometric comparisons. Validator independently recalculates these measurements.

## Source and visual review

The parent pre-export snapshot's six source hashes, catalog hash and dataset-manifest hash match final export provenance. Imported local renderer closure consists only of neuroscience_stimuli.py, natural_stimuli.py, PlainBaseline/variants.py, SpatialTaskBattery/stimuli.py and WorkingMemory/stimuli.py. Sources remained unchanged during export and validation. The suite adapter was inspected and hashed but not imported for demo seeding.

Visually inspected the all-35-condition sheet (all 13 task families visible), full N24_H5 recognition sequence, all nine contrast posters, full B28 Krauzlis sequence, and D24 binding sequence. Native corner glyphs and local cues remain inside native scenes by design; all explanatory labels are outside. N0 recognition poster is the real instruction frame; its GIF/PNG timeline still contains the full three blanks and repeated photograph probe. These contact checks do not replace the parent's browser/interaction QA.

No catalog/core API changes. Additional episode fields: raw_path, raw_file_sha256, native_ordinal, poster_frame_index, gif_palette, conversion_error, gif_quantization_error, asset_sha256, derived_fields. Seed integers are restricted to <2^53 for exact JavaScript transport. Native metadata is never rewritten to correct legacy clockwise wording.

Manifest SHA256: `10458565580760c10653aa4ffde752e5e30876dd80c83b45599ff082db283711`

Coverage SHA256: `9a139c4ca35577514c8edca8ab4c48fc3da8cbbc2ca7b00b6b19e20ad04e35e5`

Only Demo-subtree files were written. No commit, upstream edits, model/checkpoint access, training, accelerator operation or cloud action. UI/content/build integration remains with the parent.
