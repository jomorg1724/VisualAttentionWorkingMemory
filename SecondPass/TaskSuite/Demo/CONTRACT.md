# Integration contract

Scope: all writes below SecondPass/TaskSuite/Demo only. Do not commit. Do not access training/runtime/checkpoints/cloud or execute any model. Native CPU rendering only; one exporter process, <=2 numerical threads set before imports. Do not edit upstream sources or restore streams. Read the complete user plan at /Users/jonathanmorgan/Library/Application Support/Hermes/composer-pastes/pasted_content_2026-09-29_04-56-28-698_6c7a40.txt.

## Ownership
- Asset agent: export_assets.py, validate_assets.py, tests/test_manifest.py, tests/test_export.py, artifacts/manifest.json, artifacts/coverage.json, artifacts/raw/, dist/assets/, verification/export* and contact sheets. Sole renderer/export process. No frontend/content/build changes.
- Sensory content agent: content/sensory_tasks.json and content/sensory_sources.json (7 tasks), verification/sensory* only. No render runs.
- Spatial content agent: content/spatial_tasks.json and content/spatial_sources.json (ring + 5 spatial tasks), verification/spatial* only. No render runs.
- UI agent: web/index.html, web/atlas.css, web/atlas.js, DESIGN.md, tests/ui* only. No rendering/content/build writes.
- Integration parent: build_site.py, tests/test_build.py, README.md, merged content/tasks.json and sources.json, dist non-asset files/vendor, build receipts, QA, ZIP. Never race other owners.

## Manifest API (exporter -> UI/build)
Root object: schema_version, catalog (unaltered catalog object), provenance, display, episodes (array), cells (array).
Each cell: task_id, condition_id, kwargs, episode_ids (nonempty), showcase_id.
Each episode: id (globally unique safe slug), task_id, condition_id, kwargs, seed, native_trial_id, label, label_meaning, metadata (unaltered native row), frame_count, frames (array of dist-relative PNG path strings), frame_sha256 (array), float_sha256, phases (array of {index, phase, cue_visible}), gif (dist-relative string), poster (dist-relative string), frame_zip (dist-relative ZIP), metadata_path (dist-relative JSON), timing ({frame_ms: 300, ...}), covered_variants (array), curation_reason, source_split, photo_ids (array). Other fields permitted. Renderer values and derived properties clearly separated. Global display mapping clip/rint(x*255) with no gamma/per-frame normalization. Raw arrays outside dist. GIF same native frames, one stable palette, no dithering, decoded timeline validation.

## Content API
Each task file is a JSON array of objects: id, title, group (sensory / selection / retention), question, intro (connected prose string), sections (object keys appears, sequence, rules, ignore, implementation, neuroscience, network, boundary, metrics, adaptation; values arrays of substantive paragraph strings), properties (array {name, value, units, sampling, role, provenance, visibility}), equations (array of plain Unicode math strings), references (array source IDs). Inline citations use [source-id] in paragraph text; frontend links to bibliography entries. No Markdown required beyond citation tokens.
Each sources file is a JSON array of {id, authors, title, year, url, doi, type, verification, support_location, supports, does_not_support, task_ids}. Verification must distinguish full text vs abstract-only vs blocked; only verified claim depth may appear in prose. Can include additional ledger fields. Do not invent literature verification. Primary source per task, shared where justified. Properties trace imported helper defaults, not just catalog fields.

## UI data API
window.ATLAS = {manifest: <manifest>, tasks: <merged array>, sources: <merged array>, coverage: <coverage>}; loaded synchronously from data.js before atlas.js, so file:// works. Asset paths already dist-relative. UI must tolerate optional additional metadata but core schema above is fixed. Deep links #task=ID&condition=ID&example=ID plus optional mode=reader; #conditions for wall. Native PNG player shared scheduler. No remote runtime requests.

UI: Explore primary, Learn/Inspect secondary. Warm paper, dark ink, restrained blue/orange; editorial serif. Images untouched 100x100 RGB, integer nearest-neighbor default. Scene separate from annotations/answers. Gallery all 13; wall all 35; detail controls, phase timeline, observer/explanation, judgment reveal, reader/full sections, downloads, task-specific analysis displays. Hidden answers/Try mode must suppress clues in outcome selectors and technical metadata too. Independent delay-cell examples are not paired trials.
