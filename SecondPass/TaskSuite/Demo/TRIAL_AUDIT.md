# Native trial and human-play audit — October 7, 2026

The reported image-recognition example was N0_H5, an empty-list control. Its native sequence is one instruction, three blanks, and the same probe photograph five times. There are no study images to omit. Looping playback can make the final probe appear to precede the next pass's blanks. Defaulting to N0 and presenting cryptic condition names made this misleading.

## Changes

- Recognition defaults to N4_H4; explicit N0 links are preserved and prominently marked as empty-list controls.
- Separate study-image and probe-repetition controls expose all 12 recognition conditions. The fixed three-blank interval and repeated-image semantics are stated before the player.
- Other conditions describe the quantity actually varied. Controls precede the player on small screens.
- Human judgment mode starts at frame zero, plays one pass, holds the last frame, and unlocks responses only at the end. Generator answers and outcome-dependent selectors are withheld until reveal. Inspection controls remain available; no calibrated human performance is claimed.
- Frame labels count individual study images, blanks, and repeated probe exposures. Content-hashed script/style URLs prevent stale browser assets after refresh.
- Native source, task catalog, training implementation, checkpoints, frame assets, and labels were not changed.

## Existing variants

| Task family | Existing variants | Meaning |
| --- | --- | --- |
| Image recognition | N0/N4/N12/N24 crossed with H3/H4/H5 | Distinct study-image count, then three fixed blanks, then repeated exposure to one probe |
| Cued orientation, cued motion-duration, spatial binding | D0/D4/D12/D24 | Blank delay frames |
| Orientation ring | D0 only | No delayed variants in the current catalog |
| Krauzlis cued motion | B12/B20/B28 | Baseline motion transitions, not blank frames |
| Seven sensory discrimination tasks | Native mixed difficulty examples | Two input observations; difficulty parameters vary within each catalog cell |

## Direct input verification

`audit_current_native.py` replays each fixed demo seed using the current native renderer on CPU with two numerical threads; no model is imported or updated. All 115 examples, 35 primary cells, 1,580 frame rasters, and 115 labels match after the documented display conversion `round(clip(x, 0, 1) * 255)`. Stored raw arrays and PNGs also agree under this mapping. There is no per-frame normalization, interpolation, or inserted image.

93 examples are bit-exact in float32; 22 have differences no larger than 1.1920928955078125e-07. Therefore the PNG-display claim is exact, while cross-environment float replay is not universally bit-exact. Original and current dataset-manifest hashes differ; this is retained in the report rather than overwritten. Seven renderer/catalog source hashes are unchanged. Fresh seed replay and pixel comparison establish the current displayed inputs independently of that manifest mismatch.

The original full validator was not reported as passing: its provenance check flags the changed dataset-manifest hash, and current Pillow requantization does not reproduce 25 original GIF palettes exactly. Supplemental checks verify all 115 GIFs retain their original byte hashes, native duration-expanded frame-index order, and originally recorded quantization-error statistics. GIF color is lossy; HTML playback uses the verified PNGs. Neither browser colors nor illustration timing constitute calibrated laboratory presentation.

## UI verification

21 automated player/render/app/integration tests pass, including one-pass coverage of all 115 native sequences, existing explicit N0 links, all 35 condition labels, and response/answer visibility rules. Browser checks verify the default four-image trial, 12- and 24-image controls, empty-list explanation, a completed trial unlocking responses and returning correct generator feedback, and narrow-screen control ordering without horizontal overflow. See the manuscript's `research/trial_browser_qa.json` for the observed checks. Direct `file:` navigation was not browser-automated; HTTP served the exact same delivered files.

Evidence: `verification/current_native_audit.json`, `verification/gif_timeline_current_audit.json`; manuscript `research/static_audit.json` checks packaged frames and frozen implementation hashes. This atlas audit does not measure model performance or create new scientific task variants.
