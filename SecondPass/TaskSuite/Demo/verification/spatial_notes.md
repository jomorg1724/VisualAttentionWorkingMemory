# Spatial content handoff and verification

## Delivered

- `content/spatial_tasks.json`: six exact task IDs (ring plus five spatial families), all ten required connected-prose sections, equations, metadata-bound worked-example specifications, and property rows with all seven required fields.
- `content/spatial_sources.json`: collision-safe `spatial-…` IDs, primary empirical anchors for every task, claim/support exclusions, access-depth statements, literal checked excerpts, local source paths and source SHA256 values.
- `verification/spatial_validate.py`: reproducible standard-library static validation. The actual pass receipt is `spatial_validation.json`: 6 tasks / 28 catalog cells / 136 property rows / 114 section paragraphs / 13 source records / 10 checked literal excerpts.

The source files were read as text, not executed. The six native/adapter source hashes still match their initial inspection values. No native rendering, model execution, model source inspection, checkpoint access, stream restore, runtime/training-artifact read, training, cloud action, upstream edit, journal edit or commit was performed. The installed Pillow library was imported only to inspect `ImageOps.fit` source/signature; no photograph was rendered or opened by this content author.

## Source depth

| Source | Verified depth | Evidence |
|---|---|---|
| Arcizet & Krauzlis 2018 | Direct PLOS full text, Methods “Motion-direction CD task” and Figure 1 text | `spatial_arcizet_fulltext.txt` |
| Mante et al. 2013 | PMC full author manuscript plus Europe PMC XML; context task and Figure 1 | `spatial_mante-2013_retrieved.txt`, `spatial_mante-2013-xml_retrieved.txt` |
| Roitman & Shadlen 2002 | PMC full text; Direction discrimination task and Figure 2 | `spatial_roitman-shadlen-2002_retrieved.txt` |
| Brady et al. 2008 | PMC full text; repeat detection and forced-choice methods | `spatial_brady-2008_retrieved.txt` |
| Griffin & Nobre 2003 | Europe PMC bibliographic record and complete abstract only | `spatial_griffin-nobre-2003-abstract_retrieved.txt` |
| Luck & Vogel 1997 | Nature publisher abstract and Europe PMC complete abstract only | `spatial_luck-publisher_retrieved.txt`, `spatial_luck-vogel-1997-abstract_retrieved.txt` |
| BSDS500 | Official Berkeley resource page; dataset purpose, partitions and requested citation | `spatial_bsds500_retrieved.txt` |

Firecrawl search/extract failed with service-level HTTP 403. Direct HTTP retrieval recovered the four full texts and dataset page. Griffin publisher access returned 403; the metadata identifies subscription-required full text. Luck publisher text exposes abstract/access options, not full methods. Those two sources support only explicitly qualified abstract-level claims. The short initial PubMed `_retrieved.txt` files are cookie-interstitial evidence, not article text; they are not used as supporting sources. No unverified full methods, effect sizes, human capacity estimates or neurophysiological mechanisms are attributed.

The expected local BSDS500 README did not exist. No README facts were invented; source preparation code and official Berkeley provenance anchor the content. Actual demo manifest/photo integrity belongs to the asset parent. This content does not certify image redistribution rights.

## Code-trace findings preserved in the prose

- Ring: D0 only; local ring, never plus/minus despite metadata `cue_sign=1`; all four rotations nonzero; foil signs independently drawn; eight-case label/location cycle. Positive/negative numerical axial signs supersede legacy clockwise wording.
- Shared Gabors: Gaussian envelope sigma 4.5 and radius-12 support; fresh pixel noise SD .008; wavelength Uniform[5,7), amplitude Uniform[.28,.36), both shared across patches within a rendered frame but redrawn between frames; independent per-patch/frame phase. Two samples preserve angles, not pixels.
- Shared cue helper: four reserved 10×10 masks; recall/integration and phase bit glyphs; local plus/minus is translated from the original lower-left mask. No threshold or identity glyphs are passed. Gray retention frames still carry corner markers except recognition; Krauzlis does not call `visual_cues`.
- Signed orientation: the sign-relative change multiset is formed before target assignment; both signs and unchanged values exist on every trial. Negative subtype is an equiprobable draw, not an exactly balanced finite queue. `rotation_magnitude` is not the actual target change when the target is unchanged.
- Duration: eight transitions and nine dot images; first/last directions drawn independently before bounded conditioned rejection; 2048-attempt fallback fixes the six middle steps to the winner. Unique plurality is not necessarily a majority or a longest run. Exactly 16 random resets plus exits, not a ten-frame lifetime; integer raster overlap writes do not add intensity.
- Krauzlis: bilinear additive subpixel dots with clipping; offsets persist until birth, including across the event; staggered `arange(16)%10` initial ages are nonuniform. Exact 57/29/14 event cycles, side parity reversed next cycle. Sign/magnitude are equiprobable draws, not exactly balanced event subcycles. Catch has an unused sampled magnitude/sign but no physical direction change. Primary 26°/28° are monkey-specific medians, not its complete stimulus support; source cue locations were blocked while project sides are interleaved. Only this task uses a biological frame clock.
- Binding: all-trial pair exchange preserves inventory and changes exactly two orientation assignments; no target cue before query. D0 still includes query. An inventory-only memory is inadequate, but neither perfect four-slot storage nor a unique mechanism is established.
- Recognition: `ImageOps.fit` explicit bicubic, default bleed=0 and centering=(.5,.5), source RGB/255 without linear-luminance conversion or spectral filtering. Sample N+1 source IDs even on positive trials; canonical cached float32 rasters; positives exact copies; N0 overrides sampled labels; H is condition-fixed. No glyph overlay on study/probe, no delay glyph, no separate report frame beyond final probe. N0 specificity/FPR only; no BA/AUC value.

## Parent integration notes

- Local executable-source ledger records intentionally have `url: null`, `doi: null`, and `repository_path`. Do not render a misleading “Primary record / external link” for these records or invent a public code URL. Display their repository provenance/hash as local-source text. Literature and official dataset records have real URLs.
- `worked_example` objects specify exact native metadata bindings. They are not fabricated sampled trials; the parent UI's selected-episode explanation supplies actual numbers, hashes, count bars and correspondences. Source/explanation metadata must remain hidden when answers are hidden.
- Every `visibility` field is populated. Groups use the contract's `selection` / `retention` labels, not catalog reporting-group names.
- Run `python SecondPass/TaskSuite/Demo/verification/spatial_validate.py` from the repository root to repeat static validation. It performs no native imports or render calls.
- The source-depth limitation is transparent rather than an unresolved factual assertion. No extra experiment is required or authorized by this content.
