# Sensory editorial verification and integration notes

## Deliverables

- `content/sensory_tasks.json`: exactly the seven catalog sensory tasks, in catalog order. Each has tile question, accessible introduction, all ten required connected-prose sections, equations, source-ID references and detailed provenance-bearing property rows.
- `content/sensory_sources.json`: ten uniquely prefixed source records (`sensory-*`). Records separate experimental literature, official dataset documentation and implementation-source provenance. They state access depth, supporting locations, exact evidence excerpts, supported claims and non-supported inferences.
- `verification/sensory_validation.json`: real static validation receipt. The present version verifies 7 tasks, 138 property rows, 22 inline citation occurrences, 17 verbatim evidence excerpts and 31 exported sensory-episode worked explanations. Main editorial prose has 7,794 words excluding property tables and worked examples.

## Worked-example integration

Every sensory task includes `worked_examples`, an object mapping **exact exported episode ID → connected explanation string**, generated only from the current asset manifest. This is ready for the detail player: prefer `task.worked_examples[episode.id]` after answer reveal. It states actual metadata, explicitly labels calculated/rounded quantities and explains why the other answer is wrong. `showcase_worked_example` records the default episode and text; it must not be displayed as though it describes a different selected episode. `worked_example_template` documents how to regenerate explanations if the exporter changes examples. No stimulus generation or pixel analysis was done by this author.

These answer-bearing fields belong behind the same Try/answer-visibility gate as label metadata. They are not observer input. The content schema's ten required section keys are unchanged; worked explanations are additional root fields rather than an invented eleventh section.

## Source access and evidence

The configured web search/extract backend failed with Firecrawl HTTP 403. Direct HTTPS retrieval was successful for PMC and official/author PDF URLs. All source bodies and retrieval records are saved under `verification/sensory_*`.

Six experimental papers were retrieved in full and their relevant methods read: Lovejoy–Krauzlis, Zhang et al., Legge–Foley, Putzeys et al., Krauskopf–Gegenfurtner, and Field–Hayes–Hess. The latter three were scanned PDFs without a text layer; local Apple Vision OCR, with its CPU-only flag set, produced complete page-numbered transcripts. The script and transcripts are retained. OCR has ordinary typography/column-order limitations; quotations were confined to readable prose and verified against the saved transcripts. This was document OCR, not stimulus rendering or a task/model experiment.

Tadmor–Tolhurst is **abstract-only**: Europe PMC PMID 8303837 returned the published abstract; Crossref confirmed bibliography; Elsevier returned metadata only and identified non-open-access status. Prose restricts attribution to the abstract's general experiment and restricted-frequency-band account. It does not invent their methods, filter equation, image count, timings or numerical thresholds.

Campbell–Nachmias–Jukes is **metadata-only / publisher blocked**: Optica redirected to a bot-verification page; Crossref confirmed title/authors/year/DOI; Europe PMC lookup attempts failed. No ratio result or methods detail is attributed from that access. The accessible Putzeys primary psychophysics provides the actual experimental anchor for higher-frequency interval judgments. Campbell remains transparently marked as the historical starting reference, not falsely marked full-text verified.

The official BSDS500 page was retrieved in full. It supports dataset identity, disjoint subsets and requested attribution, but did **not** establish an unrestricted redistribution license. Local train-photo delivery is not public-publication permission. `PreAttentiveVision/data/bsds500/README.md` is absent; no contents were invented. The full manifest was parsed and gives 200 train, 100 validation and 200 test source identities, all 500 unique.

SciPy 1.13.1's tagged `_filters.py` was retrieved from the official repository because the exporter records that version. Its exact Gaussian default signature, normalized finite kernel, radius rule and separable application were read. This is a software source, not neuroscience evidence.

A task-scoped numerical citation ledger (`sensory_citations_ledger.json`) was populated from retrieved URLs before source-bound prose. The public content uses contract-required `sensory-*` aliases; each source carries its numeric ledger ID. Seventeen `sources.py quote` calls passed literal-text checks (`sensory_quote_checks.json`). Crossref corroboration of the scanned Field and Legge DOI records is retained. The stable author PDF URL suffices for Krauskopf where the additional Crossref request failed.

## Source closure and important corrections

Executable definitions read before drafting: `TaskStream`, `CardinalMotionStream`, `_render`, `_grating`, `_achromatic`, all six procedural task methods, `NaturalSpectrum.__init__`, `_image`, `sample`, dataset-preparation provenance, and suite dispatch/metadata/seed rules. Supplemental source notes were treated as starting points, not implementation authority. Recognition's `PhotoSet.image` line was inspected only to substantiate the contrast with its central RGB `ImageOps.fit` preprocessing. No renderer was imported or run.

- Motion: 256 domain identities, not 256 visible dots; exactly 128 survivors and 128 rebirths; common survivor direction, not a coherent-percentage sweep. Bilinear deposit → finite wrapped Gaussian → aperture → intensity clipping. No sensor-noise field.
- Orientation: numeric signed axial comparison and doubled-angle wrapping; carrier wavevector is not visible stripe orientation. Independent phases and separately peak-normalized patterns retained. No clockwise relabeling.
- Contrast: intensity modulation amplitude, **not calibrated Michelson contrast**. Shared carrier pattern, nine pedestal × increment combinations, independent noise; no post-amplitude RMS normalization.
- Spatial frequency: fixed envelope, octave ratio, independent phases; shared peak multiplier does not imply exactly matched raster RMS. Finite base-frequency ranges can provide absolute-frequency priors; random ordering alone does not prove all one-frame statistics uninformative.
- Chromatic: declared numerical RGB axis orthogonal to weighted luminance, **not calibrated isoluminance or cone isolation**. Noise is independent across channels as well as intervals; ideal-color luminance metadata is pre-noise. Only the edge mask is clipped; no gamma or final color clipping.
- Contour: path carrier is tangent + π/2 + jitter; orientation multiset is permuted while phase stays at each position. Continuous minimum spacing precedes integer rounding. No full-field mean/RMS normalization. Controls can retain partial or accidental alignment, and exact path coordinates are absent from metadata.
- Natural spectrum: native random luminance crop, not recognition's resized central RGB crop; shared augmentation, positive radial Fourier weights, zero DC, independent standardization then **pairwise common** RMS/gamut bound, no clipping or added noise. Beta is an applied multiplier exponent, not the photograph's measured spectral slope. The battery wrapper overwrites sample protocol metadata, while transform identity remains in `NaturalSpectrum.identity`.

Remaining finite-range absolute-value priors are explicitly disclosed for amplitude, frequency, chromatic and spectral tasks. Computational sections describe logical information requirements and possible solutions, not diagnoses, unique architecture requirements or observed model abilities.

## Verification command and scope

From repository root:

```
python SecondPass/TaskSuite/Demo/verification/sensory_build_content.py
python SecondPass/TaskSuite/Demo/verification/sensory_validate.py
```

Both are stdlib-only editorial/metadata operations. The validator checks exact sensory inventory, section/property/source schemas, inline reference resolution, relevant experimental anchor per task, source-line ranges, verbatim excerpts, all exported sensory example labels and arithmetic, train-only example metadata, and unchanged hashes of both sensory renderer files, suite adapter, catalog and dataset manifest. It does not render, import the suite, restore streams, read checkpoints, score a model, run training or access cloud infrastructure. The author wrote only the assigned two content files and `verification/sensory*`. No commit, journal update or skill modification was made because the write scope explicitly excludes them.
