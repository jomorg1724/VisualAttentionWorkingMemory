# Teaching revision reviewer checklist

Final PDF SHA-256: `71e0e4cedc3dd93ac2752f7dbbe6e9a9fe67e5efddeb07254bae414cf1ed5248`

## Explanatory contract
- [x] Reader question in every section; explanatory operations and takeaways, not a longer equation list.
- [x] Dedicated notation page: batch/time/scale/site/head/task indices, shapes, operations, learned weights versus input-generated activations versus derived diagnostics.
- [x] Overloaded `P` removed. Learned CNN projection is `W_proj,s H + b_proj,s`; retention transport is `T_{t←τ}`, distinct from sequence length `mathcal T`.
- [x] q/k/v roles are functional, not asserted object/angle codes; notebook analogy labeled.
- [x] D is defined from retention gates before use; decay/prediction/error/correction/read have shapes and entrywise explanations.
- [x] A and J are named only after full substitution/factoring; J is not the implemented corrective write by itself.
- [x] S_-1, S0, S1 and S2 expanded before defining transport; chronological product order and empty-product identity explained.
- [x] Coefficient formula derived by transposition with explicit product shapes; signed toy coefficients shown, not probabilities or causal final-choice attribution.
- [x] Optional norm argument defines matrix norm, assumptions, orthogonal components and successive-product inequality; not a claim about fixed behavioral lifetime.
- [x] ConvGRU write versus reset derived operationally with exact fraction example; zero bias not falsely equated with half gates.
- [x] Terminal affine maps declare weights/biases and output sizes; parameters distinguished from episode states and training-graph storage.

## Source accuracy and provenance
- [x] Stack3 oldest-to-current order, centering before zero history padding, first/second inputs, and nominal blank leakage match source.
- [x] Pointwise channel projection distinguished from convolutional neighborhood and stride-based spatial reduction.
- [x] GroupNorm spatial statistics explicitly rule out isolated-site interpretation.
- [x] Task-known head selection shown separately from visual cue processing; no task-ID inference claim.
- [x] CPU constructor re-run: 1,383,028 parameters, 13 tasks, 35 conditions; no forward/backward/checkpoint/accelerator.
- [x] All 15 model/authority hashes unchanged; inherited implementation and current training origin are distinguished.
- [x] Fresh whole-model origin and authorized same-lineage continuation preserved; no latest training or performance claim.
- [x] Existing five verified references retained without new literature-performance claims.
- [x] Cognitive limits and proposed, not run, discriminating ablations retained.

## Arithmetic and delivered PDF
- [x] `verify_toy.py` exact Fractions checks unit q/k, valid gates, all displayed toy update intermediates, third update, A/J equality, transport expansion and coefficient/value reads. This is not a model result.
- [x] Original ten-page PDF/source/scripts/receipts/renders preserved before editing in `versions/v1_10page/`.
- [x] Canonical source rebuilt with Tectonic: 15 pages including references; no TeX warnings/overflows.
- [x] `verify_pdf.py` retains all original checks and now tests 15 section/page alignments, budget, metadata, toy pass, no P overload and preserved original PDF.
- [x] Embedded fonts, replacement glyphs, page text bounds and source hashes verified.
- [x] Every page viewed; final altered pages 3/7/8 reinspected. Visual receipt bound to final PDF digest.

Limits: readability is supported by the actual reorganization and visual review, not guaranteed by a page count. This audit neither tests learned feature semantics nor establishes behavioral competence. No model/training/analysis source was changed.
