# Attention maps over immediate and remembered-scene references

[Journal index](../README.md) · [Current architecture](../ARCHITECTURE.md) · [Earlier mechanism diagnostic](15a-attention-mechanism.md)

Status: **every-timestep capture and corrected viewer completed**. Journal entry 2026-09-14T00:56:21.960683+00:00.

[Open the interactive attention-map viewer](../../WorkingMemory/AttentionMaps/index.html) · [Original report](../../WorkingMemory/AttentionMaps/report.md)

## Question and user correction

The user asked to hold off on further model experiments and inspect what the existing attention is doing: condition-averaged maps over immediate visual inputs and remembered-scene references. The broad LatentDynamics/probe queue was paused before extraction. This narrower capture uses its same unspent 1,800-second allowance, after local TrainingExposure training and evaluation had exited.

## Completed correction: every timestep of individual trials (2026-09-14T01:05:05.388626+00:00)

[Open the corrected viewer](../../WorkingMemory/AttentionMaps/index.html). It now shows **one individual movie at one exact frame**, with a frame slider/playback and task, trial and delay selection. The default displays all 13×13 receiving memory-grid sites, with a numerical value at every cell, separately for **each head and each source bank**, on a fixed 0–1 scale. It does not average across time or trials. The optional source-key view averages over queries at that selected frame only, not across frames.

The [per-frame receipt](../../WorkingMemory/AttentionMaps/perframe_receipt.json) records **48 movies**: eight paired base trials for each of three task families, presented atD 0/D24 (24 underlying base trials). Every timestep is retained, giving **912 frame observations**. The extra frozen capture took **6.76 seconds**, within the original2026-09-14T01:21:53UTCdeadline; added budget is 0. The checkpoint remained unchanged and no training or interventions occurred. This is separate from the earlier 192-base phase-summary capture, not a replacement count for that dataset.

Saved [per-frame maps](../../WorkingMemory/AttentionMaps/perframe_maps.npz), [movie/frame metadata](../../WorkingMemory/AttentionMaps/perframe_movies.json), [capture code](../../WorkingMemory/AttentionMaps/perframe.py) and [viewer code](../../WorkingMemory/AttentionMaps/frame_view.py) support replay without another model pass. Actual immediate scenes and remembered-scene references remain labelled. A reference image behind a memory grid is spatial context, not a decoded reconstruction.

The old single-query figures and phase averages below remain historical summaries. They must not be called instantaneous full-grid maps. The corrected source [report](../../WorkingMemory/AttentionMaps/report.md) makes the distinction explicit. No future model experiment, probe fit, optimizer change or cloud run is launched by this presentation correction.

## Presentation correction requested by the user (2026-09-14T01:02:13.120024+00:00)

The first promoted static figure showed connectivity for **one receiving query**, while the overview statistics averaged whole phases and then episodes. Those are valid summaries of the captured weights but **did not satisfy the user's request to see the full memory grid at each timestep**. A one-cell connectivity picture and a phase-average map should not have been presented as that deliverable. Historical averages below remain labelled as averages.

The subsequent CPU overview showed all 169 receiving cells, but it still averaged time. The user rejected phase/time averaging as the primary view. At this earlier correction point, researchers began an **individual-trial, every-timestep viewer** for all three task families atD 0/D24, with eight paired base trials per family, within the original AttentionMaps deadline2026-09-14T01:21:53UTC. Model weights/tasks stay fixed; this is a correction to the requested observation and presentation, not new model training or an intervention. The completed coverage and receipt are recorded above.

### Why one receiving-query map looked like one cell

The [cached-map locality summary](../../WorkingMemory/AttentionMaps/locality_summary.json) records trained squared-grid-distance penalty coefficients 4.03160143 and 4.02760696 for the two heads, read from the frozen checkpoint onCPU. The attention-logit term is $-\operatorname{softplus}(\lambda_h)d_{ij}^2$. Before content scores, an adjacent position is weighted by approximately$e^{-4.03}=0.0178$relative to the same coordinate.

For representative orientation sample/blank/probe matrices, approximately 93.76–94.09% of **joint visual-plus-memory attention mass** remains at the query's own spatial coordinate, averaged over 169 queries. This is evidence of strong local routing in those representatives, not a population-wide estimate across all tasks. At that coordinate, the head can still choose between current sensory and old-memory content.

Therefore a poor default picture did not show that the network “failed to attend.” It conflated the view's single-query scope with a strongly local architecture. Conversely, the locality measurement does not prove task-relevant memory content or a global spatial-selection capability. Full-grid per-timestep source allocation is the appropriate next display for the user's question.

## Initial capture (historical phase-averaged dataset)

Frozen **attention 8400**, the parent of the new allocation comparison, on unchanged existing tasks. No parameter updates, probe fits, interventions, UMAP or t-SNE were run. The model still has two learned heads, **each querying both sources jointly**. For each of 169 receiving memory queries,338 keys comprise 169 current-sensory positions and 169 old-memory positions. Visual plus memory mass sums to 1 per query.

The capture completed 192 paired base episodes:64 each for single orientation, two-location binding and eight-transition motion duration. Pairing acrossD 0/D24 gives 384 presentations, not 384 independent bases. There are 208 condition groups. The completion receipt records 30.45 seconds elapsed and an unchanged checkpoint.

## Measured phase-dependent source allocation

Each episode first contributes its average within a phase; independent episodes then receive equal weight. These totals describe **source allocation**, not spatial focus or preserved feature content.

| Task | Phase | Head | Memory source mass | Episodes |
|---|---|---:|---:|---:|
| orientation_single | instruction | 1 | 11.98% | 64 |
| orientation_single | sample | 1 | 11.91% | 64 |
| orientation_single | blank | 1 | 12.17% | 64 |
| orientation_single | query | 1 | 12.26% | 64 |
| orientation_single | probe | 1 | 12.10% | 64 |
| orientation_single | instruction | 2 | 13.36% | 64 |
| orientation_single | sample | 2 | 38.61% | 64 |
| orientation_single | blank | 2 | 77.75% | 64 |
| orientation_single | query | 2 | 77.09% | 64 |
| orientation_single | probe | 2 | 39.77% | 64 |
| orientation_binding | instruction | 1 | 11.98% | 64 |
| orientation_binding | sample | 1 | 12.11% | 64 |
| orientation_binding | blank | 1 | 12.34% | 64 |
| orientation_binding | query | 1 | 12.35% | 64 |
| orientation_binding | probe | 1 | 12.32% | 64 |
| orientation_binding | instruction | 2 | 13.36% | 64 |
| orientation_binding | sample | 2 | 76.71% | 64 |
| orientation_binding | blank | 2 | 84.71% | 64 |
| orientation_binding | query | 2 | 82.24% | 64 |
| orientation_binding | probe | 2 | 74.94% | 64 |
| motion_direction | instruction | 1 | 11.98% | 64 |
| motion_direction | moving | 1 | 11.89% | 64 |
| motion_direction | blank | 1 | 12.25% | 64 |
| motion_direction | report | 1 | 12.31% | 64 |
| motion_direction | instruction | 2 | 13.36% | 64 |
| motion_direction | moving | 2 | 27.39% | 64 |
| motion_direction | blank | 2 | 85.19% | 64 |
| motion_direction | report | 2 | 81.15% | 64 |


Head 2 shifts toward memory during blanks: single orientation 77.75%, binding 84.71%, and motion 85.19%. For single orientation it allocates 38.61% to memory at sample and 39.77% at probe; during moving-dot evidence it allocates 27.39%. Head 1 stays around 12% memory mass. This is learned source specialization in this fitted model, not a hardwired visual-head/memory-head division.

## How to read the initial viewer (historical summaries)

- Each head/source bank is displayed separately. Raw joint weights share a color scale; conditional maps normalize within a source and must be read alongside the original source mass.
- Source-key maps ask **where the receiving population obtains input**. Receiving-site allocation maps ask **which memory sites choose each source**. They summarize different axes of the attention matrix.
- Query-averaged maps may look diffuse even if individual queries are sharply local. The representative viewer exposes the full matrix for a clicked receiving query.
- The 13×13 grid is enlarged onto 100×100 scene coordinates. It denotes encoded receptive-field locations, not pixel-level attribution. The visual field itself contains opponent temporal history.
- Memory overlays use an earlier sample/final-moving scene as a labelled spatial reference. They are not reconstructed remembered images. An initial zero-valued memory can receive softmax mass without supplying useful memory content.
- Single-orientation averages split changed/unchanged, motion splits duration-winner direction, and binding keeps exact native location pairs separate. Actual small-group sample sizes remain visible. Condition means overlay a representative scene from the matching group rather than an average scene.

## Interpretation and decision

The maps make the trained routing inspectable and complement the earlier aggregate source-mass summaries. They do not independently establish causal necessity, active refresh or a decoded orientation/duration representation. The earlier frozen exclusion diagnostic supplies the separate evidence that blank-period memory-source routing matters for orientation; even that intervention entangles memory-value access with renormalization toward blank input.

No spatial-focus conclusion is inferred from source totals alone. The immediate next step is user inspection of the requested viewer. Broader latent/probe work remains paused and no new model experiment is launched.

## Evidence and closure

- [Capture README](../../WorkingMemory/AttentionMaps/README.md)
- [Report](../../WorkingMemory/AttentionMaps/report.md)
- [Interactive viewer](../../WorkingMemory/AttentionMaps/index.html)
- [Numerical summaries](../../WorkingMemory/AttentionMaps/summary.json)
- [Configuration](../../WorkingMemory/AttentionMaps/config.json)
- [Completion receipt](../../WorkingMemory/AttentionMaps/completion_receipt.json)
- [Capture implementation](../../WorkingMemory/AttentionMaps/run.py)
- [Viewer/report implementation](../../WorkingMemory/AttentionMaps/render.py)
- [Paused broader-analysis queue](../../WorkingMemory/LatentDynamics/queue_receipt.json)

Per-trial maps/metadata, condition means, representative full matrices and static figures are retained for replotting without repeating model execution. Root coordination reports no active owned TrainingExposure, AttentionMaps or LatentDynamics Python workers at closure.
