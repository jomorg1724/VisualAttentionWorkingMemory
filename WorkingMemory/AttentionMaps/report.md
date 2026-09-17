# Current viewer: every timestep, no temporal averaging

The default [viewer](index.html) now shows one individual trial at one exact frame, with a slider/playback for every timestep of all three tasks and D0/D24. Eight paired trials per task/delay were captured with the unchanged frozen model within the original attention-visualization deadline. All13×13receiving sites are shown for each head/source on a fixed0–1scale. The alternative key view averages only over queries at the selected frame.

The older figures and sections below are **phase averages**, retained as historical summaries. They must not be described as instantaneous maps. See [per-frame receipt](perframe_receipt.json).

---

# Attention maps: current visual input and memory

Frozen attention checkpoint8400; unchanged existing tasks. This captures attention weights, not a new model experiment or an intervention.

[Open the interactive scene-overlay viewer](index.html) · [Capture protocol](config.json) · [Numerical allocation summary](summary.json)

## What the network contains

Two learned heads each query both sources jointly. They are **not** a dedicated visual head and a dedicated memory head. Each head produces169 receiving queries over338 keys:169 current sensory-field keys followed by169 previous-memory keys. Per query, visual mass plus memory mass is1.

Maps are projected from the13×13 encoded grid to the100×100 scene for display. They index encoded receptive-field positions, not pixel attribution. A memory overlay on an earlier sample/reference provides spatial context; it is not a reconstruction of the remembered image.

## Phase means at D24

Each episode first contributes its mean across frames within that phase; then independent episodes are equally averaged.

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

## Averaging and interpretation

- Single-orientation maps split changed/unchanged trials. Motion maps split winner direction. Binding maps group the exact two native item locations; small groups display their actual n and are descriptive. We do not blur together displaced objects and call the result one spatial focus.
- Query-averaged source-key maps show where the whole population draws content. Averages can look diffuse even when each query is sharply local. The representative viewer also exposes the full matrix for a clicked receiving query.
- Raw joint weights use a common color scale across both heads and sources. Conditional maps explicitly normalize inside each source; their geometry must be interpreted with the displayed original source mass.
- Receiving-site allocation maps ask which memory grid sites draw from visual versus memory keys. Source-key maps answer where their inputs originate. These are different axes of the attention matrix.
- Current scenes are actual captured frames. Earlier sample/final-moving scenes behind memory maps are labelled remembered-scene references. Condition means overlay an example from the same location/condition group, not an average image.
- The model, sensory traces and tasks are unchanged; one ordinary-forward equality check confirms observation. Attention weights alone do not establish causal importance or prove a preserved feature.

Saved artifacts include per-trial maps and metadata, condition means, representative full matrices, static figures, numerical summaries and a standalone HTML viewer. No UMAP, tSNE, latent probes, additional training or interventions were run in this scoped visualization.


## Corrected overview: why the first display looked like one cell

The first promoted figure displayed connectivity from one receiving query, not the full13×13population. That was an unsuitable default. The single cell also reflects real strongly local connectivity: across representative orientation frames, approximately94% of joint weight stays at the same spatial coordinate. The trained squared-distance penalty coefficient is about4.03, giving an adjacent-coordinate factor exp(-4.03)≈0.0178 before content scores.

The corrected default shows all169 receiving memory-grid sites and their visual-versus-memory allocation, with one fixed0–1scale. [Sample/blank/probe overview](figures/phase_allocation_overview.png) shows all64orientation episodes; [query-averaged source keys](figures/phase_source_keys_overview.png) are a separate view and should not be called object saliency. [Locality measurements](locality_summary.json) use cached matrices only. The earlier blank_query_maps figure is preserved solely as advanced single-query connectivity. No new capture, GPU work or model experiment was performed for this correction.
