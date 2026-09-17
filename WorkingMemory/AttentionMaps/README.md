# Attention maps over current input and memory

**Current deliverable: [every-timestep viewer](index.html), completed.** It shows48individual trial movies:8trials per task atD0/D24,912captured frame observations including every blank and motion-reference frame. The default is the full13×13receiving-site allocation grid for each head/source, fixed0–1colors, numbered cells and exact-frame playback. There is no temporal or trial averaging. [Per-frame receipt](perframe_receipt.json) records the capture within the original allowance. `frame_view.py` is now the canonical viewer renderer; older `render.py` produces the historical phase-average view.

The original phase-averaged maps and single-query figure below are preserved historical summaries, not the default. The first single-query display was an inappropriate overview: roughly94% of that query's weight is local, so it mostly showed one grid cell. Whole-grid source allocation and a single query's connectivity are different views.

The user's current request is visualization of existing attention, with broader latent experiments paused. This directory captures ordinary frozen forward passes of selected attention8400 and maps the two heads' joint visual/memory weights back onto scene coordinates. It makes no model, training, task, loss or intervention changes.

The [queue receipt](queue_receipt.json) records the waiting/capture/completion state. It waits for local TrainingExposure training and final evaluation to exit. The pending latent-analysis allowance is reused: a finite 1800 seconds for capture and rendering, no second allowance. Queue wait expires at 2026-09-14 03:45:44 UTC.

Target64 independent base episodes per existing family, paired D0/D24: single orientation, exact-location two-item binding, and eight-transition motion duration. A profile may pin32 before capture if needed. Full weights have shape B×2×169×338. Both heads see both sources; source allocation is not a fixed head identity.

After completion, [index.html](index.html) provides the standalone viewer. Each head/source is shown separately, with a common raw-weight color scale and optional explicitly conditional normalization. Representative trials expose the full matrix for a selected receiving query. Condition averages save source-key maps and receiving-site source allocations, with native binding locations kept separate. Current scenes and earlier remembered-scene references are both displayed and labelled.

The visual source is the current sensory field H, which already contains short opponent temporal traces; it is not a direct raw-pixel projection. Memory keys index the previous rate field. Enlarging13×13 keys onto100×100 shows encoded spatial positions, not pixel-level saliency or a reconstructed remembered image. Attention allocation alone does not establish effective value contribution: for example, initially zero memory can receive softmax weight while providing zero memory values.

Capture code: [run.py](run.py); viewer/report: [render.py](render.py). Saved per-trial summaries and representative full matrices support future replotting without repeating model execution. PCA, tSNE, probes and interventions remain paused.
