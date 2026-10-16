## 19-task search-suite continuation active - October 9, 18:10 PDT

User explicitly requested training from the last completed checkpoint. MemoryGuidedSearch/LocalRuntime/run01 continues VDA best16000, preserving all86 old Adam states and exact parent weights; three new four-location heads are fresh. Shared architecture/loss/Adam3e-4/clip100/batch32micro4/native task laws unchanged. Balanced19 tasks/437conditions, up to19000 additional updates under one eight-hour local allocation from18:08:57PDT, hard deadline October10 02:08:57PDT; no automatic restart/extension/cloud. Supervisor64808/worker52248 independently live-verified. Saved19 CPU-reloaded: all92parameter tensors advanced,92finite aligned Adamstates, all19tasks updated, original parent hash unchanged,29 imported source files copied and hash verified. Initial/best/latest preserve optimizer/scheduler/streams/allRNG state. Eleven contract tests passed. Monitoring uses a prespecified71-cell panel with32trials/cell atbaseline/19/every380/endpoint, preserving equal-task AUC selection; selected model full437-cell validation64 and fresh test256 are attempted within the same cap. Initial panel duration about60seconds; this is an update ceiling, not a promise of completion. Receipt MemoryGuidedSearch/LocalRuntime/run01/independent_production_verified.json. Separate RunPod Doom run is unchanged. Previous preparation-only search entries below are historical.

## Motion-memory search added - October 9

Prepared SecondPass/MemoryGuidedSearch v3 expands to19 tasks/437conditions with a third new four-location head. Motion memory: central random-dot sample, blank delay, then four simultaneous dot patches; choose the matching direction despite independent dot layouts. S3/6/12 x D0/4/12 x H3/6/12 x direction spacing20/45/90 gives81conditions. Coherent speed1.5pixels/frame,12dots,28pixel worlds/circular apertures; no direction/flow metadata enters model inputs or losses. Counterfactual queries preserve entire arrays and initial sample frame while changing only sample motion and correct answer; duration variants use exact prefixes. Motion stream42; prior18 tasks and their204xxx seeds preserved. Total382434parameters/92tensors, all86 parent Adamstates retained. Eleven CPU contract tests passed; current forward/example receipt is in preparation_verified.json. No local training launched or queued. Related neuroscience: Mendoza-Halliday et al.2014, doi10.1038/nn.3785; four-choice search and timings are adaptations. See MemoryGuidedSearch/README.md. Earlier preparation entries below are historical.

## Search suite expanded with pop-out and independent viewing durations - October 9

Prepared SecondPass/MemoryGuidedSearch v2 now has 18 tasks and 356 conditions. Memory-guided search uses independent S1/2/4/8 sample durations, D0/4/12 delays and H1/2/4/8 array durations across three families and two similarity levels (288 conditions). New popout_search uses color differences12/36/72 or orientation differences5/15/45 degrees crossed with H1/2/4/8 (24 conditions), no sample, four-location output. Matched timing comparisons preserve exact scene pixels and answers. Two new 64-to-4 heads add520 parameters; total382174 parameters/90 tensors; original86 Adam states and VDA best16000 are preserved. Fresh204xxx streams, balanced18-task schedule. Eight CPU tests and312 new-condition forward checks passed; exact-frame examples and preparation receipt retained. No local training started or queued. Old16 task laws and completed checkpoints remain unchanged. Earlier search-v1 preparation below is historical. See SecondPass/MemoryGuidedSearch/README.md.

Latest preparation: [Memory guided search](../SecondPass/MemoryGuidedSearch/README.md), one new task with three stimulus families,17tasks total.

Current: [VDA frozen analysis complete](vda-attention-analysis.md). Training finished at 16,000 updates. All 131,616 analysis presentations verified; the requested VDA-4 reciprocal effect is present. Offline gallery, psychometrics, attention maps and interventions are ready. Browser preview was policy-blocked; static exports verified. No training queued; Doom remains unstarted. Earlier status entries below are historical.

Current: VDA continuation is running again from1280 in LocalRuntime/continuation01; saved1296 independently verified October9. Same16000-update target and unused prior local allocation. Doom remains prepared only. Previous pause entries below are historical.

Current: VDA continuation is paused at1280 at user request; checkpoints and full training state preserved. Doom remains prepared, not running. No automatic restart. See the VDA run01 pause_verified.json receipt. Earlier active descriptions below are historical.

Current: [VDA suite continuation and separate Doom preparation](vda-suite-and-deadly-corridor.md). VDA4/9/16 join the completed13-task model; continuation is running and independently verified. Doom is prepared only.

[October5 — local-comparison Krauzlis](local-comparison-krauzlis.md): user-approved targeted successor following frozen-model diagnostics.

[October5 — native Krauzlis shared attention](shared-attention-krauzlis.md): user-directed cancellation and fresh single-head experiment.

[Correspondence predictive memory](correspondence-predictive-memory.md) — next local experiment, queued October5.

# Research journal

The journal records what the project tried, how it was trained, what the saved evidence shows, and what remains unresolved. Final held-out results, live validation, frozen diagnostics, cancelled runs and design-only proposals are labeled separately. Fresh and transferred-weight lineages are not interchangeable.

## Start here

| Document | Purpose |
|---|---|
| [Current status](CURRENT_STATUS.md) | Actual present state and latest completed results |
| [Latest technical report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md) | Complete angular-contrastive model and experiment specification |
| [Research history](RESEARCH_HISTORY.md) | Narrative of the experiments and changing interpretations |
| [Experiment catalog](EXPERIMENT_CATALOG.md) | Full linked inventory, status and lineage |
| [Chronology](CHRONOLOGY.md) | Dated operational history, including interrupted runs |
| [Architecture](ARCHITECTURE.md) | Current encoder and references to historical alternatives |
| [Artifacts and reproduction](ARTIFACTS.md) | Versioned evidence, local checkpoints and execution limits |
| [Tasks and metrics](TASKS_AND_METRICS.md) | Objectives, denominators and reporting conventions |
| [Open questions](OPEN_QUESTIONS.md) | What the evidence has not established |

## Motion-learning experiments

- [Outcome-contrastive RViT with spatial GRU](outcome-contrastive-rvit.md): traininglocallytoward25024updates; full-batchterminalmemoryCLScontrast+choicefeedback;8-trialmemorizationpassed35updates, nofinalresultyet.
- [Faster-dot transformer rerun](weighted-sum-transformer-speed4.md): completed25,024updates at speeds1.5/4/8; finalBA49.772%, AUC0.498358 remainschance.
- [Direct weighted-image transformer](weighted-sum-transformer.md): completed25,024 end-to-end updates; final BA49.25%, AUC0.492292 remained at chance.
- [Weighted summed-image VAE](weighted-sum-vae.md): completed25,024 VAE/10,000 decoder updates; weighted reconstruction collapsed, final changeBA49.71%, AUC0.500107.
- [Predictive consistency VAE](predictive-consistency-vae.md): completed20,032updates; three context/three future withview consistency; finalchangeBA50.42%, AUC0.504076.
- [Pixel-derived contrastive VAE](pixel-contrastive-vae.md): pixel-loss arm completed near chance; plain-VAE control cancelled by the user, with checkpoints preserved.
- [Angular contrastive CNN](angular-contrastive-motion.md): fresh angularly supervised representations; simplified change comparison solved.
- [Frozen predictive encoder + FFN](predictive-motion-change.md): simple concatenation readout remained at chance.
- [Variational prediction](variational-motion-prediction.md): three past frames to one flattened stochastic vector to a predicted fourth frame.
- [Spatial VAE](krauzlis-three-frame-vae.md), [VAE encoder + RViT](vae-encoder-rvit.md), [input×10](vae-rvit-input10.md): reconstruction and explicitly authorized encoder transfers.
- [Weighted CNN–GRU](krauzlis-weighted-mean-conv-gru.md), [weighted CNN–RViT](weighted-mean-rvit.md), [two-epoch replay](weighted-mean-rvit-epoch2.md): temporal input aggregation and data refresh.
- [Single-stimulus diagnostic](krauzlis-single-stimulus.md), [motion energy](krauzlis-simoncelli-heeger.md), [training-path audit](krauzlis-training-path-audit.md), [upstream motion](krauzlis-upstream-motion-diagnostic.md): rendering, sensing and optimization checks.
- [RViT](krauzlis-two-frame-rvit.md), [replay](krauzlis-rvit-replay.md), [random-frame gradients](krauzlis-random-frame-gradients.md), [delayed CNN–GRU](krauzlis-delayed-frame-gru.md): temporal decision architectures.
- [Single-layer KDA](krauzlis-sequence-kda-design.md), [three layers](krauzlis-sequence-kda3-depth.md), [16 heads](krauzlis-sequence-kda16-heads.md): whole-sequence alternatives.

## Earlier program

The [catalog](EXPERIMENT_CATALOG.md) links all numbered experiments in `experiments/01-…` through `27-…`: sensory encoder comparisons, temporal accumulation, memory/attention variants, diagnostic readouts, training allocation, reset/audit and plain/KDA/ConvGRU baselines. Later [joint-suite](joint-suite-training.md), [task-suite](task-suite-assembly.md), [spatial readout](spatial-readout-convgru.md), [neuroscience analysis](joint-kda-architecture-microstimulation.md) and Krauzlis diagnostics continue that record.

The repository reset remains a boundary. This history uses retained post-reset reports; it does not reconstruct deleted earlier code or replace negative results with later successes.

## Maintenance

[Maintenance guide](MAINTENANCE.md) · [Experiment template](EXPERIMENT_TEMPLATE.md) · [Research foundations](RESEARCH_FOUNDATIONS.md)

Superseded index/status text is retained in [archive](archive/). The original `build_initial_journal.py` assembled the first journal and must not be rerun over subsequent edits. Add source links and date corrections when evidence changes; do not turn a documentation update into another training run.
