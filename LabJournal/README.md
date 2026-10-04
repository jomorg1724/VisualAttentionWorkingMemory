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
