# Visual Attention and Working Memory

A research project on visual representations, motion, attention and working memory. Jeremy Wolfe's Guided Search 6.0 is a functional scaffold; the diffuser is omitted. Treating activated long-term memory as synaptic weights is a modeling approximation. Individual successful components do not establish a complete biological implementation.

## Current result — October 4, 2026

A **fresh angular-contrastive CNN solved the simplified before/after motion comparison**: 100% balanced accuracy and AUC 1.000 on 3,072 held-out test presentations, spanning three speeds and 26°/28° direction changes. Those presentations contain 1,536 paired nuisance contexts. The network receives three ordered frames per clip, encodes them to a 128-dimensional vector, and learns an angularly graded distance objective from simulator-provided direction labels. At test time, embedding distance and one validation-selected threshold determine the change decision.

This is **not a result on the original cued, distractor-containing Krauzlis task**. The successful task uses full-field, persistent, constant-velocity dots with periodic wrapping. One training seed was run. Earlier native-task failures, the successful next-frame predictor, and the failed frozen-encoder FFN are preserved in the research record.

All current local training has finished. The latest cloud jobs were retrieved and deleted. Documentation and repository publication do not authorize a new training run.

| Start here | Purpose |
|---|---|
| [Latest technical report](SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md) | Architecture, exact loss/force, dataset, training, selection, results and limits |
| [Latest result](SecondPass/AngularContrastiveMotion/FINAL_REPORT.md) | Compact final tables and saved evidence |
| [Research history](LabJournal/RESEARCH_HISTORY.md) | What was explored, what failed, what changed and what is supported |
| [Experiment catalog](LabJournal/EXPERIMENT_CATALOG.md) | Linked inventory of experiments, diagnostics, plans and lineages |
| [Current status](LabJournal/CURRENT_STATUS.md) | Authoritative present state |
| [Lab journal](LabJournal/README.md) | Detailed experiment pages and chronology |
| [Artifacts and reproduction](LabJournal/ARTIFACTS.md) | What is versioned, what remains local and how to reproduce the current model |
| [Second-pass implementations](SecondPass/README.md) | CNN, KDA, recurrent, motion-energy, predictive and contrastive models |

## Repository organization

- [PreAttentiveVision](PreAttentiveVision/README.md): two-frame sensory encoders, the seven-task comparison and temporal accumulation.
- [WorkingMemory](WorkingMemory/TASK_BATTERY.md): memory/attention comparisons and diagnostic studies.
- [SecondPass](SecondPass/README.md): post-reset baselines, the unified task suite, later Krauzlis experiments and the new motion-learning sequence.
- [LabJournal](LabJournal/README.md): evidence-grounded history, including negative, interrupted and design-only experiments.

The reset is a boundary: deleted predecessor implementations/results are not reconstructed. The earlier [handoff](HANDOFF.md) remains a dated record of the failed lineage, rather than the current conclusion about every new architecture. Fresh initialization is the default for new architectures; trained weights are used only where explicitly requested and documented.

Research code, documentation, figures, saved scalar results and run receipts are versioned. Model checkpoints remain local under the existing `*.pt` exclusion. Browser binaries, compiler caches and disposable test directories are excluded; no research checkpoint was deleted for this publication.
