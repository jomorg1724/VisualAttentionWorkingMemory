# Joint KDA architecture and proposed microstimulation

**Date:** 23 September 2026. **Scope:** user-requested ten-page technical document, source/data audit and proposed protocol; no new training, model evaluation or intervention.

[Journal index](README.md) · [Joint-training lineage](joint-suite-training.md) · [Technical report PDF](../SecondPass/JointTraining/TechnicalReport/architecture_microstimulation.pdf) · [Editable source](../SecondPass/JointTraining/TechnicalReport/architecture_microstimulation.md) · [Reproduction instructions](../SecondPass/JointTraining/TechnicalReport/README.md)

## Question

What exactly is the current fresh-weight joint-suite architecture, how could it be computationally microstimulated, and would those interventions be comparable to Morgan, Albanna & Herman's recurrent-ViT Figure 5? Include preliminary behavioral results without presenting them as final continuation outcomes.

## Verified implementation

The model uses a causal three-frame stack, four convolution/GroupNorm/ReLU blocks, local KDA fields at 25×25, 13×13 and 7×7, a dense 256-dimensional feature projection, shared 256-unit GRU and 13 externally selected linear task heads. Every learned parameter remains trainable. Total **2,750,324 parameters**; **215,808 KDA state scalars per example**, or **216,064** including GRU state. KDA carries a separate two-head 8×16 association per site; it has no competitive spatial softmax and no GRU feedback into the encoder. GroupNorm nevertheless globally couples spatial statistics. Raw frame stacking is an additional short sensory-history route.

The report derives the prediction-error update, explains conditional contraction rather than claiming global nonlinear stability, and separates stored matrix state from emitted features and the final decision route.

## Frozen preliminary evidence

The machine-readable audit froze at **2026-09-23 06:48:15 UTC**: live training step **2647 / 84,704 episodes**, latest completed validation **2314**. Fourth validation and v3 finals were pending at that cutoff; this is a historical source snapshot, not a current process-status claim.

At validation2314, equal-task mean AUC was **0.69374** versus **0.62963** at baseline715. Spatial-frequency BA was **90.63%**, contrast/chromatic **100%**, and natural-spectrum **96.88%**. The five spatial/sequence tasks averaged AUC **0.49252**; a sensory gain is not full-suite acquisition. Orientation AUC **0.74902** coexisted with BA **50%** and an all-positive confusion matrix. Krauzlis was also all-positive: per condition 57/57 target hits, 29/29 foil false reports and 14/14 catch false positives. Empty-set recognition specificity remains separate from nonempty performance.

All 13 tasks are displayed in the report; six saved evaluations each preserve the full 35-cell inventory and denominators in the companion JSON. Completed715 tests and continuation validation have different roles/draws, so the report does not treat their contrast as a paired final-test effect. Prior tests were seen; the continuation remains exploratory.

## Source comparison and protocol

Morgan et al.'s Figure 5 forces transmission toward a spatial location. The blue legend changes location between panels: it is S1 in A/B/D/E and S4 in C/F. Bias toward the *changing* location helps; bias toward the cue is not universally beneficial. Architecture equations specify source-normalized attention, but exact clamp overwrite/renormalization and duration are incompletely documented. The direct visual residual remains. Missing supplemental references in pinned v1 are not reconstructed as verified numerical results.

Proposed KDA tests use calibrated local emission pulses or rank-one state perturbations with independent calibration, sham equivalence, matched sites/epochs/doses and paired episodes. Persistent state injection and transient emission manipulation answer different causal questions. Sensitivity, criterion, foil/catch responses and task semantics are explicit. Reference any-change detection is not interchangeable with our target-only reports or sign judgments; a fixed-final-frame classifier cannot reproduce learned reaction times without a new experiment.

The scientific conclusion is functional comparability with substantial mechanistic/task differences, not an expectation of matching curves or an identified biological stimulation mechanism. Cavanaugh, Alvarez & Wurtz (2006) supplies a primary biological anchor and motivation for explicit false-positive controls.

## Deliverables and verification

Report directory: `SecondPass/JointTraining/TechnicalReport/`. Contains ten-page PDF, Markdown/LaTeX, vector architecture/validation figures, frozen audit JSON, comparison notes, verified quotations, build scripts and checks. All39 displayed metrics match the snapshot; exact PDF page count and layout were checked. Source and saved-artifact audits were independent workstreams, followed by parent verification. No checkpoint, active training source or run state was modified.

**Next decision:** inspect the authorized continuation's completed per-task results before proposing a bounded frozen intervention run. This document does not launch or authorize that experiment, alter teaching, or extend any compute budget.
