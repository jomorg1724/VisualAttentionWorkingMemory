# Paper handoff: visual attention and working memory

> **Purpose.** This is a durable research and writing handoff, not a manuscript.
> It inventories the repository as it existed on **2026-09-15 03:59 UTC**,
> separates results from proposals and biological motivation, and defines the
> evidence and quality bar for a future paper. It does not authorize training.

## 1. Thesis, question, and evidential boundaries

### Working question

Can a compact, trainable visual system progress from parallel sensory encoding
to causal temporal integration and then to spatially structured recurrent
working memory, while retaining interpretable links to theories of visual
priority, biased competition, cortical feature coding, and evidence
accumulation?

The empirical program uses 100-by-100 RGB movies and asks where performance
fails as the system must (1) encode visual change, (2) integrate ordered
evidence, (3) retain it across delays, (4) preserve feature-location binding,
and (5) use remembered state to prioritize current input and make a decision.
The strongest narrative is not “a biological model has been validated.”
Rather:

1. lightweight encoders can support a diverse two-frame sensory battery;
2. a fixed opponent temporal computation is an effective causal front end;
3. recurrent state can retain information that the deployed decoder fails to
   use;
4. spatial E/I state and pre-update joint sensory/memory attention improve
   particular delayed and binding decisions, but tradeoffs remain;
5. global pooling is now a plausible bottleneck, while the proposed
   task-conditioned spatial priority readout has **not yet received a valid
   outcome test**.

### Goals and motivation

The scientific motivation is to build components of visual attention and
visual working memory rather than equate the entire project with
PreAttentiveVision (PAV). Jeremy Wolfe's Guided Search 6.0 provides a functional
scaffold for priority-map reasoning, with its diffuser deliberately omitted.
Activated long-term memory is approximated here by synaptic weights; that is a
project modeling choice, not a claim attributed to Wolfe. PAV, temporal
integration, recurrent memory, attention, and readout are distinct components.

Use four labels consistently:

- **Neuroscience-inspired:** a design choice motivated by a biological theory
  or qualitative mechanism.
- **Computational analogy:** a machine operation resembles a proposed
  computation (for example E/I separation, adaptation, or a priority map).
- **Measured behavior:** a metric from a named checkpoint and held-out or
  diagnostic dataset.
- **Biological validation:** correspondence to neural data or causal biology.
  The repository provides none; do not imply it.

The project has no neural recordings, behavioral-human comparison, anatomical
fit, or biological parameter identification. “E/I,” “adaptation,” “attention,”
“memory,” and “priority” are architectural names unless accompanied by a
specific measured model behavior.

## 2. Authority, precedence, and reading order

### Evidence precedence

When files disagree, use this order:

1. immutable completion, retrieval, cancellation, cleanup, and checkpoint
   receipts;
2. final held-out prediction/summary artifacts and selected-checkpoint indexes;
3. fixed run configuration, protocol, source, and launch manifest;
4. final report and chronological journal page;
5. live metrics, watcher state, validation reports, and prose snapshots;
6. README plans and design documents.

Later evidence can supersede an earlier “live” statement without invalidating
the historical snapshot. Validation is not test; a partial or cancelled run is
not a negative result. Duplicate top-level reports generally summarize the
timestamped run directory; the timestamped artifact is the plotting authority.
[`LabJournal/evidence_manifest.json`](LabJournal/evidence_manifest.json) is a
useful hash inventory but predates the newest experiments and is not exhaustive.

### Start here

1. Repository scope: [`README.md`](README.md), then the older operational
   [`HANDOFF.md`](HANDOFF.md). The latter is historical and predates later runs.
2. Journal guide and present-state layer:
   [`LabJournal/README.md`](LabJournal/README.md),
   [`LabJournal/CURRENT_STATUS.md`](LabJournal/CURRENT_STATUS.md),
   [`LabJournal/CHRONOLOGY.md`](LabJournal/CHRONOLOGY.md).
3. Cross-cutting technical summaries:
   [`LabJournal/ARCHITECTURE.md`](LabJournal/ARCHITECTURE.md),
   [`LabJournal/TASKS_AND_METRICS.md`](LabJournal/TASKS_AND_METRICS.md),
   [`LabJournal/RESEARCH_FOUNDATIONS.md`](LabJournal/RESEARCH_FOUNDATIONS.md),
   and [`LabJournal/OPEN_QUESTIONS.md`](LabJournal/OPEN_QUESTIONS.md).
4. Read the experiment pages below in order, then consult the linked protocol,
   source, receipts, raw predictions, and checkpoint manifests for any number
   used in prose.

### Complete LabJournal experiment sequence

These pages are the concise chronological index; status labels in older pages
can be superseded by newer receipts:

1. [PAV encoder screen](LabJournal/experiments/01-pav-encoder-screen.md)
2. [Saved-ensemble audit](LabJournal/experiments/02-saved-ensemble-audit.md)
3. [Hybrid continuation](LabJournal/experiments/03-hybrid-continuation.md)
4. [Contour task allocation](LabJournal/experiments/04-contour-task-allocation.md)
5. [Causal temporal accumulators](LabJournal/experiments/05-causal-temporal-accumulators.md)
6. [Sequence battery](LabJournal/experiments/06-sequence-battery.md)
7. [LSTM versus E/I](LabJournal/experiments/07-lstm-versus-ei.md)
8. [Recency diagnostic](LabJournal/experiments/08-recency-diagnostic.md)
9. [State accessibility](LabJournal/experiments/09-state-accessibility.md)
10. [Readout refit](LabJournal/experiments/10-readout-refit.md)
11. [Retention learning](LabJournal/experiments/11-retention-learning.md)
12. [Orientation accessibility](LabJournal/experiments/12-orientation-accessibility.md)
13. [Spatial memory](LabJournal/experiments/13-spatial-memory.md)
14. [Selective maintenance](LabJournal/experiments/14-selective-maintenance.md)
15. [Pre-update attention](LabJournal/experiments/15-preupdate-attention.md)
16. [Attention mechanism diagnostic](LabJournal/experiments/15a-attention-mechanism.md)
17. [Motion audit](LabJournal/experiments/15b-motion-audit.md)
18. [Training exposure](LabJournal/experiments/16-training-exposure.md)
19. [Attention maps](LabJournal/experiments/17-attention-maps.md)
20. [Unbiased attention and cancelled spatial battery](LabJournal/experiments/18-unbiased-attention-spatial-battery.md)
21. [Restored-bias spatial battery](LabJournal/experiments/19-biased-spatial-battery.md)
22. [Single-task motion](LabJournal/experiments/20-single-task-motion.md)
23. [Prospective query](LabJournal/experiments/21-prospective-query.md)
24. [Spatial priority readout](LabJournal/experiments/22-spatial-priority-readout.md)

### Sensory and temporal work

- Candidate motivation, definitions, parameters, and primary links:
  [`PreAttentiveVision/research.md`](PreAttentiveVision/research.md).
- Two-frame task definitions:
  [`PreAttentiveVision/stimuli.py`](PreAttentiveVision/stimuli.py);
  motion methods and adaptations:
  [`PreAttentiveVision/krauzlis_stimulus.md`](PreAttentiveVision/krauzlis_stimulus.md).
- Five-encoder result:
  [`PreAttentiveVision/runs/multitask_20260912_141316/report.md`](PreAttentiveVision/runs/multitask_20260912_141316/report.md).
- Hybrid attempt:
  [`PreAttentiveVision/runs/hybrids_20260912_152427/report.md`](PreAttentiveVision/runs/hybrids_20260912_152427/report.md).
- Allocation protocol and result:
  [`PreAttentiveVision/next_experiment_task_allocation.md`](PreAttentiveVision/next_experiment_task_allocation.md),
  [`PreAttentiveVision/runs/allocation_20260912_160414/report.md`](PreAttentiveVision/runs/allocation_20260912_160414/report.md).
- Temporal candidates and implementation:
  [`PreAttentiveVision/TemporalIntegration/README.md`](PreAttentiveVision/TemporalIntegration/README.md),
  [`PreAttentiveVision/TemporalIntegration/accumulators.py`](PreAttentiveVision/TemporalIntegration/accumulators.py),
  [`PreAttentiveVision/TemporalIntegration/report.md`](PreAttentiveVision/TemporalIntegration/report.md).

The seven PAV tasks are motion direction, signed orientation, contrast, spatial
frequency, chromatic increment, contour grouping, and natural-image spectral
detail. Do not merge them with either WM battery.

### Working-memory protocols, implementation, and results

- Broad sequence battery: [`WorkingMemory/TASK_BATTERY.md`](WorkingMemory/TASK_BATTERY.md),
  [`WorkingMemory/model.py`](WorkingMemory/model.py),
  [`WorkingMemory/stimuli.py`](WorkingMemory/stimuli.py), and
  [`WorkingMemory/report.md`](WorkingMemory/report.md).
- Research trail:
  [`WorkingMemory/Research/recurrent_memory_without_attention.md`](WorkingMemory/Research/recurrent_memory_without_attention.md),
  [`WorkingMemory/Research/spatial_ei_memory.md`](WorkingMemory/Research/spatial_ei_memory.md),
  [`WorkingMemory/Research/attention_as_selective_stability.md`](WorkingMemory/Research/attention_as_selective_stability.md),
  [`WorkingMemory/Research/training_exposure_and_temporal_residual.md`](WorkingMemory/Research/training_exposure_and_temporal_residual.md).
- Recurrent comparison:
  [`WorkingMemory/RecurrentComparison/model.py`](WorkingMemory/RecurrentComparison/model.py),
  [`WorkingMemory/RecurrentComparison/report.md`](WorkingMemory/RecurrentComparison/report.md).
- Diagnostics and continuations:
  [`WorkingMemory/RecurrentComparison/RecencyDiagnostic/report.md`](WorkingMemory/RecurrentComparison/RecencyDiagnostic/report.md),
  [`WorkingMemory/RecurrentComparison/StateDiagnostic/report.md`](WorkingMemory/RecurrentComparison/StateDiagnostic/report.md),
  [`WorkingMemory/RecurrentComparison/ReadoutRefit/report.md`](WorkingMemory/RecurrentComparison/ReadoutRefit/report.md),
  [`WorkingMemory/RecurrentComparison/Retention/report.md`](WorkingMemory/RecurrentComparison/Retention/report.md),
  [`WorkingMemory/RecurrentComparison/OrientationDiagnostic/report.md`](WorkingMemory/RecurrentComparison/OrientationDiagnostic/report.md).
- Spatial memory and explicit comparator:
  [`WorkingMemory/SpatialComparison/model.py`](WorkingMemory/SpatialComparison/model.py),
  [`WorkingMemory/SpatialComparison/report.md`](WorkingMemory/SpatialComparison/report.md).
- Additive controller:
  [`WorkingMemory/SelectiveMaintenance/model.py`](WorkingMemory/SelectiveMaintenance/model.py),
  [`WorkingMemory/SelectiveMaintenance/report.md`](WorkingMemory/SelectiveMaintenance/report.md).
- Pre-update attention:
  [`WorkingMemory/PreUpdateAttention/model.py`](WorkingMemory/PreUpdateAttention/model.py),
  [`WorkingMemory/PreUpdateAttention/report.md`](WorkingMemory/PreUpdateAttention/report.md),
  [`WorkingMemory/PreUpdateAttention/README.md`](WorkingMemory/PreUpdateAttention/README.md).
- Frozen mechanism evidence:
  [`WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md`](WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md) and
  [`WorkingMemory/PreUpdateAttention/MotionAudit/report.md`](WorkingMemory/PreUpdateAttention/MotionAudit/report.md).
- Matched allocation comparison:
  [`WorkingMemory/TrainingExposure/report.md`](WorkingMemory/TrainingExposure/report.md).
- Attention visualization:
  [`WorkingMemory/AttentionMaps/report.md`](WorkingMemory/AttentionMaps/report.md),
  [`WorkingMemory/AttentionMaps/index.html`](WorkingMemory/AttentionMaps/index.html),
  [`WorkingMemory/AttentionMaps/perframe_movies.json`](WorkingMemory/AttentionMaps/perframe_movies.json),
  [`WorkingMemory/AttentionMaps/perframe_maps.npz`](WorkingMemory/AttentionMaps/perframe_maps.npz).
  The corrected viewer is per trial and per timestep. Older phase averages are
  historical, not a substitute for framewise maps. Both heads attend both
  visual and memory banks; “visual head” and “memory head” are incorrect names.

### Five-task spatial suite and recent runs

The newer suite is **not** the broad seven-family sequence battery:

- Exact protocol:
  [`WorkingMemory/SpatialTaskBattery/PROTOCOL.md`](WorkingMemory/SpatialTaskBattery/PROTOCOL.md).
- Primary-source/adaptation trail:
  [`WorkingMemory/SpatialTaskBattery/SOURCES.md`](WorkingMemory/SpatialTaskBattery/SOURCES.md).
- Executable stimuli:
  [`WorkingMemory/SpatialTaskBattery/stimuli.py`](WorkingMemory/SpatialTaskBattery/stimuli.py).
- Rendered audit:
  [`WorkingMemory/SpatialTaskBattery/previews/index.html`](WorkingMemory/SpatialTaskBattery/previews/index.html) and
  [`WorkingMemory/SpatialTaskBattery/previews/movies.json`](WorkingMemory/SpatialTaskBattery/previews/movies.json).
- Bias-removal comparison:
  [`WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/report.md`](WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/report.md).
- Cancelled bias-free five-task attempt:
  [`WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/report.md`](WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/report.md).
- Restored-bias partial run:
  [`WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md`](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md).
- Motion-only continuation:
  [`WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md`](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md).
- Stopped prospective-query attempt:
  [`WorkingMemory/ProspectiveQuery/completion_report.md`](WorkingMemory/ProspectiveQuery/completion_report.md),
  with its sole validation snapshot at
  [`WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/validation_9200_report.md`](WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/validation_9200_report.md).
- Spatial-priority proposal/source:
  [`WorkingMemory/SpatialPriorityReadout/README.md`](WorkingMemory/SpatialPriorityReadout/README.md),
  [`WorkingMemory/SpatialPriorityReadout/PROTOCOL.md`](WorkingMemory/SpatialPriorityReadout/PROTOCOL.md),
  [`WorkingMemory/SpatialPriorityReadout/model.py`](WorkingMemory/SpatialPriorityReadout/model.py),
  [`WorkingMemory/SpatialPriorityReadout/report.md`](WorkingMemory/SpatialPriorityReadout/report.md).
  At cutoff the README, prose protocol, and report still describe the cancelled
  inherited v1 run and are historical; the root model had already been changed
  to scratch v2. Use the immutable run copies and receipts below to avoid
  mixing those versions.
- Run evidence:
  [`launch_artifact_index.json`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/launch_artifact_index.json),
  [`launch_receipt.json`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/launch_receipt.json),
  [`live_metrics_tail.csv`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/live_metrics_tail.csv),
  [`live_status.json`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/live_status.json),
  and the superseding
  [`lineage_correction_cleanup_receipt.json`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/lineage_correction_cleanup_receipt.json).
- Corrected scratch replacement:
  [`model.py`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/model.py),
  [`protocol.py`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/protocol.py),
  [`lineage_correction_receipt.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/lineage_correction_receipt.json),
  [`pod_ready.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/pod_ready.json), and
  [`live_status.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_status.json).
- Completed frozen-readout diagnostic:
  [`PROTOCOL.md`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/PROTOCOL.md),
  [`report.md`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md),
  [`results.json`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/results.json),
  [`run_manifest.json`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/run_manifest.json), and
  [`completion_receipt.json`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/completion_receipt.json).
  This local post-hoc analysis froze the motion-only step-12200 model and did
  not query or modify cloud training.

## 3. Experiment chronology and evidence table

“Matched” means a controlled comparison used common initialization and/or
paired examples as stated; it does not mean bitwise equivalence across
hardware. Exposures are training episodes unless noted. Percentages are held
out test values unless explicitly labeled validation/diagnostic.

| # | Hypothesis / version | Initialization, task, exposure / compute | Primary evidence and result | Interpretation and status |
|---|---|---|---|---|
| 1 | Five lightweight PAV encoders can support seven two-frame tasks. | Five separately initialized shared-weight encoders; equal 756 updates/model; local sequential GPU. | [Report](PreAttentiveVision/runs/multitask_20260912_141316/report.md): ConvNeXt-GRN was the broadest candidate; motion BA 80.6%, contour BA 60.3%. | Screening result, not biological validation. Models are exposure-matched, not parameter-matched. |
| 2 | A saved ensemble might rescue weaknesses. | Saved outputs/checkpoints only; no new acquisition. | [Journal](LabJournal/experiments/02-saved-ensemble-audit.md). | Audit did not establish a deployable superior single encoder. Historical diagnostic. |
| 3 | Architectural hybrid additions improve contour without harming other tasks. | Late-SE continuation from step 1512; equal exposure. | [Hybrid report](PreAttentiveVision/runs/hybrids_20260912_152427/report.md). Neither addition met the contour criterion. | **Rejected tested fixes**; preserve late-SE because motion remained strong. |
| 4 | Reallocating updates, not changing architecture, improves contour. | Same late-SE step-1512 parent and task-local matched prefixes; uniform versus 50%-contour; 756 added updates/arm to step 2268. | [Allocation report](PreAttentiveVision/runs/allocation_20260912_160414/report.md): contour 67.9% to 96.2%; all seven point estimates over 95% in focused arm. | Strong **matched allocation** result; selected contour-focused checkpoint became PAV reference. One seed limits generality. |
| 5 | A fixed opponent accumulator can outperform learned KDA/ConvGRU temporal cores. | Frozen selected encoder; new temporal cores/readout; 4,032 updates/model; seven PAV tasks. | [Temporal report](PreAttentiveVision/TemporalIntegration/report.md): opponent 98.44–100% across held-out tasks. | Selected causal temporal front end. This is measured task performance, not a cortical validation. |
| 6 | The opponent winner can acquire a broad sequence/WM battery end to end. | Warm-started opponent; learned weights unfrozen; 39,200 episodes; selected update 6,860. | [WM report](WorkingMemory/report.md): many new sequence rules remained near chance even at short delay. | Important negative acquisition result; does not by itself establish a capacity limit. |
| 7 | Adaptive E/I recurrence offers a neuroscience-inspired alternative to LSTM. | Same selected sequence parent and task streams; 40,000 episodes/arm; LSTM local, E/I on RTX 3090 RunPod; matched design but different hardware. | [Report](WorkingMemory/RecurrentComparison/report.md): LSTM L8 motion 79.69%, E/I 68.95%. | LSTM won the primary long-motion comparison. Not bitwise cross-platform; unequal inductive biases/state definitions. |
| 8 | Dense E/I errors reflect recency/order sensitivity. | Frozen step-5000 checkpoints; 512 controlled matched early/late pairs; no training. | [Recency report](WorkingMemory/RecurrentComparison/RecencyDiagnostic/report.md). | Ordering sensitivity diagnostic; does not uniquely identify leakage or adaptation. |
| 9 | Earlier evidence remains accessible in frozen E/I state. | Analysis-only probes with disjoint train/validation/test schedules. | [State report](WorkingMemory/RecurrentComparison/StateDiagnostic/report.md): held-out linear-readout gain 9.47 percentage points; nonlinear probe no advantage. | Recoverability supported; probe success is not proof the deployed circuit uses the feature. Slow-synapse change deferred. |
| 10 | Refit only deployed output use. | Freeze sensory/recurrent computations; train memory output and motion/orientation heads; 38,720 episodes; select 9,840. | [Readout report](WorkingMemory/RecurrentComparison/ReadoutRefit/report.md): long-motion BA 70.31% to 79.30%. | **Tested and supported:** output use was a bottleneck. Not compared with equal-exposure end-to-end continuation. |
| 11 | Recurrent-core training improves retention across inserted blanks. | Readout-refit parent; train E/I core plus output heads; sensory/opponent/input frozen; 39,680 episodes; select 14,800. | [Retention report](WorkingMemory/RecurrentComparison/Retention/report.md): motion about 80% through D24; orientation comparison near chance at D24. | Motion retention improved; orientation failure localized the next question. Baseline unfamiliar delays are generalization tests. |
| 12 | Orientation sample information remains accessible despite chance deployed output. | Frozen 14,800 model; independent 2,048/512/512 base episodes paired across D0/4/12/24; analysis-only decoders. | [Orientation diagnostic](WorkingMemory/RecurrentComparison/OrientationDiagnostic/report.md): D24 angle error 4.25 degrees; diagnostic comparator 73.44% versus deployed 50.00%. | Information is accessible in pre-probe firing rates; no causal or biological claim. Comparator has analysis-only training. |
| 13 | Spatial E/I fields plus explicit old/current comparator improve binding. | Dense and spatial competitors from retention parent; 35,200 episodes/arm; selected step 4,400; unequal prior compatibility and state sizes disclosed. | [Spatial report](WorkingMemory/SpatialComparison/report.md): strong binding improvement, but single-item D24 remained chance and motion D24 weak. | Supports spatial state for binding; not a clean architecture-only comparison because inherited compatibility differs. |
| 14 | Additive recurrent feedback controller selectively maintains useful state. | Same spatial-4400 parent; ordinary continuation versus new 32-unit E/I controller; unchanged cycle; 39,680 episodes/arm. | [Controller report](WorkingMemory/SelectiveMaintenance/report.md): only narrow D24 orientation gain; delay-only interruption is out of distribution. | Feedback package not broadly superior; cannot isolate “attention” from extra recurrent capacity. |
| 15 | Pre-update joint sensory/memory attention improves decisions. | Same spatial-4400 parent; 4,000 updates/32,000 episodes on RTX 3090 cloud; no controller. | [Attention report](WorkingMemory/PreUpdateAttention/report.md): D24 orientation 58.40% to 79.30%; immediate motion declined. | Strong delayed-orientation gain with a motion tradeoff. Platform differs from local comparator. |
| 16 | Memory-source attention during a specific phase is necessary for delayed behavior. | Frozen attention-8400; 512 paired episodes/cell; phase-specific acute exclusions and raw-drive bypasses. | [Mechanism report](WorkingMemory/PreUpdateAttention/MechanismDiagnostic/report.md): excluding memory source during blanks reduced D24 orientation 79.30% to 50%. | Causal **model intervention** implicates that computation; out-of-distribution and not a biological causal result. |
| 17 | Motion loss is caused by attention, allocation, or decision use. | Frozen artifact audit and paired interventions. | [Motion audit](WorkingMemory/PreUpdateAttention/MotionAudit/report.md): weak calibration/class bias; degradation already appears before the attention change; 10% allocation is suggestive. | Mechanism unresolved; schedule is confounded until controlled. |
| 18 | More motion exposure recovers motion while preserving the rest. | Same attention-8400 parent; local 10%-motion versus cloud 50%-motion, 4,000 updates/32,000 episodes/arm; common recipes, cross-platform. | [Exposure report](WorkingMemory/TrainingExposure/report.md): focused schedule improved motion but weakened orientation and became class-biased. | **Matched schedule test** with hardware caveat; objective tradeoff, not proof of catastrophic interference. |
| 19 | Learned attention has spatially interpretable source allocation. | Frozen attention-8400; original trials; no fitting or weight updates. | [Viewer](WorkingMemory/AttentionMaps/index.html), [report](WorkingMemory/AttentionMaps/report.md): per-timestep 13-by-13 maps for each head and source. | Descriptive visualization only; remembered-scene overlays are references, not pixel attribution. |
| 20 | Explicit source/locality logit biases are dispensable. | Bias-free migration retains Q/K/V/O; old-task arm has 4,000 updates; compared with preserved biased control on reused evaluation draws. | [Old-arm report](WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/report.md): removing both terms harmed delayed orientation and binding. | Joint ablation is supported; source and locality effects are not separately identified. Reused-test exploratory status. |
| 21 | Bias-free attention can learn the new five-task spatial suite. | Fresh five semantic heads from bias-free parent; planned five microbatches/update. | [Run report](WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/report.md). | **User-cancelled partial attempt**; not a capacity result and must not be compared as completed. |
| 22 | Restoring both biases permits the five-task suite. | Original biased attention-8400 parent; fresh five heads; five microbatches of eight/update; selected validation checkpoint 10,000; run stopped before planned completion/final test. | [Partial report](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md): binding/recognition learned; motion, signed orientation, and Krauzlis target/foil task remained weak. | **Partial, validation-only**. No final held-out five-task result; do not call it a completed battery comparison. |
| 23 | Task-only gradients can rescue cued motion. | Verified restored-bias checkpoint 10,000; only `motion_duration_cued`; 2,200 updates/17,600 fresh episodes to 12,200; local GPU. | [Completion](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md): remained near four-class chance. | **Rejected tested training fix**. Does not isolate gradient conflict because task loss weight changed from one fifth to full. |
| 24 | Current sensory evidence should enter attention queries through a learned residual. | Restored-bias parent; scalar zero-initialized gamma; intended matched continuation. User stopped after first validation at 9,200. | [Completion](WorkingMemory/ProspectiveQuery/completion_report.md): no primary motion rescue at sole validation. | **Stopped partial**; no final test. Gamma behavior is evidence about this implementation, not the general idea. |
| 25 | Replace terminal global pooling with a spatial task-conditioned priority/evidence readout. | Code migrated from attention-8400 and new readout; intended 4,000 updates. Actual launched run inherited forbidden model/Adam state and was stopped at display 8,720: 320 updates, 12,800 fresh stimulus episodes. | [Cleanup authority](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/lineage_correction_cleanup_receipt.json), [historical live metrics](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/live_metrics_tail.csv). | **Cancelled lineage-error attempt; no validation and no outcome.** Pod was deleted; the corrected replacement is a separate row. |
| 26 | Test the same spatial readout with the corrected fully scratch lineage. | `spatial_priority_readout_scratch_v2`; every model tensor and optimizer state initialized fresh from documented seeds; same five-task recipe; target 4,000 updates/160,000 episodes. | [Pinned source](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/model.py), [live status](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_status.json). RTX 3090 at USD 0.22/hour; deadline 2026-09-15T11:47:20Z. | **Live training snapshot, no validation.** At 03:51:25Z: update 14/560 episodes, batch loss 0.86850, accuracy 0.400, gradient norm 1.96260 before clipping. No result yet; scratch versus prior trained models is unmatched. |
| 27 | Determine whether a spatial output can decode cued motion evidence hidden from a capacity-matched pooled probe. | Frozen motion-only step 12200, SHA `7cec4c48c3da81b4d4935c65098b58974a38a8ff3a825dfe12e4d8d16e9a0e26`; identical cached `H_T/R_T/C_T`; 512/128/256 independent train/validation/test base episodes per delay, seeds 731091/731092/731093. Pooled 74,303 versus spatial 74,373 parameters; both selected epoch 20 (640 updates, 40,960 repeated presentations). | [Report](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md), [results](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/results.json): pooled BA/AUC 23.83%/0.4805; spatial 24.32%/0.4832; paired BA delta +0.49 pp, grouped 95% CI [-1.76,+2.64]. Spatial D0/4/12/24 BA 22.66/24.22/25.39/25.00%. Target-region priority mass 13.13% [11.88%,14.56%] versus 5.33% uniform; peak hit 20.80%. Total wall 738.5 s. | **Complete, post-hoc diagnostic:** localization without decoding rescue. It neither updates the deployed model nor predicts the independent scratch end-to-end run. |

## 4. Architecture reconstruction

### 4.1 Sensory encoder and opponent temporal computation

Input is an ordered movie `X` with shape `[B,T,3,100,100]`. Early PAV
experiments always used exactly two frames and shared encoder weights:

```math
z_1=E_\theta(X_1),\qquad z_2=E_\theta(X_2).
```

The selected ConvNeXt-GRN/late-SE lineage and the other four candidates are
defined in [`PreAttentiveVision/models.py`](PreAttentiveVision/models.py) and
explained in [`PreAttentiveVision/research.md`](PreAttentiveVision/research.md).
In the WM lineage, shared sensory stages produce approximately
`[B,24,50,50]`, `[B,48,25,25]`, and `[B,96,13,13]`; each is projected to 32
channels before opponent processing.

For each projected feature `U_t`, fixed traces use:

```math
F_t=0.25F_{t-1}+0.75U_t,\qquad
S_t=0.75S_{t-1}+0.25U_t.
```

Fixed quadrature energies and divisive normalization combine signed
fast-minus-slow responses into a fused sensory history
`H_t:[B,64,13,13]`. Exact implementation is in
[`PreAttentiveVision/TemporalIntegration/accumulators.py`](PreAttentiveVision/TemporalIntegration/accumulators.py)
and its WM integration in
[`WorkingMemory/model.py`](WorkingMemory/model.py). “Opponent” and “divisive
normalization” are computational descriptions; the learned network was not fit
to cortical response data.

### 4.2 Pre-update joint attention

Let previous memory rates be `R_(t-1):[B,64,13,13]`. Learned projections form
169 memory-site queries and a 338-token source bank (169 current-sensory plus
169 previous-memory tokens). With two heads and head width 32:

```math
Q=W_Q(\operatorname{LN}(R_{t-1})+P+e_m),
```

```math
K_s=W_K(\operatorname{LN}(S_s)+P+e_s),\qquad
V_s=W_V(\operatorname{LN}(S_s)+P+e_s),
```

```math
\ell_{hij}=\frac{Q_{hi}^{\mathsf T}K_{hj}}{\sqrt{32}}
+ b^{\mathrm{source}}_{h,s(j)}
-\operatorname{softplus}(\lambda_h)d(i,j)^2,
\qquad A=\operatorname{softmax}_j(\ell).
```

`A` has shape `[B,2,169,338]`. Source bias and locality are learned and were
retained in the original attention-8400 and spatial-priority code. The joint
bias-removal run removed both and therefore cannot identify either term
separately. Head identity is not source identity.

### 4.3 Spatial E/I recurrence and adaptation

The attended drive is projected into a recurrent spatial field. Rates and
adaptation each have shape `[B,64,13,13]`; 51 source channels are designated
excitatory and 13 inhibitory. The implementation applies sign-constrained
effective recurrent weights, state normalization/clamping, leak, and
adaptation:

```math
u_t=W_x x_t+W_E r^E_{t-1}-W_I r^I_{t-1}-\beta a_{t-1}+b,
```

```math
r_t=(1-\alpha)r_{t-1}+\alpha\,\phi(u_t),\qquad
a_t=(1-\rho)a_{t-1}+\rho r_t.
```

The source is authoritative for constants and guards:
[`WorkingMemory/SpatialComparison/model.py`](WorkingMemory/SpatialComparison/model.py).
“E/I” and “Dale-inspired” refer to the imposed sign structure, not identified
cell types; the recurrence is neither a biophysical simulation nor a fit to
spiking data.

### 4.4 Explicit comparison and historical global readout

Before the recurrent update, the comparator receives old memory and current
sensory state:

```math
C_t=\operatorname{Comparator}([R_{t-1};H_t]),
\qquad [R_{t-1};H_t]:[B,128,13,13],
```

with `C_t:[B,64,13,13]`. The historical deployed output discarded explicit
location by mean/max pooling `H_T`, `R_T`, and `C_T`, mapping each pooled branch
into a common latent, summing branches, and applying a task head. This is the
readout whose delayed-orientation and cued-motion failures motivated the later
probes.

### 4.5 Failed prospective-query residual

The sole architectural delta was:

```math
Q=W_Q\!\left(\operatorname{LN}(R_{t-1})+P+e_m+
\gamma\operatorname{LN}(H_t)\right),
```

with one scalar `gamma`, initialized to zero. All other architecture and tasks
were intended to remain fixed. The run was user-stopped after one validation
and has no final test; this rejects neither sensory-conditioned queries in
general nor other parameterizations.

### 4.6 Convolutional task-conditioned priority/evidence readout

The proposed replacement preserves spatial structure through the decision:

```math
Z=[H_T;R_T;C_T]\in\mathbb{R}^{B\times192\times13\times13},
```

```math
G=\operatorname{SiLU}\!\left(
\operatorname{GN}\!\left(\operatorname{Conv}_{3\times3}^{96\rightarrow64}
\left(\operatorname{SiLU}(\operatorname{GN}(
\operatorname{Conv}_{1\times1}^{192\rightarrow96}(Z)))\right)\right)\right).
```

For task `k`, independent `1x1` heads form selection `s_k` and class evidence
`e_k`:

```math
p_{k,ij}=\frac{\exp s_{k,ij}}{\sum_{uv}\exp s_{k,uv}},\qquad
\hat y_{k,c}=\sum_{ij}p_{k,ij}e_{k,c,ij}.
```

The implemented five-task form adds **75,153** readout parameters and removes
**51,858** old pooled-readout parameters, for **579,423 total parameters**.
There is no prospective gamma. The experimental delta is therefore spatial
convolution plus task-local selection/evidence in place of global pooling;
the original attention biases remain.

This architecture is implemented. The first cloud run was cancelled because it
inherited model/Adam state contrary to a corrected lineage requirement. In the
replacement `scratch_v2` source, `initialize_scratch` constructs every model
tensor from documented seeds, creates an empty Adam state, begins actual
training at update 0, and retains no parent tensors; the historical `8400`
counter is not model lineage. A replacement pod had reached setup, but no
training outcome existed at cutoff.

## 5. Neuroscience rationale and source trail

Use the project bibliographies as leads, then verify every central claim
against the primary paper before submission:

- Guided Search 6.0 and priority-map framing:
  [`PreAttentiveVision/research.md`](PreAttentiveVision/research.md) and
  [`LabJournal/RESEARCH_FOUNDATIONS.md`](LabJournal/RESEARCH_FOUNDATIONS.md).
  Be explicit that the diffuser is omitted.
- Motion stimuli and Krauzlis/superior-colliculus motivation:
  [`PreAttentiveVision/krauzlis_stimulus.md`](PreAttentiveVision/krauzlis_stimulus.md)
  and
  [`WorkingMemory/SpatialTaskBattery/SOURCES.md`](WorkingMemory/SpatialTaskBattery/SOURCES.md).
  The rendered tasks are disclosed adaptations, not replications.
- Cortical orientation/motion, opponent energy, normalization, and recurrent
  WM:
  [`PreAttentiveVision/TemporalIntegration/README.md`](PreAttentiveVision/TemporalIntegration/README.md),
  [`WorkingMemory/Research/recurrent_memory_without_attention.md`](WorkingMemory/Research/recurrent_memory_without_attention.md),
  and [`WorkingMemory/Research/neuroscience_task_rationale.md`](WorkingMemory/Research/neuroscience_task_rationale.md).
- Persistent versus dynamic coding, E/I balance, Dale-inspired constraints,
  and adaptation:
  [`WorkingMemory/Research/recurrent_memory_without_attention.md`](WorkingMemory/Research/recurrent_memory_without_attention.md)
  and [`WorkingMemory/Research/spatial_ei_memory.md`](WorkingMemory/Research/spatial_ei_memory.md).
- Attention/WM interaction, feedback, and task-conditioned control:
  [`WorkingMemory/Research/attention_as_selective_stability.md`](WorkingMemory/Research/attention_as_selective_stability.md)
  and [`WorkingMemory/PreUpdateAttention/README.md`](WorkingMemory/PreUpdateAttention/README.md).
- Evidence accumulation and readout are computationally motivated in
  [`WorkingMemory/SpatialPriorityReadout/PROTOCOL.md`](WorkingMemory/SpatialPriorityReadout/PROTOCOL.md);
  a future paper needs direct primary citations for any neural accumulator or
  priority-map localization claim.

Claims requiring explicit pre-submission verification include: the exact
Guided Search 6.0 formulation; any statement linking superior colliculus to
the implemented target/foil task; divisive-normalization scope; persistent
versus dynamic WM interpretations; Dale-law relevance; adaptation time scales;
and neural evidence-accumulation localization. The repository has no central
BibTeX file. A dedicated primary citation for “biased competition” was not
clearly identified during this audit; add and verify one rather than citing the
phrase by analogy.

## 6. Machine-learning and mathematical rationale

### Objectives and metrics

- Training uses task-appropriate cross-entropy, generally one task minibatch
  per update in old schedules and five eight-example microbatches with averaged
  losses for the newer five-task suite.
- Balanced accuracy is the mean class recall. Chance is 25% for four-class
  direction/duration and 50% for binary tasks.
- A useful cross-task display is
  `chance-normalized BA = (BA - chance) / (1 - chance)`, but raw BA and chance
  must appear alongside it.
- Binary AUC has chance 0.5. Multiclass AUC aggregation must be copied from the
  producing summary rather than silently redefined.
- “Composite accuracy” in live logs is training-batch aggregate accuracy, not a
  balanced held-out endpoint.

### Optimization and migration

Sequence models use full-sequence BPTT in fp32. Run configurations preserve
explicit parameter groups, lower sensory learning rates where specified,
Adam/AdamW epsilon and weight decay, and global gradient clipping (commonly
norm 1). Do not infer a universal learning rate from a prose report: retrieve
it from each `fixed_config.json`, `config.json`, or checkpoint optimizer state.
Numerical checks include finite loss/parameters/gradients, state/rate/adaptation
ranges, clipping statistics, and state-gradient diagnostics.

Architecture continuations must distinguish:

- inherited tensors and compatible optimizer slots;
- new tensors and fresh optimizer slots;
- frozen versus trainable groups;
- RNG and task-local stream restoration;
- display-step offset versus actual fresh updates/episodes;
- terminal versus selected checkpoint.

The stopped priority run is a cautionary example: source and launch artifacts
document attention-8400 inheritance, but the corrected requirement forbade
model/Adam inheritance. The cleanup receipt therefore supersedes live
training rows as an outcome.

### Attention, selection, and evidence

Attention produces a normalized distribution over sensory and memory source
tokens for each memory query. The proposed terminal readout instead produces
a task-specific spatial distribution over 169 output locations and a
class-evidence map. Report these separately:

- **selection:** entropy, peak, target-neighborhood mass, source allocation;
- **evidence:** signed/classwise local logits;
- **decision:** their weighted spatial sum.

A high selection mass near a stimulus is alignment, not proof of causal
attention or pixel attribution. A class decision can arise from diffuse
selection and structured evidence, or sharp selection and biased evidence.

### Probes and counterfactuals

Analysis-only probes show decodability under their own fitting budget. They do
not show that the deployed model uses the information or that the probed
variable is represented uniquely. Acute source exclusions and feedback
interruptions are model-causal interventions but can be out of distribution;
rescue implicates the modified computation, while no rescue does not prove
erasure.

Every probe needs grouped train/validation/test splits, train-only
standardization, validation-only hyperparameter/epoch selection, and a held-out
test. Paired episodes must remain paired in uncertainty analysis. For movies,
generate shared nuisance draws; do not create fake pairs by shuffling rendered
frames.

### Failure modes and status tags

Use these exact tags in tables/figures:

- **supported:** contour-focused sampling; opponent accumulator; readout-only
  refit; delayed-orientation gain from pre-update attention.
- **tested and rejected:** two hybrid contour additions; task-only cued-motion
  continuation.
- **partial/stopped:** bias-free five-task run, restored-bias five-task run,
  prospective-query run.
- **cancelled/invalid lineage:** priority-readout cloud attempt.
- **diagnostic only:** recency, state accessibility, orientation accessibility,
  mechanism interventions, motion audit, attention maps, and the completed
  frozen pooled-versus-spatial readout comparison.
- **untested:** a correctly initialized priority-readout training run and the
  final outcome of the ongoing correctly initialized priority-readout run.
- **confounded or unmatched:** cross-platform LSTM/E/I and local/cloud arms,
  spatial-versus-dense state-size/prior-experience comparison, any comparison
  across old PAV, broad WM, and new five-task batteries.

Most training arms have one seed. Report seeds, split identifiers, number of
independent base episodes, repeated delays/presentations, and clustering unit.
Confidence intervals must resample the independent unit, not inflate `n` by
counting paired delays or frames separately.

## 7. Plot-ready data and result inventory

### Data authority and schemas

For training curves, `metrics.csv` rows are updates; inspect each header because
columns vary. Common columns include display/update step, task, loss, accuracy,
gradient norm, learning rates, elapsed time, and state statistics. Evaluation
`summary.json` files provide aggregate task/delay BA/AUC; `predictions.jsonl`
contains episode-level labels, probabilities/logits, condition metadata, and
pair/group identifiers where implemented. `checkpoint_index.jsonl` binds
steps, selections, files, and hashes. Receipts/configs establish hardware,
budget, exposure, source hash, and lifecycle.

Primary plot sources:

- Five-encoder curves and endpoints:
  [`convnext_grn metrics`](PreAttentiveVision/runs/multitask_20260912_141316/convnext_grn_seed20271/metrics.csv),
  [`aggregate.json`](PreAttentiveVision/runs/multitask_20260912_141316/aggregate.json),
  and per-model `eval_test/summary.json`/`predictions.jsonl` under the same run.
- Allocation:
  [`contour-focused metrics`](PreAttentiveVision/runs/allocation_20260912_160414/contour_focus_seed20271/metrics.csv),
  [`uniform metrics`](PreAttentiveVision/runs/allocation_20260912_160414/uniform_seed20271/metrics.csv),
  [`paired_analysis.json`](PreAttentiveVision/runs/allocation_20260912_160414/paired_analysis.json),
  and [`aggregate.json`](PreAttentiveVision/runs/allocation_20260912_160414/aggregate.json).
- Temporal models:
  [`opponent metrics`](PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/opponent_seed30301/metrics.csv),
  [`KDA metrics`](PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/spatial_kda_seed30301/metrics.csv),
  and [`ConvGRU metrics`](PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/convgru_seed30301/metrics.csv).
- Sequence battery:
  [`metrics.csv`](WorkingMemory/runs/wm_20260912_181219/opponent/metrics.csv) and
  [`test/summary.json`](WorkingMemory/runs/wm_20260912_181219/test/summary.json).
- LSTM/E/I:
  [`LSTM metrics`](WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/lstm/metrics.csv),
  [`E/I metrics`](WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/remote_retrieval/remote_results/ei_adaptive/metrics.csv),
  [`analysis.json`](WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/analysis.json).
- Recency/state/orientation diagnostics:
  [`RecencyDiagnostic/summary.json`](WorkingMemory/RecurrentComparison/RecencyDiagnostic/summary.json),
  [`StateDiagnostic/summary.json`](WorkingMemory/RecurrentComparison/StateDiagnostic/summary.json),
  [`OrientationDiagnostic/summary.json`](WorkingMemory/RecurrentComparison/OrientationDiagnostic/summary.json).
- Output refit:
  [`metrics.csv`](WorkingMemory/RecurrentComparison/ReadoutRefit/runs/readout_20260912_220701/readout_refit/metrics.csv),
  [`analysis.json`](WorkingMemory/RecurrentComparison/ReadoutRefit/runs/readout_20260912_220701/analysis.json),
  and parent/refit test summaries in that run.
- Retention:
  [`metrics.csv`](WorkingMemory/RecurrentComparison/Retention/runs/retention_20260913_084433/retention/metrics.csv),
  [`analysis.json`](WorkingMemory/RecurrentComparison/Retention/runs/retention_20260913_084433/analysis.json),
  plus `parent_test/summary.json` and `refit_test/summary.json`.
- Spatial comparator:
  [`spatial metrics`](WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/spatial_ei/metrics.csv),
  [`dense metrics`](WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/dense_comparator/metrics.csv),
  [`analysis.json`](WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/analysis.json).
- Controller:
  [`controller metrics`](WorkingMemory/SelectiveMaintenance/runs/maintenance_20260913_124444/controller_feedback/metrics.csv),
  [`continuation metrics`](WorkingMemory/SelectiveMaintenance/runs/maintenance_20260913_124444/continuation/metrics.csv),
  [`analysis.json`](WorkingMemory/SelectiveMaintenance/runs/maintenance_20260913_124444/analysis.json).
- Attention:
  [`metrics.csv`](WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/metrics.csv),
  [`analysis.json`](WorkingMemory/PreUpdateAttention/analysis.json),
  and [`test/summary.json`](WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/test/summary.json).
- Allocation:
  [`control-10 metrics`](WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/control_10/metrics.csv),
  [`focused-50 metrics`](WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieved/remote_results/focused_50/metrics.csv),
  [`analysis.json`](WorkingMemory/TrainingExposure/analysis.json).
- Attention maps:
  [`perframe_maps.npz`](WorkingMemory/AttentionMaps/perframe_maps.npz),
  [`perframe_movies.json`](WorkingMemory/AttentionMaps/perframe_movies.json),
  [`summary.json`](WorkingMemory/AttentionMaps/summary.json),
  [`locality_summary.json`](WorkingMemory/AttentionMaps/locality_summary.json).
  Arrays are source-allocation/within-source 13-by-13 maps indexed by trial,
  timestep, head, and source as documented by the report; current RGB frames
  and remembered-scene references are separate.
- Bias removal:
  [`bias-free old-task metrics`](WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/retrieved/remote_results/old_unbiased/training/metrics.csv)
  and the `biased_parent_test`/`terminal_test` summaries below that directory.
- Restored-bias five-task partial:
  [`metrics.csv`](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/retrieved/biased_results/spatial_biased/training/metrics.csv),
  validation summaries at 9200/10000/10800/11600, and
  [`report.md`](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md).
- Single-task motion:
  [`metrics.csv`](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/runs/motion_20260913_214605/training/metrics.csv),
  parent/selected/terminal test summaries, and
  [`completion_report.md`](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md).
- Prospective query partial:
  [`metrics.csv`](WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/retrieved/prospective_query_results/prospective_query/training/metrics.csv)
  and
  [`validation summary`](WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/retrieved/prospective_query_results/prospective_query/validation_9200/summary.json).
- Invalid priority attempt:
  [`live_metrics_tail.csv`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/live_metrics_tail.csv),
  [`aggregate_latest.json`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/aggregate_latest.json),
  and
  [`cleanup receipt`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/lineage_correction_cleanup_receipt.json).
  Plot only as a clearly shaded **cancelled training trace**, never as
  validation or an architecture result.
- Corrected scratch replacement:
  [`pod_ready.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/pod_ready.json),
  [`live_metrics_tail.csv`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_metrics_tail.csv),
  [`live_status.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_status.json),
  and the pinned
  [`model.py`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/model.py).
  At cutoff these establish only early training, with checkpoint 0 and no
  validation; the rolling batch trace is not a result curve.
- Completed local frozen-readout diagnostic:
  [`features_train.pt`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/features_train.pt),
  [`features_val.pt`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/features_val.pt),
  and
  [`features_test.pt`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/features_test.pt)
  are cached float32 terminal-field data; independent base episodes are the
  grouping unit and four delay presentations per base episode are repeated
  observations.
  [`predictions_test.jsonl`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/predictions_test.jsonl)
  is the held-out episode-level prediction authority;
  [`priority_maps_test.npz`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/priority_maps_test.npz)
  stores test priority maps;
  [`results.json`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/results.json)
  contains aggregate, per-delay, bootstrap, and alignment results;
  [`run_manifest.json`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/run_manifest.json)
  hashes the data, selected probes, results, and execution sources; and
  [`completion_receipt.json`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/completion_receipt.json)
  is the immutable completion/lineage authority. Selected states are
  [`pooled_selected.pt`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/pooled_selected.pt)
  and
  [`spatial_selected.pt`](WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/spatial_selected.pt).

### Checkpoint and lifecycle authorities

Never infer a selected checkpoint from the last metrics row. The principal
hash indexes are:

- PAV:
  [`convnext checkpoint index`](PreAttentiveVision/runs/multitask_20260912_141316/convnext_grn_seed20271/checkpoint_index.jsonl),
  [`contour-focused index`](PreAttentiveVision/runs/allocation_20260912_160414/contour_focus_seed20271/checkpoint_index.jsonl),
  [`opponent temporal index`](PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/opponent_seed30301/checkpoint_index.jsonl).
- Initial WM and recurrent comparison:
  [`sequence index`](WorkingMemory/runs/wm_20260912_181219/opponent/checkpoint_index.jsonl),
  [`LSTM index`](WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/lstm/checkpoint_index.jsonl),
  [`remote E/I index`](WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/remote_retrieval/remote_results/ei_adaptive/checkpoint_index.jsonl).
- Refit and retention:
  [`readout-refit index`](WorkingMemory/RecurrentComparison/ReadoutRefit/runs/readout_20260912_220701/readout_refit/checkpoint_index.jsonl),
  [`retention index`](WorkingMemory/RecurrentComparison/Retention/runs/retention_20260913_084433/retention/checkpoint_index.jsonl).
- Spatial/controller:
  [`spatial E/I index`](WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/spatial_ei/checkpoint_index.jsonl),
  [`dense comparator index`](WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/dense_comparator/checkpoint_index.jsonl),
  [`controller index`](WorkingMemory/SelectiveMaintenance/runs/maintenance_20260913_124444/controller_feedback/checkpoint_index.jsonl),
  [`ordinary continuation index`](WorkingMemory/SelectiveMaintenance/runs/maintenance_20260913_124444/continuation/checkpoint_index.jsonl).
- Attention/allocation:
  [`pre-update attention index`](WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieved/remote_results/preupdate_attention/checkpoint_index.jsonl),
  [`control-10 index`](WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/control_10/checkpoint_index.jsonl),
  [`focused-50 index`](WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieved/remote_results/focused_50/checkpoint_index.jsonl).
- Recent attempts:
  [`bias-free old-task index`](WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/retrieved/remote_results/old_unbiased/training/checkpoint_index.jsonl),
  [`restored-bias partial index`](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/retrieved/biased_results/spatial_biased/training/checkpoint_index.jsonl),
  [`single-motion index`](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/runs/motion_20260913_214605/training/checkpoint_index.jsonl),
  [`prospective-query index`](WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/retrieved/prospective_query_results/prospective_query/training/checkpoint_index.jsonl).

Completion and transfer authority includes
[`WM completion`](WorkingMemory/runs/wm_20260912_181219/completion_receipt.json),
[`recurrent completion`](WorkingMemory/RecurrentComparison/runs/recurrent_20260912_202158/completion_receipt.json),
[`retention completion`](WorkingMemory/RecurrentComparison/Retention/runs/retention_20260913_084433/completion_receipt.json),
[`spatial completion`](WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/completion_receipt.json),
[`controller completion`](WorkingMemory/SelectiveMaintenance/runs/maintenance_20260913_124444/completion_receipt.json),
[`attention retrieval`](WorkingMemory/PreUpdateAttention/runs/attention_20260913_143459/retrieval_receipt.json),
[`focused-allocation retrieval`](WorkingMemory/TrainingExposure/Cloud/runs/cloud_20260913_163738/retrieval_receipt.json),
[`restored-bias retrieval`](WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/retrieval_receipt.json),
[`single-motion completion`](WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_receipt.json),
and
[`prospective retrieval`](WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/retrieval_receipt.json).

### Reproducible plot menu

1. **PAV encoder/task matrix:** raw BA and chance-normalized BA for five models
   by seven tasks; annotate parameters and equal update exposure.
2. **Contour allocation:** paired endpoint deltas and validation trajectories,
   with identical task-local-prefix statement in caption.
3. **Temporal accumulator matrix:** per-task BA/AUC and runtime/memory profile;
   do not collapse 98.44–100% into “perfect.”
4. **WM lineage plot:** chronological selected checkpoints and fresh episodes;
   distinguish model step from newly generated episodes.
5. **Per-delay retention:** motion and orientation BA/AUC at D0/4/12/24, with
   parent/refit paired connections and chance lines.
6. **Probe versus deployed output:** state accessibility, angle error in
   degrees, and comparator rescue; visually mark probes as analysis-only.
7. **Spatial/controller/attention deltas:** paired per-cell BA with matching and
   hardware caveats in the caption.
8. **Recognition list length:** restored-bias validation by study-set size,
   including empty-set negatives separately; label as validation/partial.
9. **Motion allocation and task-only curves:** total updates and actual
   per-task exposures on separate x axes; include confusion/class-frequency
   panels.
10. **Cue and source counterfactuals:** paired intervention deltas by phase,
    with out-of-distribution label.
11. **Gamma trajectory:** the stopped prospective run only, ending at its sole
    validation; no extrapolation.
12. **Attention atlas:** four grids per timestep (two heads times two sources),
    joint mass and within-source normalization both retained.
13. **Compute/cost:** wall time, GPU type, estimated/receipt cost, and exposure
    for completed cloud/local arms; never treat hardware as controlled.
14. **Frozen-readout decoding and priority alignment:** plot pooled versus
    spatial BA/AUC with the paired grouped interval; spatial per-delay BA; and
    target-neighborhood mass/peak hit against the 5.33% uniform reference.
    Caption this as one post-hoc frozen checkpoint, distinguish 256 independent
    test base movies per delay from 1,024 repeated delay presentations, and do
    not extrapolate it to scratch end-to-end training.

Every generated figure caption must name: run directory, checkpoint/display
step, selected or terminal status, split, independent sample count and
clustering unit, metric/chance, uncertainty method, paired/unpaired status,
training episodes and updates, seed(s), and whether the panel is primary,
validation, reused-test exploratory, post-hoc, stopped, or cancelled.

## 8. Brief for a rigorous ten-page main paper

### Audience and argument

Write for computational neuroscience and neurally inspired machine-learning
readers. Lead with the staged computational question, not a claim of brain
fidelity. The narrative should show how failure analysis successively
localized bottlenecks: temporal integration, recurrent retention, spatial
binding, memory-guided input selection, and terminal spatial decision. Preserve
negative results because they justify the next controlled change.

### Suggested page budget

| Pages | Content |
|---:|---|
| 0.75 | Abstract and concise contribution statement |
| 1.0 | Biological/computational motivation and explicit limits |
| 1.25 | Tasks, lineage, controls, metrics, and evaluation policy |
| 1.5 | Architecture with tensor shapes, attention/E-I/readout equations |
| 1.0 | PAV encoder and causal temporal results |
| 1.5 | Recurrent retention, probes, spatial memory, comparator |
| 1.5 | Pre-update attention, interventions, tradeoffs |
| 0.75 | Failed/partial recent fixes and priority-readout hypothesis |
| 0.5 | Limitations, scientific interpretation, and next decisive test |
| 0.25 | Conclusion; references continue outside main-text budget if venue permits |

Use a compact main-text table for lineage/exposure and move exhaustive
per-cell results, protocols, seeds, optimizer groups, extra confusion matrices,
rendered stimuli, checkpoint hashes, and provider receipts to appendices or
supplement. Keep neuroscience sources in the main references when they motivate
an architectural commitment; keep implementation-only citations in methods or
appendix. Create a proper bibliography rather than treating repository URLs as
scientific citations.

The paper's final sentence should identify a falsifiable next step: a valid,
lineage-correct comparison of global pooled and task-conditioned spatial
readout on common frozen or commonly initialized computations, with held-out
priority alignment and behavioral outcomes.

## 9. Figure specifications

1. **End-to-end architecture (vector).** RGB movie, shared multiscale encoder,
   fixed opponent traces/energy/normalization, `H`, Q/K/V attention, spatial
   E/I `R/a`, comparator `C`, and task output. Print tensor shapes at every
   junction and use solid/dashed edges for learned/fixed computations.
2. **Temporal/recurrent loop (vector).** One timestep expanded: previous
   `R/a`, current `H`, attention, comparator, E/I update. Indicate that
   comparator uses old memory before the update.
3. **Q/K/V panel.** Show 169 memory queries and 338 keys/values split into
   sensory and memory banks, two heads, source/locality terms, and output drive.
   Do not label heads as fixed sources.
4. **E/I dynamics panel.** Sign-constrained E and I contributions, leak and
   adaptation equations; state the 51/13 channel partition without drawing it
   as cortical cell anatomy.
5. **Old versus proposed readout.** Same `H/R/C` fields branching to (a)
   mean/max global pooling and (b) 1x1/3x3 features, task-specific selection
   and evidence, spatial softmax, weighted logits. Mark the proposed training
   result “not yet established.”
6. **Task panels.** Actual rendered frames for PAV seven-task, broad WM, and
   newer five-task suites in separate groups. Use
   [`WorkingMemory/SpatialTaskBattery/previews/index.html`](WorkingMemory/SpatialTaskBattery/previews/index.html)
   as the five-task visual authority.
7. **Timeline.** Checkpoint ancestry, fresh exposure, architecture deltas,
   completed/diagnostic/stopped/cancelled status. Avoid implying that every
   branch is a fair head-to-head comparison.
8. **Main result plots.** PAV matrix, temporal comparison, retention curves,
   spatial/attention paired deltas, mechanism intervention, and negative recent
   runs. Use common task colors and explicit chance lines.
9. **Attention/priority panel.** Select one individual trial and timestep,
   showing all four attention grids plus current frame and remembered-scene
   reference. Future priority maps must show selection and evidence separately.

Use SVG/PDF source first; rasterize only stimuli/heatmaps. Adopt a
colorblind-safe Okabe–Ito-style palette, redundant line styles/symbols, no
rainbow maps, and no red/green-only contrasts. Design at final column/page
width, with at least 7–8 pt labels after placement. Avoid truncated axes,
3-D decoration, smoothed lines without raw support, or cartoons that imply
unmeasured anatomy.

## 10. Mandatory visual QA

For every figure and compiled paper:

- Render SVG/PDF to the exact final dimensions and also to a 2x PNG preview.
- Inspect at 100% and at thumbnail/page overview.
- Check clipping, bounding boxes, panel overlap, alignment, whitespace,
  font embedding/substitution, minimum type size, legend order, equation
  baselines, raster blur, transparency/alpha, line weights, and color-only
  encodings.
- Check page overflow, table width, float order, orphan headings/captions,
  widows, reference wrapping, and hyperlinks.
- Programmatically inspect PDF page count/media boxes and image bounding boxes
  where possible; run a text-extraction check to catch missing fonts/glyphs.
- Compare plotted values against source files with an automated spot-check;
  test that chance lines and axis units are correct.
- View on light and dark backgrounds if transparency is used.
- Record source hash, render command, dimensions, reviewer, date, defects, and
  regeneration outcome in a QA report. A figure fails QA until defects are
  corrected and the replacement is re-rendered and re-inspected.

No visual should be accepted solely because source code ran. “Manual check”
must mean inspection of the actual final render, not the plotting canvas.

## 11. Scientific-integrity checklist

- [ ] Verify every neuroscience statement against a primary source; use reviews
      for context, not as sole support for central claims.
- [ ] Label preregistered/planned primary outcomes separately from post-hoc
      diagnostics and exploratory reused tests.
- [ ] Label stopped, partial, user-cancelled, and lineage-invalid runs in text,
      tables, and plots.
- [ ] State metric, chance level, split, checkpoint, sample size, and
      independent resampling unit.
- [ ] Include negative results and task tradeoffs; never select only successful
      cells.
- [ ] Never call a probe a deployed rescue or causal mechanism. Never call an
      OOD intervention a biological causal test.
- [ ] Never report validation scores as final test or assign scores to an
      unevaluated checkpoint.
- [ ] Never call a comparison matched or randomized when model initialization,
      optimizer lineage, state size, exposure, task suite, or platform differs.
- [ ] Distinguish generated episodes from repeated paired delay presentations.
- [ ] Distinguish selected from terminal checkpoints and display-step offsets
      from fresh updates.
- [ ] Preserve exact task-family names; do not combine the seven PAV tasks,
      broad sequence battery, focused retention cells, and newer five-task suite.
- [ ] State that most evidence is one-seed and synthetic-task evidence.
- [ ] Do not fabricate missing metrics, confidence intervals, citations,
      checkpoints, run completion, or biological correspondence.
- [ ] Preserve raw predictions, configuration, source hashes, checkpoint hashes,
      provider receipts, and cleanup evidence used by every claim.

## 12. Execution checklist and completion criteria

### Research and data

- [ ] Freeze an evidence manifest for all files used in the paper, extending
      [`LabJournal/evidence_manifest.json`](LabJournal/evidence_manifest.json).
- [ ] Resolve the priority-readout lineage before any new comparison; do not
      reuse the cancelled trace as an outcome.
- [ ] Verify source hashes, selected checkpoint hashes, optimizer migration,
      seeds, split names, exposure, hardware, and cost.
- [ ] Build a machine-readable experiment table with the fields in section 3.
- [ ] Extract plot tables from raw summaries/predictions; retain source path and
      row/group IDs.
- [ ] Recompute a small sample of BA/AUC/confusion/paired deltas independently.
- [ ] Verify primary neuroscience sources and create the bibliography.

### Figures and writing

- [ ] Create version-controlled scripts for every table and plot.
- [ ] Create editable vector sources for the architecture, recurrence, Q/K/V,
      E/I, and readout comparison diagrams.
- [ ] Render and inspect actual task examples.
- [ ] Draft to the section/page budget in section 8, keeping detailed protocols
      and secondary results in appendices.
- [ ] Compile the source reproducibly.
- [ ] Complete and archive the visual-QA record from section 10.
- [ ] Conduct a claim-by-claim factual audit against evidence paths.

### Required final deliverables

The future paper task is complete only when the repository contains:

1. manuscript source and a compiled PDF;
2. editable figure sources and final vector/raster renders;
3. plot/table generation scripts with locked input manifests;
4. machine-readable tables used by the manuscript;
5. verified bibliography;
6. supplemental methods/results as needed;
7. visual-QA report with render evidence;
8. final evidence/provenance manifest and reproduction instructions.

Those outputs do **not** exist merely because this handoff exists.

## 13. Current status, omissions, and unresolved ambiguities

### Superseded live snapshot

The user-supplied live snapshot for pod `mbi39b005jp884` recorded RTX 4090
24 GB at USD 0.34/hour; supervisor PID 399, worker PID 478; display step 8692
(292 fresh updates, 11,680 freshly generated episodes); loss 0.70260,
composite batch accuracy 0.625, gradient norm 0.99744; GPU
2,014/24,564 MiB, 29%, 141.9 W; and deadline
`2026-09-15T10:56:06Z`. It also correctly described the proposed readout as
`H/R/C -> 1x1 192->96 -> 3x3 96->64 -> task selection/evidence -> weighted
logits`, with 75,153 new and 51,858 removed parameters, no gamma, and original
attention biases.

That snapshot is now **historical, not live**. A later local watcher artifact
recorded display step 8720 (320 updates; 12,800 fresh episodes), loss 0.69737,
batch accuracy 0.575, gradient norm 0.52782, GPU 2,014 MiB, 29%, 144.9 W at
`2026-09-15T03:41:38Z`. Most importantly, the
[`lineage-correction cleanup receipt`](WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/lineage_correction_cleanup_receipt.json)
states that the run inherited attention-8400 model/Adam state contrary to the
corrected requirement, both local monitors were stopped, the pod was stopped
and deleted, and absence was confirmed by provider HTTP 404 around
`2026-09-15T03:42:55Z`. There was no completed validation. Therefore:

- the rows are an invalid-lineage training trace, not a priority-readout result;
- the `8400` counter offset alone may be retained for schedule/evaluation naming
  in a future corrected design;
- no model or Adam inheritance is authorized by the correction;
- no replacement from-scratch run was verified in repository artifacts at
  the cleanup time.

### Corrected scratch replacement now training

Newer artifacts then record a separate corrected attempt. At
`2026-09-15T03:47:20Z`,
[`pod_ready.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/pod_ready.json)
records pod `ce00y2ooosl7wc`, NVIDIA GeForce RTX 3090, listed
USD 0.22/hour, with deadline `2026-09-15T11:47:20Z`.
[`model.py`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/model.py)
identifies `spatial_priority_readout_scratch_v2`; its initialization function
loads zero parent tensors, zero optimizer states, and records all component
seeds. At the `2026-09-15T03:51:25Z` documentation snapshot,
[`live_status.json`](WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_status.json)
recorded worker PID 332, lifecycle PID 36128, watcher PID 36812, update 14,
560 episodes and 11,744 rendered frames. The latest minibatch had loss
0.86850, aggregate accuracy 0.400, and pre-clip gradient norm 1.96260; GPU
snapshot was 1,852/24,576 MiB, 50% utilization and 253.32 W. Checkpoint 0 was
indexed, and no validation had completed. This is an **ongoing early-training
snapshot, not an outcome**; later lifecycle/metrics/validation/cleanup artifacts
must supersede it. The scratch run is not an initialization-controlled
comparison to the trained attention lineage.

### Other material gaps

1. Several prose files still describe the priority run as live. The cleanup
   receipt takes precedence.
2. The restored-bias five-task run has validation snapshots but no final
   held-out test; it is partial.
3. The bias-free five-task attempt was cancelled and is not evidence of task
   capacity.
4. ProspectiveQuery has one validation only and no final test.
5. The completed local frozen diagnostic shows spatial localization without
   cued-motion decoding rescue; it does not answer whether end-to-end scratch
   training can use the new readout.
6. There is no central BibTeX database, and a dedicated primary
   biased-competition citation remains to be verified.
7. Some comparisons differ in initialization, inherited optimizer state,
   recurrent state size, platform, or task battery; preserve the qualifications
   in section 3.
8. Most results are single-seed synthetic experiments. No human or neural-data
   validation has been performed.
