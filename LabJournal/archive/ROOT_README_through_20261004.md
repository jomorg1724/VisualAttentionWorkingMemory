> Historical snapshot through October 4, 2026. Earlier present-tense launch statements are not current status. Relative links have been adjusted for this archive location.

# Visual Attention and Working Memory

**Current: user-authorized ConvGRU FROM SCRATCH is training on RunPod.**
All components freshly initialized;10400updates/332800episodes pinned, native
13tasks/35conditions unchanged. Saved production and local checkpoint retrieval
verified. See [live status](../CURRENT_STATUS.md) and
[fresh harness](../../SecondPass/SpatialReadout/FreshRun/README.md). Earlier cancellation
below still applies to the transformer run, not this separately authorized run.

**Current: training CANCELLED; architecture rolled back to the original
CNN + spatial KDA + global GRU.** See [active baseline](../../SecondPass/ACTIVE_BASELINE.md).
The RunPod GPU is stopped; its artifact disk is retained and still billable.
No new run is authorized. New training must start entirely from scratch unless
the user specifically authorizes inherited weights. Older status entries below
are historical, not current launch authority.

[Next-agent handoff](../../HANDOFF.md).

**Latest (2026-09-23):** the user explicitly authorized a new **eight-hour local continuation from terminal715**. The sole MPS worker has advancing optimizer progress; checkpoint728 is independently hash-verified and loaded. The pinned target is **2,145 additional updates / 68,640 additional episodes**, or **2,860 total updates / 91,520 episodes**. Hard deadline: **2026-09-23 09:08:50.062308 UTC**; no renewal. Parent owns completion guardian `proc_5cbdb07600dd`. Prospective selection uses mean validation AUC; the old selected117 result remains historical. See [current status](../CURRENT_STATUS.md), [v3 continuation contract](../../SecondPass/JointTraining/AMENDMENT_V3.md), and [preserved completed715 results](../../SecondPass/JointTraining/RESULTS.md).

**Analysis protocol:** follow [ANALYSIS_SOP.md](../../ANALYSIS_SOP.md) for the next trained-model analysis. Grounded in Morgan, Albanna & Herman's recurrent vision-transformer paper, it requires cueing psychometrics, cued-versus-uncued attention allocation, and spatial inhibition and microstimulation with behavioral readouts. Model diagnostics support these scientific comparisons; they do not replace them.

A fresh component-by-component project using [Jeremy Wolfe's Guided Search6.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/) as a functional scaffold.

- **PreAttentiveVision:** first component; compare convolutional encoders on two-frame sensory tasks, starting with four-way random-dot motion direction classification.
- **Sensory temporal integration:** completed comparison of three causal accumulators; the opponent model is the current sensory reference. See [design and execution allocation](../../PreAttentiveVision/TemporalIntegration/README.md).
- **Memory capacity battery:** completed with the opponent model and all learned weights unfrozen. See [results](../../WorkingMemory/report.md). New sequence rules were weak even at minimal delays, so this run did not cleanly isolate memory capacity.
- **Visual working memory:** the LSTM/E/I comparison is complete. Frozen-state diagnostics, readout refitting and retention training followed; a spatial E/I competitor then substantially improved feature-location binding while exposing a motion trade-off. See the [lab journal chronology](../CHRONOLOGY.md) for the evidence and decisions.
- **Visual attention:** pre-update joint sensory/memory attention is implemented and trained. It substantially improved delayed orientation, with motion regressions that prompted targeted diagnostics and the current training-allocation comparison. This is a component experiment, not a complete visual-search system.
- **Current state (2026-09-16):** the lineage is closed as a failure. Every
  from-scratch arm on the five-task battery stayed at chance, including on
  tasks the warm-started models had solved; the warm starts had hidden that
  the stack could not learn end to end. Nothing is training. The next phase
  keeps the tasks as benchmarks, audits the environments and training logic,
  and rebuilds from first principles with all weights trainable. See
  [HANDOFF.md](../../HANDOFF.md) and [current status](../CURRENT_STATUS.md).
- **Integration:** later, as components become useful.

The diffuser is excluded. Activated long-term memory is provisionally represented by synaptic weights, as requested by the user. These choices do not claim a complete implementation or neuroscientific validation of GS6.

Sensory work and results belong in [PreAttentiveVision](../../PreAttentiveVision); memory experiments belong in [WorkingMemory](../../WorkingMemory). The previous repository, including Git history, was deleted; this is a new repository.
