# Visual Attention and Working Memory

[Next-agent handoff](HANDOFF.md).

A fresh component-by-component project using [Jeremy Wolfe's Guided Search6.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965574/) as a functional scaffold.

- **PreAttentiveVision:** first component; compare convolutional encoders on two-frame sensory tasks, starting with four-way random-dot motion direction classification.
- **Sensory temporal integration:** completed comparison of three causal accumulators; the opponent model is the current sensory reference. See [design and execution allocation](PreAttentiveVision/TemporalIntegration/README.md).
- **Memory capacity battery:** completed with the opponent model and all learned weights unfrozen. See [results](WorkingMemory/report.md). New sequence rules were weak even at minimal delays, so this run did not cleanly isolate memory capacity.
- **Visual working memory:** the LSTM/E/I comparison is complete. Frozen-state diagnostics, readout refitting and retention training followed; a spatial E/I competitor then substantially improved feature-location binding while exposing a motion trade-off. See the [lab journal chronology](LabJournal/CHRONOLOGY.md) for the evidence and decisions.
- **Visual attention:** pre-update joint sensory/memory attention is implemented and trained. It substantially improved delayed orientation, with motion regressions that prompted targeted diagnostics and the current training-allocation comparison. This is a component experiment, not a complete visual-search system.
- **Current state:** after cancelling and deleting an invalid inherited-lineage
  attempt, the corrected spatial-priority-readout architecture is training
  fully from scratch on a fresh RTX 3090 RunPod. Construction/profile checks
  passed; current metrics are launch-health evidence, not validation. Earlier
  partial cloud and prospective-query runs remain stopped/retrieved/deleted. See
  [live status and evidence](LabJournal/CURRENT_STATUS.md).
- **Integration:** later, as components become useful.

The diffuser is excluded. Activated long-term memory is provisionally represented by synaptic weights, as requested by the user. These choices do not claim a complete implementation or neuroscientific validation of GS6.

Sensory work and results belong in [PreAttentiveVision](PreAttentiveVision/); memory experiments belong in [WorkingMemory](WorkingMemory/). The previous repository, including Git history, was deleted; this is a new repository.
