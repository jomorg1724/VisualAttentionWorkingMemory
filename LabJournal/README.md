# Visual Attention and Working Memory — lab journal

[Next-agent handoff](../HANDOFF.md) ·
[paper-writing research handoff](../PAPER_HANDOFF.md).

This is the project's research wiki: what we built, what actually happened, why the next experiment followed, and what remains uncertain. It covers extant work after the user-authorized repository reset, through the dated snapshot below. Deleted pre-reset projects and plans have not been recovered.

**Current state:** a third independent full-scratch dual-attention priority
experiment is live on separate RTX 3090 pod `f7y0zw02f4fzum`; construction,
profile and first finite metrics are verified, but no validation result exists
yet. It does not share lifecycle or artifacts with scratch control pod
`ce00y2ooosl7wc` or comparator pod `h1ygacz4zcsfj3`. The inherited
spatial-priority-readout attempt was cancelled
after a lineage correction and its RunPod was deleted; it completed no
validation and is not an architecture result. A separate fully scratch
replacement pod was provisioned at 2026-09-15T03:47Z; remote construction and
profiling passed, and by its verified launch snapshot scratch training had
reached update 46 / 1,840 episodes, with no validation or result yet. The
completed local frozen-readout diagnostic found target localization without a
decoding rescue and did not touch cloud training. The older prospective-query
run remains stopped and unchanged. See
[current status](CURRENT_STATUS.md) and
[experiment 22](experiments/22-spatial-priority-readout.md).

## Start here

- [Current state, capabilities and unresolved failures](CURRENT_STATUS.md)
- [Chronology: observations, decisions and corrections](CHRONOLOGY.md)
- [Current architecture: tensors, equations and what learns](ARCHITECTURE.md)
- [Tasks, timing, datasets and metric definitions](TASKS_AND_METRICS.md)
- [Open questions and what evidence would answer them](OPEN_QUESTIONS.md)
- [Research foundations and limits of biological claims](RESEARCH_FOUNDATIONS.md)
- [Completed local frozen-readout diagnostic](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
- [How to update this journal](MAINTENANCE.md) and [new experiment template](EXPERIMENT_TEMPLATE.md)
- [Source inventory and hashes](evidence_manifest.json)

## Experiment catalogue

| ID | Experiment | Status at journal entry |
|---|---|---|
| 01 | [Five convolutional encoders: the first corrected sensory screen](experiments/01-pav-encoder-screen.md) | Complete |
| 02 | [Can complementary encoder errors be combined cheaply?](experiments/02-saved-ensemble-audit.md) | Complete |
| 03 | [Gabor and late-SE additions did not solve contour](experiments/03-hybrid-continuation.md) | Complete |
| 04 | [Contour succeeds when it receives enough training allocation](experiments/04-contour-task-allocation.md) | Complete |
| 05 | [One frame at a time: opponent traces beat the tested KDA and ConvGRU fits](experiments/05-causal-temporal-accumulators.md) | Complete |
| 06 | [The broad sequence battery did not isolate memory capacity](experiments/06-sequence-battery.md) | Complete |
| 07 | [Focused recurrent memories learn the rules; LSTM leads on motion duration](experiments/07-lstm-versus-ei.md) | Complete |
| 08 | [Ordering sensitivity: the E/I model benefits from later winning evidence](experiments/08-recency-diagnostic.md) | Complete |
| 09 | [Earlier motion evidence remains in the firing-rate population](experiments/09-state-accessibility.md) | Complete |
| 10 | [An existing output-only refit recovers much of motion performance](experiments/10-readout-refit.md) | Complete |
| 11 | [Retaining a motion judgment improves; delayed orientation comparison remains weak](experiments/11-retention-learning.md) | Complete |
| 12 | [Orientation survives the delay but the ordinary comparison fails](experiments/12-orientation-accessibility.md) | Complete |
| 13 | [Keeping a spatial field helps binding but does not solve every task](experiments/13-spatial-memory.md) | Complete |
| 14 | [Additive controller feedback gives a narrow gain, not a maintenance explanation](experiments/14-selective-maintenance.md) | Complete |
| 15 | [Joint attention substantially improves delayed orientation, with a motion cost](experiments/15-preupdate-attention.md) | Complete |
| 15a | [The orientation improvement depends on memory-source routing during blanks](experiments/15a-attention-mechanism.md) | Complete |
| 15b | [Motion errors include decision bias and predate attention](experiments/15b-motion-audit.md) | Complete |
| 16 | [Training exposure:10% versus50% motion](experiments/16-training-exposure.md) | Complete |
| 17 | [Frozen attention maps and scene overlays](experiments/17-attention-maps.md) | Corrected per-frame viewer complete |
| 18 | [Bias-free attention: old-task control and new spatial battery](experiments/18-unbiased-attention-spatial-battery.md) | Old arm complete; new battery cancelled |
| 19 | [Five-task acquisition with original biases restored](experiments/19-biased-spatial-battery.md) | User-stopped partial; retrieved and pod deleted |
| 20 | [Focused cued motion-duration continuation](experiments/20-single-task-motion.md) | Complete; no task-acquisition gain |
| 21 | [Prospective sensory-conditioned attention queries](experiments/21-prospective-query.md) | User-stopped partial; retrieved and pod deleted |
| 22 | [Spatially preserving task-conditioned priority readout](experiments/22-spatial-priority-readout.md) | Scratch cloud training live; complementary frozen-core diagnostic complete |
| 23 | [Dual pre/post attention with exclusive priority-map decoding](experiments/23-dual-attention-priority.md) | Independent scratch cloud training live |
| 24 | [AV-context v2: five combined changes, local scratch run](experiments/24-av-context-v2.md) | Local training live; pre-registered gate at 2,400 |
| 25 | [Battery audit: ideal observers, streams, one-step training diagnostics](experiments/25-battery-audit.md) | completed 2026-09-16 (audit only, no training) |
| 26 | [Plain baseline, rung 1: standard CNN+GRU from scratch, recipe sweep, difficulty ladder](experiments/26-plain-baseline-rung1.md) | completed 2026-09-17 (orientation family; other families record runs only) |
| 27 | [Accumulator states inside the conv stack vs plain baseline: gate, curriculum, delay ladder](experiments/27-accumulator-conv-stack.md) | completed 2026-09-17 (RunPod, both lanes, two seeds) |

The useful trajectory is not a sequence of architectures declared permanently good or bad. It contains task acquisition failures, output-use failures, genuine improvements, and regressions that triggered targeted diagnostics. Notably, contour was solved by changing allocation; earlier motion information survived in E/I rates and benefited from output refitting; spatial memory helped binding but harmed motion; pre-update attention improved delayed orientation, and its blank-period routing is functionally important.

## Scope and provenance

The broader project follows Jeremy Wolfe's Guided Search 6.0 as a functional scaffold, omits its diffuser, and provisionally treats activated long-term memory as synaptic weights at the user's request. PAV is one component. The current trained model contains sensory temporal integration, spatial recurrent memory and pre-update attention; it is not a complete GS6 implementation or a validated biological model.

This wiki adds an explanatory layer over retained code, reports, scores, checkpoints and receipts. It does not launch experiments, change models or manufacture missing evidence. Completed pages link primary local artifacts. Live statuses are explicitly dated; they require refresh rather than being treated as permanent facts.

Initial documentation compiled 2026-09-14T00:21:31.576956+00:00. Scientific claims describe the saved experiments, not independent replication across seeds or general human performance.
