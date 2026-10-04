# Removing cue selection and the competing patch

2026-10-03 UTC. The user requests the same complete sequence lengths with just one
stimulus in an existing location, binary change detection, a conv-encoder model,
and pod training. The decision is whether the simpler sensory task is learnable
without cue-based selection and competition from another patch.

Use the older RViT architecture unchanged and wholly fresh: 7,270,290 parameters
across92 learned tensors, full BPTT, no carry/sampled-gradient changes. Match its
replay schedule and exposure target:1,000trials/tenepochs,2,310updates /70,000
presentations /7,000unique trials. Adam1e-4/no clipping,FP32,effective32/micro4.

The [adapter](../SecondPass/SingleStimulusRViT/stimuli.py) retains source target
pixels at(20,50) or(80,50), removes the other aperture, and replaces two cue-time
frames with fixation. B12/B20/B28 retain exactly29/37/45frames, motion onset,
baseline/event/report timing, all dot properties and signed26/28-degree changes.
Labels stay57%change/43%nochange: removing an invisible foil event leaves the
retained target unchanged. Former source-event identities are analysis metadata;
actual events are change/nochange, with no foil. No metadata enters the model.

This removes both cue selection and the competing stimulus, so improvement would
not isolate the cue glyph alone. Architecture, objective and sampling policy are
preserved relative to the older native-task run. Failure on this simplification
would not prove that cues are irrelevant to the original failure.

[Focused check](../SecondPass/SingleStimulusRViT/check_results.json) confirms
pixel identity on the retained patch, unchanged timing, correct actual direction
labels, cue/other-patch absence and exact stream restore for all three conditions;
one full native movie backward succeeds. No repeated broad audit or training tests.

New singleA40 cloud cap is eight hours/$5 including setup/profile/evaluation/
retrieval; no existing budget is extended. Target exposure is pinned before
production from measured CUDA rates. Validation100/250/every500/terminal,
100trials/condition, meanAUC thenBA; fresh paired selected/terminal200/cell finals.
Provisional success thresholds stay70%BA acquisition /90%BA solved in every
condition. Current random-frame cloud and structured-motion local runs continue.
Production status requires persisted optimizer evidence.

[Run and evidence](../SecondPass/SingleStimulusRViT/RUN_STATUS.md).


## Actual production — 2026-10-03 03:31 UTC

Pod`jxbmb44y9wamhl`,oneA40/$0.49h. Full2,310 updates /70,000 presentations /7,000
unique movies prospectively pinned. Production checkpoint3/96 independently
downloaded and CPU reloaded, all92learned tensors/Adam states advanced; initial
state matches direct fresh constructor with empty Adam. Snapshot6/192,loss0.68173.
First validation100 pending. Fixed cap03:28:02→11:28:02UTC /October3,4:28:02AM
Pacific; guard/mirror/wake independently active and bounded. Source archive
contains no trained checkpoints. The other two active runs remain unchanged.
