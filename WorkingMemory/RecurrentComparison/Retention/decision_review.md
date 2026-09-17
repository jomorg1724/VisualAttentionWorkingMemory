# One read-only implementation/protocol review

Reviewer: `/root/temporal_neuroscience_design`.

Reviewed actual `protocol.py`, `train.py` and `sweep.py`, strict selected-parent lineage, resumed Adam/stream/RNG, full r/a graph, ten-cell schedule, validation-first baseline and paired held-out tests.

Response: “No substantive issue found. Recurrent calls are outside no_grad with r/a graph preserved across all frames; sensory/opponent/memory_input/trunk stay frozen eval. Exact full parameter inventory keeps Adam mapping stable, resumed counters replace correctly, and seeded80-cycle reconstructs schedule from global offset. Eval restores complete sampler state across delays and compares correct retained frame slices: motion first10+report, orientation first3+query/probe. This preserves actual evidence and comparison timing; report D as inserted cue-marked blank frames, with motion ageD+1/orientationD+2. Proceed with accounted profile and authorized fixed allocation; no additional gate requested.”

Protocol qualifications adopted: orientation has two sample rasters plus a fixed query frame before the probe; `post_delay=0` removes query-to-probe waiting rather than a post-probe interval. E/I-core optimizer moments retain their older per-parameter histories from before readout-only fitting. Untrained-delay baseline scores do not establish failure or a biological retention constant. No new circuit change or interpretation of probe failure as erasure is authorized by this experiment.


Final result interpretation confirmed: the existing system learned an observed near-flat motion-decision curve through24 inserted blanks, without proving perfect retention. One-item orientation improved atD12 and remained unresolved atD24 under this horizon/frozen interface. The motion choice is computable before delay; orientation comparison requires the later probe. Parent motionD24 AUC0.789 excludes an all-information-erased narrative. No further checks or experiments requested.
