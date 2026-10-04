# Single-stimulus, no-cue change detection

This diagnostic keeps the older cloud conv-encoder RViT architecture unchanged
and initializes the whole model freshly. The training change is the stimulus:
one moving-dot patch, no cue ring, and no competing patch. All original models
and their artifacts remain preserved.

Each trial retains the exact original target patch at either **(20,50)** or
**(80,50)** in a 100×100 RGB frame. The other aperture is removed. The two cue-time
frames show ordinary fixation instead of a ring, so the movie still has exactly
**29,37,45 frames** for B12,B20,B28. Keep the seven initial fixation frames, motion
reference, baseline durations, eight post-event transitions and final fixation
report frame. Retained moving-dot pixels are identical to the source movie.

Dot number, speed, lifetime, angular dispersion, subpixel rendering, randomly
varying baseline directions and signed 26/28-degree changes are unchanged.
Positive trials contain a direction change in the sole patch. Native foil-only
trials become no-change trials when that foil is removed, keeping the original
**57% change / 43% no-change** labels. Locations retain the native balanced schedule.
There are no privileged angles, labels, frame indices or trial metadata as model inputs.

Internally the existing task/head identifier is retained for harness compatibility.
Actual event rows contain `target=change`, `catch=no change`, and zero foil trials;
the report names change hit rate and no-change false-positive rate. Source event
types remain analysis metadata. This is a modified task, not a result on the native
cued two-patch benchmark. It removes both cue selection and visual competition;
success would not identify the cue glyph alone as the cause of failure.

The unchanged model has **7,270,290 learned parameters / 92 tensors**: ordered
previous/current RGB CNN, 169×256 visual tokens, separate visual and previous-memory
attention, and the existing 256→16 token reduction/flattened classifier. Full
sequence BPTT, FP32/TF32 off, Adam1e-4/no clipping, effective batch32/micro4.
No new carry gate, structured-motion frontend or sampled-frame gradient policy.

Replay retains 1,000 fresh trials per pool and ten shuffled epochs. Target
**2,310 updates / 70,000 presentations / 7,000 unique movies**, matching the older
native-task RViT's completed exposure. Pin a feasible number of complete pools
before production from native-length CUDA profiling. Validation100/250/every500/
terminal,100 trials/condition; mean AUC then BA selects a checkpoint; fresh paired
selected/terminal tests use200 trials/condition. Provisional acquisition criterion
is at least70% BA in every condition;90% is the solved target. No early-score stop.

The new A40 run has an eight-hour/$5 cap including setup, profiling, evaluation
and retrieval. Independent shutdown and artifact mirror are retained. Existing
random-frame cloud and structured-motion local runs continue on their original caps.

[Focused evidence](check_results.json): complete100-trial samples in each condition
retain original target pixels and exact timing, have no cue or foil, match ground
truth from the retained patch's actual direction change, and restore streams
exactly. One full native-movie CPU backward passes. Source checks alone are not
training evidence; see [run status](RUN_STATUS.md).
