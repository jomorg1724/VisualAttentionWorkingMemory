# Random-frame gradients and a direct memory carry

The user asked whether gradient normalization could correct the RViT's severe
temporal attenuation, then proposed learning from one independently sampled
frame per complete trial. They requested cloud training in the slot of the
nearest-finished model. KDA16 had just completed; verified retrieval and deletion
freed its slot. The remaining cloud RViT and local structured-motion run continue
under their original caps.

The [carry variant](../SecondPass/GatedMemoryRViT/README.md) changes the memory
update to H_t = a_t H_previous + (1-a_t) proposal. One shared 256-channel sigmoid
gate per token starts at a=0.98, with normalized observations and an unnormalized
identity path for old memory. Fresh paired native 29/45-frame checks give
early/final memory-gradient ratios around 0.57/0.42. This checks the gradient
route; it does not establish task acquisition. No separate full-BPTT carry arm was launched.

[RandomFrameRViT](../SecondPass/RandomFrameRViT/README.md) combines that carry
architecture with the structured-motion frontend. Independently sample one
uniform timestep per trial. Prefix computations have no graph; suffix weights
are detached while memory Jacobians stay live. Multiply only the selected step's
parameter derivative by sequence length. The final decoder trains normally.
Averaging all possible choices recovers the full-BPTT gradient at fixed weights.
Sampling increases variance, and Adam updates need not match a full-BPTT run.
No temporal labels, task changes or additional losses were introduced.

[Focused proof](../SecondPass/RandomFrameRViT/check_results.json) enumerates a
three-frame movie and matches every full-BPTT parameter gradient, including
with activation checkpointing. On native B28, selecting frame zero retains the
gradient through all 44 subsequent memory updates; only that raw appearance
frame has a nonzero input gradient. Selected CNN steps are batched across trials,
and recurrence is grouped into prefix, selected and suffix subsets.

## Actual launch — 2026-10-03 UTC

Fresh model has 7,428,978 learned parameters across 94 tensors. The analytic
motion filters remain fixed buffers. Adam 1e-4/no clipping, FP32/TF32 off,
effective batch 32/microbatch 4, native B12/B20/B28 unchanged. Replay uses
1,000-trial pools and ten epochs per pool; frame choices are resampled on every
presentation. CUDA profiling pinned **3,960 updates / 120,000 presentations /
12,000 unique movies** before production. The measured profile is approximately
two seconds per update; no matched speedup comparison is claimed.

Pod `5us0rp5jwmg5bu`, one A40 at $0.49/hour, created 02:39:56 UTC. New eight-hour/
$5 total cap ends 10:39:56 UTC / October 3, 3:39:56 AM Pacific, with the last ten
minutes reserved for retrieval. Independent authenticated shutdown and off-pod
mirror are active. Fresh production checkpoint 3 / 96 presentations was downloaded
and CPU reloaded: all 94 Adam states and learned tensors advanced from direct
fresh initialization; fixed buffers unchanged. No predecessor/profile state transfer.

Validation at 100/250/every 500/terminal, 100 trials per condition, selects by mean
AUC then balanced accuracy. Fresh paired selected/terminal tests use 200 per
condition. First live snapshot: update 25 / 800 presentations, loss 0.70018.
No validation result or acquisition conclusion yet. This experiment combines
the carry correction and sampled gradient policy; task performance alone would
not isolate their respective effects. No automatic cap renewal or extra arm.

[Run status and evidence](../SecondPass/RandomFrameRViT/RUN_STATUS.md).
