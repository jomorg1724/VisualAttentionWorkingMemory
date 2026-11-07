# One randomly selected frame update per movie

The entire unchanged native movie is processed to its ordinary final binary
cross-entropy decision. Independently for each trial, choose one timestep uniformly
from all T frames (including cues, motion, blanks and final report). Re-sample on
every replay presentation; effective batch 32 supplies 32 independent frame draws.

Before the selected timestep, run the learned encoder and recurrence without a
gradient graph. At the selected timestep, differentiate the encoder and memory
update. After it, encode visual observations without gradients but keep the
memory-to-memory Jacobians differentiable using detached copies of learned
weights. No memory detach is permitted after the chosen frame. Train the final
decoder normally on every trial.

The chosen step's local shared-parameter derivative is multiplied by T while
forward values and memory derivatives remain unchanged. Consequently the mean
sampled gradient equals the full-BPTT gradient of the same model at the same
parameters. The decoder gradient is not multiplied by T. This is stochastic
sampling of temporal parameter contributions, not truncated BPTT or extra labels.
It has higher variance than using every frame. Adam's nonlinear update does not
make a sampled run identical to full-BPTT training.

`RandomFrameRViT` uses the fresh motion-energy RViT with the learned 0.98 initial
[memory-carry path](../GatedMemoryRViT/README.md), so gradients can cross the suffix
without the original repeatedly projected memory path. All 7,428,978 learned
parameters train; the published analytic filters remain fixed buffers. The
ordered previous/current frame and nine-frame motion features are conditioning
inputs to the selected processing step. Their existence does not make the other
timesteps' learned computations receive parameter gradients.

The full forward sequence is still required. Backward skips earlier recurrence
and all unselected CNNs, but must traverse the later recurrence. Its speedup must
be measured on the pod. Selected CNN steps are batched together across the microbatch; later recurrence
is grouped by which trials have already reached their selected frame.

[Focused evidence](check_results.json): exact enumeration on a three-frame movie
matches all full-BPTT parameter gradients; on native B28, selecting frame zero
retains gradients through all 44 later updates and gives no raw appearance-input
gradient to the other frames. Zero gradients for three memory-attention tensors
at frame zero are expected because its previous memory is zero, not a graph break.

Training uses fresh whole-model weights, Adam1e-4/no clipping, FP32, effective
batch32/micro4, unchanged native B12/B20/B28 and 1000-trial pools reused ten epochs.
The new cloud run has an eight-hour/$5 total cap, measured CUDA profiling before
pinning exposure, validation100/250/every500/terminal at100 trials/condition, and
fresh selected/terminal tests200/condition. The completed/deleted KDA16 frees its
slot; the other cloud RViT and local run retain their original caps.
