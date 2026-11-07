# Weighted-mean CNN and standard GRU

An untrained queued candidate processes each movie through the fixed causal pixel mean
`0.5 X[t] + 0.4 X[t-1] + 0.1 X[t-2]`. Missing startup history repeats frame zero.
One shared residual CNN encodes each resulting RGB image. Its learned spatial output
has 16 channels at 13×13 locations; all 2,704 values enter one standard GRU with
256 hidden units directly. A learned 256→128→2 classifier reads the terminal hidden
state. The GRU starts at zero for every movie. There are no learned motion filters,
attention blocks, additional recurrent modules or inherited weights.

The CNN follows the existing encoder's residual convolutions and space-to-depth
compression, with a three-channel RGB input. Encoder activation checkpointing uses
nonreentrant recomputation during training. Full temporal gradients reach every
frame and learned weight. The weighted mean is fixed arithmetic, not a trainable
motion detector or an auxiliary objective.

The stimulus adapter is exactly [SingleStimulusStream](../SingleStimulusRViT/stimuli.py):
one stimulus at its original (20,50) or (80,50) center, no cue or second stimulus,
and unchanged 29/37/45-frame lengths and original dot dynamics. Change labels are
57% positive and no-change labels43% negative. Source foil/catch examples are
ordinary no-change trials; no visible foil exists. This is the modified single-stimulus
change task, not the original cued benchmark.

The prepared worker starts the entire model, Adam, RNG and streams fresh. It retains
FP32, unclipped Adam1e-4 and effective batch32/micro4. Each fresh1,000-movie pool is
reused for10 shuffled epochs, including correctly normalized partial batches;
330 updates yield10,000 presentations per pool. The requested2,310 updates are
70,000 presentations of7,000 unique movies. A disposable native CUDA profile must
pin feasible complete pools before production within the parent-owned new8h/$5
cap, with no automatic extension or transfer of profile state.

Validation uses100 movies per condition at updates100,250,every500 and terminal;
mean AUC then balanced accuracy selects the checkpoint. Paired fresh final tests
use200 per condition for selected and terminal models. Reports show change hit and
no-change false-positive rates, with the absent-foil category explicitly empty.
Immutable checkpoint readback verifies the fresh constructor, every learned
parameter update and persisted Adam state. The source bundle contains no weights,
training state, credentials or budget. The parent owns queueing and provisioning;
this implementation does not launch a worker or spend compute.

`python -m SecondPass.WeightedMeanConvGRU.check` runs a short CPU check of exact
startup arithmetic, causality, the spatial flatten/GRU interface and full temporal
gradients. The model has 6,999,474 trainable parameters in70 tensors. Counts are also
verified from the actual fresh constructor in the worker and bundle manifest.
