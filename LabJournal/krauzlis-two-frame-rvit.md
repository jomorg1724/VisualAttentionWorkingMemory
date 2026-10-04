# Krauzlis: two-frame CNN with recurrent visual-query attention

**First live validation100 completed:**100trials/condition, BA50%allthreeconditions, meanAUC0.442132; all-positive decisions. Trainingcontinuesunderunchangedcap; earlyscoresare notfinalresultsor astopcriterion. [Validation](../SecondPass/TwoFrameRViT/CloudRuntime/artifacts/validation_000100.json).

**Training update — separate cloud launch authorized and verified.** Userrequested
this model on its own pod plus faster validations. NewA40pod7f27p6jxpitihn,
NEW8h/$5cap21:16:04UTC→05:16:04UTC /October2 10:16:04PM PDT. Nativeprofilepinsfull
4216updates/134912episodes; GPUpeak1,173,951,488bytes. Checkpoint3/96episodes
CPUverified withall92Adamstatesadvanced/allparameterschanged,freshwholemodel
/optimizer/RNG/streams and discarded profile state. Validation100/cell at100,
250,500,...terminal; exact18looks costed beforeproduction. Selection sees all
scheduledlooks; freshpaired selected/terminalfinal200/cell remainsindependent.
No accuracy claim yet. Threefocusedworker tests and exactparallel-pod gate passed;
no exhaustiveCPUcampaign. The existing16headKDA/localCNN-GRU continue, with extra
CPUearlycheckpointsnapshotsoutsideactive run directories. [Live evidence](../SecondPass/TwoFrameRViT/RUN_STATUS.md).
Earlierimplementation-only notes below are historical.

## Question and requested architecture

The user returns to a recurrent vision transformer, asking for a sophisticated
CNN over the past two frames and 256-channel spatial tokens. Current visual
inputs query two separate key/value streams: current vision and previous memory.
The final decision reduces each token to16 features, flattens the spatial field
and uses an FFN. This is an implementation request, with no new training launch
or compute allowance. The existing three-layer KDA cloud run, delayed-frame
CNN-GRU local run and sixteen-head KDA cloud queue continue unchanged.

## Implemented choices

[TwoFrameRViT](../SecondPass/TwoFrameRViT/model.py) concatenates ordered previous
and current RGB frames into a six-channel image. At the first step it repeats the
current image rather than introducing artificial motion from a synthetic prior.
A full-resolution 32-channel residual stage precedes three learned space-to-depth
compression stages, producing 50 x50 x64,25 x25 x128 and13 x13 x256. Two residual blocks
per stage, GroupNorm8 and GELU provide an expressive pair encoder. All20 convs
have stride1; pixel rearrangement plus learned projection reduces resolution.
The last stage pads25 to26 by replicating its bottom/right boundary. There is no
pooling, optical-flow oracle or handcoded frame difference. Pixel rearrangement
preserves phases; later learned compression can still discard information.

Learned spatial row/column positions and token normalization give169 visual
vectors of256 dimensions. Exactly one transformer block is shared across time.
Eight32-dimensional heads in each of two independent attention modules query
current X. Visual attention reads X; cross-attention reads H_(t-1). They have
separate projections and softmaxes. Their outputs are merged with1/sqrt2 scale and
an X residual, then a pre-LayerNorm 256->1024->256 GELU FFN residual produces H_t.
Initial memory is fixed zeros. No CLS, GRU, memory gate or growing token cache.
The state contains 43,264 FP32 values/trial. Current queries can therefore change
when the visual cue changes, while reading previous learned memory content.

Only final H reaches the decoder: LayerNorm, shared per-token 256->16+GELU,
flatten 169*16=2704, then 2704->512->256->2 FFN. All 7,270,290 parameters/92 tensors
are new and trainable. CNN activation checkpointing uses nonreentrant backward
recomputation during training without detaching any temporal state. Streaming
requires the caller to carry the prior raw frame and H; the module has no cache.

## Evidence and limitations

Three focused CPU checks passed in 1.04 seconds: exact dimensions/readout layout,
sequence-stream equivalence (maxabsolute difference 0), reset/order behavior,
previous-memory influence, finite gradients through both attention branches and
CNN, all 92 parameter tensors advancing, and an earliest-frame gradient through
checkpointed full BPTT. A complete native45-frame B28 movie passed a separate
CPU forward/interface check in 0.66 seconds, returning finite1 x2 logits. These are
implementation checks; no trained accuracy or native GPU feasibility is claimed.
No experimental weights were saved for subsequent training.

The compact 13 x13 grid keeps 169^2 attention pairs/head/frame instead of10,000^2,
while retaining a spatial memory field. This is a practical modeling choice,
not evidence that the CNN preserves every small displacement. Recurrent memory
attention can learn retention; its presence does not guarantee it, or establish
a biological memory mechanism. Full native GPU profiling and a fresh optimizer
will be needed under a separately authorized launch budget before training.
Keep native B12/B20/B28 cues, labels, rendering and event proportions unchanged.

[Design and use](../SecondPass/TwoFrameRViT/README.md) ·
[Exact protocol](../SecondPass/TwoFrameRViT/protocol.json) ·
[Focused checks](../SecondPass/TwoFrameRViT/verification/model_cpu.json) ·
[Native integration](../SecondPass/TwoFrameRViT/verification/native_movie_cpu.json).
