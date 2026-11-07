# Correspondence Predictive Memory

Next local experiment requested October 5, 2026. Implemented; user subsequently requested immediate activation after
cancelling OutcomeContrastiveRViT/run01. Supervisor14972 launched6:01PM PDT. Runtime state lives in `LocalRuntime/run01`;
this document describes the prospective experiment, not a trained result.

## Scientific question

Can local visual correspondence, causal transport prediction, and appearance-
invariant transformation learning support a generalizable moving-dot change
decision? The complete model has **432,924 learned parameters in 43 tensors**.
All learned parameters start fresh. Its explicit fixed correspondence operation
is an inductive bias: this experiment does not ask a generic network to discover
the entire motion computation from terminal binary feedback.

No true direction, angle difference, displacement, dot coordinates or simulator
state enters the forward model or becomes a regression target. Binary labels
supervise the final choice. Two independent renderings share the same sampled
dynamics during training; this is disclosed synthetic pairing supervision.

## Neuroscience inspiration and precise limits

1. **Early motion-sensitive populations:** local comparisons across space and
   time, followed by divisive energy normalization. This draws on the general
   local spatiotemporal computation motivating motion-energy theories. Our
   implementation is delay correlation, not an exact Adelson-Bergen or
   Simoncelli-Heeger circuit reproduction.
2. **Pooling across local evidence:** displacement populations retain spatial
   structure and also supply an aggregate motion representation, loosely
   analogous to pooling in motion-processing hierarchies. This is not a claim
   that the model's cells correspond one-to-one with V1 or MT neurons.
3. **Prediction and residual error:** earlier motion transports visible content
   forward; unexplained observations supply a prediction-error signal. This is
   motivated by predictive coding, not a fitted cortical predictive-coding model.
4. **Recurrent memory:** a spatial ConvGRU and separate CLS GRU maintain visual
   state, with recurrent attention feedback. Gates and backpropagation are
   engineering choices; biological learning plausibility is not established.

Sources:
- [Adelson & Bergen, 1985, spatiotemporal motion energy](https://persci.mit.edu/pub_pdfs/spatio85.pdf).
- [Rao & Ballard, 1999, predictive coding](https://www.nature.com/articles/nn0199_79).
- [Teed & Deng, 2020, RAFT correlation volumes and recurrent inference](https://arxiv.org/abs/2003.12039).
- [Jonschkowski et al., 2020, UFlow/self-supervised flow](https://research.google/pubs/what-matters-in-unsupervised-optical-flow/).
- [Bardes et al., VICReg](https://arxiv.org/abs/2105.04906).

The last three are engineering precedents. This is a new combination, not a
reproduction of their architectures, objectives or published performance.

## Inputs and local correspondence

Six consecutive raw 100x100 frames: three at constant motion, then three with
continued motion or a +/-26/28 degree change. Dots remain persistent, with
toroidal wrapping; speed is 1.5/4/8 pixels per frame, count16-48, sigma0.7,
contrast0.42 on gray0.5. Frame3 continues the old displacement; transitions3->4
and4->5 use the second clip's velocity. No cues or distractors.

RGB channels are identical in this renderer, so storage uses one channel and
the model also accepts RGB by averaging its channels. New namespaces are
194101/194102/194103 for train/validation/test. Training has two independently
positioned and independently counted dot layouts for each sampled dynamics.
Evaluation pairs change/no-change trials with matched nuisance draws and uses
one view per decision.

For each neighboring frame pair, take16 overlapping50x50 receptive fields on a
4x4 grid, spaced25 pixels. Subtract each patch mean and apply a Hann window.
For each of441 candidate displacements delta=(dx,dy), dx,dy in[-10,10], compute

    C_t(r,delta) = sum_x A_t,r(x) A_t+1,r(x+delta)
                   / sqrt(sum_x A_t,r(x)^2 sum_x A_t+1,r(x)^2).

The denominator is bounded below by1e-8. FFT64 correlation computes this fixed
operation efficiently; Fourier coefficients are not learned. The search range
covers the training displacement range and is a stated design assumption.
Circular boundary handling matches the renderer.

A shared small1->8->1 convolutional correction on each21x21 displacement map
learns to refine the evidence. The last convolution starts at zero. Add0.1 times
this correction to C, then softmax over candidate displacements with a learned
temperature initialized0.04 and bounded0.015-0.2. This gives local population
p_t(r,delta), with inferred flow v_t(r)=sum_delta p_t(r,delta) delta.

The inferred flow is computed from pixels; it is never compared to ground-truth
flow during optimization. Local receptive-field windowing and a finite search
range can bias estimates, especially in sparse regions.

## Prediction, comparison and memory

Upsample the4x4 inferred flow bilinearly. Periodic backward image sampling
implements W(I,v)(x)=I(x-v(x)). The genuine causal predictions are:

    frame2_hat = W(frame1, flow estimated from frames0,1)
    frame5_hat = W(frame4, flow estimated from frames3,4).

Neither prediction's flow sees its target frame. For prediction P of target T
from source S, use normalized squared error

    E(P,T,S) = mean((P-T)^2)
               / max(0.5[mean((S-.5)^2)+mean((T-.5)^2)],1e-5).

The image-energy denominator is fixed by the observed frames. Predicting gray
cannot become an easy solution by averaging over the background alone.
L_prediction is the mean of the before/after causal errors.

An additional expectation transports frame3 to frame4 using the average of the
first two inferred flows. Its error is a decision feature, **not** a loss that
forces changed trials to remain predictable under their old motion.

All five observed transition populations feed a shared recurrent block. Each
4x4x441 population becomes16 width64 tokens, with a visual CLS and learned
positions. One4-head self-attention and previous-memory cross-attention block,
plus width256 FFN, is followed by a64-channel4x4 ConvGRU and a64D CLS GRU.
Spatial gates follow the repository's existing SpatialGRU equations. Memory
starts at zero for every movie; all five steps use full backpropagation.

The nonlinear decision head receives the CLS memory after transition1, final
CLS memory, their absolute difference, and five explicit comparison features:
before/after pooled-population L1 distance, cosine distance, the two causal
prediction errors, and the old-motion prediction error in the second clip.
This explicitly supplies a generic compare/predict operation; the readout learns
the binary decision. The three-frame boundary is part of this scoped task.

## Appearance-invariant transformation objective

Average transition populations over space and over the two internal transitions
of each clip. A shared441->128 GELU->64 projector forms before/after transform
embeddings. Multiply pooled populations by10 before projection (fixed scaling).
The projector does not consume the binary label.

For each clip, the same dynamics in an independent dot layout provide a positive
pair a,b. Minimize their mean squared difference. Pool all before/after embeddings
from both appearances to compute dimensionwise standard deviation and covariance:

    L_invariance = mean((a-b)^2)
    L_variance   = mean_j relu(0.5 - sqrt(var(z_j)+1e-4))
    L_covariance = sum_(j!=k) Cov(z)_(j,k)^2 / 64
    L_representation = L_invariance + L_variance + 0.04 L_covariance
    L_total = CE(choice,label) + 0.5 L_prediction + 0.1 L_representation.

CE averages both rendered views. Statistics use the full batch. This is a
VICReg-inspired adaptation; no negative pairs are used because different clips
can legitimately share motion. Constant embeddings incur a variance penalty,
but that alone cannot guarantee useful information or prevent every failure.

## Finite local training and evaluation

One local GPU, FP32/TF32off, full batch32 dynamics pairs (64 rendered movies),
Adam3e-4, no weight decay or clipping, two CPU threads. A1000-pair pool is reused
for two independently shuffled epochs, balanced16/class per full batch and4/class
per tail. Each pool has64 updates. Requested25,024 updates =391 pools =391,000
unique dynamics pairs /782,000 unique rendered movies /1,564,000 rendered-movie
presentations. Runtime counters called `presentations` count dynamics-pair
presentations; completion also explicitly records rendered-movie counts.

One new finite14,400-second allowance starts only with native GPU profiling,
including a bounded tiny-fit check, evaluation and reporting. Native timing pins
complete64-update pools prospectively. No cloud, automatic cap extension or
additional arm. Disposable check/profile weights are discarded; production is
fresh with initialization194211 and shuffle194212. Only best/latest checkpoints,
with optimizer/RNG/pool state, are retained. No automatic recovery/retry.

Validation384 trials at64/5000/10000/15000/20000/terminal selects minimum decision
CE. Final test3072 fresh trials/1536 matched contexts; fixed argmax with tie0.
An additional768 test trials evaluate unseen speeds0.75/3/6 and changes18/42,
using the same selected checkpoint and no recalibration. These are never used
for selection. New textures, occlusion, cueing, distractors and the native task
battery remain untested. Success here would not establish unrestricted transfer.

## Evidence before activation

`check_results.json` records CPU checks of correlation/warp sign, prediction
causality, matched evaluation nuisance draws, independent training appearances,
all43 finite parameter gradients and all43 learned tensors changing after two
updates. `runner_check_results.json` exercises production save/select/final-test
and transfer-test control flow on a disposable two-update CPU fixture.

A fixed population-difference baseline, with no learned parameters, achieved
**97.92% BA /0.99590 AUC** on384 diagnostic test trials after choosing its threshold
on192 separate validation trials. Diagnostic offsets60M/72M are disjoint from
production validation12M/test36M. This demonstrates accessible pixel evidence,
not success of the trainable model. The initial diagnostic before the final
namespace layout used a different sample and achieved96.35%/0.98831; its receipt
is preserved as `diagnostic_initial.json`. No model hyperparameter was selected
using either diagnostic test; the second check followed addition of the separate
transfer-test namespace.

The CPU-only queue verifies predecessor completion, process exit, released GPU
lock and frozen source hashes before dispatching a guarded local supervisor.
`queue_status.json` means queued/dispatched; only saved optimizer progress proves
that production training actually started.
