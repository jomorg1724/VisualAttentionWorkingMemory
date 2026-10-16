# Outcome-supervised contrastive RViT: next experiment

Status: architecture clarified by the user and implemented; objective, temporal
gradients and production path CPU-checked. Local activation is recorded separately
in `LocalRuntime/run01`; no result is implied by these checks. The faster-dot
transformer completed October5 at4:36PM PDT with finalBA49.772% andAUC0.498358.

## Requested computation

Process the six raw frames in order, carrying state across all six: three frames
of one motion direction, then three frames continuing or changing direction.
Visual tokens plus a visual CLS pass through the ViT and receive recurrent
feedback. Their output enters the spatial recurrent network, which maintains
spatial memory tokens and a corresponding persistent CLS memory token. The
terminal memory CLS supplies both the two-choice belief head and the contrastive
representation. Backpropagation spans the whole trial; no direction angles,
velocities, trial phases or correctness labels enter the forward model.

The user clarified that memory must be a patch-based recurrent network, such as
a spatial GRU/LSTM. This implementation uses the existing **FinalConvGRU** gate
equations from `SecondPass/SpatialReadout/model.py`, also used by
`PreAttentiveVision/TemporalIntegration/accumulators.py::SpatialConvGRU`.
The new `SpatialGRU(64)` matches the reference exactly with the same weights.
We adapt width/grid to128channels/10x10patches and add an independent linear CLS
GRU with the same gate equations. The CLS is not inserted as a fictitious grid
cell. This is a new composition from the existing components, not an assertion
that the previous TwoFrameRViT already contained the missing GRU.

Each raw100x100RGBframe becomes100 learned10x10patch embeddings ofwidth128.
Prepend visualCLS and learnedpositions; use one shared4-head ViT block with
separate visual self-attention and previous-memory attention, combine their
residuals divided bysqrt2, then a128->512->128 GELU FFN. This follows the previous
RViT feedback organization. Its101 outputs drive the spatial and CLS GRUs:

```text
write, reset = sigmoid(W_gate * concat(visual, previous))
candidate = tanh(W_candidate * concat(visual, reset * previous))
next_memory = (1-write) * previous + write * candidate
```

For spatial tokens, `*` means3x3convolution over the10x10grid; forCLS it is a
linear map. AllGRUbiases initiallyzero; memory startszero at eachtrial. Both
newstates feed back on the next frame with no detach. FinalrawmemoryCLS is
normalized for contrast; LayerNorm->128GELU->2 gives the beliefs. Every learned
parameter trains jointly. Total1,316,994parameters in36tensors. No pretrained
weight transfer, pooling/flattened-memory decision shortcut or weighted-frame
sum is used. Patchwidth/headcount/grid choices are engineering settings.

## Bounded local experiment

Six-frame geometry preserves speeds1.5/4/8 and26/28degreechanges, with fresh
193101/102/103train/val/test namespaces and no privileged forward inputs.
The firstafterframe continues the oldvelocity; the followingtwo transitions
use the secondclip's direction. Training nuisances independent per movie;
evaluation same/change movies share each basecontext. This is a simplified dot
task, not a nativeKrauzlis replication.

FreshAdam3e-4, epsilon1e-8, zero weightdecay, no clipping, FP32/TF32off, CPUthreads2,
one localGPUworker. Requested25,024updates in complete64-update pools. Eachpool
contains1000freshmovies usedfortwoshuffledepochs; balancedfullbatches32(16/class)
andtail8(4/class) retain cross-example gradients. Target391,000unique movies/
782,000presentations if nativeprofiling fits within the newfinite14,400second
allowance; otherwise pin fewercompletepools beforeproduction. No automatic
extension, cloud or additionalproductionarm.

Within thatsameallowance, a bounded8-trial memorization check runs foratmost256
updates and discardsitsweights beforefreshprofiling/production. It can establish
some learning capability, not generalization. Validation384presentations at64,
5000,10000,15000,20000andterminal selectsminimumdecisionCE. Final3072presentations/
1536pairedcontexts uses the selectedcheckpoint and actualargmax (no threshold
fitting). ReportTP/TN/FP/FN, BA,AUC,CE andallsixcells. Onlybest/latest model
checkpoints persist, includingAdam/RNG/samplerstate. Independentdeadlineguard,
source/config receipts and savedfirstupdate verification are used.

## Choices, feedback, and the four outcomes

Let `p_i = softmax(logits_i)` be terminal beliefs, `a_i = argmax(p_i)` the choice,
and `r_i` the environment's score (+1 correct, -1 incorrect). For two choices,
these reveal the correct answer exactly:

`y_i = a_i if r_i = +1 else 1-a_i`.

This is binary supervision through outcome feedback. It requires no motion-angle
target, but is not unsupervised learning or a policy-gradient estimator. With
more than two choices, negative feedback alone would no longer identify the
correct alternative; that extension would require a different objective.

The four nonredundant groups are TP (hit), TN (correct rejection), FP (false
alarm), and FN (miss). FP and FA are two names for the same outcome.

| Anchor group | Positive answer group | Opposite answer group | Anchor weight |
|---|---|---|---:|
| TP | Other TP and FN | TN and FP | 1 |
| FN | TP and other FN | TN and FP | 2 |
| TN | Other TN and FP | TP and FN | 1 |
| FP / FA | TN and other FP | TP and FN | 2 |

Correctness alone must not define a cluster: a hit and a correct rejection
require different answers. Errors are directed toward their correct answer
group, not made into a permanent separate error class. The weight2 is an
explicit engineering choice for this candidate, not a published optimal value.

## Full-batch objective

Normalize the terminal memory CLS directly: `z_i = h_i / max(||h_i||, 1e-8)`.
No extra projection head is currently specified. For the full batch of B trials,
let `P(i) = {j != i : y_j = y_i}` and temperature `tau = 0.1`.

```text
ell_i = -(1 / |P(i)|) sum_{j in P(i)} log[
    exp(z_i dot z_j / tau) / sum_{k != i} exp(z_i dot z_k / tau)
]
w_i = 1 if correct else 2
L_contrast = sum_i w_i ell_i / sum_i w_i
L_choice = mean_i -log p_i[y_i]
L_total = L_choice + 0.1 L_contrast
```

Argmax, rewards, masks and weights are detached decisions. Gradients flow through
the beliefs and through every CLS vector both as an anchor and as a candidate.
The decision term gives the belief head a training signal; the contrastive term
shapes the shared memory representation. Joint training, direct memory CLS,
single views, outcome-derived targets, mistake weighting and coefficient0.1 are
adaptations, not an exact reproduction of Khosla et al.'s two-stage procedure.

Use a class-balanced full batch with at least two examples per answer; proposed
B=32 gives16 per class and15 positives per anchor. Exclude only the anchor itself
from its denominator. If forward passes are chunked for memory, concatenate the
still-differentiable CLS vectors/logits before evaluating this loss once.
Independent losses on microbatches are not equivalent and are not permitted.
No stale memory queue or detached negative bank is part of this candidate.

## Relation to CPC and limits

Supervised contrastive learning is the closer match for trial-outcome grouping.
CPC encodes past observations into a causal context and distinguishes its actual
future latent from negative future samples. A genuine CPC extension would need
an explicit future target encoder/predictor and temporal contrastive terms;
terminal outcome grouping alone is not CPC. We do not claim a CPC mutual-
information bound for this reward-weighted supervised objective.

For these continuous dot movies, future matching can exploit persistent scene
identity without solving direction-change detection. Also, the random change
cannot be determined before evidence of it arrives. These are reasons to start
with the requested terminal outcome objective, not proof that CPC cannot help.
No separate CPC term has been added to the candidate.

This loss is applicable to other binary tasks with correctness feedback, but
generalization across task/stimulus families remains untested. It may compress
within-class sensory distinctions; success on change/no-change would not prove
a general-purpose sensory memory. No performance claim is made before training.

## Evidence and sources

`loss.py` implements the objective. `check_loss.py` / `loss_check_results.json`
verify all four outcomes, recovery of binary targets, finite nonzero gradients,
permutation invariance, lower loss for separated answer classes than collapse,
and full-batch gradient equivalence when forward outputs are chunked before
concatenation. These are objective checks, not a learning-capability test.

- Khosla et al. (2020), *Supervised Contrastive Learning*, Eq.2:
  https://arxiv.org/abs/2004.11362
- Author implementation, `SupConLoss`:
  https://github.com/HobbitLong/SupContrast/blob/master/losses.py
- van den Oord, Li and Vinyals (2018), *Representation Learning with Contrastive
  Predictive Coding*, causal context and InfoNCE Eqs.3-4:
  https://arxiv.org/abs/1807.03748
