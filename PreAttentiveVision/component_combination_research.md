# Combining PAV computations without discarding the working motion path

Read-only source and saved-result study, 2026-09-12. No encoder, decoder or stimulus was changed; no model loading, inference, GPU work or training was performed. These are at most two proposed development experiments, not an execution plan or renewed compute allowance.

## What the completed comparison supports

The seven-task run in `results_multitask.json` completed756 updates and24,192 training pairs per encoder, **3,456 pairs/task**, with one training seed. All selected checkpoints were step756. Each test task contained448 pairs. The relevant saved results are:

| Encoder | Motion BA / macro OVR-AUC | Contour BA / AUC |
|---|---|---|
| ConvNeXt/GRN | 80.58% / .9356 | 60.27% / .6251 |
| Mobile/SE | 27.68% / .5215 | 71.65% / .8301 |
| VOne/residual | 23.88% / .5116 | 72.10% / .8040 |

ConvNeXt is the most useful current starting point for motion and broad performance. Mobile and VOne provide **candidate sources of complementary computations**, not proof that SE or Gabors produced their contour scores. Their full encoders also differ in normalization, nonlinearities, downsampling, residual blocks, widths of intermediate expansions and optimization behavior.

The separately completed saved-score analysis in `combination_analysis.json` provides a more direct complementarity observation. Equal probability averaging of ConvNeXt+Mobile gave74.78% contour BA versus Mobile's71.65% (paired95% gain interval1.1–5.3percentage points), but reduced motion BA to70.76% from ConvNeXt's80.58%. ConvNeXt+VOne likewise gave74.78% contour BA and77.68% motion BA. Thus their predictions can combine usefully for contour while blind averaging sacrifices motion. This motivates a limited feature addition rather than adopting an ensemble as the new architecture. It remains post hoc analysis of already inspected test examples, conditional on one set of trained models; it neither identifies SE/Gabor causality nor promises that a hybrid will inherit both strengths.

There is strong evidence that the initial run did not reach a flat trajectory: between validation steps504 and756, Mobile contour BA rose55.80→75.45%, VOne54.02→74.55%, and ConvNeXt motion51.79→81.25%. These are descriptive development changes, not independent replications. **An unchanged continued ConvNeXt arm is essential** to distinguish a useful addition from the benefit of additional training; there is no evidence here of a permanent structural bottleneck.

## Actual transferable components

`models.py` already contains the following distinct operations:

- ConvNeXt: learned 7x7 depthwise spatial mixing; per-location channel LayerNorm; 4x channel expansion; GELU; GRN; residual addition. GRN compares each channel's spatial L2 response with the mean channel response. It is not an SE gate or a demonstrated biological normalization circuit. [Woo et al., 2023](https://arxiv.org/abs/2301.00808); [official GRN](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py).
- Mobile: 3x inverted expansion, learned depthwise5x5 convolution, h-swish, GroupNorm, SE and linear projection. SE uses a learned nonlinear function of channel means to modulate the retained map. That function is mathematically portable; Mobile's trained channels are not interchangeable with ConvNeXt's. [Howard et al., 2019](https://arxiv.org/abs/1905.02244); [Hu et al., SE](https://arxiv.org/abs/1709.01507).
- VOne adaptation: a fixed chromatic Gabor bank, simple rectification and quadrature energy, followed by an anti-aliased learned residual encoder. Fixed front-end filters are independent of the downstream learned channel basis and can be reused directly. The current bank contains no neuronal stochasticity and is not the original fitted VOneBlock. [Dapello et al., 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html); [official simple/complex implementation](https://github.com/dicarlolab/vonenet/blob/master/vonenet/modules.py).

The bank frequencies are0.12 and0.25 cycles/pixel. The current contour elements use a5-pixel period, or0.20 cycles/pixel; orientation gratings use5–10 cycles/image, or0.05–0.10 cycles/pixel. This makes frequency coverage a **specific possible explanation** for the VOne task profile. It is not proof: Gabor bandwidths overlap, the bank includes a raw-RGB route, and the following layers learn. Do not silently broaden the bank while claiming to test transfer of the existing component.

`decoder_multitask.py` already has the valuable ordered comparison: signed difference, absolute difference, mean, product and25 local offset correlations per scale. It uses50/25/13-pixel spatial maps before decoder pooling. Preserve these terms and resolutions. A pure energy-only or swap-symmetric replacement would remove information relevant to signed motion; more global pooling could erase small displacements.

## First proposal: residual oriented-feature side branch

**Hypothesis:** fixed oriented local measurements can improve the representation available for contour grouping while retaining the learned ConvNeXt path that already supports motion. This tests an engineering addition; it does not test whether Gabors explain the original VOne advantage.

Compute the original ConvNeXt pyramid \(H_1,H_2,H_3\) without changing its learned stem, reductions, GRN blocks or connections. Independently compute the existing24 simple plus24 complex fixed features \(G(x)\) at100×100. Use a branch-only binomial low-pass/decimation to50×50, channel LayerNorm and a learned1x1 projection48→24:

\[
B(x)=P\!\left(\mathrm{LN}_{C}\!\left(\downarrow_2[h*G(x)]\right)\right),\qquad
\widetilde H_1=H_1+\alpha\odot B(x).
\]

Return \([\widetilde H_1,H_2,H_3]\) to the same decoder. **Do not feed the altered \(\widetilde H_1\) back into the deeper ConvNeXt stages.** Thus its original deeper feature path remains present independently of the side branch. The branch keeps local spatial structure; normalization is across channels at each location, not a spatially pooled replacement. It may suppress absolute branch amplitude, but the original pathway retains that information.

Initialize the24 channel scales \(\alpha\) to zero and the projection to ordinary nonzero random weights. The initial function then equals the trained base model. The scale receives gradients first; projection gradients become available as the scale moves away from zero. Do not zero both scale and projection. This preserves the initial function, **not a guarantee that later training cannot harm motion**. The unchanged motion/decoder path still needs outcome monitoring.

**Analytical cost, not profiled timing:** approximately1,296 new trainable parameters:96 LayerNorm parameters,1,176 projection parameters,24 residual scales. The existing fixed bank adds11,664 stored coefficients. Computing48 filters over3 channels with9x9 kernels at100×100 costs116.64 million multiply-accumulates per frame; blur and projection bring the branch to approximately120.6 million MAC/frame, excluding nonlinearities and normalization. A pair doubles this. Small parameter count therefore does **not** imply negligible GPU cost. This count uses the current dense filter implementation; no unimplemented separable optimization is assumed.

**Control and useful result:** compare with unchanged ConvNeXt continued from the same checkpoint, with the same new source pairs, recipe, decoder, update exposure and evaluation examples. Improvement in contour scores without an unacceptable motion loss would support this branch as an engineering extension. It would not isolate fixed orientation tuning from added capacity or an extra pathway; a random-bank/capacity control is only warranted later if that narrower causal claim becomes the question.

## Second proposal, optional: one late SE-style residual channel gate

**Hypothesis:** learned image-dependent channel modulation can help group contour evidence without changing the high-resolution representations used for small motion. This is an alternative small experiment, not a request to combine it with the Gabor branch immediately.

Add one gate only to the96-channel13×13 output. Leave the50×50 and25×25 outputs unchanged. Let \(z=\operatorname{mean}_{h,w}H_3\), with a96→24→96 MLP:

\[
\widetilde H_3=H_3\odot\left[1+\tfrac12\tanh\{W_2\operatorname{ReLU}(W_1z+b_1)+b_2\}\right].
\]

Initialize \(W_2,b_2\) to zero and \(W_1\) normally. This starts at identity and permits gains between0.5 and1.5. It is a deliberately centered **SE-inspired adaptation**, not the literal MobileNet h-sigmoid suppression gate. Do not copy Mobile's SE weights across unrelated channel bases. The gate is an encoder operation shared across frames and tasks; task labels or a contour-only switch do not enter it.

**Analytical cost:**4,728 trainable parameters including biases, about4,608 MLP MAC/frame, plus roughly16,224 feature reductions and16,224 multiplications. These are operation counts, not latency measurements. Its global summaries modulate rather than replace the spatial map. Because GRN already performs response-dependent channel rescaling, the addition may be redundant; the saved Mobile result cannot resolve that. Nor can channel gating itself establish a contour-binding mechanism.

Use the same unchanged continued ConvNeXt control and evaluation rule. Keep or reject this addition based on actual task tradeoffs. Do not infer that an SE benefit explains all of Mobile's behavior.

## Minimal comparison and decision

Prefer the Gabor side branch first if paired-error evidence supports real contour complementarity. The SE option is the cheaper computational alternative if branch cost is impractical. At most two hybrid candidates, each separately compared with a shared continued baseline; no five-model ensemble, stage-swapping lottery, simultaneous normalizer changes or expanded sweep is proposed.

For a trained-parent development experiment, transfer the common ConvNeXt/decoder parameters and compatible optimizer state into explicitly identified sibling runs, initializing only new parameter state. Keep their initial functions matched. The interpretation is improvement of this trained parent, not superiority of architectures trained from scratch. Fix the finite allocation and equal-update target before running; measure actual hybrid throughput only as part of that allocation.

Before observing new results, specify a useful contour improvement and a practically acceptable motion-regression margin. Report paired differences, uncertainty, all seven task scores and added cost. If both hybrid and baseline improve similarly, prefer the simpler baseline. If contour improves but motion deteriorates materially, do not present the tradeoff as combining strengths. If nothing improves, preserve the result without claiming either component is intrinsically ineffective.

The earlier test results have now informed design. New generated evaluation pairs can measure generalization to fresh draws, but new crops from the same BSDS test photos do not create new independent source photographs. Use development evaluation for this exploration and label reused-source evidence accordingly. No new neuroscience, VWM or attentional claim follows from a useful PAV hybrid.
