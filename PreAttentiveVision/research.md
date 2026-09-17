# PAV: five spatial encoders for learned visual comparison

This is the first component of the visual attention and working-memory project. It implements per-frame representations, leaving VWM, attention and integration for later work. In Guided Search 6.0, early visual encoding and the restricted features that actually guide attention are distinct: encoding orientation does not establish a functioning priority map. These encoders are candidates for the former, not complete implementations of preattentive guidance. We omit the diffuser and approximate activated long-term memory using learned synaptic weights as requested; that approximation is ours, not a claim by Wolfe. [Wolfe, 2021, Guided Search 6.0](https://search.bwh.harvard.edu/new/pubs/Wolfe2021_GS6.pdf).

We selected five complementary computational biases rather than five width settings of one CNN. They are compact engineering adaptations trained from scratch, **not published architectures reproduced at their original scale**, and not established biological models. Published results motivate experiments; they do not predict the ranking here.

## Common interface and comparison

`models.py` exports `MODEL_NAMES`, `build_encoder(name)`, `MODEL_DESCRIPTIONS`, `model_configs()` and `parameter_count(model)`. Every model exposes `.out_channels`, `.config`, `.description` and `.parameter_count`.

For a frame \(x\in\mathbb R^{B\times3\times100\times100}\), outputs are

\[
E(x)=[F_1,F_2,F_3],\qquad
F_1\in\mathbb R^{B\times24\times50\times50},\quad
F_2\in\mathbb R^{B\times48\times25\times25},\quad
F_3\in\mathbb R^{B\times96\times13\times13}.
\]

Widths are 24/48/96 and block depths are 1/2/2. There is no global spatial collapse or classification head. Internal SE channel summaries and GRN response norms modulate intact feature maps. No encoder has temporal state, a task phase input, a previous-frame buffer or added process noise. The same encoder instance processes both frames. Pairwise comparisons and spatial reduction belong to the common decoder.

The runtime applies the same fixed RGB normalization \((x-0.5)/0.5\) to every candidate. Learned layers use fresh PyTorch initialization, except GRN's zero gain/bias and Inception residual scale 0.1 described below. No pretrained weights or source-code downloads occur during construction. GroupNorm replaces running-statistic BatchNorm in compact residual, Inception and mobile blocks; ConvNeXt uses channel-wise LayerNorm. This small-batch engineering choice is not a neuroscientific claim.

| Stable name | Trainable parameters | Distinct computation | Main risk to test |
|---|---:|---|---|
| `aa_resnet` | 479,160 | Dense residual convolutions and filtered decimation | Smoothing could reduce sensitivity to small positional changes |
| `convnext_grn` | 262,032 | Wide depthwise spatial mixing, channel expansion, GRN | Global response statistics might obscure useful absolute contrast |
| `inceptionnext` | 245,574 | Identity plus square and directional spatial branches | Axis-oriented mixing may introduce directional preferences |
| `mobilenet_se` | 329,730 | Expanded depthwise blocks and learned channel gating | Global channel gates may suppress weak local evidence |
| `vone_resnet` | 489,528 | Fixed oriented simple/complex features plus learned hierarchy | Fixed frequency/orientation grid may underserve some images |

Parameter counts exclude fixed buffers. The VOne-inspired front-end additionally stores 11,664 fixed Gabor coefficients. Equal output widths make the decoder topology identical; encoder parameter counts and computation are not matched. Report actual throughput, exposure and held-out family scores rather than calling this an isolated component ablation. Fixed-time comparisons answer resource efficiency; equal-update comparisons answer a different question.

## 1. Anti-aliased residual hierarchy

A residual block computes \(y=\operatorname{SiLU}(x+f(x))\), with two learned 3x3 convolutions. Identity shortcuts support optimization without forcing each block to relearn the identity. This adopts the residual principle rather than an original ResNet stage schedule. [He et al., 2016](https://arxiv.org/abs/1512.03385).

Before each factor-two reduction, our stride-one convolution, normalization and GELU are followed by

\[
y=\downarrow_2(h*z),\qquad h=\frac1{16}[1,2,1]^\top[1,2,1].
\]

The fixed filter attenuates high spatial frequencies before subsampling, following the anti-aliasing motivation in [Zhang, 2019](https://proceedings.mlr.press/v97/zhang19a.html). Reflect padding yields sizes 100→50→25→13. This does not guarantee exact shift invariance. For change detection, total translation invariance would be undesirable: a position change may be the signal. Returning all three spatial maps preserves local evidence for the decoder; whether smoothing helps noisy inputs more than it hurts positional sensitivity is empirical.

## 2. ConvNeXt with global response normalization

Each residual branch uses depthwise 7x7 convolution, per-location channel LayerNorm, 4x channel expansion, GELU, GRN and a pointwise projection. For expanded activations \(X\), GRN is

\[
g_{bc}=\sqrt{\sum_{h,w}X_{bchw}^2},\quad
n_{bc}=\frac{g_{bc}}{C^{-1}\sum_jg_{bj}+\epsilon},\quad
Y_{bchw}=X_{bchw}+\gamma_cX_{bchw}n_{bc}+\beta_c.
\]

Zero-initialized \(\gamma,\beta\) make GRN initially an identity transformation. Relative channel response is the adopted bias; this is **not** biophysically validated divisive normalization. We do not use masked-autoencoder pretraining, stochastic depth, the original stride-four stem or original model sizes. Standard 3x3 stride-two reductions maintain the shared pyramid. [Woo et al., 2023](https://arxiv.org/abs/2301.00808); formulas checked against the authors' [GRN implementation](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py).

## 3. InceptionNeXt directional mixing

Channels are split into an identity branch and three groups of \(\lfloor C/8\rfloor\) channels. The learned depthwise operators are 3x3, 1x11 and 11x1:

\[
M(X)=\operatorname{concat}(X_0,K_{3\times3}*X_1,
K_{1\times11}*X_2,K_{11\times1}*X_3).
\]

A normalized pointwise expansion/projection MLP mixes these channels, with a residual connection. This supplies local and long directional receptive fields without applying a dense large square kernel to every channel. Our three shallow stages, GroupNorm and residual scale initialized to 0.1 differ from the published configuration; the larger scale avoids making this short scratch-training experiment nearly an identity network initially. No speed advantage on this GPU is assumed. [Yu et al., InceptionNeXt](https://arxiv.org/abs/2303.16900); [official implementation](https://github.com/sail-sg/inceptionnext/blob/main/models/inceptionnext.py).

## 4. Mobile inverted residuals with SE

Each block expands channels threefold, applies depthwise 5x5 convolution, then projects linearly back to the residual width. Hidden nonlinearities use

\[
\operatorname{h\!swish}(x)=x\,\frac{\operatorname{ReLU6}(x+3)}6.
\]

Squeeze-excitation forms a channel gate from spatial means:

\[
z_c=\frac1{HW}\sum_{h,w}X_{chw},\qquad
a=\operatorname{h\!sigmoid}(W_2\operatorname{ReLU}(W_1z)),\quad
Y_{chw}=a_cX_{chw}.
\]

SE here is feedforward, image-dependent channel modulation; it is not the future attention system or a GS6 priority map. Expansion ratio, shallow stage schedule and GroupNorm are our adaptations. There is no architecture search or claim of reproducing MobileNetV3's hardware-optimized configuration. [Howard et al., 2019](https://arxiv.org/abs/1905.02244); implementation structure checked against [Torchvision MobileNetV3](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py).

## 5. Deterministic VOne-inspired front-end

VOneNet motivates combining fixed Gabor filtering with simple/complex nonlinearities before a learned hierarchy. Its published front-end also uses biologically constrained parameter distributions and neuronal stochasticity. We adopt only a deterministic oriented-feature bias, not those fitted distributions or noise. [Dapello et al., 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html).

Our exact bank uses 9x9 kernels, orientations \(0,\pi/4,\pi/2,3\pi/4\), frequency/sigma pairs \((0.12,2),(0.25,1.5)\) in cycles/pixel and pixels, and three unit-length color axes proportional to \((1,1,1),(1,-1,0),(-0.5,-0.5,1)\). With rotated coordinates \(u,v\),

\[
k_\phi(u,v)=e^{-(u^2+0.5v^2)/(2\sigma^2)}\cos(2\pi f u+\phi).
\]

Each kernel is mean-subtracted and L2-normalized. For quadrature responses \(a=k_0*x,b=k_{\pi/2}*x\), the outputs are \(\operatorname{ReLU}(a)\) and \(\sqrt{a^2+b^2+10^{-6}}-10^{-3}\). The latter reduces phase dependence while remaining numerically differentiable. There are 24 simple and 24 complex maps; concatenating raw RGB produces 51 channels before learned reduction. Fixed opponent axes and the RGB route are deliberate color-preserving additions. Compare the original [VOneBlock code](https://github.com/dicarlolab/vonenet/blob/master/vonenet/modules.py).

All later residual layers learn from data. No process noise, cortical dynamics, foveation or biological cell-distribution match is claimed. The fixed bank supplies useful orientation structure but may favor Gabor stimuli; diverse stimulus families matter. A good change score alone would not establish V1 correspondence.

## What this screen can teach

The common decoder tests whether each representation supports learned comparison of separate frames. Spatially retained low/high-level features support testing local texture, color, position and more contextual changes. Two-frame dot comparisons are not evidence for a motion-energy system, velocity coding or temporal memory: each encoder processes one static image.

A positive candidate supplies an engineering PAV starting point. Family-specific strengths can guide the next small development step; no candidate is selected merely for being newer or more biologically named. A negative bounded run is evidence about that trained setup and exposure. Neither result implements the later VWM or attention modules.

## Focused implementation verification

`C:/Python310/python.exe -B -m unittest PreAttentiveVision.test_models -v` passed on CPU for all five encoders (1.101 seconds of tests, 2026-09-12). It checks the shared three-level API, finite random/blank outputs, gradients through both frame branches and learned weights, deterministic repeated forward passes, fixed Gabor buffers and parameter counts below one million. This is an integration check, not training, GPU timing or evidence of visual competence. The encoder agent performed no GPU work.
