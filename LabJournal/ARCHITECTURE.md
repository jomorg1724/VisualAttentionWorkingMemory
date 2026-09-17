# The current model: causal sensory traces, spatial E/I memory and joint attention

[Index](README.md) · [Training status](CURRENT_STATUS.md) · [Task definitions](TASKS_AND_METRICS.md)

This describes the selected **attention 8400** architecture and its unchanged Stage1 continuations. It is not the original vector E/I model, the frozen-encoder temporal screen, or the additive-controller arm. Current implementation is [PreUpdateAttention/model.py](../WorkingMemory/PreUpdateAttention/model.py), inheriting [SpatialComparison/model.py](../WorkingMemory/SpatialComparison/model.py) and [RecurrentComparison/model.py](../WorkingMemory/RecurrentComparison/model.py).

## 1. Shared convolutional encoding of the current frame

At each time t the input is (I_t\in\mathbb R^{B\times 3\times 100\times 100}). The learned contour-focused ConvNeXt-GRN/late-SE lineage produces three scales, with learned 1 × 1 projections to 32 channels:

| Scale | Encoder tensor | Projected current tensor $U_t^{(i)}$ |
|---|---|---|
|1|B×24×50×50|B×32×50×50|
|2|B×48×25×25|B×32×25×25|
|3|B×96×13×13|B×32×13×13|

The same encoder is used every frame. This sensory path has no feedback from the spatial memory. All learned sensory parameters are currently trainable at the inherited lower learning rate; “encoder frozen” applied to the original temporal comparison, not the present experiment.

## 2. Fixed fast and slow traces at each scale

For each projected field,

$$F_t=.25F_{t-1}+.75U_t,\qquad S_t=.75S_{t-1}+.25U_t.$$

Both traces initialize to the first field, $F_0=S_0=U_0$, and each has the same shape as U. Retention coefficients remain fixed; gradients still propagate through these differentiable equations to learned upstream features. The response to a blank is not necessarily zero: the CNN can produce a nonzero blank representation.

## 3. Fixed quadrature energies preserve ordered temporal interactions

Eight zero-mean unit-norm 7 × 7 Gabor kernels represent horizontal/vertical coordinates, two frequencies(.125,.25 feature-site cycles), and even/odd quadrature phases. The fixed bank filters every feature channel of both traces with reflection padding. For one axis/frequency pair let $f_e,f_o,s_e,s_o$ be the even/odd filtered responses. The directional opponent energies are

$$E_+=(f_e+s_o)^2+(f_o-s_e)^2,\qquad E_-=(f_e-s_o)^2+(f_o+s_e)^2.$$

Each energy averages across the 32 feature channels. Four axis/frequency pairs × two signs yield eight channels. A 3 × 3 average pool smooths each channel; normalization is

$$\widehat E_k(x,y)=\frac{\operatorname{Pool}_{3\times3}E_k(x,y)}{10^{-4}+\frac18\sum_{j=1}^{8}\operatorname{Pool}_{3\times3}E_j(x,y)}.$$

The learned temporal output at each scale is a 1 × 1 convolution of

$$X_t=[(F_t+S_t)/2,\;F_t-S_t,\;\widehat E_t]\in\mathbb R^{B\times72\times h_i\times w_i},\quad O_t=\operatorname{Conv}_{72\to32}(X_t).$$

The Gabor bank, squares, smoothing and normalization are fixed computations; which learned features they filter and how their results are mixed learn. Fixed equations are an inductive bias, not a fixed entire representation. See [accumulator source](../PreAttentiveVision/TemporalIntegration/accumulators.py).

## 4. Fuse scale-specific sensory history into a 64-channel field

Concatenate each current U and emitted O to 64 channels, apply its learned local convolutional block to 32 channels, and adaptive-average-pool to 13 × 13. Concatenate the three outputs to 96 channels and apply the learned fusion block:

$$H_t\in\mathbb R^{B\times64\times13\times13}.$$

H already combines current visual features and short temporal history. Calling it “current image only” would be inaccurate.

## 5. Joint sensory/memory attention supplies the memory input

Flatten spatial sites only for the attention matrix: sensory V and old rate memory M are each B× 169 × 64. The model retains the spatial field as its recurrent state; tokenization is not the old 64→8 channel compression into one 128 vector.

With learned site encoding P and source encodings $e_v,e_m$, the source code implements

$$Q=W_Q(\operatorname{LN}_q(M)+P+e_m),$$
$$K=W_K(\operatorname{LN}_k([V;M])+[P+e_v;P+e_m]),\quad \mathcal V=W_V[V;M].$$

Split 64 channels into two 32 dimensional heads:

$$A_h=\operatorname{softmax}_{338}\left(\frac{Q_hK_h^\top}{\sqrt{32}}+b_{h,\mathrm{source}}-\operatorname{softplus}(\lambda_h)d_{ij}^{2}\right).$$

Q has shape B× 2 × 169 × 32, K and values B× 2 × 338 × 32, and A has B× 2 × 169 × 338. The learned distance bias is an initial/local preference, not a hard spatial mask. Both heads may read both sources. Learned source bias initially favors vision; locality initializes to coefficient 4 in grid-site squared distance. The two heads' outputs concatenate and receive a learned 64→64 projection, producing attended field (J_t\in\mathbb R^{B\times 64\times 13\times 13}).

The input to recurrent memory is $Z_t=\operatorname{LN}_{channels}(J_t)$ at each spatial site. Attention replaces the raw sensory drive here. It does not overwrite the separate sensory/comparator routes below. There is no explicit instruction that blanks must attend to memory, no phase input, and no growing key/value cache.

## 6. Spatial excitatory/inhibitory dynamics with adaptation

Persistent rate R and adaptation A each have B× 64 × 13 × 13 shape. There are 51 excitatory and 13 inhibitory source channels. A learned 3 × 3 kernel has sign constrained by its presynaptic channel:

$$K_{o,i,\Delta}=\operatorname{softplus}(\theta_{o,i,\Delta})s_i,\qquad s_i\in\{+1,-1\}.$$

For learned per-channel time constants and adaptation strength,

$$\tau_r=1+31\sigma(\theta_r),\quad \tau_a=4+124\sigma(\theta_a),\quad g=.5\sigma(\theta_g),$$
$$\alpha=1-e^{-1/\tau_r},\quad\beta=1-e^{-1/\tau_a},$$
$$D_t=W_{in}*Z_t+K*R_{t-1}-g\odot A_{t-1}+b,$$
$$R_t=(1-\alpha)\odot R_{t-1}+\alpha\odot\operatorname{ReLU}(D_t),$$
$$A_t=(1-\beta)\odot A_{t-1}+\beta\odot R_{t-1}.$$

Updates use old states synchronously. Both start at zero. Rates and recurrent currents are not layer-normalized; normalizing nonnegative rates would change their circuit interpretation. Sign constraints, finite time-constant ranges and leak do not mathematically guarantee stability of the full recurrent loop.

## 7. Compare old memory with current sensory history and produce the decision

Before the memory update, a learned comparator receives `[old R, H]`, B× 128 × 13 × 13. It applies 1 × 1 Conv 128→64, SiLU,3 × 3 Conv 64→64, SiLU, then concatenated spatial mean/max pooling to B× 128. This gives a direct learned route for comparing remembered content with a new probe without first overwriting the rate state.

At the final frame three 128 vectors are added:

$$y_T=f_{sens}([\operatorname{mean}H_T,\operatorname{max}H_T])+W_{mem}[\operatorname{mean}R_T,\operatorname{max}R_T]+W_{cmp}c_T,$$
$$\ell_T=W_{task}y_T+b_{task}.$$

The motion head has four logits; native orientation and binding have two each. Diagnostic branch-logit decomposition adds final head bias once. The comparator uses old memory and original H, not the attended field, so a probe-only memory-input lesion leaves this comparator route intact by construction.

## State, parameter count and learning

The selected attention model has 556,128 learned parameters; attention adds 27,590 over spatial 528,538. Stage1 introduces zero architecture parameters. Spatial R/A contribute 21,632 persistent scalars per example =84.5 KiB in fp 32. Opponent traces add 64 (50²+25²+13²)=210,816 scalars ≈823.5 KiB. Together these explicit persistent histories are about 908 KiB/example, excluding temporary activations, gradients, optimizer state and runtime memory. Attention probabilities are temporary per-frame computations.

Persistent state size does not grow with frame number at inference. Total sequence computation grows approximately linearly with frames because each must be processed. Training with full BPTT retains/recomputes information for gradients across time; activation checkpointing trades compute for memory. This differs from an inference cache that stores every old frame.

The old vector model instead used 64→8 channels, flatten 1352→128, then 256-unit dense memory with 512 state scalars (2 KiB). Spatial state is 42.25 times that memory state. These 2 KiB and 141,968 trainable-parameter figures belong to distinct earlier configurations, not the current system.

Stage1 uses fp 32, full BPTT, Adam epsilon 1e-10, clipping norm 1 and inherited LR 3e-4 for memory/attention/comparator/binding head versus 3e-5 for sensory and ordinary task heads. The code preserves compatible optimizer history. Fixed sensory filter/trace computations remain fixed, while learned inputs and outputs can adapt.

Neuroscience-inspired parts include local oriented energy computations, signed recurrent excitation/inhibition and adaptive rate dynamics. Dot-product attention, LayerNorm, exact channel counts, Adam and BPTT are engineering choices; performance is not validation of a literal cortical circuit.
