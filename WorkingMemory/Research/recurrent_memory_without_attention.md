# Proposed recurrent working memory without attention

Date: 2026-09-12. Design only; no implementation or training launch. The user explicitly excludes attention until its contribution is separately motivated. Existing opponent/PAV computations are the sensory component, not an architectural commitment for the whole project.

## Recommendation and reason

Add one dense, gated recurrent memory block after the opponent's per-frame feature computations. Start with a conventional LSTM-style cell with separate input, retention and output gates and recurrent feature mixing. This is a proposed engineering baseline, not a claim that cortex implements LSTM equations. No attention weights, queries/keys, content-addressable memory, object slots, token routing, or feedback into sensory processing are added.

The current SequenceOpponent forwards CNN features into fixed fast/slow traces every frame. Its motion energy, temporal output projections and fused decoder field do not influence subsequent state; forward() computes them only at the report because intermediate emissions would be unused. Thus the architecture has no persistent state downstream of its interpreted motion/change features. It can learn a final function of sensory traces, but does not explicitly update an ongoing computation from each detected change. This is a structural observation, not an impossibility theorem about its scores.

The completed battery had strong sensory ranking but weak new-task performance, including minimal-delay conditions. That does not distinguish inadequate acquisition, cue processing, decision bias, temporal computation or retention limitations. A memory addition is a hypothesis to test, not a demonstrated necessity.

## Research informing the design

1. **Machine learning: xLSTM (NeurIPS 2024).** Its sLSTM branch develops recurrent memory mixing and stabilized exponential gating; its mLSTM branch uses matrix memory. The relevant lesson here is to let the previous recurrent representation influence new updates, rather than using only input-dependent linear traces. Start with bounded sigmoid gates, not an untested reproduction of the full xLSTM stack. Exponential gating is a later alternative if a concrete limitation warrants it. [Paper](https://arxiv.org/abs/2405.04517), [official implementation](https://github.com/NX-AI/xlstm).

2. **Machine learning: Gated DeltaNet (ICLR 2025), Gated DeltaNet-2 (May 2026 preprint).** These distinguish retention from targeted updates; the newer work separates erase and write controls. Their language/retrieval results motivate considering independent update operations, not assuming transfer to our visual tasks. Because the user excludes attention, do not implement their associative matrix state or key/query addressing. The proposed LSTM already has independent retain/write gates; it is not a DeltaNet implementation. [2025 paper](https://arxiv.org/abs/2412.06464), [2026 preprint](https://arxiv.org/abs/2605.22791).

3. **Neuroscience: Soni and Frank (2025).** Their PFC/basal-ganglia computational model learns gating policies for working-memory storage and adaptive resource use. This motivates learned protection/update control at a functional level. It does not establish literal LSTM gates, fixed object slots, or the superiority of our proposed implementation. We are not adopting its RL training or chunking mechanism. [Version of record](https://doi.org/10.7554/eLife.97894).

4. **Neuroscience: Panichello and Buschman (2021).** Monkey recordings distinguish maintaining information from selecting and transforming it for use. This is a reason to keep selection as a separately tested future component; the current design must not quietly introduce attention to solve the entire task battery. [Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8223505/).

5. **Neuroscience: Bellafard et al. (2024), Volatile working memory representations crystallize with practice.** Longitudinal recordings and perturbations in a mouse olfactory task examine the development of memory representations. Learning history matters when interpreting a weak newly trained task. This does not supply a sufficient episode count or directly validate visual memory in our model. [Primary study](https://www.nature.com/articles/s41586-024-07425-w).

6. **Neuroscience: Inagaki et al. (2019).** Recordings and perturbations in mouse frontal cortex support discrete attractor dynamics underlying persistent activity in a delayed-response task. This supports investigating maintained recurrent state; it does not establish a general visual item store or literal LSTM dynamics. [Primary study](https://www.nature.com/articles/s41586-019-0919-7).

Independent read-only neuroscience consultation: researcher `/root/temporal_neuroscience_design` inspected current model/report and supported a single 256-cell LSTM without attention, noting the need for per-frame opponent emissions, spatial cue preservation and retention initialization. The recommendation is a design choice; no empirical component benefit is asserted.

## Concrete forward computation

Batch dimension B; one RGB image x_t has shape [B,3,100,100]. Preserve the trained CNN, projections and opponent fast/slow updates. All learned weights may adapt; fixed opponent kernels/coefficients remain as before. Explicit episode resets clear all recurrent states. Blank, cue and probe images pass through the same computations as other frames.

1. At every frame compute projected current features U_t and opponent outputs O_t at all three scales, not just at the final report.
2. Reuse the current readout's local convolutions, adaptive pooling to 13x13 and fusion to obtain F_t with shape [B,64,13,13]. This fused field contains current and temporal features.
3. Form a spatially ordered memory input: a learned 1x1 convolution reduces 64 channels to 8; flatten [B,8,13,13] to [B,1352]; apply a learned Linear(1352,128), LayerNorm and SiLU to obtain z_t [B,128]. No attention or image-dependent spatial weighting is used. Retaining a fixed spatial ordering avoids making global pooling the only route for corner cues and item identity, but does not guarantee those features are learned or survive compression.
4. Maintain two recurrent vectors c_t and h_t, each [B,256]. For a in {i,f,o}, let a_t = sigmoid(W_a z_t + R_a h_(t-1) + b_a). Let g_t = tanh(W_g z_t + R_g h_(t-1) + b_g). Every W has shape [256,128], every R [256,256], and every gate/proposal [B,256]. Update:

   c_t = f_t * c_(t-1) + i_t * g_t

   h_t = o_t * tanh(c_t)

   Products are elementwise. The dense R matrices mix the previous representation into future updates. The gates act on one distributed memory vector; they do not choose spatial locations, tokens or slots.

5. Keep the existing 128-dimensional sensory feature s_T and task heads. Add a learned residual projection P h_T, P of shape [128,256], so the report is Head_family(s_T + P h_T). Initialize P small but nonzero, with zero bias, so the new branch begins near the parent output while receiving gradients from the first update. Exact parent-logit equality is not claimed for this expanded architecture. Do not use previous predictions as another persistent state. Family chooses only the same task head as before; the core receives images, not latent protocol/rule/phase labels.

The added persistent state is 512 fp32 values per example, 2 KiB, in addition to existing opponent traces. A straightforward implementation adds approximately 0.60 million learned parameters; this is an analytical estimate, to be checked once implemented. Activations and per-frame opponent emission/fusion add training cost; small persistent state alone does not establish cheap training.

## What the update permits

- Hold: f near 1 and i near 0 approximately preserve the cell vector while new frames arrive.
- Replace: f near 0 and i near 1 replace selected feature coordinates.
- Accumulate: f near 1 permits repeated additions i*g, including signed contributions. Success on long counts is not guaranteed; bounded proposals and tanh output can still make numerical representation/optimization difficult.
- Context-dependent processing: R h_(t-1) allows a previously represented rule or item to influence later updates.

The retain and input gates are independent. A gate that forces i=1-f would restrict a unit to interpolation, making retention and addition inseparable. No whole-cell normalization is proposed because it could erase magnitude information useful for accumulation. Gate logits/candidates can use normalized input z without normalizing the stored c.

Use a retention-biased initialization, with candidate nominal time constants spanning approximately 4-128 observed frames: f_j=exp(-1/tau_j), b_f,j=logit(f_j) before input/recurrent contributions; input gate bias -2, output gate bias 0. These are engineering initial conditions, not biological time constants or hard horizons; the gates learn. Full-sequence BPTT, no cell-state dropout, fp32.

The project convention activated LTM = learned synaptic weights remains intact: the episode-specific c/h are activations, not changes to trained weights. This is a functional WM candidate and makes no claim about consciousness or a literal PFC microcircuit.

## Small next comparison, when execution is authorized

Begin with motion-duration integration and one-item orientation match/change, with the shortest and a modest longer sequence/delay in each. These cover accumulation of detected events and retaining an item for comparison. Include familiar sensory anchors. Keep actual stimulus rules unchanged; reducing the task mix is an explicit new acquisition experiment, not continuation of the original broad joint schedule. A cue still enters as pixels; no supplied cue embeddings or correct memory vectors.

Updated comparison after the user's request for a neuroscience-inspired competitor: use LSTM and the excitatory/inhibitory adaptive rate network described below as the two primary trained arms, initialized from the same selected sequence-model parent with the same per-frame sensory interface and task streams. Preserve trained progress; these are versioned warm starts with new optimizers, not strict optimizer continuation. Resetting the added state at evaluation is a cheap intervention on each trained model; it is not equivalent to a separately trained parameter-matched memory-free control. Such a trained control remains a possible follow-up if a benefit needs attribution, not a silently added third arm.

Fit the minimal and modest-delay conditions together under one finite, profiled allocation; inspect their acquisition during the planned run without stopping on early chance scores. Improvement would support the usefulness of an expanded model under these conditions; the shared new front-end/readout or focused training could contribute. If neither improves, do not automatically add attention; inspect cue/rule acquisition and optimization. Capacity curves become interpretable when minimal-condition behavior is competent.

Only after this focused comparison should we return to the full seven-family battery. No expanded sweep, architecture lock-in, attention module, compute-cap renewal or training launch is part of this research note.

## Added competitor: adaptive excitatory/inhibitory rate network

The subsequent user requested a more neuroscience-inspired competitor and specifically emphasized numerical stability and gradient flow for both arms. Proposed state: r and a, each [B,256], for nonnegative population rates and adaptation. A provisional 205 excitatory / 51 inhibitory split approximates 4:1; this is an engineering population choice, not a measured ratio for a particular region. With the same z_t [B,128]:

    r_t = (1-alpha)*r_prev + alpha*ReLU(W_in*z_t + W_rec*r_prev - g*a_prev + b)
    a_t = (1-beta)*a_prev + beta*r_prev

All products outside matrix multiplications are elementwise. Use synchronous old-state updates, g>=0, 0<alpha,beta<1. Parameterize W_rec = softplus(A) diag(d), with d_j=+1/-1 for excitatory/inhibitory presynaptic units; signs are fixed throughout optimization, not only at initialization. This constrains the recurrent core; unconstrained sensory/readout projections and BPTT remain engineering abstractions. Both new models have 512 state scalars, but parameter counts and compute differ. Keep attention, addressing and sensory feedback absent.

The variant is inspired by recurrent E/I mechanisms and adaptation-based dynamics, not a reproduction of every component in [Liu et al., 2025](https://www.nature.com/articles/s42003-024-07282-3). It does not prescribe a sustained versus transient solution. Do not report successful circuit memory without checking what the trained state actually retains.

## Stability and gradient-flow design

**Normalization placement.** Share per-frame LayerNorm on the projected sensory input z, with statistics within each example, never across time. For LSTM gate/candidate preactivations, use independent LayerNorm modules, with retention/input biases added AFTER normalization so their intended initialization is not centered away. Keep the stored c unnormalized. For the E/I arm, normalize the feedforward input/interface, not r, a, or the recurrent E/I current. Centering a nonnegative rate state destroys its rate interpretation; normalizing recurrent current introduces population-dependent coupling and changes the dynamics being tested. LayerNorm is a numerical design choice, not biological evidence. [Layer Normalization](https://arxiv.org/abs/1607.06450).

**Initialization.** Use separate orthogonal recurrent matrices per LSTM gate/proposal, scaled input initialization and the retention bias range above. For the E/I network, initialize nonnegative magnitudes with fan-in scaling and account for the much smaller inhibitory population when balancing total E/I drive. Do not set softplus raw weights near zero (that would produce dense magnitudes near 0.69); initialize desired small positive magnitudes then inverse-softplus them. Start adaptation weak enough to leave usable stimulus responses, and use finite heterogeneous leak/adaptation timescales. alpha=1-exp(-dt/tau_r), beta=1-exp(-dt/tau_a) keep isolated leak updates in (0,1); this does not prove stability of the full recurrent loop. Time is in observed-frame units until physical timing is specified. A one-frame network update remains fixed cost. Do not apply an orthogonal initialization to a signed matrix and then take absolute values as if orthogonality survived.

**Gradients and optimizer.** Use fp32 and full-sequence BPTT, with activation recomputation if needed; never detach memory between frames. Use a small nonzero residual output projection for both new branches. Give transferred sensory weights a lower learning rate than new memory/interface weights. Initial implementation candidate: new weights 3e-4, transferred weights 3e-5, Adam, global gradient-norm clipping at 1.0. These are starting choices, not guarantees. Log unclipped norms and clipping frequency; excessive clipping can conceal an unsuitable learning rate and does not fix forward-state explosion. Exclude normalization parameters, gate biases, timescale/adaptation parameters and raw sign-constrained recurrence parameters from generic weight decay. Avoid default penalties that drive the biological memory toward a silent solution or the LSTM toward rapid forgetting.

**Focused measurements within training.** Record compact summaries at normal logging intervals: loss, recurrent-state RMS/max, pre-clip gradient norm and clipping frequency; LSTM gate saturation; E/I inactive-unit fraction and mean E/I activity; gradient contribution to early versus late sequence representations on a small planned diagnostic batch. Temporal-gradient attenuation is evidence about optimization, not by itself proof of information loss. Check the full recurrent/adaptation behavior on real streamed sequences, not only a weight-matrix spectral radius. A short finite forward/backward check is an implementation prerequisite, not an acquisition test or a new validation campaign.

**Fair opportunity.** Match fresh task streams, curriculum, exposure and evaluation; allow justified model-specific initialization/learning rates rather than mechanically forcing identical settings. If numerical tuning is needed, give both arms a comparably bounded opportunity and select settings from training/validation only. Record costs and preserve attempts; no automatic sweep or budget expansion. Healthy early chance-level training proceeds through its agreed finite allocation. Stop for actual nonfinite/unsafe computation, preserving the affected checkpoint/evidence. Do not equate a hard-to-optimize constrained RNN with a disproven neuroscience mechanism. Relevant ML work explicitly identifies this training difficulty: [Soo, Goudar and Wang, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html).
