# Causal temporal integration: three accumulator designs

2026-09-12. Status: user authorized implementation and training with "proceed with training". Researchers are implementing the specified comparison; run receipts and production metrics establish actual launch and completion. This experiment supersedes the suggestion to move immediately to delayed working-memory tasks.

Execution allocation: a new finite7200-second local wall-time cap, beginning with the first GPU profile and including profiling, training, evaluation and analysis. This is separate from the previous experiment's unused378.55393593191866seconds. Target4032 updates per arm; if measured cost cannot fit, reduce equally in1008-update increments before production and record the actual target. One GPU worker at a time, no automatic extension. The exposure is exploratory, not a claim of sufficient acquisition.

## Question and placement

Can a bounded, continuously updated visual state replace direct access to the two frame encodings while preserving the seven sensory skills? Compare three computational ideas: associative updating, gated spatial recurrence, and sustained/transient motion opponency.

The existing encoder already applies the same weights to each frame independently. Its decoder explicitly receives both spatial pyramids and constructs pair differences and correlations. Batching the encoder computations was an efficiency choice; the direct two-encoding decoder is the component being replaced.

Using temporal information is essential for direction; simultaneous access to a frame stack is optional. Causal filtering can implement a temporal receptive field using internal state. A sum or temporal mean alone is unsuitable because it is invariant to frame order. Direction requires order-sensitive state dynamics and spatial interactions.

Use the successful frozen `convnext_se_residual` encoder from:

`../runs/allocation_20260912_160414/contour_focus_seed20271/checkpoint_002268.pt`

Keep all three returned feature maps: 24x50x50, 48x25x25, 96x13x13. Attach a separate accumulator to EACH returned map, after its encoder computation. The deepest map already includes the trained SE gate. No accumulator feeds back into the encoder in this comparison. Thus placement is held constant while update mechanisms differ.

Let H_t^s be encoder scale s at time t. A learned affine 1x1 projection produces U_t^s with 32 channels. Do not globally pool first: the finest scale matters for 1-3 image-pixel motion. Do not normalize away every channel's amplitude: contrast and color are tasks too. Each candidate has its own trainable projection, with identical initial projection tensors across candidates.

At each step:

\[
H_t=E(I_t),\quad U_t^s=P_sH_t^s,\quad (O_t^s,S_t^s)=A_s(U_t^s,S_{t-1}^s).
\]

Each accumulator emits a 32-channel spatial field O. A common new readout receives only [U_t^s,O_t^s] at each scale: 3x3 convolution 64->32, pool to13x13, concatenate scales, 3x3 fusion96->64, spatial mean/max, MLP128->128, seven task heads. Use the existing decoder's activation/dropout conventions and the same initial new readout weights across arms. It never receives H_1 or U_1 through a separate cache. Current-frame appearance is available equally to all candidates; past information must arrive through their declared state.

The old ordered pair decoder remains an evaluated reference, not a hidden component of these new models. The new readout and accumulator require training; the encoder's existing optimizer state is not an optimizer state for these new parameters.

## 1. Spatial KDA: associative update

Use the Kimi Delta Attention recurrence as a temporal module at each spatial site, rather than treating a raster scan as time. This is a new spatial adaptation, not a recovered prior implementation or the complete Kimi Linear architecture. The user's earlier favorable KDA result motivates including it; its exact historical configuration and scores have not been verified here.

For each scale and site, use two heads, key/query dimension8 and value dimension16. Produce q,k,v, decay and write gates from a learned3x3 convolution of the CURRENT U map. This supplies local spatial context without pooling. Normalize q and k across their key dimension with epsilon1e-6. Constrain alpha and beta to(0,1), initially around .9 and .5 respectively. Values remain signed.

With S in R^(8x16) for each head:

\[
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1},\qquad
e_t=v_t-\bar S_t^\top k_t,
\]
\[
S_t=\bar S_t+\beta_t k_te_t^\top,\qquad o_t=S_t^\top q_t.
\]

Concatenate the two16-dimensional outputs and use an affine1x1 output projection32->32. Read AFTER writing. Initialize S_0=0. Weights share across sites; states do not. The3x3 projections let a location query motion-relevant neighboring features, while the recurrent association supplies history. No explicit two-frame correlation bank is added.

This decay-then-error-correction update follows KDA equation1, with an explicitly stated spatial parameterization. Finite state does not imply cheap state: two8x16 matrices per site means256 scalars/site. Across3294 sites this is843,264 floats, about3.22MiB per sequence in fp32 (about103MiB at batch32), excluding activations and gradients. Normalized keys and bounded gates make the old-state linear transition nonexpansive in spectral norm; this is not a guarantee of bounded values or training gradients.

Use a transparent fp32 recurrent implementation initially. The official language-model package requires a newer Torch stack than this project's established runtime; no environment upgrade or custom kernel is needed to express two recurrence steps.

Primary sources: [Kimi Linear, equation1](https://arxiv.org/html/2510.26692v1#S3), [official implementation repository](https://github.com/MoonshotAI/Kimi-Linear).

Expected strength: selective retention and replacement of feature associations. Main risk: spatial motion interactions may be harder to acquire than in an explicitly motion-sensitive model. Biological status: associative-learning analogy only; not an MT circuit model. Its fast state is an episode-local computation, separate from the project's provisional interpretation of trained long-term weights.

## 2. ConvGRU: learned spatial recurrence

Use one32-channel convolutional GRU per scale, with learned3x3 kernels and zero initial hidden state. All gates depend on the current map and previous hidden map:

\[
z_t=\sigma(W_z*U_t+V_z*h_{t-1}+b_z),\quad
r_t=\sigma(W_r*U_t+V_r*h_{t-1}+b_r),
\]
\[
\tilde h_t=\tanh(W_h*U_t+V_h*(r_t\odot h_{t-1})+b_h),
\qquad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
\]

Emit O_t=h_t. In this convention z is the WRITE fraction, initialized near.5. Use the same update at each frame; no task/phase-dependent gate. Recurrent kernels mix neighboring past states, allowing learned displacement-sensitive computations. Each scale has55,392 cell parameters (three32x64x3x3 kernels plus96 biases), or166,176 across scales, excluding common projections/readout. Persistent state is32 scalars/site:105,408 floats, about.40MiB per sequence; full training memory is larger.

This adapts convolutional recurrence over spatial CNN features from [Ballas et al., ICLR2016](https://arxiv.org/abs/1511.06432). It does not reproduce that paper's full architecture or published results.

Expected strength: flexible local temporal computations covering motion and nonmotion changes. Main risk: it must learn useful temporal filters and gates from the sensory objectives, and can simply overwrite its history. Biological status: broadly recurrent and spatially local, but sigmoid gates and this exact learning rule are engineering choices.

## 3. Opponent temporal filters: neuroscience-inspired

Keep fast and slow feature traces at each scale:

\[
F_t=aF_{t-1}+(1-a)U_t,\qquad L_t=bL_{t-1}+(1-b)U_t,
\quad a=.25,\ b=.75.
\]

At the first frame after reset, initialize F_1=L_1=U_1. This is a declared boundary condition, not a hidden phase input. The same causal rule applies thereafter. Preserve the sustained trace and signed transient D_t=F_t-L_t so the model can represent contrast/color/frequency changes as well as motion.

Apply matched even/odd spatial filters to the fast and slow traces. For each orientation/frequency channel, let their responses be f_e,f_o,l_e,l_o. Form:

\[
E_+=(f_e+l_o)^2+(f_o-l_e)^2,\qquad
E_-=(f_e-l_o)^2+(f_o+l_e)^2,
\]
\[
E_+-E_-=4(f_e l_o-f_o l_e).
\]

Use locally pooled energies with a shared divisive normalizer; opponent signals are differences of the resulting energy channels. Keep the unnormalized sustained field (F+L)/2 and transient F-L. Concatenate these64 channels with eight normalized energy channels, then learn a1x1 projection72->32 to emit O_t. The companion [neuroscience specification](neuroscience_design_notes.md) fixes filter-bank dimensions, normalizer and pooling details.

At startup and on repeated identical frames the opponent term is zero. At frame2, D_2=.5(U_2-U_1), and the opponent response reverses sign if the two frames are exchanged. Thus this is direction-sensitive temporal filtering, not a sum of frames. Direction labels follow image coordinates and filter phase convention; the learned task head maps these signals to the four labels.

Persistent state is F and L:64 scalars/site,210,816 floats or about.80MiB per sequence. Energy responses can be computed from that state; they do not require another stored image. This does not match KDA/GRU state size, and transient filter-bank activations must also be profiled.

The source principles are oriented spatiotemporal energy and opponency from [Adelson and Bergen1985](https://persci.mit.edu/pub_pdfs/spatio85.pdf), and pooling/rectification/normalization in [Simoncelli and Heeger1998's V1-to-MT model](https://www.cns.nyu.edu/pub/lcv/simoncelli96-reprint.pdf). The chosen leaky filters, learned CNN inputs and small output adapter are our engineering adaptation. These two temporal kernels are not exact temporal quadrature filters, so this is an energy/opponency-inspired model, not a faithful reconstruction of either paper.

Expected strength: builds temporal order and direction sensitivity into the computations, with interpretable timescales. Main risk: imposed filters may miss useful learned temporal features or sacrifice fine appearance information. A motion-only energy bottleneck would be inappropriate for our seven-task objective; the sustained/transient route is part of the design.

## Shared experiment specification

1. Retain the existing seven stimulus generators, image size100x100x3, labels and difficulty distributions. Present their existing two frames in order, ONE encoder call/update per frame. Reset state between independently generated pairs. No blank delay, repeated-frame warmup, interpolation or extra observed frame is silently introduced. Score after the second frame.
2. Freeze the successful encoder in eval mode. Train the new projections, accumulators and common readout with full two-step gradients, fp32, batch32, cross-entropy, Adam lr.001, weight decay.0001 and clip norm5. Model-dependent state is the intervention; count parameters, state bytes, measured runtime and peakVRAM separately.
3. Reuse the successful contour-focused12-update schedule for ALL arms: contour interleaved with motion, orientation, contrast, spatial frequency, chromatic increment and natural spectrum. Share task-local streamed training draws and fresh evaluation pairs across arms. The earlier sampling result is preserved rather than rediscovered. Freeze a source/config manifest at implementation.
4. Proposed first exposure is4032 updates/arm (129,024 fresh pairs;64,512 contour and10,752 per other task), with planned validation at1008/2016/3024/4032. This is a concrete exploratory proposal for learning new temporal modules AND readouts, not an established sufficient horizon. Compare trajectories and do not turn an early low score into an architecture verdict. Before any launch, measure costs and settle an explicit finite compute allowance; this design does not renew the old allowance or spend its residual automatically.
5. Select by minimum task balanced accuracy, then mean task AUC, exactly as in the successful allocation comparison. Use224 validation pairs/task at planned looks and448 shared fresh test pairs/task. Evaluate the preserved two-frame reference on those same new test pairs. Use new held-out seeds; keep BSDS source-photo splits and clustered uncertainty. Report all7 scores, confusion matrices, AUCs, difficulty and paired differences. The95% per-task point target remains an engineering screen, not a guaranteed population bound.
6. Primary temporal-replacement screen: every task point BA>=95%, together with paired changes versus the trained pair reference. Do not select by average accuracy alone or conceal losses in contrast/contours. A two-percentage-point retention margin is a descriptive engineering tolerance, not a statistically established noninferiority claim. State uncertainty and seed limitations. The first comparison uses one shared training seed; conclusions concern these fitted systems, not population architecture rankings.

## Small checks that answer the temporal question

Implementation checks cover only changed behavior: streaming versus the same two-step unroll; reset isolation; no future-frame access; finite state updates and nonzero training gradients. For the fixed neuro operator, check static-frame zero opponency and reversal of its raw opponent term under frame swap.

At final evaluation, reuse saved/generated evaluation pairs to measure:

- State reset immediately before frame2. This removes temporal information. For neuro, reset reinstates F=L=current. Report all scores without declaring a particular chance threshold mandatory: an interval-label task might be partly inferable from one frame's marginal statistics.
- Reverse frame order and transform labels using the actual task definition: cardinal directions invert; binary higher-interval/presence-interval decisions flip; signed orientation flips. Verify these mappings against the generator before implementing the diagnostic.

These are targeted diagnostics, not another training sweep. A direct state reset can be out of the trained state distribution, so performance loss supports history use but does not identify a neural mechanism.

## What this establishes, and the next boundary

This is sensory temporal integration before a separate working-memory phase. A successful two-step recurrence establishes causal replacement of the direct pair comparison. It does not establish persistent memory, longer-horizon evidence accumulation, speed tuning, motion coherence integration, or MT pattern-motion selectivity.

In particular, two fast/slow traces can retain invertible mixtures of two inputs. At two frames, a recurrent system may function as an online pair comparator. That is a legitimate first result, not evidence of sustained accumulation. Longer fresh motion sequences and pattern-motion stimuli would be a later, explicitly versioned question if we want stronger MT correspondence. They are not extra work included in this three-model design.

No candidate is preselected as the winner. KDA tests associative state, ConvGRU tests learned spatial recurrence, and the opponent model tests a concrete sensory temporal prior. Compare accuracy, acquisition trajectory and cost, then preserve whichever trained computation proves useful.
