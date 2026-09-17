# Causal opponent-energy accumulator: neuroscience design notes

Status: design only, 2026-09-12. No implementation, training, profiling or new compute authorization is claimed. This is one candidate for the requested three-model comparison, not an additional model sweep.

## Scientific position

Processing successive frames through temporal state is a sensible next step between spatial PAV and a working-memory experiment. A biological visual system does not need simultaneous access to two externally retained images: temporally extended responses can combine present input with recent activity. A feedforward spatiotemporal filter and a recurrent implementation of temporal filtering can share computations; merely choosing a recurrent implementation does not establish a new biological mechanism.

Adelson and Bergen describe spatial-temporal filtering, quadrature energy and directional opponency. We adopt those computational principles, while replacing their temporal filters with two simple causal traces. This is an engineering adaptation, not their published model. [Primary paper, 1985](https://persci.mit.edu/pub_pdfs/spatio85.pdf); [publisher DOI](https://doi.org/10.1364/JOSAA.2.000284).

Simoncelli and Heeger model V1-to-MT processing through weighted pooling, rectification and divisive normalization, including velocity selectivity. We adopt local pooling and response normalization as design inspiration. We do **not** implement their fitted velocity-plane pooling or claim MT pattern-motion selectivity. [Primary paper, 1998](https://www.cns.nyu.edu/pub/lcv/simoncelli96-reprint.pdf); [DOI](https://doi.org/10.1016/S0042-6989(97)00183-1).

## Location and tensor contract

Use the successful frozen spatial encoder. Its actual outputs are 24x50x50, 48x25x25 and 96x13x13 channels/spatial dimensions. Put the accumulator after each returned spatial map, avoiding a change to the trained encoder's internal distribution. A learned 1x1 projection produces u_t at d=32 channels per scale. Do not apply per-frame normalization that discards global amplitude or color in this projection: contrast and chromatic tasks need those cues. The comparison can change d consistently across candidates if a state-size constraint is chosen before training.

The accumulator receives only the current frame's projected map and its own state. All equations apply independently to every scale. The task identity selects only the final task head, not the state update. There is no frame-pair argument, raw first-frame cache, future input or privileged task metadata.

## Two causal time scales

After an episode reset the state is uninitialized. On the first actual frame, set:

\[
f_1=s_1=u_1.
\]

This steady-state initialization treats the first observed image as the initial condition. It avoids a fictitious motion transient caused by a zero-to-image onset. It is a declared modeling boundary condition, not a claim about biological trial onset. The reset/initialization bookkeeping is not supplied to the learned readout.

For every subsequent actual frame:

\[
f_t=a f_{t-1}+(1-a)u_t,\qquad
s_t=b s_{t-1}+(1-b)u_t,
\quad a=0.25,\quad b=0.75.
\]

Fix these two coefficients in the first comparison. They deliberately impose distinguishable temporal scales; they are not fitted neuronal time constants. In physical time a=e^{-Delta t/tau_f} and b=e^{-Delta t/tau_s}; without a declared frame interval, report frame units rather than milliseconds. Do not insert blank frames or repeat frames in this initial two-frame task.

Retain both sustained appearance and a signed transient:

\[
p_t=(f_t+s_t)/2,\qquad d_t=f_t-s_t.
\]

Exactly at the second frame:

\[
p_2=(u_1+u_2)/2,\qquad d_2=(u_2-u_1)/2.
\]

Thus the appearance route retains color/contrast/spatial organization while the transient route retains temporal sign. An orderless average by itself would not suffice for cardinal direction or signed changes. A single leaky trace with unequal weights is order-sensitive, but it does not by itself compute direction and its compressed endpoint need not distinguish arbitrary frame pairs.

## Causal quadrature energy and opponency

Apply fixed even/odd spatial Gabor pairs G_e and G_o to each channel of both traces. Use horizontal and vertical spatial frequency axes, two frequencies per axis (initial design: 0.125 and 0.25 cycles per feature-map site), 7x7 support and Gaussian sigma=1.5 sites. Remove each kernel's spatial mean, normalize its L2 norm, and use the same kernels for f and s. These finite filters approximate spatial quadrature; frequencies/support are engineering choices in feature coordinates, not a V1 population fit.

For one axis/frequency and one feature channel, write:

\[
f_e=G_e*f_t,\quad f_o=G_o*f_t,\quad
s_e=G_e*s_t,\quad s_o=G_o*s_t.
\]

Define opposing directional-energy channels:

\[
E_+=(f_e+s_o)^2+(f_o-s_e)^2,
\]
\[
E_-=(f_e-s_o)^2+(f_o+s_e)^2.
\]

Their difference is:

\[
E_+-E_-=4(f_e s_o-f_o s_e).
\]

For two frames, the expression in parentheses reduces to:

\[
(b-a)(u_{2,e}u_{1,o}-u_{2,o}u_{1,e}).
\]

This is the crucial temporal-spatial interaction. It changes sign when the two frames are swapped, unlike squared temporal differences alone. At initialization, or for identical repeated frames, f=s and the opponent difference is exactly zero. Which algebraic sign denotes right/up depends on the kernel phase, convolution convention and image coordinates; establish the label convention using translated synthetic patterns rather than assigning it from notation.

Average each E channel across the d learned feature channels, then locally pool with a fixed 3x3 nonnegative averaging kernel K. With j indexing the eight channels (two signs x two axes x two frequencies), compute:

\[
A_{j,t}=K*\operatorname{mean}_{c} E_{j,c,t},\qquad
\widetilde E_{j,t}=\frac{A_{j,t}}{\epsilon+\operatorname{mean}_{k} A_{k,t}},
\quad \epsilon=10^{-4}.
\]

The common positive denominator preserves opponent sign and introduces response competition. Keep the amplitude-preserving p,d routes beside it; normalization alone could erase task-relevant contrast. Border handling should be reflection padding, with no circular wrap introduced into feature maps.

## Output and common readout

At each scale form [p_t, d_t, normalized E_t], with 2d+8=72 channels when d=32. A learned local 1x1 adapter emits O_t with 32 channels. The common decoder receives only the current projected feature U_t=u_t (32 channels) and this emitted O_t (32 channels), using the same spatial fusion/readout for all three candidates. Optional opponent maps are obtained by subtracting the corresponding normalized E channels; do not add a fourth candidate to evaluate this algebraic redundancy.

Only the current projected feature and current accumulator output are read at t=2. The previous ordered decoder cannot receive u1 and u2 directly: that would bypass the temporal component. The seven task heads remain separate because the existing battery has distinct labels, but the temporal computation is shared. Episode state is reset before every pair and must not cross batch/example boundaries.

Persistent trace state at d=32 is 2*32*(50*50+25*25+13*13)=210,816 scalars per example, approximately 0.804 MiB in fp32. Temporary filter responses and training activations are additional. Count both states when comparing against KDA and ConvGRU; equal output width does not mean equal state or compute. Fixed filter coefficients should be distinguished from trainable parameters.

## What this comparison would and would not establish

- The biological hypothesis is useful direction-opponent, temporally filtered spatial computation, not that this network is area MT.
- The measured question is whether a causally updated state can support the same seven two-frame tasks as the direct pair reference.
- Two frames supply one temporal transition. They cannot establish sustained evidence accumulation, coherent-motion integration over time, adaptation time courses or robustness of a recurrent state over long streams.
- At t=2 the two traces preserve the projected pair invertibly: u1=1.5s2-0.5f2 and u2=1.5f2-0.5s2. Consequently, success alone would not demonstrate temporal compression or a novel mechanism beyond pair computation. The learned output adapter reduces dimensionality, but the internal state still contains that information.
- Fixed quadrature filters applied to learned nonlinear CNN channels are less directly interpretable than filters of retinal luminance. This buys reuse of the competent PAV checkpoint; it limits claims about V1 tuning.
- Cardinal random dots do not test component-versus-pattern motion or speed-plane pooling. Plaids/longer streams would be distinct future experiments requiring new scope, not silent additions here.
- Biological relevance must be judged separately from accuracy, parameter count or adoption of names such as energy, recurrence and normalization.

## Focused checks when implementation is authorized

One two-frame causal stepping check, one no-leak reset check, and algebraic checks for zero static opponency and sign reversal on swapping translated patterns suffice to establish the changed computational contract. These checks do not justify a large validation campaign or establish neural correspondence. Preserve task-level BA/AUC and equal paired held-out examples in the eventual three-candidate comparison; report state size and exposure rather than concealing their differences.

