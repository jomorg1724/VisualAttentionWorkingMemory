---
title: "What would make our model primate-like?"
subtitle: "Mathematical mechanisms, neuroscience comparisons, and falsifiable hypotheses"
date: "24 September 2026"
fontsize: 11pt
mainfont: "Times New Roman"
sansfont: "Arial"
monofont: "Menlo"
geometry: "a4paper,left=23mm,right=23mm,top=23mm,bottom=23mm"
toc: true
toc-depth: 1
colorlinks: true
linkcolor: ink
urlcolor: ink
---

# Research judgment

**The architecture is a plausible computational model of spatially organized visual memory and selective evidence use. It is not yet established as a model of primate attention, and its components should not be assigned brain-area identities from their names.** The most useful next question is whether the model solves its tasks through computations that resemble biological selection, maintenance and comparison, rather than through shortcuts that happen to give correct labels.

My leading architecture-specific hypothesis is a partial division between **content storage** in spatial KDA and **selection/comparison** in the final recurrent readout. A competing account is that the final readout stores most task-relevant information and KDA acts mainly as a temporal feature transform. A third is substantial distributed redundancy. The analyses below are designed to distinguish these accounts, not assume the first one.

Three kinds of evidence need to converge: task-appropriate behavior, internal representations, and selective causal perturbations. Decoding alone establishes accessible information for a particular decoder. A lesion alone can establish dependence under the tested intervention. Neither, by itself, establishes a biological mechanism or a one-to-one anatomical correspondence.

**Scope.** This is literature research and experimental design, not a report of new model results. The earlier reading treated the new ConvGRU as pending. On this read-only inspection, \nolinkurl{SecondPass/SpatialReadout/model.py} now contains the described implementation. I use that source for structural claims; I have not verified its training progress or evaluated a checkpoint. No checkpoint inference, training, intervention or change to another agent's work was performed. Algebraic identities were checked separately with synthetic arrays; these checks are not model results. Proposed evaluation variants and future task extensions are explicitly separated from existing suite conditions.

# 1. Mathematical connections: what is shared, and why?

The comparisons below distinguish **exact identities in our implementation**, **reductions that require stated assumptions**, and **biological hypotheses that require experiments**. An operational analogy should specify the state, update, readout and observable consequence. Merely finding an exponential decay or a multiplication in two models is too weak: the variables must play comparable roles in the computation.

## 1.1 Visual processing: filtering, rectification and sensory history

A convolutional feature before normalization has the form

$$a_c(p,t)=b_c+\sum_{d,\delta}W_{cd}(\delta)x_d(p+\delta,t).$$

Here $p$ is position, $\delta$ is an offset within the kernel, and $d$ indexes input channels. This is a weighted receptive field: the response depends on the match between a local stimulus pattern and a learned filter. Weight sharing repeats that filter across positions. Subsequent rectification preserves one sign of the match. The computational connection to a simple-cell-like feature detector is therefore a **spatial projection followed by a nonlinearity**, not merely the presence of an image input. Orientation-selective monkey striate receptive fields provide the biological motivation; the learned filters still have to be measured. [Hubel and Wiesel, 1968](https://pmc.ncbi.nlm.nih.gov/articles/PMC1557912/?page=1).

For example, an elongated alternating-sign filter has a large inner product with a suitably oriented grating and a smaller one after rotation. A filter whose positive and negative lobes cancel uniform brightness can retain sensitivity to contrast structure while rejecting a constant luminance offset. Neither property is guaranteed by convolution alone. Fit orientation, spatial-frequency and phase response curves, and test whether their predictions generalize to held-out images rather than naming a layer V1 from its position.

Our stacked input makes the first convolution a short spatiotemporal filter:

$$a_c(p,t)=b_c+\sum_{\ell=0}^{2}\sum_{d,\delta}
 W_{cd\ell}(\delta)x_d(p+\delta,t-\ell).$$

Lag-specific weights let it distinguish temporal order. A positive current-frame lobe paired with a negative earlier-frame lobe computes a temporal difference; offsetting those lobes also makes the response sensitive to displacement. Reliable direction-selective energy generally needs nonlinear combinations of such features. This equation establishes the available operation, not a measured motion-energy mechanism. Reverse the frame order and inspect direction codes; a truly order-insensitive representation cannot by itself distinguish opposite displacements in otherwise matched pairs.

The convolutional paths combine increasingly broad neighborhoods, but there is an important exception to a purely local interpretation: **GroupNorm pools statistics over spatial positions within each channel group**. Its statistics can couple distant locations within one update. A feature-map cell therefore has local convolutional support plus a potentially nonlocal normalization dependence. Spatial KDA state is locally indexed, but the entire encoder is not spatially independent.

## 1.2 The exact KDA computation: online associative correction

At one spatial site and one head, omit those indices. The implementation has $S\in\mathbb R^{8\times16}$, keys and queries $k,q\in\mathbb R^8$, values $v\in\mathbb R^{16}$, row retention $\alpha\in(0,1)^8$, and write strength $\beta\in(0,1)$. Define $D_t=\operatorname{diag}(\alpha_t)$. Then

$$\bar S_t=D_tS_{t-1},\quad \hat v_t=\bar S_t^\top k_t,\quad e_t=v_t-\hat v_t,$$
$$S_t=\bar S_t+\beta_t k_te_t^\top,\qquad o_t=S_t^\top q_t.$$

The key specifies which association to test; the value is the desired associated feature; the residual measures its mismatch with the retained association. The query chooses how to read the updated matrix. These are learned feature coordinates, not necessarily named stimulus variables such as location or orientation.

**Why this is genuinely an error-correcting memory.** Hold $k_t,v_t$ fixed and define a local loss on a candidate matrix $M$:

$$L_t(M)=\tfrac12\|v_t-M^\top k_t\|_2^2,
\qquad \nabla_M L_t=k_t(M^\top k_t-v_t)^\top.$$

One gradient step starting at the retained state gives

$$\bar S_t-\beta_t\nabla_M L_t\big|_{M=\bar S_t}
 =\bar S_t+\beta_t k_te_t^\top=S_t.$$

This is an exact algebraic identity for our update. It does not mean the training objective explicitly includes this loss: the network learns how to produce keys, values and gates through its task loss. The local identity describes the state transition used even during inference.

The outer product can also be separated into

$$\Delta S_t=\beta_t k_tv_t^\top
 -\beta_t k_tk_t^\top\bar S_t.$$

The first term writes a key-value association. The second subtracts the association already predicted along that key. Thus KDA is closer to a **delta-rule associative memory** than to an unconstrained Hebbian sum. Calling the first term Hebbian-like refers to the product of two feature populations; the residual subtraction and learned representations prevent a direct identification with a particular synaptic learning rule.

**What changes, and what is protected?** For a unit-norm key,

$$S_t^\top k_t=(1-\beta_t)\bar S_t^\top k_t+\beta_tv_t.$$

The current-key prediction moves a fraction $\beta_t$ toward the value. For a different query $a$,

$$S_t^\top a-\bar S_t^\top a
 =\beta_t(k_t^\top a)e_t.$$

A query orthogonal to the key is unaffected by this correction; a similar query is changed strongly; an oppositely aligned query changes with the opposite sign. This gives a concrete interference mechanism. The code normalizes keys and queries, except that vectors below its numerical epsilon need not have norm one. In that case the same-key coefficient is $\beta_t\|k_t\|^2$ instead of $\beta_t$.

**Prediction.** Measure interference as a function of learned key overlap, not only stimulus similarity. A controlled new write should alter old retrieval in proportion to overlap and residual direction when the other quantities are held fixed. Failure of the task-level effect would not refute this local identity; it would show that downstream processing compensates, ignores the affected feature, or violates the control assumptions.

## 1.3 Why a KDA matrix is operationally like a dynamic synaptic efficacy

A conventional fixed linear pathway computes $y=W^\top a$. If recent activity changes the effective matrix, the same later input can produce a different output. In KDA, with stored $S$ held fixed,

$$o=S^\top q,\qquad \frac{\partial o}{\partial q}=S^\top.$$

The memory is therefore a history-dependent **input-output operator**. It does not merely add a remembered vector to the current input; it changes the transformation applied to that input. This is the precise sense in which it acts like fast weights, although software stores $S$ as a recurrent tensor rather than as optimizer-managed parameters.

A standard rate approximation to short-term facilitation/depression uses presynaptic activity $r_j$, utilization $u_j$, available resources $x_j$ and fixed connectivity $W^0_{ij}$:

$$\dot u_j=\frac{U-u_j}{\tau_f}+U(1-u_j)r_j,
\qquad
\dot x_j=\frac{1-x_j}{\tau_d}-u_jx_jr_j,$$
$$I_i=\sum_jW^0_{ij}u_jx_jr_j.$$

This is a representative Tsodyks-Markram-type rate formulation, not a transcription of the complete spiking implementation in Mongillo et al. Facilitation raises utilization; use depletes resources; both relax between inputs. The resulting effective pathway is $W^{\mathrm{eff}}=W^0\operatorname{diag}(ux)$. [Itskov et al., 2011, model equations](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2011.00040/full).

Mongillo, Barak and Tsodyks proposed that calcium-mediated facilitation could retain a trace that later spiking reads or refreshes, so continuously elevated firing need not carry the entire memory. [Mongillo et al., 2008](https://barak.net.technion.ac.il/files/2012/11/synapticmemory.pdf).

\Needspace{19\baselineskip}

The comparison is now operationally explicit:

| Operation | Dynamic-synapse description | KDA description |
|:--|:--|:--|
| Write | Activity changes $u,x$, hence efficacy | Current features change $S$ through $ke^\top$ |
| Retain | State relaxes over facilitation/depression timescales | Old associations are transformed by retention and subsequent corrections |
| Read | Later activity is transformed by modified efficacy | Later $q$ is transformed by $S^\top$ |
| Express weakly | Low activity can yield weak output despite an altered efficacy | A query can expose little of a stored association |

**The shared computation is history-dependent effective connectivity that can outlast its currently expressed output.** It is not an assertion that $S=u$, $\alpha$ is calcium, or $\beta$ is release probability. In the rate example, changes are constrained by fixed connectivity and bounded presynaptic variables. KDA can write signed, cross-feature corrections into a full matrix. There is no global invertible change of variables making these systems equivalent.

A useful limited correspondence concerns relaxation. With no drive, the rate equations give

$$u(t)-U=[u(0)-U]e^{-t/\tau_f},\qquad
x(t)-1=[x(0)-1]e^{-t/\tau_d}.$$

For KDA, **if writes are experimentally disabled** and one row has constant retention $\alpha$, its stored contribution obeys

$$s_{t+n}=\alpha^ns_t=e^{-n\Delta t/\tau_{\mathrm{eff}}}s_t,
\qquad \tau_{\mathrm{eff}}=-\frac{\Delta t}{\log\alpha}.$$

Both retain a history-dependent state with a finite relaxation time. This mapping concerns deviations in a biological state and retained KDA components, not their absolute numerical baselines. Moreover, blank images do not automatically disable KDA writes: learned biases, keys, values and higher-scale inputs can still drive updates. Time in seconds requires an explicit frame-to-time mapping; a gate alone yields a timescale in updates.

**Test implied by the analogy.** Present two distinct histories, then the same diagnostic input. Ask whether their outputs differ because of the carried matrix, whether a matched matrix replacement transfers that difference, and how it decays during controlled delays. This tests retained effective connectivity more directly than comparing raw activation magnitude.

## 1.4 Hidden storage versus expressed activity: a linear-algebra justification

Suppose two histories leave states differing by $\Delta S$. Under the same read-only query their raw outputs differ by

$$\Delta o=\Delta S^\top q.$$

It is possible to have $\Delta S\ne0$ but $\Delta S^\top q=0$. For example, if $\Delta S=av^\top$ and $a^\top q=0$, the memory difference is invisible to this query even though it is still present in the matrix. Changing the query to $a/\|a\|$ exposes it. This is a concrete storage-access dissociation, rather than an appeal to a generally “hidden” state.

For a frozen matrix, an analysis-only bank of the eight coordinate queries $e_i$ retrieves its eight rows: $S^\top e_i$. Thus complete access through that artificial query bank follows algebraically. The trained network need not be able to generate those queries or use the recovered values. Report artificial accessibility separately from ordinary readout and behavior. The learned output projection and downstream recurrence can hide or recover still other combinations.

Our normal KDA step reads **after** writing. Consequently, a physical image impulse changes keys, values and gates as well as the query. It is not the same experiment as a read-only query bank. Keep a frozen-state probe analysis and an ordinary input-impulse analysis separate. Human impulse experiments motivate this distinction between latent information and its expression, but do not establish that human brains contain our matrix mechanism. [Wolff et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5446784/).

**Sharper hypothesis.** Content can be weak in ordinary emissions yet strong in the carried matrix, and a later cue may expose it without rewriting the original stimulus. Compare matrix, emission and behavioral decoding across cue phases. Test whether a matrix intervention changes the recovered content. Mere success of an unconstrained decoder is insufficient.

## 1.5 Prediction error: an exact connection, with a different target

The KDA residual $e=v-\bar S^\top k$ really is a prediction error in the mathematical sense: an observed target feature minus an internally predicted target feature. Calling it only “not predictive coding” understates that connection. The critical questions are **what is predicted, which variable changes to reduce the error, and where the error goes**.

A simple linear generative model writes a sensory vector as $x\approx Wz$. Its reconstruction objective and gradient updates are

$$E(z,W)=\tfrac12\|x-Wz\|^2,\quad \epsilon=x-Wz,$$
$$\Delta W=\eta\epsilon z^\top,\qquad
\Delta z=\eta W^\top\epsilon.$$

Under $W=\bar S^\top$, $z=k$, $x=v$, and $\eta=\beta$, the **weight update** is exactly KDA's transposed correction. The common computational motif is residual-driven adjustment of an associative/generative map. These equations are a minimal illustrative derivation, not the complete equations of a cited cortical theory.

Hierarchical predictive coding additionally assigns predictions and residuals to inter-area communication and updates latent causes using feedback. Rao and Ballard's account explicitly uses descending predictions and ascending residual errors. [Rao and Ballard, 1999](https://doi.org/10.1038/4580). Our KDA keys and values are computed from the current feature field; it does not iterate the displayed $\Delta z$ rule to infer its key, and it emits $S^\top q$, not the residual as an explicit ascending error population. Its target is a present key-conditioned value, not necessarily a future frame or a lower visual area's activity.

**A prediction we can derive.** With fixed unit key $k$, fixed value $v$, $q=k$, no decay ($D=I$), and constant $\beta$, repeated writes satisfy

$$S_n^\top k=[1-(1-\beta)^n]v\quad\text{if }S_0=0.$$

The pre-update residual shrinks geometrically. Thus repeated predictable feature pairs produce smaller **write residuals**, while the associated readout approaches $v$ and can grow. Residual suppression is not automatically neuronal-response suppression.

With scalar retention $D=\alpha I$, the aligned steady-state readout instead is

$$S_\infty^\top k=\frac{\beta}{1-\alpha(1-\beta)}v.$$

Decay continually removes part of the association, leaving a nonzero pre-update residual when $\alpha<1$. Normal repetition therefore need not drive error to zero. This derivation assumes constant representations and gates; a repeated image does not guarantee that at higher scales.

**Test.** Record predicted value, residual, correction norm and emitted response separately. Compare matched repeated versus changed features while controlling gate changes. Geometric residual shrinkage would support the local correction account; expecting every downstream feature to suppress would be an unjustified extra prediction.

## 1.6 Attention as selective influence: derive the coefficients

Rearrange the exact KDA update as

$$S_t=A_tS_{t-1}+\beta_tk_tv_t^\top,
\qquad A_t=(I-\beta_tk_tk_t^\top)D_t.$$

Starting at zero and unrolling one site/head gives

$$S_t=\sum_{s=1}^{t}(A_tA_{t-1}\cdots A_{s+1})\beta_sk_sv_s^\top,$$
$$o_t=\sum_{s=1}^{t}c_{t,s}v_s,\qquad
c_{t,s}=\beta_s q_t^\top(A_t\cdots A_{s+1})k_s.$$

The empty product at $s=t$ is the identity. Matrix order matters: retain-then-correct does not generally commute with correct-then-retain. Each earlier value reaches the current emission through its initial write strength, all intervening retention/correction operators, and the current query. **This is the mathematical basis for describing KDA as selective access to history.**

The coefficients may be negative and need not sum to one. They are temporal coefficients at a specified spatial site/head, not probabilities distributed across all image locations. A larger $\alpha$ preserves a key-coordinate row; whether that preserves target evidence depends on what that row contains. A cue can in principle influence writing, survival, or reading. These are distinct forms of selection that should have different epoch-specific intervention effects.

The expansion is exact conditional on the realized keys, values and gates. It is not a total causal attribution from pixels: changing an old image can also change later gates, higher-scale features and the final readout. In the implementation each KDA's gates come from its current input rather than directly from its own previous matrix; higher-scale inputs can nevertheless depend on lower-scale memory emissions.

For behavior, a useful descriptive model is

$$\operatorname{logit}P(y=1)=b(c)+\sum_p w_p(c)e_p,$$

where $c$ is the cue and $e_p$ is independently varied evidence at location $p$. Cue-dependent $w_p$ means selective evidence use; cue-dependent $b$ alone is a response bias. This is an analysis model, not a claim that the network implements a logistic circuit. Estimate it on matched valid scenes with enough independent evidence variation, then perturb the proposed route. Biased-competition physiology motivates asking whether a paired-stimulus representation shifts toward the attended singleton, but the behavioral weighting and representational tests answer different questions. [Reynolds et al., 1999](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782185/).

A final-state ConvGRU intervention cannot affect earlier KDA on later steps through a descending connection, because that connection is absent. This graph constraint rules out one top-down explanation; it does not rule out selective encoding or choice within existing pathways. The external task ID selects a head and is not a learned task instruction entering the recurrence.

\Needspace{12\baselineskip}

## 1.7 Gain control: the meaningful overlap with normalization

A schematic divisive-normalization response is

$$r_i=\frac{a_iE_i}{\sigma+\sum_jw_{ij}a_jE_j},$$

for nonnegative drives, weights and attentional gains. Rectification and the specific spatial/feature pooling are omitted here for clarity. Multiplying a target's drive can alter both numerator and shared suppression, so effects depend on the stimulus and gain field. [Reynolds and Heeger, 2009](https://www.cns.nyu.edu/heegerlab/content/publications/Reynolds-Neuron2009.pdf).

For illustration, if the same gain $a$ applies throughout a pool, this expression becomes $E_i/(\sigma/a+\sum_jw_{ij}E_j)$. Its effect depends on the relative size of the constant and stimulus-dependent denominator. This algebra explains why a multiplicative internal gain need not produce an equally multiplicative output at every contrast.

Our GroupNorm instead computes, over a channel-and-space group $G$,

$$\operatorname{GN}(h_i)=\gamma_i\frac{h_i-\mu_G}{\sqrt{\nu_G+\varepsilon}}+b_i.$$

Ignoring epsilon, a common positive scale and offset $h'_i=ah_i+d$ cancel in its standardized component. Both operations make a feature response depend on the surrounding population and regulate scale; that is a substantive shared computation. However, GroupNorm uses centering and variance, rather than the nonnegative suppressive drive above, and has no separately specified attentional field. It can redistribute responses without enforcing a conserved attention budget.

**Test.** Vary target contrast, distractor contrast, and spatial extent independently; compare response-versus-contrast curves at matched cues. Inspect whether cue effects enter numerator-like feature drive, normalization statistics, KDA retention, or final evidence weighting. Fit competing gain models on held-out conditions. Similar psychometric gain alone does not identify the denominator mechanism.

## 1.8 ConvGRU and recurrent neural dynamics: when memory becomes integration

Our final ConvGRU obeys

$$z_t,r_t=\sigma(\mathrm{Conv}([x_t,h_{t-1}])),\quad
\tilde h_t=\tanh(\mathrm{Conv}([x_t,r_t\odot h_{t-1}])),$$
$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.$$

Rearranging yields $h_t-h_{t-1}=z_t\odot(\tilde h_t-h_{t-1})$. A leaky rate circuit has the generic form $\tau\dot h=-h+\phi(W_xx+W_hh)$; forward Euler discretization gives the same relaxation form with $z=\Delta t/\tau$ when the gate is fixed and the candidate matches the circuit's nonlinearity. This is a conditional dynamical correspondence. A learned gate changes the rate of movement toward the candidate; the reset gate changes how previous activity contributes to that candidate. The model's signed tanh coordinates need not be literal nonnegative firing rates.

If a scalar candidate is constant, differences between trajectories decay as $(1-z)^n$, giving the exact discrete relaxation time $-\Delta t/\log(1-z)$. It is only approximately $\Delta t/z$ for small $z$. That simple formula fails when recurrent feedback or changing gates dominate.

Linearize about a trajectory, with candidate derivative $C_h$ and gate derivative $Z_h$. The one-step state Jacobian is

$$J_t=\operatorname{diag}(1-z_t)
 +\operatorname{diag}(\tilde h_t-h_{t-1})Z_h
 +\operatorname{diag}(z_t)C_h.$$

Small perturbations propagate as $\delta h_t\approx J_t\delta h_{t-1}+B_t\delta x_t$. A stable, approximately constant mode with $0<|\lambda|<1$ has decay-envelope timescale $-\Delta t/\log|\lambda|$. A positive eigenvalue close to one preserves a component; a negative/complex mode may alternate or rotate it. Time-varying and non-normal systems require finite-time products or measured impulse responses, not eigenvalues alone.

\Needspace{9\baselineskip}

For a fixed-gate scalar candidate $\tilde h=ah+be$, the effective recurrence is

$$h_t=\underbrace{[1-z+za]}_{\lambda}h_{t-1}+zb\,e_t.$$

If $a=0$, it is an exponentially weighted filter. If $a\approx1$, it can approximate an evidence integrator over its operating range. If feedback yields multiple stable fixed points, it can instead maintain a categorical decision. These are different computations available to recurrence; a GRU label does not tell us which was learned. Primate motion-pulse experiments motivate testing persistence of an evidence contribution. [Huk and Shadlen, 2005](https://pubmed.ncbi.nlm.nih.gov/16280581/).

**A particularly important KDA contrast.** Fixed-key KDA with a fixed direction value approaches that association, as derived above. Repeatedly writing the same direction is not automatically incrementing a direction counter. Solving the duration task through accumulated counts would require another code, varying keys/values/gates, or downstream dynamics such as the ConvGRU. Test equal-count reordered sequences and estimate time-dependent evidence weights before describing KDA itself as the accumulator.

The 3x3 recurrent kernels allow spatial neighbors in the 7x7 hidden field to interact repeatedly, making local spread, competition or stabilization possible. They do not impose any particular sign pattern, bump attractor or excitatory/inhibitory circuit. Measure a perturbation's spread and recovery before claiming those mechanisms. The full input pathway also includes the nonlocal normalization described above.

## 1.9 Binding, interference and capacity: what the matrix geometry predicts

An idealized associative store illustrates the computation:

$$S=\sum_i k_iv_i^\top,\qquad
S^\top k_j=v_j+\sum_{i\ne j}(k_i^\top k_j)v_i$$

for unit keys. Retrieval contains the intended value plus cross-talk weighted by key overlap. This is an explanatory reference model, not our exact delta-rule trajectory. Our delta correction modifies the cross-talk, but the geometric question remains: how separable are associations under the queries actually used?

Spatial KDA has a separate matrix at each feature-map site, so physical location can provide part of the address without being stored in the key. Convolutional receptive-field overlap, normalization and later spatial recurrence can mix nearby items. Thus both **feature-key overlap** and **spatial overlap** are candidate predictors of errors. The eight-dimensional key space does not imply an eight-item behavioral capacity: there are many spatial sites, heads and scales, continuous superposed codes, and a separate final recurrent state.

For orientation, a natural analysis target is the axial embedding

$$v(\theta)=(\cos2\theta,\sin2\theta),$$

which respects the equivalence of $\theta$ and $\theta+\pi$. If sample and probe are represented this way, a bilinear comparison can compute

$$\sin2(\theta_p-\theta_s)
=\sin2\theta_p\cos2\theta_s-\cos2\theta_p\sin2\theta_s.$$

For changes strictly within $(-90^\circ,90^\circ)$ its sign gives the rotation direction, with zero requiring separate treatment. Multiplying by the instructed sign gives an idealized signed-change decision. The implementation is not forced to use this representation, but it shows exactly how remembered feature content can support the task rather than decoding only the final label.

A post-sample location query could change access to a site's stored feature without globally increasing memory magnitude. Decode every site's actual angle before and after the query, then replace one site's state with a matched donor. Selective transfer of that location's reported content would support associative binding more strongly than a global accuracy drop. Control the suite's fixed-spaced orientation inventory: global offset plus assignment may suffice, so four successful reports would not demonstrate four independent arbitrary-angle stores.

## 1.10 Stable information in changing activity: the decoding consequence

Let a remembered variable $m$ be represented at time $t$ as $h_t=R_tm+\eta_t$. Even if $m$ is preserved, a changing embedding $R_t$ can cause a decoder trained at one time to fail at another. For a scalar feature, the simple noiseless example $h_t=m(\cos\omega t,\sin\omega t)$ preserves $|m|$ and its signed coordinate relative to time, while a decoder using only the first coordinate periodically fails.

At each time a suitable linear decoder can recover $m$ in this example; a fixed decoder cannot. Conversely, a stable subspace can coexist with dynamic components. This motivates separate within-time decoding, cross-time generalization and behavioral-use tests. Primate PFC population analyses provide evidence that stable mnemonic information and dynamic single-unit responses can coexist. [Murray et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5240715/).

Fit temporal alignment or subspace models only on training trials, then test on held-out trials. A decoder needing time-specific alignment shows accessible transformed information, not proof that the actual readout performs that alignment. Link the preserved direction to behavior and targeted perturbations. This distinction also prevents mistaking poor cross-time decoding for complete forgetting.

**What these derivations earn.** We can already justify describing the architecture as a spatially organized, error-correcting associative state coupled to a gated recurrent dynamical system. It has mathematically specified mechanisms for history-dependent transformation, selective access, interference, relaxation and potential integration. Whether training recruits them in the task-specific, primate-like ways proposed below remains an empirical question.


\clearpage

# 2. What the existing suite can actually establish

The suite can test elementary visual discrimination, selective use of a designated location, temporal integration, delayed feature comparison, location-feature binding, and exact image membership. It cannot, without an extension, establish classical probability-based cue-validity benefits, spontaneous saccadic selection, reaction-time effects, a universal item limit, or human-like general scene recognition.

| Existing task | Candidate functional ability | Most informative first analysis |
|:--|:--|:--|
| Motion direction | Ordered short-range motion processing | Direction decoding across displacement; frame-order control |
| Signed orientation | Axial orientation representation and comparison | Decode actual angles; generalize over phase and base angle |
| Contrast | Contrast-sensitive visual code | Pedestal x increment curves; response/contrast functions |
| Spatial frequency | Frequency-sensitive visual code | Generalization across phase/orientation and octave increment |
| Chromatic increment | Defined colour-axis discrimination | Sensitivity to the increment versus luminance/noise nuisance |
| Contour | Spatial grouping | Structured versus scrambled response with orientation inventory matched |
| Natural spectral detail | Image-frequency processing | Spectral difference versus mean/RMS controls, held-out photos |
| Ring orientation | Cue-dependent spatial comparison | Target versus foil evidence use; no long-delay claim |
| Signed cued orientation | Location x sign rule plus retention | Separate cue/sign, stored target angle, and post-probe comparison |
| Cued motion duration | Selective accumulation and decision retention | Cumulative counts, temporal evidence weights, suffix controls |
| Krauzlis motion change | Relevant-change selection | Target hits, foil false reports and catch false positives separately |
| Spatial binding | Retrospective access to location-feature associations | Per-location angle memory before query; query-dependent access |
| Scene recognition | Exact study-membership memory | Novel/old matched probe responses, serial position, load and hold |

Two details change what an analysis means. First, the first two nominal blank updates still contain prior sample frames in the raw input stack. A recurrent-maintenance claim must use later sample-free updates. Second, the binding task uses an orientation inventory with a fixed 45-degree spacing and a random offset. A compact representation of global offset plus assignment could solve it; success does not prove four independent, arbitrary-precision orientation stores.

The recognition task repeats the same probe three to five times. A negative probe remains negative despite becoming familiar during those repetitions. The critical distinction is study-list membership versus familiarity acquired during the test itself. The empty-list cells have only negative labels and measure specificity, not balanced recognition discrimination.

# 3. Hypothesis A: the model develops meaningful visual feature codes

**Prediction.** On tasks it solves, early/intermediate maps should represent orientation, frequency, contrast, colour direction or motion in a way that generalizes across the task's nuisance variation. Deeper stages may combine local evidence into contour or relational signals. This is a functional visual-processing hypothesis, not an assignment of cortical areas.

**Experiment.** Start with native task stimuli and record encoder outputs before memory concatenation, KDA emissions, the deepest combined field and final recurrent state. Fit held-out encoding models and simple linear decoders for stimulus variables. For orientation, regress cosine and sine of twice the actual rendered angle; for direction use circular or four-way representations as appropriate. Hold out phase, base angle and source identity where supported, instead of randomly splitting nearly duplicate rasters.

**Response mapping extension.** Independently vary orientation, position, contrast and spatial frequency with a factorial stimulus set. Plot tuning curves, effective receptive fields and phase dependence. These would be new diagnostic stimuli, not additional primary suite cells. Use first-frame responses and controlled reset states to distinguish visual tuning from inherited sequence context.

**Evidence against.** A decoder or behavioral decision fails when irrelevant phase or source identity changes, despite high aggregate training-like accuracy. Apparent selectivity explained entirely by label-correlated low-level artifacts would weaken the intended visual-code interpretation. Untuned-looking individual channels do not rule out a distributed feature code.

**Causal follow-up.** Occluding the stimulus tests available evidence; suppressing a calibrated internal feature direction tests a representation. Keep them separate. A direction-specific perturbation should alter relevant judgments more than unrelated task controls if the feature representation is behaviorally used.

# 4. Hypothesis B: cues select evidence, rather than merely biasing reports

**Prediction.** With the physical scene held fixed and the instruction changed, the model should change which location's evidence determines its answer. Target evidence should influence reports more strongly than matched foil evidence. Cue identity should remain recoverable after it disappears when later computation still needs it.

**Existing-suite test.** Build matched counterfactual evaluations for ring orientation, signed orientation, motion duration and Krauzlis change: preserve the scene/evidence and vary a valid target instruction, recomputing labels under the task's rule. Verify that matched generation preserves intended nuisance distributions; never simply relabel a scene with incompatible rendering constraints. These are new matched evaluations of existing rules, since ordinary suite streams do not supply such pairings automatically.

Fit a held-out behavioral evidence-use model. For signed cued orientation, target evidence is the instruction-relative rotation at the cued location; include foil evidence, unchanged/opposite negatives, position and magnitude. For motion duration, include each patch's count evidence and recent suffix. For Krauzlis, retain target/foil/catch categories rather than collapsing all physical changes. Use model choices as behavior and logits as a supplementary continuous measure, not confidence masquerading as reaction time.

**Two competing mechanisms.** Under early selection, cueing changes the fidelity or downstream accessibility of target features in encoder/KDA stages. Under late selection, target and foil features remain represented, but the final state or head preferentially uses the target. Measure both. Context-dependent population dynamics in macaque PFC and a trained RNN provide a precedent for selection and integration occurring within recurrent computation rather than requiring complete early filtering. [Mante et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/4121670/).

**Evidence against selective use.** Changing only foil evidence shifts the answer as much as changing target evidence; changing the cue produces a fixed class bias irrespective of the newly selected stimulus; or cue effects vanish once physical image differences are controlled. Merely decoding the cue from visible cue pixels does not support persistent internal selection.

## A separate test is needed for classical cueing benefits

Our cue normally defines which item must be reported. An “invalid cue” would often redefine the target, and deleting the cue can leave the correct response unspecified. Therefore cued-minus-uncued accuracy here is not automatically a Posner-style attentional benefit.

For that question, add a separately specified **any-change detection task** with equal reportability at all locations, probabilistic validities such as 25/50/75/100%, and a spatially uninformative condition. Match change magnitude, site and nuisance variables. Predict enhanced sensitivity or lower detection thresholds for more informative valid cues, and possible costs for unexpected locations. These are hypotheses, not guaranteed effects of recurrence. Existing weights tested on this protocol face distribution shift; matched training would be a separate experiment.

Morgan, Albanna and Herman's recurrent ViT is a computational comparison: it uses cue-validity conditions, memory-guided attention and wait/declare actions. It also illustrates why solving a task and reproducing its cueing signatures are separable. Our fixed final classifier cannot supply learned reaction times without an additional action-timing protocol. [Morgan et al., 2025, pinned v1](https://arxiv.org/html/2502.10955v1).

# 5. Hypothesis C: KDA preserves content that current readout may hide

**Prediction.** During genuinely sample-free delay frames, sample orientation or local feature-location information remains recoverable from KDA state. It may be weaker in the emitted field, then become more accessible when the query/probe changes. Final ConvGRU state may instead emphasize the instruction, a transformed comparison basis or a decision variable. The reverse distribution of roles is a serious competing hypothesis.

Human fMRI work showed decodable remembered orientation in early visual areas; other work found that mnemonic and incoming sensory information can coexist in visual population patterns. These observations motivate testing storage within visual processing, but they do not identify our tensors with BOLD or establish one universal storage locus. [Harrison and Tong, 2009](https://www.nature.com/articles/nature07832); [Rademaker et al., 2019](https://escholarship.org/uc/item/71z76126).

**Decoders.** At each native trial phase, compare (a) local KDA matrix, (b) raw queried value, (c) projected KDA emission, (d) deepest map, and (e) final vector-GRU/ConvGRU state. Decode each location's actual sample angle using a regularized doubled-angle linear model. Also decode location cue and sign separately. Do not substitute final class label for sample content: a remembered answer and a remembered sensory item are different quantities.

**Timing.** Compare the last sample, the first genuinely sample-free blank, end of delay, post-query/pre-probe where available, and post-probe. In binding, post-query is particularly informative because it occurs before new orientation evidence. For orientation, sample angles decoded after probe can exploit sample-probe correlations; pre-probe analysis is the primary maintenance evidence.

**Access test.** If state decoding is strong but normal emission decoding is weak, apply a fixed analysis-only set of queries or a nonspecific input impulse on separate trials and test whether its responses carry the prior item. Compare against shuffled-state and no-memory controls. Human impulse-response experiments motivate this logic, but a positive model result would show an access/storage dissociation, not prove biological activity-silent synapses. KDA state is itself a nonzero computational activation. [Wolff et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5446784/).

**Causal test.** Suppress or replace local state during the late blank, then carry that changed state forward; compare with a one-frame emission perturbation and with final-state perturbation. Calibrate site/scale doses and include sham. Content-specific donor-state swaps between matched trials can test whether the later report moves toward donor content, but swaps can create unnatural state combinations; use matched donor controls and graded mixtures.

**Evidence against KDA storage dominance.** Good retention survives well-verified local and multi-scale KDA manipulations while final-state perturbations reliably disrupt it, with sensory processing otherwise spared. This favors another route, although no single null lesion proves irrelevance. Strong KDA decoding without behavioral influence supports accessible but unused or redundant content. A failed decoder alone is inconclusive.

# 6. Hypothesis D: retrospective queries change access to bindings

**Prediction.** Before the binding query, the system retains enough information about all initially eligible locations to support a later arbitrary query. After the query, the selected location's remembered feature becomes more accessible to the decision route. This might be a change in readout geometry without wholesale erasure of uncued content.

Human work found recovery of mnemonic reconstructions after a retrospective cue, with behavioral relevance. It motivates a distinction between stored information and its current priority, not a requirement that every model layer show rising mean activity. [Sprague, Ester and Serences, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4978188/).

**Test.** Decode all four sample angles by location just before and just after the ring query, using a common decoder trained independently as well as time-specific decoders. Compare correct-location assignment with an inventory-only decoder. Include matched global offsets across different location permutations, so the fixed inventory cannot solve the analysis target by itself.

Then fit an analysis-only comparator using pre-probe state plus independently encoded probe features. Train and select it on independent splits, and compare its performance with the unchanged deployed head on held-out trials. Improvement establishes recoverable information for that comparator, not success of the original model or a definitive identification of the bottleneck.

**Binding-specific perturbation.** Exchange calibrated state regions corresponding to two locations during the delay, keeping the probe unchanged. A systematic tendency to judge the exchanged remembered bindings is more informative than a generic accuracy drop. Map input locations to actual feature support; overlapping receptive fields and GroupNorm prevent exact pixel-local interpretation.

**Evidence against.** The representation supports only the global orientation inventory, cue effects reflect current ring pixels without access to the remembered feature, or apparent binding generalization collapses when assignment varies. A later evaluation with independently drawn orientations would test whether the model relies on the structured inventory; it changes the stimulus distribution and must be labeled accordingly.

Work on continuous visual-memory reports distinguishes imprecision from reports of other items, motivating error-type analysis. Our binary swap task cannot directly fit a standard continuous-report mixture model or identify a universal slot/resource account. It does not vary the number of Gabors. [Bays, Catalao and Husain, 2009](https://www.paulbays.com/pdf/BayCatHus09.pdf).

# 7. Hypothesis E: recurrence integrates relevant evidence across time

**Prediction.** On motion duration, the state should represent accumulated target evidence across eight transitions, not merely the last direction or the winner in a foil patch. Integration need not produce monotonic ramping in every unit; a distributed count or decision representation can change non-monotonically.

Macaque LIP studies related activity to developing motion decisions, and motion-pulse experiments tested persistence of brief evidence in the decision process. These provide functional precedents for integration, not evidence that a four-way count-winner task is identical to a two-choice motion-coherence task or that ConvGRU is LIP. [Shadlen and Newsome, 2001](https://journals.physiology.org/doi/10.1152/jn.2001.86.4.1916); [Huk and Shadlen, 2005](https://pubmed.ncbi.nlm.nih.gov/16280581/).

**Decode competing variables.** Fit matched-capacity decoders for current direction, cumulative counts per direction, count margin, running winner, final report and cue location. Early states cannot know a future random suffix; decoding eventual labels early can reflect schedule constraints rather than forecasting. Use sequence controls to separate these correlations.

**Behavioral controls.** Compare sequences with the same counts and final answer but reordered evidence; then compare sequences with the same recent suffix but different earlier counts and different correct winners. Regenerate continuous movies with matched nuisance draws instead of shuffling finished frames, which would create unphysical transitions. Evaluate whether early evidence changes the final decision and whether that effect survives blank retention.

**Temporal weights.** Fit regularized multinomial evidence-use models with direction indicators at each transition for target and foil patches. Control count margin, final direction and shared base episode. Uniform weights are an ideal count-based baseline; primate-like integration need not be exactly uniform. Strong exclusive last-step dependence would favor a recency shortcut.

**Perturbation.** Suppress the target-region final recurrent state after an early evidence block and measure whether later reports underweight that block. Compare with KDA suppression, sham, foil sites and late-epoch suppression. If a representation of the final winner alone survives the subsequent blank, that supports decision retention, not necessarily storage of the full motion sequence.

# 8. Hypothesis F: memory codes are transformed, not simply frozen

**Prediction.** The model may preserve task content in a stable subspace while individual coordinates change, or transform its code between encoding, maintenance and report. Nonconstant activation does not imply loss of memory.

Population work in macaque PFC supports coexistence of stable mnemonic coding and heterogeneous temporal dynamics. Mixed-selectivity studies also motivate testing combinations of stimulus, rule and response rather than hunting only for one-variable “memory units.” [Murray et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5240715/); [Rigotti et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/4412347/).

**Temporal generalization.** Train a decoder at each time and test it at every other time on held-out episodes. A broad off-diagonal band supports a reusable code. Strong diagonal but weak cross-time decoding supports an accessible yet changing code, subject to signal-to-noise and decoder-power controls. Weak decoding everywhere cannot distinguish absent content from inaccessible nonlinear coding.

**Mixed selectivity.** Use a factorial encoding model for location, cue sign, angle, phase and their interactions. Compare additive and interaction models on held-out trials. Distinguish actual response from correct class where errors permit. Train-only PCA or demixed methods can summarize the geometry, but a beautiful projection is not a mechanism. The task head's external identity does not establish an internal abstract task-rule signal.

**Timescale extension.** Test whether perturbation effects or retained task information persist longer in higher/final recurrent stages than in early emissions. A cortical hierarchy of intrinsic timescales motivates this question, but task-evoked feature autocorrelation in a deterministic network is not the same statistic as across-trial fluctuations in spiking. Measure matched-input impulse decay or controlled state differences; report time in model frames unless a renderer defines a clock. [Murray et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4241138/).

**Evidence against.** A claimed stable code fails cross-time generalization with adequate power; a claimed timescale hierarchy disappears when common input and frame-stack carryover are controlled; or apparent rule mixtures are only consequences of unbalanced stimulus/label sampling. These weaken those specific hypotheses, not all possible working-memory accounts.

# 9. Hypothesis G: recognition separates study membership from recent repetition

**Prediction.** A probe should evoke different responses depending on whether it appeared in the study list, after matching its physical identity. Old/new evidence should remain appropriately separated across probe repetitions. A negative probe must not become positive just because the model has now seen it repeatedly during testing.

Primate IT recordings during delayed matching with intervening items found match-related response modulation, often suppression but sometimes enhancement. This motivates looking for a membership-sensitive signal; it does not require global suppression in our signed tensors or identify the recognition computation with one anatomical structure. [Miller, Li and Desimone, 1993](https://pmc.ncbi.nlm.nih.gov/articles/PMC6576733/).

**Test.** Construct matched trials placing a given probe identity inside versus outside the study set while maintaining load and scene statistics. Decode membership after the first probe separately from later repetitions, stratify by the studied item's serial position, and analyze loads 4/12/24 separately from empty-list specificity. Evaluate source-identity-held-out readouts or similarity-based matching, not an identity classifier trained and tested on the same finite photograph labels.

Record KDA residual norms, state/readout changes and final hidden responses as candidate correlates. A smaller KDA write residual for a repeated image follows naturally only if the learned key/value code behaves that way; the architecture does not guarantee it. Pure novelty could confound membership with probe repetition.

**Evidence against study-memory interpretation.** Behavior is predicted by last-k-image matching alone, collapses for early study items, or drifts toward positive as negative probes repeat. Compare last-item, recent-window and simple similarity baselines. Exact-image matching success does not establish semantic recognition or a hippocampal episodic-memory mechanism.

# 10. Causal spatial attention tests

Use **site x epoch x dose** as the core design: target/queried site, matched foil site and background; cue, encoding, genuinely sample-free delay and probe; sham plus graded suppression or calibrated signed perturbations. Freeze checkpoint selection before viewing intervention results. Calibrate directions and dose on separate data. Keep paired episodes across conditions and record achieved local and downstream changes.

Subthreshold FEF stimulation can alter responses in retinotopically corresponding V4 representations. This motivates spatial specificity and propagation checks. Our architecture cannot reproduce the literal descending FEF-to-V4 pathway through a final-ConvGRU feedback connection it does not have. Its local perturbations instead test computational sufficiency or necessity within its actual graph. [Moore and Armstrong, 2003](https://pubmed.ncbi.nlm.nih.gov/12540901/).

| Manipulation | What it tests | Important competing explanation |
|:--|:--|:--|
| Input occlusion | Need for visual evidence | Changed stimulus distribution |
| KDA emission pulse | Influence of current retrieved features | Downstream recurrence can prolong a transient pulse |
| Persistent KDA-state pulse | Effect of altered stored association | Query may not expose the altered state; donor mismatch |
| ConvGRU-state suppression | Dependence on final recurrent representation | Generic damage or changed comparison context |
| Cue-state replacement | Whether retained instruction redirects evidence use | Changed content as well as priority |

For suppression, zero is not a universal neutral biological baseline in a signed representation. Compare graded scaling with a calibrated reference or matched replacement. Equal elementwise amplitude does not equal equal perturbation energy across unequal ROIs or tensor sizes. Use the same scene-space masks and report actual covered sites and achieved norms. A perturbation at a feature site need not remain local after convolution and GroupNorm.

**Interpretation.** Selective disruption when a site is relevant, with timing and content specificity, is stronger than uniform degradation. A target pulse that raises hits and false alarms together may shift criterion rather than sensitivity. A foil perturbation that reduces inappropriate reports can reflect suppression of distractor evidence. Effects can differ by task because an uncued event is correct to report in an any-change task but incorrect in our target-specific Krauzlis task.

For binary decisions estimate hits and false alarms under explicit signal/noise definitions, with a declared correction at rates zero/one; report sensitivity and criterion only where the signal-detection assumptions are meaningful. Four-way motion requires confusion matrices and class-specific evidence effects. No conversion of confidence, time-to-threshold in an untrained intermediate head, or wall-clock runtime into biological reaction time.

# 11. Decoder design that can survive scrutiny

**Readout sites.** Capture block outputs before concatenation; KDA pre-update and post-update state; queries, keys, values, retention and write gates; raw read and projected emission; final recurrent state; and final logits. Start with selected phases and ROIs rather than saving every enormous array for every condition. Instrumentation must reproduce ordinary forward outputs before interpretation.

**Information targets.** Use renderer ground truth only as analysis labels: actual axial angles, locations, cue sign, motion counts, event type, study membership and eventual choice. Metadata must not be passed into the model or used to create a label-directed intervention on evaluation trials. Matched scene IDs and true phase indices are for analysis scheduling, not learned inputs.

**Capacity matching.** The KDA matrices have many more coordinates than their emissions or final recurrent state. Use identical train/validation/test partitions, train-only scaling, regularization selected on validation, and dimension/sample learning curves. Report native-size decoding alongside fixed-dimensional projections or matched feature counts. Superior raw-state decoding with many more features is not, by itself, superior biological storage.

**Splits.** Group all frames, delays, cue variants and perturbations from one base episode. Split photographs by original source identity across tasks. For analyses pooled over delays, prevent the same scene from appearing in training at one delay and testing at another. Shuffle labels at the episode/group level within appropriate nuisance strata, not frame by frame.

**Uncertainty.** Estimate paired effect sizes and intervals by resampling independent base episodes; use source-aware resampling for photograph analyses. Distinguish uncertainty over test stimuli from variability over trained seeds. Correct or preregister primary comparisons when many layers/times/sites are examined. Failure to reject a null is not evidence of absence without a precision or equivalence argument.

**Baseline competence.** Interpret an attention or memory lesion on a task only if sham behavior supplies a meaningful measurable effect range. A chance observer has little room to reveal selective impairment; a ceiling observer may require harder diagnostic stimuli. Changing difficulty outside the trained support is an explicit generalization experiment, not a silent correction to the primary task.

**Counterfactual leakage.** Before probe onset, a decoder of “the final answer” can exploit sample/probe scheduling correlations. To measure storage, decode sample content, then ask whether the normal model uses it. To measure cue retention, exclude pixels that still contain the cue via the stack. To measure binding, hold inventory/global angle fixed while varying assignments.

# 12. A staged plan for this project

## First pass: three questions, one checkpoint

1. **Does it select the instructed evidence?** Use Krauzlis event-specific behavior and matched cue/scene comparisons on signed orientation or motion duration. Plot psychometrics/evidence weights first, with target versus foil effects and errors visible.
2. **What survives true absence of the sample?** On D0/D4/D12/D24 orientation and binding, compare pre-probe content decoding across KDA state, emission and final recurrent state. Include a temporal-generalization matrix and cue decoding after visual carryover has ended.
3. **Which route makes that information behaviorally useful?** Run one calibrated local suppression family, comparing a KDA state intervention with an emission or final-state intervention. Use sham, matched sites and epoch controls; do not start with a whole-network reset atlas.

These are proposed future experiments. Freeze primary metrics and a practically meaningful effect/precision target before evaluation. Use a small extraction pilot to establish decoder sample requirements, instrumentation equivalence and costs; then set the finite run allocation. Do not launch repeated training or broad intervention sweeps merely because the literature suggests many possibilities.

## Second pass: tests driven by the first result

If content survives but the answer fails, prioritize comparison/readout and retrocue access. If target and foil representations remain strong but only target evidence affects choice, investigate late selection rather than treating absent sensory gain as failure. If both storage and use are strong, test interference, structured binding errors and novel-location generalization. If duration performance is dominated by the suffix, prioritize temporal evidence weighting before claiming an accumulator.

A fair test of the new ConvGRU's causal architectural benefit requires a continuing old-readout control with matched parent, exposure, task streams and selection, ideally replicated over seeds. Its current warm-start branch alone can show acquisition in that branch, not isolate architecture from extra training or changed representation size. No additional training comparison is conducted or silently authorized by this research note.

## The first results document should have six figures

**Figure 1: Behavior and cue-dependent evidence use.** Trial schematic; task-appropriate psychometrics; target/foil/catch decomposition; classical cue-validity effects explicitly marked unavailable in the existing suite.

**Figure 2: Spatial and temporal selection.** Actual feature/gate/readout maps with their mathematical quantity labeled; matched ROI time courses; behavioral influence distinguished from internal magnitude.

**Figure 3: Content maintenance and access.** Per-location pre-probe angular decoding; KDA versus emission versus final state; cross-time generalization; pre/post-retrocue access with leakage controls.

**Figure 4: Spatial inhibition.** Paired site x epoch effects and achieved suppression; preserve sensory anchors and false-report categories.

**Figure 5: Calibrated perturbation.** Dose and direction effects on discrimination and criterion, with sham and equal-energy controls. A null effect is interpreted only after checking the manipulation.

**Figure 6: Mechanistic alternatives.** Integration versus recency, binding versus inventory, and study membership versus probe repetition, prioritized according to the first-pass findings.

## What would justify the language “primate-like”?

A modest claim would be: **the model exhibits specified functional signatures also measured in primates**. It would require reproducible cue-dependent evidence use, useful content over genuine delays, and selective causal effects that agree with the declared behavioral hypothesis. A stronger claim requires matched behavioral protocols and direct comparison to neural recordings, using held-out encoding or representational-similarity analyses with stimulus/nuisance controls and data reliability limits.

Neither solving thirteen tasks nor matching one signature proves an anatomical or physiological account. Conversely, failure of a sensory-gain hypothesis does not rule out all primate-like selection. The aim is to identify which computations the model actually uses, which resemble specific biological findings, and which are clearly consequences of its engineering design.

# Implementation sources and evidence boundary

Code and task definitions inspected for this note:

- \nolinkurl{PreAttentiveVision/TemporalIntegration/accumulators.py}: KDA recurrence and gate inputs; historical accumulator ConvGRU.
- \nolinkurl{WorkingMemory/PlainBaseline/accum.py}: hierarchy, frame stacking, old final vector GRU, no descending final-state feedback.
- \nolinkurl{SecondPass/SpatialReadout/model.py}: newly present final spatial ConvGRU route; final-only flatten/projection.
- \nolinkurl{SecondPass/SpatialReadout/BRIEF.md} and `README.md`: intended branch comparison and warm-start limits; launch instructions were read as context, not executed.
- \nolinkurl{SecondPass/TaskSuite/README.md}, `catalog.json`, and native renderer definitions discussed in the textbook: task rules and 35 conditions.
- `ANALYSIS_SOP.md`: local behavioral-first protocol, cue-validity distinctions, inhibition and perturbation controls.

Primary papers are linked beside the claims they support. Human studies, macaque studies and computational theories are identified as such. Literature findings motivate the proposed model hypotheses; none is reported as an observed result of our network. No source document's historical training or experiment instructions were treated as the current user's request to run it.
