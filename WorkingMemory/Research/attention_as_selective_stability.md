# Attention as learned selective stability of memory

Research/design, 2026-09-13. The original proposals below are historical design discussion, not the active execution protocol.

**User correction:** after authorizing a test, the user explicitly rejected changing how the models are taught. The delayed continuous-report loss, new retrocues, feature-selection tasks, cue-only controller and proposed generalization curriculum below are superseded. None of those tasks were implemented or launched. The active implementation is an architecture-only comparison on the unchanged SpatialComparison battery: ordinary continuation of the selected spatial4400 model versus that same parent with a shared full-sensory/memory controller supplying additive recurrent feedback. The controller has no direct path to the classifier; the existing comparator, heads, losses, stimuli, cues and 80-update training cycle remain unchanged. See WorkingMemory/SelectiveMaintenance for the executable protocol when available. This tests whether the feedback addition improves existing failures; it does not establish attention-specific benefits independently of added recurrent capacity.

## Recommendation

Test cue-conditioned E/I memory dynamics: a learned representation of the task-relevance cue supplies additive currents to the memory population, changing which recurrent modes persist. Preserve the spatial E/I backbone and its existing comparator. Compare the new feedback connection with a readout-only control that has the same cue controller and training objective. Do not replace memory with an LSTM gate, force a hold during blank frames, prescribe an orientation ring, or impose a global contraction that would make all memories converge to the same state.

The operational target is preservation of task-relevant feature precision during blank intervals. Lower accuracy after blanks establishes a behavioral failure, but not its unique mechanism. The earlier frozen Retention14800 diagnostic recovered sample angle at D24 with 4.25-degree mean error and rescued comparison; it demonstrated surviving information in that model. This should not be transferred untested to the new spatial checkpoint. Blank frames also contain the established visual glyphs, and the sensory encoder can produce nonzero input from an otherwise uniform image. Autonomous memory drift, adaptation, continuing sensory drive, and probe processing remain distinct possible contributors.

## What attending to a memory could compute

Let theta be a relevant visual feature, x the complete recurrent state, and d a feature decoder. A useful maintenance mechanism preserves d(x) as the network evolves, while allowing irrelevant aspects of x to change. Stable firing rates alone are insufficient: an identical fixed point for every orientation is maximally uninformative. Conversely, time-varying rates can preserve usable information.

For a continuous-valued feature, the desired geometry is a learned family of distinguishable states or trajectories. Dynamics should correct perturbations away from this family and drift slowly along the direction that changes the remembered feature. Locally, a discrete-time linearization is

\[
\delta x_{t+1}=J_t\delta x_t.
\]

Content-related directions should preserve discriminability over the required horizon; nuisance directions should contract. In a simple fixed-point approximation, relevant modes have eigenvalues near +1 while nuisance modes have magnitude below 1. This is a local explanatory target, not a guarantee based on a single Jacobian or a universal spectral penalty: time-varying, nonnormal and coupled dynamics require care. A neutral continuous attractor still permits diffusion along its content direction under noise. A strong discrete attractor can instead bias continuous orientations toward a few preferred values. The test must measure angular drift, bias and distinguishability rather than activity magnitude alone.

Attention, in this hypothesis, changes the dynamical operating regime according to the current goal. It favors persistence of selected content and permits unneeded content or nuisance activity to fade. It need not continuously boost all rates or freeze an entire vector. Nor can it reconstruct arbitrary information already lost from every state; apparent recovery can reflect information available through another population or improved access.

## Evidence and what is adopted

1. **Ester, Nouri and Rodriguez (2018):** retrospective cues reduced loss of spatial information reconstructed from human EEG, with different effects depending on cue timing. Adopt the hypothesis that internal selection can influence maintenance, not only the final response. This is not a direct measurement of our proposed E/I current mechanism. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596221/).
2. **Panichello and Buschman (2021):** monkey neural recordings showed enhanced selected information and a transformation of its population representation into a behaviorally useful format. Adopt goal-dependent reconfiguration, while distinguishing maintenance from output preparation. [Primary paper](https://www.nature.com/articles/s41586-021-03390-w).
3. **Bays and Taylor (2018):** a population-coding model fit retrospective-attention benefits through changes in population gain and swap frequency. It explicitly did not implement the recurrent mechanism that maintains memory. Gain changes are a motivation, not a ready-made retention solution. [Primary model](https://pmc.ncbi.nlm.nih.gov/articles/PMC5788052/).
4. **Driscoll, Shenoy and Sussillo (2024):** task-trained RNNs reused context-dependent dynamical motifs, including ring attractors for continuous memory tasks. Adopt training a shared recurrent system to enter an appropriate memory regime under a learned context. The work does not establish that every successful RNN must use an attractor or that its weights are biologically learned by BPTT. [Primary model and analyses](https://www.nature.com/articles/s41593-024-01668-6).
5. **Sagodi et al. (NeurIPS 2024):** approximate continuous attractors can remain useful as attractive slow manifolds over finite horizons despite structural perturbations. Adopt slow content drift plus transverse stability rather than requiring a mathematically perfect infinite-duration ring. [Primary theoretical/ML paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b78a2a7360d5a9ad750834dc5a33bfb-Paper-Conference.pdf).
6. **Schmitt et al. (2017):** mediodorsal thalamic support sustained cortical attentional-rule representations through effects on functional connectivity. **Guo et al. (2017)** demonstrated dependence of motor preparatory persistence on a frontal thalamocortical loop. These motivate distributed control, but not a literal claim that our cue controller is thalamus or that their task was visual-feature storage. [Schmitt](https://www.nature.com/articles/nature22073), [Guo](https://www.nature.com/articles/nature22324).
7. **Phillips et al. (2025):** primate recordings and a circuit model linked thalamic populations to abstract-rule selection and prefrontal processing modes. This supports continued investigation of recurrent rule control; it does not specify the implementation below. [Primary paper](https://www.sciencedirect.com/science/article/pii/S0896627325002211).

One tempting alternative is ORGaNICs-style divisive normalization. Rawat, Heeger and Martiniani (2024/2025) establish useful stability results and train the circuit on ML sequence tasks. However, the high-dimensional local stability theorem assumes identity recurrence, and the positive-baseline, zero-drive identity circuit relaxes toward zero. Stability of a normalization equilibrium is not retention of arbitrary samples. Learned convolution, additional attention control and discrete numerical integration do not inherit that theorem automatically. Do not select a normalization circuit solely because it is stable. [Primary mathematical/ML manuscript](https://arxiv.org/html/2409.18946v3).

## Candidate: cue-conditioned E/I attractor dynamics

Retain R,A in [B,64,13,13] and all existing E/I source signs. Add a small cue/rule controller C in [B,16], with nonnegative rates and its own learned ungated dynamics. Sixteen units is a lightweight engineering starting point, not a biological estimate. A possible 13E/3I allocation keeps its recurrent and outgoing signs explicit.

The controller is driven by a learned encoding h_cue of the visually presented cue region, containing no sample/probe content. A fixed cue region is an explicit stimulus/interface convention; it is not an internally supplied label. The cue disappears during the blank interval. The controller must therefore retain the cue through its own learned recurrence; no latent task ID, blank detector, copied cue vector or clock is supplied to the update.

\[
C_t=(1-\eta)\odot C_{t-1}
 +\eta\odot\operatorname{ReLU}(U C_{t-1}+Vh_{\mathrm{cue},t}+b_c).
\]

U has signed presynaptic columns, eta is derived from positive bounded learned time constants, and C_0=0. Its state and parameters count in the model budget. Restricting its input to cue pixels prevents a direct extra visual-feature storage route; indirect leakage through future architectural changes must not be silently introduced.

At each memory position i with fixed normalized coordinate p_i, produce a context current

\[
I^{\mathrm{ctx}}_{t,i}=B(p_i)C_{t-1},
\qquad B(p_i)\in\mathbb R^{64\times16}.
\]

B(p) may be a small shared coordinate network generating smooth connection magnitudes, with source signs set by controller cell type: B_oj(p)=softplus(f_oj(p))d_j. This retains spatially varying control without hardcoding a left/right attention mask or an orientation template. The numerical dimensions and coordinate parameterization must be profiled before any later launch. They are not claims about cortical anatomy.

The existing memory update becomes

\[
J_t=K_H*H_t+K_R*R_{t-1}+I^{\mathrm{ctx}}_t
    -\gamma\odot A_{t-1}+b,
\]
\[
R_t=(1-\alpha)\odot R_{t-1}+\alpha\odot\operatorname{ReLU}(J_t),
\qquad
A_t=(1-\beta)\odot A_{t-1}+\beta\odot R_{t-1}.
\]

All right-hand sides use previous recurrent states synchronously. R,A remain full fields. Input and recurrent projections learn; timescale/adaptation bounds initially follow the current model. Rates/adaptation are not centered by LayerNorm. Cue representation changes E/I excitability and which units participate in recurrent computation. It does not multiply a memory vector by a hold gate.

For the rectified rate model, the direct R-to-R Jacobian block is

\[
J_{RR}=I-D_\alpha+D_\alpha D_{\phi'(J_t)}K_R.
\]

The cue can change D_phi by shifting operating points, even though K_R itself remains fixed within a trial. The full stability analysis must also include adaptation, cue state and sensory traces. Training must discover whether such reconfiguration can sustain the selected feature; we do not assume that any additive context current produces useful attractors.

The same controller feeds the task readout in both comparison arms so both can interpret a retrocue. The experimental difference is whether its learned current also reaches memory during the blank interval. This makes readout-only attention the control and maintenance feedback the candidate. Memory state size, cue state, sensory initialization, task supervision and added training exposure are shared. The feedback pathway adds parameters; report that difference and avoid claiming a universal attention necessity from one positive comparison.

## Learning objective and task

Use two localized Gabors with independent orientations. After encoding, a brief unpredictable visual retrocue identifies the target location; it then disappears. Insert variable blank delays. Request the target's continuous orientation using a neutral report cue, with no visual orientation probe. This removes probe comparison from the primary maintenance measurement. Train the identical objective in the readout-only control and maintenance-feedback candidate.

Use a single shared decoder of recurrent rates, with cue context for target selection, to predict u=(cos(2 theta),sin(2 theta)). A simple loss is

\[
L_{\mathrm{report}}=\|\hat u(R_T,C_T)-u(\theta_{\mathrm{target}})\|_2^2.
\]

Ground-truth angles are targets for training, never recurrent inputs. This is additional feature supervision compared with the current binary task and must be given to both arms. Randomize delay and report time so preserving the selected feature at many horizons is useful. Initial tested delays can reuse D0/4/12/24; a modest longer untrained delay is a distinct generalization result, not a failure criterion. Train all new cue meanings and nuisance variation, with the same lower sensory LR in both arms and full BPTT.

Vary phase, contrast and the irrelevant item's orientation independently. Balanced location cues make the same physical item relevant on some trials and irrelevant on others. This distinguishes attention from a globally slower circuit. Retain a single-item condition, and later evaluate the existing same/different task with the same trained storage/readout interface. Do not infer a load effect by comparing our old full-field grating with the easier localized swap stimuli.

Start with delayed report loss and a modest common activity penalty for numerical control. Do not penalize all raw state changes or contract every Jacobian direction. If direct supervision during blank frames is later considered, give it to both arms and document it as an objective change. No template replay or sample-image reconstruction is fed back into memory.

## Minimal evidence that would support the mechanism

- The maintenance-feedback arm has a flatter pre-probe/report angular-error-versus-delay curve than the readout-only control, with useful zero-delay acquisition in both.
- The cue determines which independently sampled feature is preserved better. Irrelevant variability should not substantially alter target recall. Loss of uncued information is possible, not a required proof of attention.
- A common cross-time decoder continues to distinguish nearby relevant angles; time-specific diagnostic decoders check whether a failure instead reflects changed access. Keep the standard decoder as the primary outcome and do not search many probes.
- In a small paired intervention after encoding, disable context-to-memory feedback only during the delay while preserving the cue controller and readout. A lost retention benefit supports a maintenance contribution, but this acute intervention can be out of distribution and is secondary to the trained-arm comparison.
- A small state perturbation tests recovery of decoded content without sensory re-presentation. Recovery toward a preferred wrong angle does not count as preserving the sample. Tangential noise cannot be magically corrected without additional information.

No extra architecture sweep is needed. A separate cortex-thalamus-style relay population is a reasonable later hypothesis if this fails, but would add both memory capacity and a new feedback route; it is not the first recommendation. The current proposal directly tests learned relevance-dependent dynamics in the memory system we have.
