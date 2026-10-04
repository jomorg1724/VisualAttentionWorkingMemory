# Morgan–Albanna–Herman Figure 5 versus the current joint-suite KDA

**Status:** source-grounded architecture comparison and proposed experiments only. No checkpoint inference, calibration, perturbation, training, or accelerator work was performed. These notes are not the final report or a shared citation ledger. References use source keys for the parent to register.

## 1. Sources and verification boundary

**M — pinned primary source:** Morgan, J., Albanna, B., & Herman, J. P. (2025), *A recurrent vision transformer shows signatures of primate visual attention*, arXiv:2502.10955v1. PDF: https://arxiv.org/pdf/2502.10955v1 ; HTML: https://arxiv.org/html/2502.10955v1 . Local PDF: `SecondPass/papers/pdf/Morgan_Albanna_Herman_2025_Recurrent_ViT_arxiv_2502.10955v1.pdf`. SHA-256: `946e1033a7f15d03ad36a6768cae559b2272b25fff7c4d7a82e0bee0b6724da9`. All page numbers below refer to this **23-page PDF**, with PDF page and printed page agreeing. Read its full extracted text; visually inspected Figure 5, including legends, from a render of p. 8. Saved extraction: `paper_evidence/morgan_v1_pages.json`; render: `paper_evidence/morgan_v1_page08.png`. HTML independently checked for §§4.3 and 6.7.3.

**B — one directly cited biological source, M reference 45:** Cavanaugh, J., Alvarez, B. D., & Wurtz, R. H. (2006), *Enhanced Performance with Brain Stimulation: Attentional Shift or Visual Cue?* J. Neurosci. 26(44):11347–11358. https://www.jneurosci.org/content/26/44/11347 ; https://doi.org/10.1523/JNEUROSCI.2376-06.2006 . Read primary full-text HTML, especially Materials and Methods / “SC stimulation,” “Task specifics,” “Performance measures,” Results / base experiment and Test 2, and Figure 2. Biological pointers below use exact HTML section/figure names, not unverified within-article page assignments.

**Repository sources:**

- **C1:** `PreAttentiveVision/TemporalIntegration/accumulators.py:19–66` (`kda_update`, `SpatialKDA`).
- **C2:** `WorkingMemory/PlainBaseline/accum.py:40–75` (`AccumulatorBaseline`).
- **C3:** `SecondPass/JointTraining/worker.py:33–38,51–60`; `continuation_v3.py:201–202` (current construction/trainability).
- **C4:** `WorkingMemory/SpatialTaskBattery/stimuli.py:69–91,121–142` (actual labels and timing).
- **C5:** `SecondPass/JointTraining/core.py:37–66` (existing behavioral metric definitions).

C1, C2 and C4 were byte-compared with `SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v3_8h/locked_source/`; all matched. Current Git HEAD was `ac351be944fc698e52578b64995f38b083b579f7`, but the worktree contains untracked joint-suite files, so HEAD alone is not a complete provenance identifier. The report brief was read. The requested report-directory SOP does not exist; the actual repository-root `ANALYSIS_SOP.md` was read instead. SOP was treated as guidance, not substituted for primary-source verification.

## 2. What the reference experiment actually does

### Task and reporting rule

M §2.1, pp. 2–3 / Figure 1: seven discrete timesteps, indexed 0–6; cue at t=1, blank at t=2, four Gabors appear at t=3, change at t=5 on change trials, and final frame at t=6. Half the trials are no-change. Cue locations are S1 or S4. The four visually encoded validity conditions are **25%, 50%, 75%, 100%**, meaning the probability that the change is at the cued site **conditional on a change trial**. At 25%, the four sites are equally probable. A response to an uncued change is a legitimate hit, not a distractor false alarm. The actor chooses wait/declare-change at each timestep; declaration terminates the trial. Successful declaration is rewarded at t≥5 on change trials. [M §2.1, pp. 2–3]

M §4.1, p. 4 explicitly says testing used a trained model “with fixed weights,” including uncued changes under a 100%-valid cue that were absent during training. Thus the 100%-cue/uncued-change comparison is an acknowledged counterfactual, not evidence that the training validity was lower. Figure 5 likewise contains that counterfactual condition. Do not silently reinterpret its 100% label as empirical validity of the plotted test subset. [M §4.1, p. 4; Figure 5C,F, p. 8]

### The manipulated variable and what normalization does—and does not—establish

The paper's model combines current visual features X with previous activated patch memory H. In its main multiplicative model:

\[
Q=(XW_{XQ})\odot(H_{t-1}W_{HQ}),\quad
K=(XW_{XK})\odot(H_{t-1}W_{HK}),\quad
V=(XW_{XV})\odot(H_{t-1}W_{HV}),
\]
\[
A=\operatorname{Softmax}_{\mathrm{source}}(QK^\top),\qquad
Z=X+AV.
\]

These are M §7.2, p. 14, Eqs. **11–15**; §6.7.3, p. 13, Eqs. **7–8** gives the equivalent indexed multiplicative feedback formulation. For each destination/query i, attention is normalized over source patches j. §6.5, p. 11 explicitly imposes \(\sum_j a_{ij}=1\), with positive weights. §6.5, p. 12 describes the winner-take-all limit: selected source approaches one and other source weights approach zero. This is **row-normalized competition**, not a single unit of mass over all entries of the entire attention matrix.

Figure 5, p. 8 describes a selected spatial bias \(\alpha_i^{(t_{change})}=1\), saying transmission Z is “completely biased toward” that site's internal representation ξ. **Architecture-level implication:** if the intervention preserves the stated row-stochastic normalization and forces a selected source weight to one in a row, other source weights in that row must be zero. **Implementation-level uncertainty:** the pinned paper does not provide a perturbation algorithm establishing exactly which rows/heads are overwritten, how its single-index plotted α is reduced from matrix αᵢⱼ, or an explicit post-overwrite renormalization step. Accordingly, report “forced spatial allocation within a normalized-attention architecture,” not “the authors demonstrably renormalized every other location using operation X.” Simply overwriting one coefficient and leaving the rest unchanged would not preserve the published normalization. No ±50 logit clamp is specified. [M §3, p. 3; §§6.5,6.7.3, pp. 11–13; Figure 5 caption, p. 8]

**Residual-path qualification:** Eqs. 8 and 15 retain X. Even ideal one-source AV routing does not mathematically erase all nonselected immediate sensory information in Z. “Completely biased” is the caption's description of routing; it must not be restated as total silencing of all other visual input. The paper's local LSTM states are independent across patches, and §7.3, p. 15 identifies self-attention as its cross-patch communication mechanism. This differs from the KDA conv hierarchy.

### Figure 5: exact panel conditions and supported qualitative outcomes

All panels fix cue location at **S1**. Each point averages **500 trials**. Black is “No Artificial Bias.” Crucially, **blue denotes stimulation toward the changing location, not a fixed anatomical location across panels**. [M Figure 5, p. 8, visually verified]

| Panels | Fixed cue condition | Change site | Blue intervention | Red intervention | Qualitative comparison with black |
|---|---|---|---|---|---|
| A,D | S1, 25% | S1 | α1=1 | α4=1 | Bias toward changing S1 increases responses at intermediate magnitudes and shortens trial-ending time; bias toward S4 markedly reduces responding and keeps times near the long end. |
| B,E | S1, 100% | S1 | α1=1 | α4=1 | Bias toward S1 adds little over the already strongly cued baseline; bias toward S4 strongly impairs responding and prolongs trial-ending time. |
| C,F | S1, 100% | S4 | **α4=1** | **α1=1** | Bias toward the uncued-but-changing S4 improves responding and shortens trial-ending time; bias toward the cued-but-unchanging S1 suppresses responses. |

These are visual qualitative readings, not digitized numerical estimates or significance claims. Most importantly, “stimulation at the cued site helps” is **not** a valid summary of this figure: in C,F it hurts detection of a valid uncued change under the any-change rule.

### Timing, duration, false positives, sensitivity, criterion, reaction time

- **Timing established:** §4.3, pp. 6–7 contrasts cue-time and change-time manipulation; Figure 5 labels the latter \(t_{change}\), corresponding to t=5 in §2.1. Cue-time corresponds to t=1. The authors report minimal cue-time effects and change-time effects on both sensitivity and criterion, depending on validity and change location. [M §4.3, p. 7]
- **Duration not fully specified:** the notation identifies a targeted discrete timestep, but no explicit pulse-duration protocol, millisecond conversion, or statement that the clamp persists through t=6 is supplied in the pinned document. Distinguish duration of imposed clamping from persistent consequences through recurrence. A one-update pulse is a clean proposed KDA design, **not a fully verified reproduction of source code**.
- **False positives:** the no-change/zero-magnitude endpoint and premature declarations matter. Figure 5 does not tabulate a separate false-positive effect or show that stimulation invariably leaves false positives unchanged. Do not import that claim from the biological study. Nor do increased responses at positive Δ alone prove enhanced sensitivity. [M §2.1, p. 3; Figure 5, p. 8]
- **d′/criterion evidence limitation:** §4.3 says both change, but refers to “Supplemental Figure X,” “Supplemental Figure A,D,” and “Supplemental Figure 16.” Those panels are **not included in this pinned 23-page PDF**, which ends with references. Quote the textual claim, not nonexistent accessible numerical plots. No direction or size of d′/criterion changes should be fabricated.
- **Reaction-time definition:** Figure 3 caption, p. 4 defines the mean of trial-ending times across all trials, **including waiting through the final timestep**. This is not a mean latency conditional on a correct hit. Figure 5 uses the same response/time presentation without a separate formula; its time axes visually run about 6–7 although §2.1 indexes frames 0–6. Do not silently assign physical units or resolve the apparent indexing offset without source code. Present these as the paper's reported trial-ending-time curves.
- **Other source inconsistencies:** Figure 5 references “Figure 4G,” but Figure 4 has A–D; §4.2, p. 6 once calls t=5 stimulus onset, conflicting with §2.1 and the same paragraph's correct t=3 onset. Use the consistent task timeline, and mark the source's unfinished cross-references rather than inventing supplemental contents.

## 3. Current KDA: the actual substrate, not an old E/I model

C2/C3 construct **all-trainable** `AccumulatorBaseline(stack=3, center=True, accumulator='kda')`, trained with supervised cross-entropy and a shared GRU decoder plus task-specific heads. Do not confuse this with `StreamingPAVClassifier` later in C1's file, which freezes its encoder and uses different spatial scales/readout. No explicit excitatory/inhibitory neuronal populations, firing-rate constraints, electrode model, or biological-current units are present in the current KDA.

For 100×100 RGB inputs, the conv backbone has widths 32/64/96/128. Three projected 32-channel spatial accumulators sit at 25×25, 13×13, and 7×7. Each emitted field is concatenated with the current conv features and fed onward; the deepest 160-channel field is flattened to a 256-dimensional feature, processed across time by the shared 256-dimensional GRU, and classified only from its terminal hidden state. [C2:43–51,59–75]

At each scale, spatial site and head, C1 implements a signed key×value matrix \(S\in\mathbb R^{8\times16}\), two heads, with:

\[
D_t=\operatorname{diag}(\alpha_t),\quad \widetilde S_t=D_tS_{t-1},\quad
e_t=v_t-k_t^\top\widetilde S_t,
\]
\[
S_t=\widetilde S_t+\beta_t k_t e_t^\top,\qquad o_t=q_t^\top S_t.
\]

Queries/keys are L2-normalized learned signed vectors; values and states are signed. α is an eight-component sigmoid **row-retention gate**, β a scalar sigmoid **delta-write gate** per head/site. A 3×3 convolution on the current projected field produces q,k,v,α,β; the head outputs are concatenated and passed through a learned 1×1 output projection. Initial α=.9 and β=.5 are trainable initialization, not measured final gate values. [C1:19–66]

**Same letter, different mechanism:** M's α is a source-patch routing probability; KDA's α is local temporal retention. Setting KDA α=1 does **not** allocate all spatial attention to that site. There is no softmax across spatial sites in this update and no conserved spatial allocation budget. β increases the residual correction, which may subtract old content as well as write new content; it is not a monotonic “excitation” control.

Direct same-scale state dynamics are site-local, but the **whole network is not spatially isolated**. Convolutions, overlapping receptive fields, downstream normalization, lower-scale emitted fields entering deeper scales, flattening and the shared GRU propagate or combine information. q/k/gates do not directly receive the same core's old S; deeper-scale current inputs can nevertheless contain lower-scale memory-dependent output. The shared GRU does not feed back into the conv stack in C2. These distinctions prevent falsely equating KDA recurrence with M's explicit H→Q,K,V multiplicative feedback.

Three-frame stacking supplies previous sensory frames in early nominal blanks. Specifically, the first two blank updates can still contain a preceding sample; retention-specific interventions should also include a window after those frames have flushed. A local KDA intervention leaves the shared GRU as another potential memory pathway. Failure of a local lesion is not proof that the perturbed location contained no useful information. [C2:54–58,69–75]

## 4. Scientifically comparable intervention families — proposals, not results

Use one fixed checkpoint and paired physical scenes across sham/perturbed conditions. Preregister layer, head, spatial mask M, pre/post-update placement, pulse duration, dose and channel direction. Stimulate target, matched non-target, and off-target regions; retain signed arrays and achieved perturbation norms. Calibrate doses/directions on an independent split **only after separate authorization**. Local perturbation of a signed vector means a computational change of representation, not identified neuronal excitation.

| Family | Explicit candidate operation | Scientific question and comparability limit |
|---|---|---|
| **Emitted-field gain / directional pulse** | After C1's output projection, before C2 concatenation: \(O'_t=(1+\lambda M)O_t\), or \(O'_t=O_t+\lambda M\sigma_O d\), with independently chosen unit direction d and calibration scale σO. | Tests causal use of local transmitted memory-dependent features; closest existing-interface test of increased local influence. Gain enlarges both positive and negative components. Does not directly edit that core's stored S, although downstream memory can change. Not normalized spatial competition. |
| **Persistent-state pulse** | At a declared point, \(S'_t=S_t+\lambda M D\), with calibrated matrix direction D, optionally a rank-one key/value pattern. Recompute the readout if immediate same-step influence is intended. | Tests memory-content causality; can persist into later updates. A post-read edit alone influences later reads, whereas a pre-read edit can act immediately. Do not call arbitrary all-positive matrix addition a feature-neutral attention increase. |
| **Retention/write gate modulation** | Locally shift α or β logits by λM, separately. Use explicit limiting controls only with declared semantics. | Tests retention versus evidence updating, not attention routing. α=1 removes row decay but leaves delta writing; β=0 removes the corrective write but leaves decay; α=0 discards previous state before a possible new write. Any behavioral benefit or harm is hypothesis-dependent. |
| **Explicit competitive routing control** | Add a documented analysis-only gain operator: over N specified sites, \(w_r(\lambda)=\exp(\lambda M_r)/\sum_u\exp(\lambda M_u)\), \(O'_r=Nw_r O_r\). At λ=0 it is identity. | Separates local gain from gain with compensating suppression elsewhere. It introduces competition absent from native KDA; it is a new intervention operator, not discovery of existing attention. Its limiting mask redistributes gain over selected cells; it is not M's all-query source-value broadcast and does not eliminate the H conv bypass. Large gain can be out of distribution. |
| **Suppression companion** | \(O'=(1-\gamma M)O\) or \(S'=(1-\gamma M)S\), 0≤γ≤1, with the reference state explicitly zero. | Tests localized necessity; state suppression and output suppression answer different questions. This is an additional proposed study, not a verified separate inhibition experiment in M. A global reset is not the spatial control. |

Start with sham and graded emitted-field gain/suppression plus matched-norm directional controls; treat state/gate interventions as mechanistic follow-ups rather than interchangeable “stimulation.” Include opposite-sign d, shuffled/channel-rotated d, and matched off-target masks to distinguish representational content, location, nonspecific norm changes and damage. An image-space flash is a distinct sensory-cue control, not internal stimulation. Report whether the pulse is one update or sustained; compare cue encoding, sample encoding, genuinely blank retention, first change/probe update, and later readout. Do not infer biological milliseconds from KDA update counts.

## 5. Report semantics and expectations

**Signed orientation:** `orientation_cued` is positive iff the target rotates in the cued sign; negative includes target zero **and opposite-sign rotation**. All scenes contain aligned, opposite and unchanged locations. Plot P(report-positive) against **signed, cue-relative target rotation**, conditioned on foil rotations. “No-change false alarm” alone is the wrong label for its entire negative class. Moving/removing its cue can change the report instruction itself; classical valid/neutral/invalid tests need a separate, fixed report-target/sign instruction, not silent relabeling. [C4:74–91]

**Krauzlis-style motion:** positive means cued-target change; foil-only changes and catches are negative. Existing metrics already distinguish target hit rate, foil false-alarm rate and catch false-positive rate. Unlike M, improving detection/reporting of an uncued foil can be **worse performance**. The task has a fixed end-of-trial report, despite stimulus frame-clock metadata; there is no trained wait/declare policy or measured reaction time. [C4:121–142; C5:58–66; C2:69–75]

**Conditional hypotheses, not promised outcomes:**

1. If an intervention selectively amplifies task-relevant evidence in a competent observer, target-site gain could increase target evidence use; if it instead adds response-aligned content, it could shift criterion without improving discriminability. Both can occur together.
2. Foil-site gain could increase foil-induced false reports under target-only semantics, while the analogous gain at an uncued changed site can improve M's any-change detection. Conversely, foil suppression could reduce interference, but may also remove contextual evidence or disrupt normalization; improvement is not guaranteed.
3. Change/probe-time effects larger than cue-time effects would resemble M's timing dissociation. This is **not an architecture-derived prediction**: encoding/retention interventions in KDA can matter, and repeated cues/frame stacking differ from M. M's near-saturated cue-time allocation is also a plausible ceiling limitation, not proof of cue-time irrelevance.
4. State retention or write changes can help or hurt depending on stale versus useful memory and the signed delta error. A zero or reversed effect need not refute attention generally; a null can reflect insufficient baseline competence, ineffective perturbation, another memory pathway or the wrong feature direction.
5. M §4.4, p. 7 / Table 1, p. 9 reports task performance without comparable cueing in some supervised alternatives. That motivates separating competence from cueing; it does not establish that **all** supervised KDA observers must lack attention, nor that RL is universally necessary.

For binary comparisons, estimate d′=Φ⁻¹(H)−Φ⁻¹(F) and c=−[Φ⁻¹(H)+Φ⁻¹(F)]/2 only after declaring the signal/noise classes, aggregation weights and finite-rate corrections. For Krauzlis, distinguish target-versus-catch from target-versus-foil comparisons rather than concealing their difference in a pooled F. For signed orientation, disclose that discriminability is sign-relative target classification, not the paper's unsigned any-change detection. Report hits and each false-report category regardless of d′. Use paired trial-level uncertainty and distinguish trial uncertainty from checkpoint-to-checkpoint variability.

**Unsupported now:** equal effect signs/magnitudes across architectures, a guaranteed cue-time null, sensitivity-only enhancement, unchanged false positives, improved accuracy from stimulation, neuroanatomical homology, a biological current dose, spatial softmax equivalence, reaction-time effects, or any already-measured causal result for the current joint-suite model. Preliminary accuracy/AUC can establish only the particular tested competence, not those causal signatures. No preliminary metric values were extracted in this source-comparison task.

## 6. Biological anchor: what is, and is not, being emulated

M itself reports **in-silico perturbations**, comparing them with previous primate interventions; it is not a new electrode experiment. Its specific cue/change timing analogy cites B (reference 45), not only the FEF paper by Moore and Armstrong. [M §4.3, pp. 6–7; §5.4, p. 9]

B used real SC electrodes in macaques. After locating the movement field with 200-Hz stimulation, investigators used **70 Hz**, typically **15 or 20 μA**, without evoking saccades; these are biological experimental parameters, not an α=1 conversion. Change-time stimulation began **150 ms before the end of the first motion stage** and lasted **600 ms**; a premotion timing condition also used 600 ms. [B Materials and Methods / “SC stimulation,” “Task specifics”; Figure 2A]

B's base experiment reports mean hit improvement of 15.2% and mean false-positive change of −0.5% (latter not significant), while cue replacement could increase both hits and false positives; premotion SC stimulation failed to provide the comparable benefit. Importantly, only sites showing a positive base effect were included (8 of 10), and its performance measures retained false positives to guard against direct motor effects. These specific biological results should **not** be transferred to M or KDA as predictions. B's target/distractor and saccadic-response protocol also differs from M and from the repository. [B Methods / “SC stimulation,” “Performance measures”; Results / base experiment, Tests 1–2; Figures 2–4]

## 7. Short verbatim evidence for report citation assembly

Quotes are deliberately short; complete context is in the exact sources above. `paper_evidence/verified_quotes.json` records whitespace-normalized literal-match verification against retrieved HTML, with PDF page pointers for M.

| Evidence key | Exact reference | Verbatim excerpt |
|---|---|---|
| M-task | §2.1, p. 3 | “Declaring a change always ended the trial.” |
| M-fixed | §4.1, p. 4 | “Using the trained model with fixed weights” |
| M-time | Figure 3 caption, p. 4 | “either by the agent declaring a change or waiting through the final timestep” |
| M-pulse | Figure 5 caption, p. 8 | “Artificial modulation involves inducing a high bias in a single spatial region” |
| M-n | Figure 5 caption, p. 8 | “All data points are the result of an average over 500 trials.” |
| M-cue | §4.3, p. 7 | “manipulating attention at the time of cue presentation had minimal effects” |
| M-sdt | §4.3, p. 7 | “complex, interrelated effects on both sensitivity and criterion” |
| M-normalization | §6.5, p. 11 | “To ensure a proper probability-like weighting” |
| M-feedback | §6.7.3, p. 13 | “the top-down feedback pathway multiplicatively gates the bottom-up signals” |
| B-duration | Figure 2A caption | “Stimulation began 150 ms before the first stage of dot motion ended and lasted a total of 600 ms.” |
| B-fp | Results / base experiment | “increase the proportion of hits while leaving the proportion of false positives relatively unchanged” |
| B-motor | Methods / “Performance measures” | “it was critical to keep track of false positives explicitly, to rule out a direct motor effect of SC stimulation” |

**Recommended report wording:** “We propose localized, timed perturbations of KDA transmission and memory to test the functional site-by-epoch logic of Morgan et al.'s artificial-bias experiment. Native KDA gates do not implement the paper's normalized spatial routing, and its signed states are not neuronal firing rates. Comparability therefore concerns matched behavioral consequences under explicit report rules, not identical intervention variables or guaranteed biological effect signs.”
