# Can value act on RViT's per-location code? An existence proof (rev 4, 2026-11-06)

**Why.** A central NMI claim is [[value-acts-on-readout]]: in the trained original RViT, value changes how a location's
evidence is read out, not whether the change is encoded (SENS-C: AUC 0.899 vs 0.899, pooled-decoder separation 1.85 vs
1.82, readout slope 3.76 vs 0.86; Baruni: encoding equal across values; which-loss: value reaches encoding only at zero
reward, via an attention shift). The anticipated reviewer objection: *"the architecture forces this; it is a limitation,
not a finding about learning."* Herman approved an existence proof on 4 Nov ("Please do!"): show that the same
architecture **can** carry value in the per-location representation, like cortical gain modulation, with an L&M-like
value effect, and test whether training keeps or abandons that route.

**History.** rev 1 drafted. rev 2 after an independent review (`RVIT_EXISTENCE_PROOF_PLAN_review.md`, 15 findings,
"revise before running"); responses in §10. rev 3 (5 Nov, §11): Part B as planned is moot (FROZ has no value effect),
so the release test moves to the hand-built solution (Part D), and FROZ is extended (CONT). rev 4 (6 Nov, §12):
Part E, what erodes the hand-built route in Part D (follow-up approved in spirit by Herman, 6 Nov).

**Scope.** [[rvit-original]] only (scope decision 9), L&M sensitivity task, SENS-C reward table. Planned analyses, revised
openly; exploratory, not a pre-registration. A supporting result; a few days.

## 1. What the architecture allows (from the code)

`VWMNETNOMOD.py` (model-0 package; `zk/orig_hooks.py`). Per frame t and location i:

```
x_i  = LN2(s_i + fc2(elu(s_i + LN(fc(s_i)))))       s_i = [VAE(patch_i) (128), position one-hot (4), time one-hot (8)]
q,k,v_i = W_{q,k,v} x_i ⊙ W_C{q,k,v} H_i(t-1)       a_i = Σ_j softmax(q_i·k_j/√d) v_j
H_i, C_i = xLSTM(u_i = norm1(x_i + a_i), state_i)    actor, critic read concat(H_0..H_3)
```

1. **Every per-location weight is shared across locations.** A location-specific (in L&M, value-specific) change of the
   code can only come through the **position one-hot**, through **attention** (the only channel between locations), or
   through the location's own history. An encoding-level value mechanism in RViT is therefore a *position-gated gain
   field* or an *attention reallocation*.
2. **Two normalizations cancel the simplest gains.** LayerNorm removes uniform scaling of a location's input. The xLSTM
   normalizer (H = O·C/N, with C and N both sums of I·ΠF terms from zero) removes uniform scaling of the input gate. A
   gain must act on the output gate, the forget-vs-input balance, the q/k/v products or attention.
3. **Gain on H is behaviorally equivalent to a readout gain** (review #5). The network is deterministic given the
   input; the within-class variability that sets a decoder's AUC comes from the stimulus and scales with the signal.
   The only noise downstream of H is the policy's sampling. A gain on H therefore lowers d′ by the same algebra as a gain
   on the actor's first layer, and the tested-slot AUC cannot see it. Only an amplitude measured on a common axis
   (pooled decoder, P9) can. Lowering *information* needs a nonlinear compression or a change of what is routed into
   H_i. This bears on the manuscript claim itself: SENS-C's equal AUC rules out information loss, and its equal pooled
   separation (1.85 vs 1.82) rules out an amplitude gain on the decoder axis. This experiment checks whether those
   measures would have caught a gain-type value mechanism if there had been one.
4. **Information-level value is already known to be expressible, at abandonment.** The which-loss positive control
   (low-location rewards × 0) costs 15% of neural d′ (+0.154 [+0.113, +0.195]), with attention moving away (+0.38),
   but FA is 0.91. The open questions are narrower: (a) at a graded value effect, can value sit in the code, in which
   form; (b) does the training objective keep it there, or move it to the readout?

## 2. Method

**Options weighed.**

| option | shows | weakness |
|---|---|---|
| Hand-written network | expressibility | a hand-built change detector in a 1024-d multiplicative xLSTM is its own project; "not the model you trained" |
| Targeted hand edit of trained weights (Part C) | expressibility in the model's own weights, transparent, cheap | any location lesion "implements value" in L&M; says nothing about learning |
| Tied actor, from scratch | learnability | the actor can still read location from H and gate it nonlinearly |
| **Frozen readout, fine-tuned under the real objective** (Part A) | the optimizer must put value into H; same forward pass, losses, rewards | a learning constraint; the form is not forced, and controls are needed (§3) |
| **Reversed-value fine-tune, unconstrained** (Part A, FREE-REV) | where the unconstrained learner puts a *new* value asymmetry | — |
| **Release from the frozen solution** (Part B) | whether the objective keeps or abandons an encoding route | needs a perturbation to tell "kept" from "not pushed" (review #3) |

Both parts start from the same models. **Part C** (hand edits) runs first because it is cheap, and gives the
expressibility curve directly.

**Start models: reward ×10 (`whichloss/rscaled10`, 10 counterphased pairs, s42–51, 200k).** SENS-C trained from
scratch with every reward ×10: no behavioral value effect (DiD d′ −0.05; d′ 2.65 / 2.70), equal encoding (AUC 0.906 /
0.904), DiD c −0.37. Switching to the SENS-C table (every reward ×0.1, ratio unchanged) lowers the low location's
absolute reward below the entropy temperature ([[lm-exchange-rate-temperature]]) and produces the value effect in a
free learner.

**The reader asymmetry (review #1).** The start actor reads the old low-value location with a smaller dv gap (5.12 vs
8.96, saturated, so behavior is equal). A location-*symmetric* shrink of the change signal in H would desaturate the
low location first and give DiD d′ > 0 under a frozen reader. Three things handle this:
- **Reversed arms (-REV):** the same start models with `--high-loc` flipped. The reader's strong slot becomes the
  low-value one, so a symmetric H change gives DiD ≤ 0. A positive DiD in FROZ-REV needs a location-specific change in
  H that beats the reader. The congruent/reversed average cancels the reader's asymmetry.
- **Procedure null (FROZ-NULL):** frozen actor, start table kept (×10). It measures what 50k constrained steps, a new
  critic head and a restarted buffer do with no value change.
- **A location-specific representational signature** is required (§4), not just behavior.

## 3. Arms

All [[rvit-original]], zk tree, `zk/ep_train.py` wrapping `zk/train_orig.py` unchanged. Common: `luo2015_sensitivity`,
SENS-C table (value ratio 25, K 0.25, H:CR 1.0 / 0.44, mean 3) unless stated, 1,201 particles, γ 1, no fixation, entropy
0.02, seed = the start model's seed. **In every arm:** the same two-group transformer optimizer with **q/k weight decay
0** (review #8: avoids the CRIT-DET collapse under weak q/k gradients; in intact models decay is ≈ 0 of the update), and
**θ frozen** at the start value (`--curr-threshold 1.01`; review #10). Run names keep the *start* model's name
(`loc<hl>_s<seed>`); in -REV arms the high-value location is 3 − hl.

**Part A (stage 2, 10 pairs each, 50k, snapshots 10k / 25k):**

| arm | warm-up (5k, new 1,201-particle critic head, critic only) | learner |
|---|---|---|
| FREE | W (SENS-C) | standard: the unconstrained answer |
| FREE-REV | W-REV (SENS-C, reversed) | standard, reversed value: must undo a readout asymmetry and build the opposite one |
| FROZ | W | actor frozen (no step; gradients pass through it into H) |
| FROZ-REV | W-REV | actor frozen, reversed value |
| FROZ-NULL | W-NULL (start table ×10) | actor frozen, no value change |

FROZ-CD (critic detached) from rev 1 is dropped (review #8: confounded; little gain).

**Part B (stage 3, from FROZ and FROZ-REV final, only where Part A shows a value effect):**

| arm | pairs | what |
|---|---|---|
| REL | 10 | release: standard learner, same table |
| REL-K | 10 | release *and* a step in the low location's reward (K 0.25 → 0.125): the learner must re-adjust the value effect, and the locus of the *new* adjustment shows its preference (review #3c) |
| CONT | 5 | constraint kept (drift reference for the restart) |

Horizon: max(50k, 3 × the episodes FREE needs to reach 80% of its DiD d′); snapshots 5k / 10k / 25k / end.

**Part C (hand edits, no training; first).** On the 20 W models, at the low-value location: a rank-1, position-gated edit
of the LSTM's input weights along location L's position direction in u-space (direction = mean u at L minus mean u
elsewhere; bias offset so other slots are unchanged on average), swept over strength k (multiples of the pre-activation
SD at L):
- **C-F:** forget-gate pre-activation +k·SD: more weight on the stored past, so the test is diluted. Tier-I candidate.
- **C-O:** output-gate pre-activation −k·SD: a gain on L's memory output. Tier-II (gain) candidate, for contrast.

Measured at each k: 18° assay (d′, c at both locations), tested-slot and whole-state AUC, pooled separation. It gives
ℓ_N vs ℓ_beh curves: is tier I expressible at a graded behavioral effect? Expressibility only; nothing is learned.

**Compute.** Part A ≈ 100 fine-tunes of 50k (≈ 6 h each, three per L40S) + 30 short warm-ups; Part B ≤ 100; Part C a
few GPU-hours. ≈ 350 GPU-h (≈ 2,800 SU) at most. CRC `-M gpu`, l40s, `--exclude=gpu-n72`, requeue-safe.

## 4. Definitions

**Value effect (behavior).** 18° `lm_assay` (2,000 trials per location); [[counterphased-did]]: DiD d′ = d′_high − d′_low
(and DiD c), averaged within a pair. Present if the DiD d′ 95% lower bound is > 0 and the estimate ≥ 0.3. Every arm is
read against its null where one exists (FROZ against FROZ-NULL). **L&M-like:** the change of DiD c from the warm-up lies
within ±0.2 (90% CI) (review #7), also benchmarked against FREE; if FREE itself is not L&M-like, arms are judged by
locus only. Runs must have learned (HR ≥ 0.10, FA ≤ 0.90 at both locations). High-location d′ change from W is reported
in every arm.

**Value in the representation: tiers** (end state, tested slot at t3, the only primary frame; t4 secondary):

| tier | name | test (pair level) |
|---|---|---|
| I | **information** | neural proportional loss ℓ_N = 1 − dN_low/dN_high (twin-grouped AUC on H at the tested slot, dN = √2·Φ⁻¹(AUC)), 95% lower bound > 0.077 (the which-loss minimum effect), also against the null arm; ℓ_N/ℓ_beh reported as a continuous quantity |
| II-gain | **amplitude** | not I; pooled-decoder separation loss ℓ_P = 1 − pooled_dn_low/pooled_dn_high, lower bound > 0.077, *and* the change direction kept (cosine between the end-state and W class-mean differences at L ≥ 0.9) |
| II-rot | **rotation / routing** | not I; the change direction at L rotated away from the reader (cosine < 0.9, or the frozen reader's projection J_W·Δμ falls more than the amplitude) |
| III | **offset / other** | value effect present, none of the above |

Only II-gain is the analog of cortical gain modulation; tier I is the analog of value changing what is encoded.
Additionally: **whole-state AUC** (change decoded from concat(H_0..H_3), per tested location; review #6b): the
information available to the reader, which can differ from the tested slot when attention moves evidence.

**Where the value lives (every arm).**
1. **2×2 hybrid decomposition:** behavior (18° assay, DiD d′) of actor ∈ {ref, end} × transformer ∈ {ref, end};
   Δ_rep, Δ_ro (Shapley averages), rep share, the **interaction** (E,E) − (W,E) − (E,W) + (W,W), and per-location d′ of
   each hybrid. Uninterpretable if |interaction| > ½ the smaller main term (review #9).
2. **On-manifold first-order decomposition** (probe trials): dv gap at L ≈ J·Δμ (J = mean actor Jacobian ∂dv/∂H,
   Δμ = class-mean difference of the full state). Δgap = (J_E − J_W)·Δμ̄ + J̄·(Δμ_E − Δμ_W): readout vs representation.
   Also |Δμ_L| ratio (end/W) and its cosine, per location (the II-gain vs II-rot split).
3. **P9 specificity check:** if a *frozen* actor yields a lower P9 "readout slope" at the low location, then P9 cannot
   localize the locus, and the manuscript's "readout gain 4×" wording must be softened to the joint result
   (encoding and pooled amplitude equal; slope lower).

**Trivial routes (excluded or flagged).**
1. *Input degradation* (pixels, VAE, environment): excluded; nothing outside the network changes.
2. *Input-stage learning* (token projection fc / fc2 and their LNs, before memory or attention): closer to a retinal or
   thalamic gain. A module transplant (§5) says which stage carries the effect; "input-stage only" is flagged.
3. *Symmetric degradation read by an asymmetric reader:* handled by -REV and FROZ-NULL (§2).
4. *Abandonment:* excluded by the learned-run rule.

## 5. Measures (per run; the warm-up measured as the baseline)

- **Behavior:** `zk.lm_assay` (18° at snapshots, all thetas at the end), `zk.lm_perturb` (18°, end).
- **Encoding / readout:** `zk.lm_readout_probe` unchanged (600 twin pairs): tested-slot AUC, pooled decoder, P9 slope /
  intercept, attention by location (P10).
- **`zk.ep_geom`** (new): whole-state AUC, actor Jacobian decomposition, |Δμ| and cosine, against the warm-up.
- **`zk.ep_hybrid --mode cross`:** the two cross networks, 18° assay; ref = warm-up (and the stage-3 start in Part B).
- **`zk.ep_hybrid --mode transplant`** (secondary, frozen arms): probe-level d′ with one module group from the end
  state grafted into the warm-up network, or reverted (G1 token projection; G2 q/k; G3 v; G4 norm1 + LSTM);
  "revert all" must reproduce the warm-up exactly (check).
- **Checks:** actor bit-identical to the warm-up in frozen arms; `zk/tests/test_ep_train.py` (two-group Adam ≡ legacy,
  including converted state; frozen actor passes the same transformer gradient); `resume_count` logged (each preemption
  empties the buffer).

## 6. Predictions (before any Part A data)

| arm | behavior | representation | confidence |
|---|---|---|---|
| FREE | a value effect (DiD d′ +0.5 to +1.0); criterion may move liberal, as in the dose-response fine-tunes | ℓ_N ≈ 0, ℓ_P ≈ 0, rep share < 0.2 | value locus high; c low |
| FREE-REV | a reversed value effect | the actor re-learns it; rep share < 0.2 | moderate |
| FROZ-NULL | no DiD change | no change in ℓ_N, ℓ_P | high |
| FROZ | a value effect, smaller than FREE | II-gain or II-rot; tier I unlikely (the critic's KL keeps information) | low |
| FROZ-REV | smaller than FROZ, possibly none | if present, location-specific (it must beat the reader) | low |
| REL / REL-K | value effect kept | REL: little movement (nothing pushes, review #3); REL-K: the new adjustment goes mostly to the actor | moderate |
| Part C | C-F lowers ℓ_N and d′ together (tier I expressible at graded d′); C-O lowers d′ with ℓ_N ≈ 0 and ℓ_P > 0 | | moderate |

## 7. Statistics and decision rules

- Unit = counterphased pair (n = 10; 5 for CONT). t-intervals; ratios by pair bootstrap (20,000) with Fieller as a
  check; arm contrasts by Welch. Intervals throughout, no multiplicity correction (an existence claim needs one arm);
  the decisive arm, frame and tier are named. Paired sign-flip p for sign-consistency claims.
- **Existence proof (per frozen arm, end state, against FROZ-NULL):**
  - **strong:** tier I or II-gain with a value effect, in FROZ-REV (beats the reader) or in both FROZ and FROZ-REV;
  - **moderate:** the same in FROZ only (reader asymmetry not excluded), or tier II-rot in FROZ-REV;
  - **weak:** tier III, or input-stage only;
  - **not shown:** no value effect beyond FROZ-NULL.
  L&M-likeness is reported alongside and does not gate the existence verdict.
- **Preference of the unconstrained learner:** FREE / FREE-REV rep share (hybrid and Jacobian), with CI. "Readout
  preferred" if the upper bound < 0.3.
- **Keep vs abandon (Part B):** retention relative to the FROZ end state: Δ_REL = measure(REL end) − measure(FROZ end),
  with CONT as the drift reference, on the defining measure (ℓ_N, ℓ_P or the Jacobian representation term) and on the
  hybrid rep share relative to the FROZ start. Absolute differences (not ratios) where the measure is < 0.05.
  **REL-K is the decisive test:** the share of the *new* adjustment (REL-K vs REL) carried by the actor vs H (hybrid and
  Jacobian decomposition relative to the FROZ end). "Readout preferred" if the actor's share ≥ 0.7 (lower bound ≥ 0.5);
  "representation preferred" if ≤ 0.3; else "both".
- **Readings.** If the frozen arms show value in the code and REL-K's new adjustment goes to the actor, the architecture
  can carry value in the code, but the objective puts new value in the readout. If REL-K keeps adjusting H, both routes
  are live and history selects ([[training-history-selects-mechanism]]). Either way "the architecture forces it" is
  rejected. If no arm shows value in the code (and Part C fails), the result is reported as such.

## 8. What a reviewer could still say

- "Freezing the actor is not how the agent learns." Part A's frozen arms are expressibility tests under the real
  objective; FREE-REV and REL-K are the learning tests.
- "Fine-tunes, not from scratch." The start models are from scratch; FREE shows the fine-tune reproduces the readout
  locus. A from-scratch, in-distribution steep-but-graded test is the running low-location reward sweep (which-loss,
  CRC 4660210, ×0.05 / ×0.1); read it before Part B.
- "Linear decodability is not what the brain uses." Information, amplitude, direction and use (hybrids, Jacobians) are
  reported separately; §1.3 states what each can and cannot see.
- "L&M only." In the Baruni analog ([[2026-10-baruni-analogue]]) value is *cued in the input*, so a cue-gated gain route
  was available to the unconstrained learner and was not used (encoding equal across values). That is a second,
  independent rebuttal. A dynamic, cue-gated existence proof in Baruni is a follow-up, not run here.

## 9. Outputs

- Code (new files only; shared modules untouched): `tree_zk/zk/ep_train.py`, `ep_hybrid.py`, `ep_geom.py`, `ep_edit.py`,
  `ep_stats.py`, `ep_figs.py`, `zk/tests/test_ep_train.py`; slurm `ep_*.slurm`, `run_ep.sh`, tables (local mirror
  `experiments/zenon-krauzlis/code/`).
- Runs: CRC `rvit_zk/runs/existence/<arm>/loc{0,3}_s{42..51}`.
- Results: `analyses/2026-11-04_rvit-existence-proof/` (README, `stats/*.json`, figures PNG + PDF).
- Wiki: experiment page `2026-11-rvit-existence-proof`; update [[value-acts-on-readout]] and scope-decisions entry 15.

## 10. Responses to the review

| # | finding | response |
|---|---|---|
| 1 | start-actor asymmetry confound | FROZ-REV and FREE-REV added; congruent/reversed comparison; location-specific signatures (Jacobian projection, cosine) required |
| 2 | no procedure null | FROZ-NULL (frozen, start table kept). Not FROZ-EQ: a ratio-1 table changes both locations' rewards, which is itself a value manipulation |
| 3 | kept vs not pushed | Part B redefined: retention against the FROZ end with CONT as drift reference; REL-K (release + reward step) is the decisive preference test; horizon from FREE's time course |
| 4 | FREE-REV missing | added |
| 5 | tier II ≡ actor gain | stated in §1.3; tier II split into gain and rotation; P9 specificity check made explicit |
| 6 | tier I bar unreachable; tested slot only | bar = lower bound > 0.077; ℓ_N/ℓ_beh continuous; whole-state AUC added |
| 7 | FREE criterion | prediction lowered; L&M-likeness = change from the warm-up, benchmarked against FREE; does not gate existence |
| 8 | FROZ-CD confounded | dropped; q/k decay 0 in every arm |
| 9 | hybrids off-manifold | interaction term and per-location hybrid d′; on-manifold Jacobian decomposition added |
| 10 | live curriculum | θ frozen (`--curr-threshold 1.01`) |
| 11 | extension rule | dropped; Part B horizon set from FREE instead; `resume_count` reported |
| 12 | code checks | tests added; route override only on `--init-from`; parameter-count assert; "revert all" check |
| 13 | Part C wrong tier, run first | C-F (forget gate, tier-I candidate) added beside C-O; Part C runs first, on CRC (the lab GPU is reserved) |
| 14 | sequencing; Baruni | read the K sweep before Part B; Baruni in §8 |
| 15 | statistics | t3 the only primary frame; decisive arm named; CONT at 5 pairs |

## 11. rev 3 addendum (2026-11-05): Part B replaced by Part D

**What happened.** Part C ran first (20 models). A first pass used a position direction that leaked about 0.75 of the
offset onto the other stimulus slot (it encoded stimulus-vs-blank); it was discarded as not location-specific (kept in
`partC/`) and replaced by a direction on u's four position coordinates, exact on all four slot means (`partC2/`).
In Part A, FROZ (15/20 runs by 5 Nov 00:00) shows no value effect at 50k: DiD d′ +0.04 [−0.05, +0.13] (6 pairs) against
the warm-up's −0.05, with every representation measure unchanged. Part B's precondition (a value effect in a frozen arm,
§3) is therefore not met, and REL / REL-K from FROZ cannot test anything.

**Part D (replaces Part B): does the standard learner keep a hand-built encoding route?** This is the original brief's
"fine-tune from the constructed weights" test, and it no longer depends on the frozen arms.
- **EDIT1 / EDIT2:** the Part-C output-gate edit (C-O, k = 1 and k = 2) written into each W checkpoint at the SENS-C
  low-value location (`zk/ep_make_edit.py`; the rest of the training state unchanged). In Part C these were the
  L&M-like amplitude case (k = 1: d′ at L 2.70 → 2.16, other location unchanged, c unchanged, information intact,
  pooled amplitude −46%) and the information case (k = 2: d′ at L 0.89, tested-slot information −15%).
- **EDREL1 / EDREL2** (10 pairs each, 50k, snapshots 5k / 10k / 25k): the standard learner (actor trained, critic
  attached, the common settings of §3) on the SENS-C table, from EDIT1 / EDIT2. Reference arm: FREE (same learner and
  table, from the unedited W).
- **CONT** (5 pairs): FROZ continued frozen for 50k more (now an extension of FROZ: does a value effect appear slowly?).

**Measures.** As §5; the hybrid and Jacobian decompositions against W *and* against the edit start.

**Decision rules (per EDREL arm, end vs its EDIT start).** Retained fraction r of the edit's defining representation
measure (EDREL1: ℓ_P, the amplitude loss; EDREL2: ℓ_N and ℓ_P), r = mean(end) / mean(start), 90% pair-bootstrap CI:
**kept** if r ≥ 0.7 with lower bound ≥ 0.5; **abandoned** if r ≤ 0.3 with upper bound ≤ 0.5; else **partial**. Also: is the
behavioral value effect kept, and where does it sit at the end (hybrid representation share against W; EDREL vs FREE)?
- *Kept* → the constructed encoding route is stable under the RL objective (the learner does not undo it), and the
  readout locus of trained models reflects the path learning takes, not an architectural limit.
- *Abandoned, with the value effect rebuilt in the actor* → the objective actively prefers the readout.

**Predictions (before Part D data).** EDREL1: the amplitude loss is largely kept (the critic's magnitude-free KL can
re-scale its own reading, so nothing restores amplitude; review #3), while the actor adds its own value effect, so DiD d′
ends above EDIT1's. EDREL2: the information loss is partly restored (the critic needs the low location's outcomes) and
the amplitude loss partly kept. Confidence low.

**Responses to the addendum review (`RVIT_EXISTENCE_PROOF_PLAN_review.md`, "Addendum review").**
1. *"Kept" is the default under SENS-C.* Added **EDIT1H / EDREL1H** (10 pairs): the same k = 1 edit at the **high**-value
   location, where a d′ loss costs about 25× more reward. "Kept" in EDREL1 is read as value-consistent retention only if
   EDREL1H restores its edit; if both keep it, the result is reported as inertia. EDREL2 cut to 5 pairs (s42–46).
2. *Retention measure.* Primary: the amplitude of the class-mean difference at the edited location on a fixed axis,
   |Δμ_L| (end / W, `ep_geom` amp_slot), divided by the same ratio at the other location. Retention
   r = (m_EDREL,end − m_FREE,end) / (m_EDIT − m_W), paired by model (FREE from the same W carries the drift); differences
   reported next to r, and ℓ_P kept as secondary. "Kept" is joint: the representation measure is retained *and* the
   hybrid (W actor × end transformer) still carries a share of the DiD d′.
3. *Off-equilibrium critic.* Each EDIT checkpoint gets a 5k critic-only re-warm on the edited network (as W did); EDREL
   starts from that, with an extra 1k snapshot.
- CONT dropped (no null at 100k; FROZ's own 10k / 25k / 50k assays give its trajectory).

## 12. rev 4 (2026-11-06): Part E, the restoring force

**Question.** In Part D the standard learner undid about 60% of the hand-built amplitude effect within 50k episodes, at
the same rate whether the edit sat at the low- or the high-value location (0.43 vs 0.42 retained). Which force does
this? Candidates (coordinator, 6 Nov): (1) Adam's L2 weight decay (1e-6 at lr 1e-5; value-independent; it drove q/k to
the float floor in CRIT-DET); (2) the critic's gradient into the shared trunk; (3) the actor's gradient into the trunk;
(4) anything visible in the gradients themselves.

**Scouting fact (6 Nov, before this plan; 4 runs).** The edited weights themselves are almost untouched. The coordinate
of the trunk along the edit direction, e·(θ_t − θ_W)/|Δθ_edit| over RO's four position columns and bias, is 0.9997 /
0.998 / 0.994 at 1k / 10k / 50k in both placements. The edit entries are about 20× the typical RO weight, and
280k learn steps of pure Adam-normalized decay could have removed them, so decay is not what acts on them. **The
erosion is therefore compensation elsewhere in the network**, not reversal of the edit. This reframes candidates 1–4: the
question is which loss drives the compensating change, and where it sits.

**E0 (no training; all Part-D runs).**
- *Bookkeeping:* the edit coordinate at every snapshot for all EDREL runs (and the same projection for FREE runs onto
  that model's edit direction, as drift), plus the analytic pure-decay prediction for the edited entries from the
  checkpoint's Adam second moments (rate lr·λ/(√v̂ + ε) per step, integrated over the logged learn steps).
- *Time course:* fixed-axis amplitude (`ep_geom`, against W) at the 1k / 5k / 10k / 25k snapshots of EDREL1, EDREL1H and
  FREE, so the erosion's time constant is known.
- *Localization:* on the EDREL end states, revert one trunk module group at a time to its EDIT-start value (the edit is
  kept) — G1 token projection, G2 q/k, G3 v, G4a norm1, G4b LSTM input weights R* (other than the edited entries), G4c
  LSTM recurrent weights W* and biases — and measure the amplitude at the edited slot (`ep_geom`) and DiD d′ on the probe
  trials. The group whose reversion brings the edit's amplitude loss back carries the compensation. "Revert all" must
  return the EDIT start (check).
- *Gradient attribution:* per-loss (critic KL, MPO, entropy, decay) Adam-preconditioned updates of the trunk at the
  EDIT start and the 1k / 5k snapshots (zk.grad_attrib's method), projected onto the observed compensation direction
  Δθ_comp = θ_EDREL(5k) − θ_EDIT restricted to the localized group, minus the same projection in the paired FREE run
  (drift). Report each part's share of the projected update, both placements.

**E1 (causal arms, 50k from the Part-D EDIT checkpoints, the Part-D settings otherwise).** For each arm: the edit at the
low-value location (k 1), at the high-value location (k 1), and the unedited FREE control from W with the same
manipulation, so retention r_x = (EDREL_x − FREE_x)/(EDIT − W) is drift-corrected within the manipulation. 5 pairs
(s42–46) per cell (Part D's between-pair SD of the amplitude measure gives 90% CIs of about ±0.03 on r at 5 pairs).
- **CD (critic detached; `--grad-route actor`, the CRIT-DET machinery, q/k decay already 0):** only the actor's MPO and
  entropy terms reach the trunk.
- **AD (actor detached; `--grad-route critic`, ACT-DET):** only the critic's KL reaches the trunk.
- **ND (no weight decay anywhere in the trunk):** trimmed unless E0 contradicts the scouting fact. If the edited
  weights are not being decayed and decay cannot produce a location-specific compensation, an ND arm cannot change
  retention. If E0's localization or attribution gives decay a share above 0.2 of the projected update, ND runs (same
  design) before the report.

**Predictions (before E0 / E1 data).** The critic's KL drives the compensation: AD retention ≈ Part D (≈ 0.4); CD
retention high (≥ 0.7) in both placements. The compensation sits in the LSTM's recurrent or input weights (G4), not in
attention. Gradient attribution gives the KL ≥ 0.7 of the projected update. Confidence moderate (Phase 0 of which-loss
found the KL is 0.84–1.00 of the BASE transformer update).

**Decision rules.** Per arm and placement: r_x with 90% pair-bootstrap CI. A force "carries the erosion" if removing it
raises retention to ≥ 0.7 (lower bound ≥ 0.5) in both placements; "contributes" if it raises retention by ≥ 0.2 over
Part D's (lower bound > 0); otherwise "not needed". Value-independence is re-checked within each arm (low vs high
difference in r, 90% CI). Caveat to state up front: detaching a loss changes the dynamics (the CD learner is slower and
its attention is weaker, as in which-loss), so a retention change shows sufficiency of the remaining force, not the
size of the removed one in the intact learner; the E0 attribution gives the intact-learner shares.

**What it means for the manuscript.** If the critic's KL is the force, the readout account becomes specific: the
location-symmetric, magnitude-free critic keeps every location's code equally informative and equally scaled, and the
actor, which sees reward magnitude, is the only place value can live. If the actor is the force, "learning prefers the
readout" needs restating, because the reward-sensitive loss itself would be removing value from the code.

**Responses to the Part E review (`RVIT_EXISTENCE_PROOF_PLAN_review.md`, "Part E review").**
1. *Projection onto the summed update is circular.* Replaced: each loss's Adam-preconditioned update is projected onto
   the gradient of the measure itself, log A = log|Δμ_L| − log|Δμ_O| (class-mean change difference at the edited slot
   over the other slot, t3 probe trials; the token projection is included in the graph). Predicted per-step change
   d log A_k = ⟨∇_θ log A, −lr·u_k⟩, at the EDREL 1k and 5k checkpoints (and FREE 10k for drift), per group (proj, qk,
   val, norm1, LSTM R*, LSTM W*) with the edited entries masked separately. Shares are read only if Σ_k predicts the
   observed 1k → 5k change of log A within a factor of 2.
2. *CD's decay pathology; ND premise.* CD now runs with trunk weight decay 0 (`--trunk-decay 0`; "actor losses alone").
   The edit coordinate is logged for every E1 run. ND is triggered if decay's share of the predicted d log A exceeds 0.2
   (not its share of the weight update). The scouting statement is narrowed: decay does not act on the edited entries;
   decay of u's position coordinates (norm1 / token projection) is covered by the d log A attribution per group.
3. *Reversion off-manifold.* Localization now uses revert (end with one group from EDIT) and graft (EDIT with one group
   from the end), each minus the same operation in the paired FREE run (relative to W); disagreement is reported as an
   interaction. Added the direct measures: the realized output-gate offset at L on the probe trials (end vs EDIT) and the
   change of u's position coordinates at L.
4. *CD may retain because nothing moves.* Trunk displacement per group (relative to W, in units of the FREE run's) and the
   change of H in FREE_CD (amplitude, direction, DiD c) are reported; retention is compared with Part D on the same seeds.

