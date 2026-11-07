# Can value act on RViT's per-location code? An existence proof (4–6 Nov 2026)

**Architecture:** [[rvit-original]] only (multiplicative q/k/v gating, the preprint model; scope decision 9).
**Task:** L&M sensitivity session, SENS-C reward table (value ratio 25, K 0.25, H:CR 1.0 / 0.44).
**Plan:** `documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md` rev 3 (planned analyses, revised openly; not pre-registered),
with two independent reviews in `documents/plans/RVIT_EXISTENCE_PROOF_PLAN_review.md`. Approved by Herman on 4 Nov
("Please do!").

## The question, in plain words

The manuscript says that in trained RViT, value changes how a location's evidence is **read out** (the actor's gain),
not how well the change is **encoded** in that location's own state H. The SENS-C models encode the change equally
well at both locations (AUC 0.899 vs 0.899), and the actor's gain on that code is about 4× lower at the low-value
location. A reviewer could say: "the architecture forces this; value has no way into the code". This analysis asks
whether that is true, in three ways:

1. **Can the architecture express it at all?** (Part C) Write a value-like, location-specific modulation directly into a
   trained model's own weights, and see whether it produces an L&M-like behavioral effect through the code.
2. **Will training put value into the code when the readout cannot change?** (Part A) Fine-tune with the actor frozen,
   so that the only way to produce a value effect is to change H.
3. **Does training keep a value route in the code once one exists?** (Part D) Start the standard learner from the
   hand-built solution and see whether it keeps the encoding route or moves the value effect to the actor.

**Short answer.**
- **The architecture can carry value in the code.** One rank-1, position-gated edit of the LSTM's own output-gate
  weights at the low-value location gives a graded value effect that is location-specific and, at a mild strength,
  L&M-like (d′ 2.70 → 2.16 at that location, the other location 2.65 → 2.68, criterion unchanged). It works by a gain
  on that location's code (change amplitude ×0.54 on a fixed axis) with the change information intact (AUC loss 0.03).
  Made stronger, the edit also removes information (tested-slot loss 0.15, lower bound 0.11 > 0.077).
- **Training does not put it there.** With the readout frozen, 50k episodes of the real objective give **no**
  location-specific value effect on sensitivity: value term +0.011 [−0.010, +0.033] in DiD d′, against +0.888
  [+0.861, +0.915] for the same fine-tune with the actor free. The trajectory is flat (10k / 25k / 50k). Yet the frozen
  learner did move the **criterion** through H (value term in DiD c −0.29 [−0.33, −0.26]). So the transformer was
  learning; it just did not learn a sensitivity effect.
- **With the actor free, the value effect goes entirely into the actor**, also when the value assignment is reversed
  (FREE / FREE-REV: share of the change carried by the transformer −0.02 and −0.08).
- **Part D** (does the learner keep the hand-built route?): see §4.

- **What undoes it** (Part E, §5). Not weight decay, and not one loss.
  - The edited weights stay 99.5% intact. The learner disarms the edit by re-mapping the position signal in the token
    projection, which cuts the edit's realized gate offset to about 30%.
  - Either loss alone does this at about the intact rate. Retention is 0.38 / 0.29 with only the actor's gradients
    (no decay) and 0.45 / 0.46 with only the critic's KL, against 0.46 / 0.42 intact.

So "value acts on the readout" is a property of what this learning rule does, not a limit of the architecture. A
reviewer who says the architecture forces it can be shown a single weight edit that does otherwise.

## 1. What the architecture allows (from the code)

Every per-location weight is shared across the four locations; the actor reads the four hidden states H_0..H_3
together. A location-specific change of the code can enter only through the position one-hot in each token, through
attention, or through the location's own history. Two normalizations cancel the simplest gains (LayerNorm on the LSTM
input; the xLSTM normalizer, which makes a uniform input-gate gain cancel), so a gain has to act on the output gate, the
forget-vs-input balance, the q/k/v products or attention. The network is deterministic given the input, so a pure gain
on H cannot change how linearly decodable the change is: it changes only the **amplitude** of the code, and the policy's
own randomness downstream turns that into a d′ change. That is why the tested-slot AUC alone cannot detect a gain-type
value mechanism, and why the amplitude measures below matter.

## 2. Part C: hand-written edits (expressibility; nothing trained)

**What was done** (`zk/ep_edit.py`; CRC job 4665127; 20 start models = the reward ×10 models, whose behavior is equal at
both locations). At the SENS-C low-value location L, a rank-1 edit of the LSTM's input→gate weights along L's position
direction (the four position coordinates of the LSTM input, solved so the added bias is exactly β at L and 0 at the
other three slots on average; trial-by-trial jitter SD ≈ 0.55β everywhere):
- **C-O:** output gate, β = −k·SD(pre-activation at L): a gain on L's memory output.
- **C-F:** forget gate, β = +k·SD: more weight on the stored past, so the test is diluted.
k ∈ {0.5, 1, 2, 4, 8}. Measured: 18° assay (2,000 trials per location), the twin-grouped tested-slot AUC, the
whole-state AUC (all four H) and the pooled-decoder separation, all at the first test frame t3.

**A first pass was discarded.** Its direction contrasted L with the mean of the other three slots and put about 0.75 of
the offset onto the other stimulus slot (it mostly encoded "stimulus vs blank"), so its effects were not
location-specific. Those files are kept in `data/runs/partC/` (identical copy in `data/runs/partC_v1/`).

**Results** (20 models; mean [95% CI]; `stats/ep_stats.json` → `partC`; `figs/fig_partC.png`):

| edit | d′ at L | d′ other | c at L | info loss, slot | info loss, whole | amplitude loss (pooled) | behavioral loss |
|---|---|---|---|---|---|---|---|
| none | 2.70 | 2.65 | +0.32 | 0.01 | −0.02 | 0.01 | −0.02 |
| C-O k 0.5 | 2.62 | 2.70 | +0.28 | 0.01 | 0.03 | 0.22 | 0.03 |
| **C-O k 1** | **2.16 [2.07, 2.25]** | **2.68** | **+0.32** | **0.03 [−0.01, 0.08]** | 0.10 | **0.46 [0.44, 0.49]** | 0.19 |
| **C-O k 2** | 0.89 [0.76, 1.02] | 2.49 | +0.78 | **0.15 [0.11, 0.20]** | 0.39 | 0.77 | 0.64 |
| C-O k 4 | 0.09 | 1.88 | +1.09 | 0.49 | 0.67 | 0.92 | 1.22 |
| C-F k 1 | 1.79 | 2.28 | +0.72 | 0.08 | 0.03 | 0.19 | 0.23 |
| C-F k 2 | 0.51 | 1.31 | +0.90 | 0.17 | 0.09 | 0.30 | 0.76 |

- **The output-gate edit is a working encoding-level value mechanism.** At k = 1 it lowers d′ at L by 0.54 and leaves
  the other location and the criterion at L unchanged: the L&M signature. It acts through the code: the change's
  amplitude at L falls to 0.54 of the warm-up on a fixed axis (the other slot 1.04), while the change stays as decodable
  as before. At k = 2 the effect is large (d′ 0.89) and information is lost too (tier I by the plan's rule), with a
  conservative criterion shift (+0.46) and a small spill-over to the other location (−0.16).
- **The forget-gate edit is not location-specific:** it costs the other location almost as much (k 2: 1.31 vs 0.51),
  because the actor reads all four states and the edited state changes on every trial.
- **What this says about the measures.** At C-O k 1, the paper's P4 rule (tested-slot AUC) would call the effect
  "readout" (information loss 0.03, less than half the behavioral loss), although the value effect lives in H. The
  pooled-decoder separation (0.46) and the fixed-axis amplitude (×0.54) see it, and the P9 readout slope at L does not
  move (2.09 → 2.22): a gain placed in H shows up as lower amplitude, not as a lower slope (`figs/fig_locus.png`). The
  trained SENS-C models show the opposite pattern: pooled separation equal at both locations (1.85 vs 1.82) and the
  slope 4× lower. So the existing evidence already rules out the gain route in the trained models, but the claim must
  cite the amplitude measure, not the AUC alone.

## 3. Part A: fine-tunes with the readout frozen or free

**Design** (`zk/ep_train.py` wrapping `zk/train_orig.py` unchanged; CRC jobs 4664614 warm-ups, 4664615 training,
4673533 / 4673535 assays). Start models: the 10 counterphased reward ×10 pairs (`whichloss/rscaled10`, s42–51, trained
from scratch on the SENS-C table with every reward ×10: no value effect, DiD d′ −0.05, equal encoding). Each got a 5k
critic-only warm-up with a fresh 1,201-particle head on the new table (W, W-REV, W-NULL). Then 50k episodes, θ frozen at
the start value, q/k weight decay 0 in every arm:

| arm | table | actor |
|---|---|---|
| FREE | SENS-C (rewards ×0.1: the low location falls below the entropy temperature) | trained |
| FREE-REV | SENS-C with the value assignment reversed | trained |
| FROZ | SENS-C | frozen (no step; gradients still pass through it into H) |
| FROZ-REV | reversed | frozen |
| FROZ-NULL | the start table kept (×10) | frozen: the procedure null |

The start actor already reads the old low-value location with a lower gain (P9 slope 2.09 vs 3.76, saturated, so
behavior is equal). A symmetric change of H could exploit that and look like a value effect. The reversed arms put the
reader's strong slot at the new low-value location; the mean of the congruent and reversed changes (the **value term**)
cancels the reader's asymmetry, and their half-difference estimates it.

**Results** (10 pairs per arm; mean [95% CI]; t3; `stats/ep_stats.json`; `figs/fig_partA.png`):

| | FREE | FREE-REV | FROZ | FROZ-REV | FROZ-NULL |
|---|---|---|---|---|---|
| DiD d′ at 50k | +0.90 [+0.86, +0.94] | +0.87 [+0.83, +0.91] | +0.04 [−0.02, +0.09] | −0.02 [−0.05, +0.02] | −0.18 [−0.23, −0.13] |
| change from the warm-up | +0.95 [+0.85, +1.05] | +0.82 [+0.77, +0.88] | +0.09 [+0.02, +0.16] | −0.07 [−0.14, +0.01] | −0.13 [−0.24, −0.02] |
| DiD c at 50k | −0.07 | −0.09 | −0.27 | −0.32 | −0.56 |
| d′ high / low | 2.73 / 1.83 | 2.72 / 1.85 | 2.67 / 2.63 | 2.69 / 2.70 | 2.48 / 2.66 |
| info loss (tested slot) | −0.000 [−0.026, +0.026] | −0.033 | +0.018 [−0.005, +0.041] | +0.005 | −0.036 |
| amplitude loss (pooled) | +0.009 | −0.021 | +0.009 | −0.022 | −0.025 |
| P9 slope high / low | 4.74 / 0.90 | 3.36 / 0.89 | 4.18 / 2.38 | 3.06 / 3.39 | 3.62 / 2.07 |
| attention high − low (t3–t4) | −0.001 | +0.003 | −0.001 | +0.003 | −0.003 |
| transformer share of the change (hybrid) | −0.02 [−0.04, +0.00] | −0.08 [−0.11, −0.06] | 1.01 (by construction) | 1.01 | 1.00 |

**Value term (congruent + reversed)/2, change from the warm-ups:**
- FREE: DiD d′ **+0.888 [+0.861, +0.915]**; DiD c −0.08; information loss −0.011.
- FROZ: DiD d′ **+0.011 [−0.010, +0.033]**; DiD c **−0.29 [−0.33, −0.26]**; information loss +0.017 [−0.003, +0.036];
  amplitude loss −0.006. Reader term in DiD d′ −0.08 [−0.15, −0.01].

**Readings.**
- **The unconstrained learner puts the whole value effect in the actor**, as the from-scratch SENS-C models do: equal
  encoding and amplitude at both locations, the P9 slope 5× lower at the low-value location, and a hybrid
  decomposition in which the W actor reading the end-state transformer reproduces essentially none of the change
  (representation main term −0.02; interaction −0.15). Reversing the value assignment, which forces the learner to
  undo an existing readout asymmetry and build the opposite one, gives the same answer. The end-state criterion is
  L&M-like (DiD c −0.07 and −0.09).
- **With the readout frozen, no sensitivity value effect appears.** The congruent arm's small gain (+0.09, or +0.22
  over FROZ-NULL) is the reader asymmetry: the reversed arm moves the other way and the value term is 0.01, with an
  upper bound of 0.03, a tenth of the plan's 0.3 minimum. It does not grow (FROZ DiD d′ +0.03 / +0.03 / +0.04 at
  10k / 25k / 50k). Nothing changes in the code either: information, amplitude, direction and attention are all as at
  the warm-up.
- **But the frozen learner did change H in a value-specific way: the criterion.** The value term in DiD c is −0.29,
  i.e. the low-value location became more conservative, as its H:CR table (0.44, CR worth more than hit) rewards. So the
  transformer was not stuck. The objective found a value-dependent *bias* in the code, but not a value-dependent *gain*,
  even though Part C shows a gain is one weight edit away.
- **Why, most likely:** the which-loss analysis found that the critic's magnitude-free KL makes up about 95% of the
  transformer's update, and wants the change encoded equally well at both locations. The actor's push on sensitivity at
  the low-value location is tiny (its reward is below the temperature), so through a frozen actor it cannot outweigh the
  critic. This was not tested directly here (the critic-detached arm was dropped after review).

## 4. Part D: does the standard learner keep a hand-built encoding route?

**Design** (plan §11 and the addendum review; CRC jobs 4671962 edits + critic re-warm, 4671963 fine-tunes). The C-O
edit was written into each W checkpoint (`zk/ep_make_edit.py`), the critic was re-warmed for 5k episodes on the edited
network (critic only), and the standard learner then ran 50k episodes on the SENS-C table:
- **EDREL1:** from the k = 1 edit at the low-value location (10 pairs);
- **EDREL2:** from the k = 2 edit at the low-value location (5 pairs; the information case);
- **EDREL1H:** from the k = 1 edit at the **high**-value location (10 pairs). Here the edit works *against* the reward
  table (a d′ loss there costs about 25× more), so it is the control that tells "kept because it is value-consistent"
  from "kept because nothing pushes it" (inertia), and both from a value-independent restoring force.

Retention of the edit's effect at 50k is r = (EDREL end − FREE end) / (EDIT − W), paired by seed, so FREE (same learner,
same table, unedited start) removes the drift every fine-tune has. Primary measure: the amplitude loss at the edited
location on a fixed axis (|class-mean difference| end/W, relative to the other location).

**Results** (`stats/ep_stats.json` → `retention`, arms `EDIT*`, `EDREL*`; `figs/fig_partD.png`):

| | EDREL1 (edit at low, k 1) | EDREL2 (edit at low, k 2) | EDREL1H (edit at high, k 1) |
|---|---|---|---|
| edit's amplitude loss at the start (vs W) | 0.48 | 0.76 | 0.47 |
| left at 50k, beyond FREE | 0.21 | 0.28 | 0.19 |
| **retained fraction r, amplitude** | **0.43 [0.41, 0.45]** | **0.37 [0.36, 0.39]** | **0.42 [0.39, 0.43]** |
| information loss at the start → retained | 0.03 (no information effect to retain) | **0.14 → r −0.05 [−0.29, +0.21]: fully restored** | 0.05 (none) |
| DiD d′: start → 1k → 10k → 50k | +0.52 → +0.34 → +0.86 → +0.91 | +1.59 → +0.98 → +0.88 → +0.84 | −0.51 → −0.08 → +0.70 → +0.90 |
| the end transformer read by the W actor (DiD d′; EDIT start in brackets) | +0.17 (+0.57) | +0.23 (+1.70) | +0.08 (−0.46) |
| transformer share of the end-state value effect (hybrid vs W) | 0.06 [0.04, 0.08] | 0.10 [0.07, 0.14] | −0.03 [−0.06, −0.00] |
| P9 slope high / low at 50k | 4.32 / 1.01 | 4.17 / 1.08 | 4.83 / 0.82 |

(FREE for comparison: DiD d′ +0.90, transformer share −0.02, P9 slope 4.74 / 0.90.)

**Readings.**
- **The learner does not keep the hand-built route.** Within 50k, about 60% of the edit's amplitude effect is undone, and
  the information loss of the k = 2 edit is undone completely. By the planned rule the amplitude result is "partial"
  (r ≈ 0.4: neither kept, ≥ 0.7, nor abandoned, ≤ 0.3), but the direction is clear and the remaining part carries
  little of the behavior: at 50k the transformer carries 6–10% of the value effect and the actor the rest, as in FREE.
  All three arms end at FREE's value effect (DiD d′ +0.84 to +0.91) and FREE's readout pattern (slope about 4.3 vs 1.0).
- **The undoing does not depend on whether the edit helps or hurts.** The value-consistent edit (EDREL1) and the
  value-opposing one (EDREL1H) lose the same fraction of their amplitude effect (0.43 vs 0.42 retained). So what erodes
  the edit is a value-independent pull back toward equal coding at all locations, not reward. The likeliest source is
  the critic's location-symmetric, magnitude-free KL, which dominates the transformer's update (which-loss Phase 0).
  This was not tested directly.
- **Reward acts fast, but on behavior and through the actor.** The value-opposing edit's behavioral cost is gone within
  1k episodes (DiD d′ −0.51 → −0.08), long before its amplitude effect has decayed, so the fast correction is in the
  actor. By 50k the W actor reading the EDREL1H transformer gives DiD d′ +0.08, the same as for FREE's transformer
  (+0.06): the edit's behavioral trace in the code is gone. In the value-consistent case the actor builds the value
  effect on top of the decaying edit, as it does from an unedited start (the code's own contribution falls from +0.57
  to +0.17).

So Part D answers the reviewer's question in the strongest form available here: even when value is already in the
per-location code, the RViT learning rule moves it out of the code and into the readout, at a rate that does not depend
on reward. The readout locus is something the objective does, not something the architecture forces.

## 5. Part E: what erodes the hand-built route? (6 Nov)

**Why.** Part D's erosion was the same for a value-consistent and a value-opposing edit. Herman approved a follow-up
(6 Nov) to identify the force. Plan: `RVIT_EXISTENCE_PROOF_PLAN.md` rev 4 §12 and its review. Candidates: Adam's weight
decay, the critic's gradient into the trunk, the actor's gradient into the trunk, or something visible in the gradients.

### 5a. E0: diagnostics on the Part-D runs (no training; 10 runs per placement, s42–46; CRC 4696871)

| | edit at the low-value location (EDREL1) | edit at the high-value location (EDREL1H) |
|---|---|---|
| edited weights left at 50k (coordinate along the edit) | 0.995 [0.994, 0.996] | 0.995 [0.994, 0.996] |
| the same, predicted by Adam-normalized weight decay alone | 0.998 | 0.998 |
| amplitude at the edited slot, edit → 1k → 5k → 10k → 25k → 50k (W = 1) | 0.53 → 0.62 → 0.68 → 0.70 → 0.74 → 0.79 | 0.54 → 0.65 → 0.70 → 0.72 → 0.77 → 0.81 |
| realized output-gate offset at the edited slot, end / edit | **0.32 [0.27, 0.38]** | **0.29 [0.24, 0.34]** |
| the same in the paired FREE run (no edit learned around) | 0.89 | 0.93 |
| share of the compensation undone by reverting the token projection to its edit-time value (net of FREE) | **0.98 [0.71, 1.25]** | **0.82 [0.70, 0.94]** |
| share reproduced by grafting the end-state token projection onto the edit | **0.98 [0.89, 1.07]** | **0.94 [0.88, 1.00]** |
| the same for attention q/k, v, norm1, LSTM input and recurrent weights (revert / graft) | all ≤ 0.07 | all ≤ 0.07 |

- **Weight decay is not it.** The edit's weights are 99.5% intact after 50k episodes, which is what decay alone
  predicts (99.8%), and the paired FREE runs, which have the same decay, show no change of the position code.
- **The edit is not undone; it is disarmed.** The edit is a bias on the output gate that is switched on by the position
  coordinates of the LSTM's input at the edited slot. The learner re-maps those coordinates in the **token projection**
  (the layers that turn each patch's VAE code and position one-hot into its token), so the edit's trigger fires at only
  about 30% of its original strength. Reverting the token projection alone brings back 82–98% of the edit's effect, and
  grafting it alone onto the edit reproduces 94–98% of the compensation; every other trunk group carries ≤ 7%.
- **It is fast.** About 40% of the 50k recovery happens within the first 1k episodes.
- **The planned gradient attribution failed its own validity check** (plan §12, review fix 1). The per-loss
  Adam-preconditioned updates projected onto the gradient of the amplitude measure predicted a per-step change of the
  wrong sign and about 190× too large (one run; 580× with the checkpoint's second moments). A first-order expected step
  does not describe this noisy trajectory, so per the plan its loss shares (critic KL 0.94–0.98 of the projected update)
  are **not** read. The causal arms carry the attribution.
- **The no-decay arm was not run.** Its trigger in the plan was decay's share of the predicted change, which is not
  readable. The direct evidence above (edit weights intact, matching the decay-only prediction; FREE unchanged under the
  same decay; the compensation is location-specific) rules decay out without it.

### 5b. E1: which loss drives the compensation? (5 pairs per cell, s42–46; CRC 4696758, diagnostics 4700032)

Each arm fine-tunes from the same EDIT checkpoints as Part D (edit at the low- or the high-value location), plus an
unedited control from W with the same manipulation. Retention r = (arm − its own FREE control) / (EDIT − W) of the
fixed-axis amplitude loss (`stats/ep_stats.json` → `retention`; `figs/fig_partE.png`).
- **CD:** critic detached from the trunk (`--grad-route actor`), with trunk weight decay 0, so only the actor's MPO and
  entropy gradients reach the trunk.
- **AD:** actor detached (`--grad-route critic`), so only the critic's KL reaches it.

| arm | r, edit at low (90% CI) | r, edit at high | low − high (paired, 90% CI) | change vs Part D, same seeds (low / high) |
|---|---|---|---|---|
| intact learner (Part D, s42–46) | 0.46 | 0.42 | +0.04 [+0.00, +0.08] | — |
| **CD: actor losses only, no decay** | **0.38 [0.32, 0.44]** | **0.29 [0.23, 0.34]** | +0.10 [−0.04, +0.24] | −0.08 [−0.18, +0.02] / −0.12 [−0.22, −0.03] |
| **AD: critic KL only** | **0.45 [0.42, 0.49]** | **0.46 [0.44, 0.48]** | −0.01 [−0.08, +0.06] | −0.01 [−0.08, +0.07] / +0.04 [+0.01, +0.08] |

- **Neither loss is needed; each alone is enough.** With the critic cut off from the trunk, the actor's gradients alone
  erode the edit at least as fast as the intact learner. With the actor cut off, the critic's KL alone erodes it at the
  intact rate. By the plan's rule neither removal "carries the erosion" (no arm retains ≥ 0.7). The force is redundant
  across the two losses.
- **Weight decay is not needed either.** The CD arm had no trunk decay at all and eroded the edit fastest.
- **Mechanism, the same in every arm.** The edited weights stay 99.5–99.8% intact. The realized gate offset at the edited
  slot falls to 0.31–0.36 of its edit value (CD at the low-value location: 0.52). The unedited controls keep it at
  0.87–0.97 (FREECD, at the high-value location's direction: 0.66).
- **Positive control for the detached arms.**
  - Both trunks moved: CD moves about 2/3 as far as FREE in the LSTM weights, and AD moves as far as FREE.
  - Both learned the value effect: FREECD DiD d′ +0.93, FREEAD +0.90; P9 slopes 4.7 / 0.8–0.9.
  - So "retention" is not explained by a frozen trunk.
- **A hint of value dependence only when the critic is removed.** Under CD, the value-opposing (high-location) edit is
  undone somewhat more than the value-consistent one (+0.10, CI includes 0). With the critic present (intact, AD), the
  rates are equal (+0.04, −0.01). The critic's location-symmetric, magnitude-free KL therefore sets the value-independent
  rate seen in Part D. The reward-sensitive actor adds a small value-dependent push of its own, against the costly edit
  more than the value-consistent one.

### 5c. What this means

- **The restoring force is not a single loss.** Any task gradient that reaches the trunk, the critic's or the actor's,
  disarms a location-specific perturbation of the code. It does so by re-mapping the token projection's position
  signal, the cheapest place to cancel a position-gated edit. Weight decay plays no part.
- **Why value-independent, in the intact learner:** the critic dominates the trunk's update and is blind to reward
  magnitude, and Adam normalizes step sizes. The rate at which a perturbation is removed is then set by structure,
  not by how much reward it costs.
- **This is untested:** that the edit is detected through its spill-over on all trials, including those where the
  edited location is blank, is a plausible reason and was not tested.
- **For the manuscript's "learning prefers the readout" account:** the account should be stated as a property of shared-trunk gradient learning, not of
  one magnitude-free loss. The trunk is shaped by gradients that, together, keep each location's code uniform and do not
  carry value into it: the critic's KL by construction, and the actor's push because it, too, removes the edit. Value
  then has only one stable home, the actor's readout. The which-loss result (each loss alone leaves encoding equal)
  and this one (each loss alone removes value that was put into the encoding) are the two halves of that statement.
- **Planned but not readable:** the per-loss gradient attribution (5a) is not readable. Its 0.94–0.98 critic share should not be quoted.


## 6. Caveats and what a reviewer could still say

- **Part C is a construction, not a learned solution.** It shows the architecture can carry value in the code; it says
  nothing about whether learning would find it. Parts A and D address learning.
- **Fine-tunes, not training from scratch.** The start models were trained from scratch; FREE reproduces the
  from-scratch locus, so the fine-tune itself is not the reason value lands in the actor.
- **50k episodes.** FREE reaches 80% of its effect within 10k; FROZ is flat from 10k to 50k. A much slower route
  cannot be excluded.
- **The frozen-readout test changes the learning dynamics.** Its result is "the objective does not move sensitivity
  into H when it cannot use the actor", not a statement about the unconstrained learner, which Parts A (FREE-REV) and D
  address.
- **The first-order Jacobian decomposition (plan §4) was not usable:** the actor's Jacobian changes with its operating
  point, so its readout and representation terms are large and opposite even where the actor did not change (EDIT1).
  It is in `ep_stats.json` but not interpreted; the exact 2×2 hybrid is used instead.
- **The II-gain / II-rot cut (cosine 0.9)** was not calibrated: arms with no change already sit at 0.90 (50k of drift).
  Amplitude and direction are reported as numbers instead.
- **Procedure null.** FROZ-NULL drifts (DiD d′ −0.13 from its warm-up), mostly at the reader's strong slot; the
  congruent/reversed design absorbs this kind of term.
- **Operational incident.** Replacing `run_ep.sh` on CRC while jobs were running (a `mv` over the file on the shared
  file system) ended the running scripts right after training, silently, before their assays. The assays were rerun from
  the saved checkpoints (jobs 4673533, 4673535); nothing was retrained. One task was requeued once early on (to pick up
  a thread limit), so FROZ loc0_s42, loc3_s42 and loc0_s43 resumed from their ep-5000 checkpoints with an empty buffer.
- **Baruni is not run.** In the Baruni analog, value is cued in the input, so a cue-gated gain was available to the
  unconstrained learner and was not used (encoding equal across values). That is an independent rebuttal; a dynamic,
  cue-gated version of this existence proof is a natural follow-up.
- **Primate parallel (to keep soft in the text).** The C-O edit is loosely analogous to a top-down gain on one
  location's sensory code. RViT is its own system; the point is only that its architecture admits such a mechanism.

## Where it lives

- **Code** (new files only; shared modules untouched): `experiments/zenon-krauzlis/code/tree_zk/zk/ep_train.py`,
  `ep_hybrid.py`, `ep_geom.py`, `ep_edit.py`, `ep_make_edit.py`, `ep_e0.py` (Part E diagnostics), `ep_stats.py`, `ep_figs.py`, `zk/tests/test_ep_train.py`
  (two tests: two-group Adam ≡ legacy Adam, frozen actor passes the same transformer gradient; both pass on CRC);
  slurm `experiments/zenon-krauzlis/code/slurm/{run_ep.sh, ep_cells.slurm, ep_edit.slurm, ep_smoke.slurm}`, tables
  `slurm/tables/ep_table_*.txt`.
- **Runs (CRC):** `/ix/jpherman/hermanj/rvit_zk/runs/existence/<arm>/loc{0,3}_s{42..51}` (checkpoints there);
  Part C in `existence/partC2/` (and the discarded first pass in `existence/partC/`).
- **Local data:** `data/runs/` (JSON, CSV and logs only). **Statistics:** `stats/ep_stats.json` (Part E: `partE0`, `retention`; per-run E0 diagnostics in `data/runs/partE0/`), printout
  `stats/ep_stats_print.txt`. **Figures:** `figs/fig_partC`, `fig_partA`, `fig_locus`, `fig_partD`, `fig_partE` (PNG + PDF). In `fig_partD` the
  information-retention points for EDREL1 / EDREL1H are meaningless (those edits had no information effect to retain).
- Reproduce: `python -m zk.ep_stats --root data/runs --out stats/ep_stats.json` then
  `python -m zk.ep_figs --stats stats/ep_stats.json --out-dir figs` (from `tree_zk/`).
