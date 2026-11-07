# Adversarial review: RVIT_EXISTENCE_PROOF_PLAN.md (rev 1, 2026-11-04)

Independent reviewer, 4 Nov 2026. Covers the plan, the related wiki/README pages, and the zk training, wrapper and
probe code.

## Verdict: revise before running

The frozen-readout idea is reasonable, but as specified the most likely outcome cannot be interpreted. The rscaled10 start
actor already reads the low location with about half the gain. Lowering every reward ×0.1 pushes *both* locations
toward a softer policy. So the cheapest FROZ solution is a location-symmetric shrink of the change signal in H, which
the asymmetric frozen reader turns into a positive DiD d′. The plan would score this as a representational value
effect (rep share 1, tier III). Stage 3 has a second problem: "kept" cannot be told apart from inertia. Both problems
can be fixed cheaply (findings 1, 2, 4). With those fixes the experiment is worth running.

## Findings

**1. (major) The start actor's value asymmetry confounds every behavioral and hybrid measure in the constrained arms.**
The rscaled10 actor's dv gap is 5.12 (low) vs 8.96 (high). The actor (`fc1 → LN → …` over concat(H₀..H₃)) has
location-specific weights. A uniform reduction of the change signal at both slots therefore desaturates the low
location first, giving DiD d′ > 0 with no location-specific change in H. The trigger is location-symmetric too:
×0.1 on every reward shrinks the high location's ΔQ from −5.06 to about −0.5, so MPO softens both locations.
Counterphasing does not remove this, because the reader's asymmetry follows the value assignment. §4 "trivial
route 3" (DiD removes generic degradation) is false under a nonlinear, asymmetric reader. The hybrid decomposition
cannot detect it either: FROZ gets rep share 1 whatever the cause.
*Fix:* add a **reversed-assignment arm (FROZ-REV)** from the same rscaled10 checkpoints, with `--high-loc` flipped so
the reader's *strong* slot becomes the low-value location. Under reversal, a symmetric H change gives DiD ≤ 0, so a
positive DiD needs a location-specific H change that beats the reader. Averaging congruent and reversed cancels the
reader's asymmetry. Also require a location-specific representational signature: report the change in the projection
onto the frozen actor's Jacobian ∂dv/∂H_L, high vs low. Report high-location d′ change from W in every arm.

**2. (major) There is no null arm for the procedure itself.** W is a baseline *state*, not a control for the
*process*: 50k episodes of constrained training under a new critic head, a restarted buffer, Adam's normalized L2
on weights with small task gradients, and drift. The CONT arms control only the stage-3 restart.
*Fix:* add **FROZ-EQ**: frozen actor, same mean reward and critic support, value ratio 1. Every DiD, ℓ_N and ℓ_P in
FROZ should be read against FROZ-EQ, not against W or zero. FROZ-EQ is also the direct test of finding 1: with a
congruent start, FROZ-EQ should show DiD > 0 if the confound is real.

**3. (major) Stage 3 cannot separate "kept" from "not pushed".** Phase 0 shows the actor's MPO share at the low
location is 0.00 at convergence. The critic KL (about 95% of the transformer update) is location-symmetric and
magnitude-free. If FROZ ends near the behavioral fixed point, then after release no gradient moves value from H into
the actor. "Kept" would then mean a flat loss surface, not "two stable optima". For tier II/III there is no restoring
force at all, because the critic can read a rescaled code. "Abandoned" within 50k is unlikely for the same reason.
r = REL/CONT also has an unstable denominator (CONT can keep growing; ℓ near 0 under tier III).
*Fix:* (a) set the stage-3 horizon from FREE's own time constant (≥ 3× the episodes FREE needs to reach 80% of its
DiD); (b) define retention relative to the FROZ end state (REL_end/FROZ_end), with CONT as a drift reference, and use
differences with absolute margins rather than ratios when ℓ < 0.05; (c) add a **perturbed release**: release *and*
change the low-location K by a step (e.g. ×0.5). The learner then has to adjust the value effect, and the locus of
the *new* adjustment (actor vs H, by hybrid or Jacobian decomposition) shows its preference. That is the test of the
reviewer's question, and it is cheap.

**4. (major) A cleaner preference test is missing: FREE-REV.** Fine-tune the rscaled10 models *unconstrained* with
the value assignment reversed. The learner has to undo a readout asymmetry and build the opposite one. If it does
this in the actor and leaves H equal, that is direct evidence that the objective prefers the readout, with no
freeze to argue about. It costs one arm (20 runs) and arguably answers "the architecture forces it" better than FROZ.

**5. (major) Tier II is behaviorally identical to an actor gain, and the plan should say so.** The network is
deterministic given the input (P4). The within-class variability that sets AUC comes from the stimulus (pixel noise,
orientations) and is scaled along with the signal. Policy sampling is the only noise downstream of H. A gain on H
therefore lowers d′ by the same algebra as a gain on the actor's first layer. Calling tier II "cortical gain
modulation" invites the reply: "a readout gain moved one layer up". It is also an issue for the manuscript claim
itself. The tested-slot AUC can never detect a gain-type value mechanism, so "encoding equal" does not by itself
rule out value in H. The real discriminator is ℓ_P / P9 (amplitude on a common axis).
*Fix:* (a) state this in §1.3 and §8. (b) Split tier II into **gain** (amplitude along W's change direction,
direction cosine ≈ 1) and **rotation/routing** (change direction rotated away from the reader's axis, cosine falls).
Only gain is the cortical analog. (c) Use FROZ as a specificity check on P9. If a frozen actor yields a lower P9
"readout slope" at the low location, then P9 cannot localize the locus, and the manuscript's "readout gain 4×"
wording must be softened. This is the plan's most useful by-product; make it an explicit measure.

**6. (major) Tier I is unreachable by construction, and ℓ_N measures the wrong quantity.** (a) The tier-I bar is
ℓ_N ≥ ½ ℓ_beh. The positive control (full abandonment, attention moved away) reached only ℓ_N 0.154 against
ℓ_beh > 1, so it would fail the bar. The P4 rule was conservative for "readout"; reversed, it is an almost
impossible bar for "encoding". Set tier I as ℓ_N lower bound > 0.077 (the which-loss minimum effect) and report
ℓ_N/ℓ_beh as a continuous quantity. (b) ℓ_N uses only the tested slot, but the frozen actor reads all four slots.
FROZ can move or remove low-location change evidence in *other* slots (via attention), which the tested-slot AUC does
not see. Add a per-location AUC from concat(H₀..H₃): the information available to the reader.

**7. (major) FREE's prediction and the L&M-likeness rule ignore known fine-tune behavior.** The dose-response fine-tunes
(including all0.1, which matches this design) gave **FA-driven, liberal-criterion** value effects (c low −0.48 to
−0.89), not flat c. The start also has DiD c −0.37. So an absolute end-state |DiD c| < 0.2 requires c to move
by ≥ 0.17 from W for reasons unrelated to locus. FREE may fail L&M-likeness, which would leave "strong" unreachable.
*Fix:* lower FREE's confidence; define L&M-likeness as ΔDiD c from W within ±0.2 **and** benchmark each constrained
arm's c against FREE. If FREE is not L&M-like, judge the arms by locus only.

**8. (major) FROZ-CD and REL-CD are confounded and add little.** FROZ vs FROZ-CD differs in two factors (critic
detached *and* q/k decay 0), which breaks the "arms differ only in the stated factor" rule. REL-CD vs CONT-CD differs in
three (actor released, critic re-attached, decay restored). Re-attaching the critic brings back the dominant KL update,
which on its own could restore information, so "abandoned" in REL-CD would be uninterpretable. Separately, with the
critic detached and the low-location MPO near 0, Adam-normalized L2 decay dominates every weight with a weak task
gradient. This is the CRIT-DET mechanism, now affecting non-q/k weights. Any information loss could then be decay,
not value.
*Fix:* run every stage-2 arm with q/k decay 0 (the decay share is about 0 in intact models, so FROZ/FREE barely
change). Drop REL-CD/CONT-CD, or release *only* the actor in REL-CD. Interpret FROZ-CD only against a FROZ-CD-EQ null
(finding 2).

**9. (minor) Hybrids are off-manifold.** In FREE, the transformer keeps drifting under the KL while the actor
co-adapts. A (W-actor, end-transformer) network can lose d′ at both locations, and the asymmetric reader turns that
into DiD (finding 1 again). *Fix:* report the interaction term (E,E)−(W,E)−(E,W)+(W,W) and the per-location d′ of
each hybrid. Declare the decomposition uninterpretable if |interaction| > ½ the smaller main term. Add the
on-manifold alternative: a first-order decomposition of Δ(dv gap_L) into Δw·μ + w·Δμ, using the actor Jacobian
averaged over W and the end state, computed on the probe trials.

**10. (minor) The curriculum stays live during the fine-tunes.** `train_orig` restores θ and keeps the shared
per-location curriculum on. FREE, FROZ and REL can then lower θ at different times, so the arms train on different
stimulus distributions. *Fix:* pass `--curr-threshold 1.01` (θ frozen at the start value) in every arm, and log θ.

**11. (minor) The extension rule cannot run as written.** `train_orig.main` returns "final checkpoint exists; nothing to
do" if `ckpt_final.pt` exists. Extending a 50k run to 100k needs a new out-dir with `--init-from`, which restarts the
buffer and resets the episode counter, or a documented rename of `ckpt_final.pt`. Specify which, and match CONT the
same way. Also log `resume_count`: each preemption empties the buffer, and this differs across arms.

**12. (minor) Code checks before launch.**
- The `ep_train.py` freeze is correct as far as it goes: the step is a no-op, `requires_grad=False` still lets
  gradients flow into H, and there is no Adam decay on the actor. The Polyak target converges to the frozen actor
  (W runs about 25k learn steps at τ 0.01). Add a CPU test like `test_grad_route`: the actor is bit-identical after N
  steps; the transformer gradient is nonzero and equals the "both" route's transformer gradient; the two-group Adam
  with decay 1e-6 is bit-identical to the legacy single group over 4 steps, including after `_convert_one_group` on a
  real rscaled10 state dict.
- `_convert_one_group` assumes the legacy optimizer was built over `tr.parameters()` in the same order. That holds
  for `TransformerNetwork.__init__`, and `hook()` registers no parameters, but assert that the parameter count equals
  the size of the saved state.
- The `--route-change-ok` patch also applies on *resume*. Restrict it to `--init-from` so that a mislaunched resume
  still raises.
- In `ep_hybrid --mode transplant`, the four groups omit `linear1/linear2/norm2`. Assert that these are unused in the
  forward pass, and that "revert all groups" reproduces the reference network's d′ exactly.

**13. (minor) Part C targets the wrong tier.** A rank-1 gain on the output gate scales H_i, which is mostly tier II
(finding 5): it cannot test whether *tier I* is expressible. For information loss, edit a position-gated forget/input
balance (more pre-change memory diluting the change), or a position-gated attention-logit bias (the route the
positive control actually used). Part C is also a cheap local job on the RTX 3090. Run it first and unconditionally,
not as a fallback: it gives the expressibility curve (ℓ_N vs ℓ_beh) in a day.

**14. (minor) Sequencing with existing evidence.** The low-location ×0.05/×0.1 sweep (job 4660210) is a from-scratch,
in-distribution test of whether a steep-but-graded value reaches encoding. Its result may settle the reviewer
objection more naturally than any constrained fine-tune. Read it before launching stage 3. Also use the Baruni
result in §8: there value is *cued in the input*, so a cue-gated gain route was available and unused. That rebuts
"the architecture forces it" more strongly than L&M.

**15. (minor) Statistics and compute.** Ten pairs is ample (DiD d′ between-pair SD ≈ 0.07; ℓ_N CI ±0.05). For an
existence claim multiplicity is harmless, but keep t3 as the only primary. Findings 1, 2 and 4 add about 60 runs.
Dropping REL-CD/CONT-CD and cutting CONT to 5 pairs keeps the total near 300 GPU-h.

## Suggested minimal design
Stage 2 (q/k decay 0, θ frozen, from W): FREE, FREE-REV, FROZ, FROZ-REV, FROZ-EQ. Stage 3 from FROZ/FROZ-REV:
REL, CONT, REL+K-step. Part C locally, first. Add these measures: whole-state AUC, actor-Jacobian amplitude and
direction cosine, the hybrid interaction term, and P9 in the frozen arms.

## Addendum review (rev 3, 2026-11-04)

Independent reviewer; covers §11, `zk/ep_make_edit.py`, `zk/ep_edit.py`, `slurm/run_ep.sh` (EDIT1/2, EDREL1/2, CONT) and
the Part C numbers in `ep_stats.json`.

**Verdict: run Part D, but add one control and fix the retention measure first.** Dropping Part B is right, and starting from
the hand-built solution is the cleanest form of the brief. As written, though, "kept" is almost guaranteed and cannot be
interpreted.

**1. (major) Under SENS-C, "kept" is the default outcome, not a finding.** The edit lowers d′ at the *low*-value location,
which is the direction the value table already pushes. That location's reward is below the entropy temperature, and its
MPO share is about 0 at convergence. The critic's KL does not depend on magnitude, so it can rescale its own reading
of a shrunken H. §11's own EDREL1 prediction says as much ("nothing restores amplitude"). So r ≈ 1 tells "the objective
keeps an encoding route" apart from "no gradient reaches it" only if we know the learner *can* undo the edit when undoing
pays. *Fix (priority 1):* **EDREL1-HI**: the same k = 1 edit placed at the SENS-C **high**-value location, with the same
table and the same learner. There a d′ loss costs about 25× more reward. If EDREL1-HI restores the edit and EDREL1 keeps
it, then "kept" means value-consistent retention. If both keep it, the result is inertia, and §11's "kept" reading
must not be used. This beats a reward-×10 EDREL: it needs no new warm-up (it uses W), changes only the edit's location,
and makes the cost much larger. Code: `ep_make_edit` hard-codes L = 3 − hl, so either pass `--hl $((3-TH))` in a new
`EDIT1H` case or add `--at high|low`. Ten pairs (20 runs). If compute is short, drop EDREL2 to 5 pairs before cutting this.

**2. (major) ℓ_P is the wrong retention measure, and r needs a baseline and a drift reference.** At O_k1, ℓ_P = 0.46
comes from pooled_dn_low 1.86 → 1.20 *and* pooled_dn_high 1.88 → 2.25. The pooled decoder is refit per checkpoint and
re-weights toward the intact location, so ℓ_P moves whenever either location or the decoder's mixing drifts, even with
nothing happening at L. *Fix (priority 2):* (a) make the primary measure the per-location amplitude on a **fixed axis**,
|Δμ_L| (from `ep_geom`, end/W) divided by the same ratio at the other location; keep ℓ_P as secondary. (b) Use a
baseline-subtracted and drift-corrected ratio: r = (m_EDREL,end − m_FREE,end) / (m_EDIT − m_W). It is paired by
model, so the FREE run from the same W gives each pair its own drift. W's baselines are small (ℓ_N 0.006, ℓ_P 0.01).
For EDREL2's ℓ_N, though, the start effect (0.154) is only about 3× its CI half-width, so report differences next to
r there. (c) Make "kept" joint: the representation measure is retained *and* the DiD d′ still has a representation
share in the hybrid (W actor × EDREL transformer). Otherwise "H stays edited while the actor re-learns its gain at L"
would be scored as kept when the route has in fact been replaced.

**3. (moderate) Off-equilibrium critic.** W's critic was warmed on the *unedited* H. After the edit, its predictions at L
are wrong, and the critic KL (about 95% of the transformer update) gives an early, large gradient that can restore H. That
transient comes from the surgery, not from what the objective prefers. A trained-in route would come with a matched
critic. *Fix (priority 3, cheap):* a 5k critic-only re-warm (`--critic-only`, transformer and actor frozen, as in W) on each
EDIT checkpoint before EDREL, with EDREL started from that. Add a 1k snapshot, so that a fast transient is visible even
if (3) is skipped.

**Minor / no change needed.** The empty buffer, Adam moments carried over from rscaled10, and the episode reset are the
same as in FREE, so they are matched. `run_ep.sh`'s EDIT/EDREL cases are correct: TH = HL, `${ARM#EDIT}` → 1/2, the atomic
write, idempotent. The edit is recomputed rather than copied from Part C. Check that each EDIT's `roc_ckpt_final.json`
reproduces that model's Part-C O_k1/O_k2 d′; if `make_pairs_nofix` is unseeded, β differs slightly. CONT restarts the
buffer and counter through `--init-from` and has no null at 100k. If it shows a slow DiD, it needs a FROZNULL
continuation before it can be read. Do not launch that now.

## Part E review (rev 4, 2026-11-06)

Independent reviewer; covers §12, Part D results (README §4), `zk/grad_attrib.py`, `ep_train.py`, `ep_edit.py`,
`ep_geom.py`, `agent_orig.py` (grad_route) and `slurm/run_ep.sh`.

**Verdict: proceed after fixing majors 1–2.** The reframing (edit intact, so the erosion is compensation) is right, and
so is r_x drift-corrected within each manipulation.

**1. (major) The gradient projection is close to circular.** Δθ_comp = θ_EDREL(5k) − θ_EDIT is the summed update, mostly
value learning and noise. FREE's projection onto EDREL's weight path is not a drift correction, because the two runs
take different paths. Whatever dominates the total update (the KL, 0.84–1.00 in which-loss) dominates the projection,
so §12's "KL ≥ 0.7" is nearly guaranteed. *Fix:* project onto the gradient of the measure itself. Make
A = amp_slot(L)/amp_slot(other) on the `ep_geom` probe trials differentiable (W's axis held fixed), and compute
dA/dt ≈ −lr·⟨∇_θA, u_k⟩ for each loss k (u_k from `grad_attrib`; linear given v̂). Do the same in the paired FREE run.
Before reading any share, require Σ_k to predict the observed 1k→5k rate within a factor of 2. `grad_attrib` also needs
the G4a/b/c groups (norm1 currently falls in "ffn") and a mask for the 5,120 edited entries.

**2. (major) CD inherits the CRIT-DET pathology, and the ND trim rests on a false premise.** With the KL detached, v̂
decays (β2 0.999) to the scale of the tiny MPO/entropy gradients. Adam-normalized decay (1e-6, still on for every
non-q/k weight; `ep_train` exposes only `--qk-decay`) then takes steps of order lr on weak-gradient weights, the edited
entries included (§12: 280k steps could remove them). Low CD retention could be decay, not the actor. And decay *can*
compensate at L alone without touching the edit. The edit's offset is β·(w·u_L + b0), so shrinking u's position
coordinates (norm1 affine on 128–131; the token projection of the one-hot) erodes it only at L. *Fix (no extra cells):*
run CD with trunk decay 0 (a new `--rest-decay` flag), labeled "actor losses alone", and log the edit coordinate in every
E1 run. Keep ND conditional, but trigger it on decay's share of the predicted dA/dt (fix 1), not of the weight update.

**3. (moderate) Reversion is off-manifold.** The other groups co-adapted to the reverted one, and reverting also undoes
that group's ordinary drift. *Fix:* apply the same reversion in the paired FREE run (to W) and subtract; add the
complementary graft (EDIT + one end group). If revert and graft disagree, call it an interaction, not a locus. More
direct still: measure on the probe trials the *realized* output-gate offset at L (Õ with minus without β·1⊗w) and the
change in u's position coordinates at L. Together they show whether the compensation sits upstream of the gate or around it.

**4. (moderate) CD may retain because nothing moves.** *Fix:* report trunk displacement per group, plus a positive
control that the CD trunk still changes H (e.g. an H-borne DiD c change in FREE_CD, as FROZ showed). Compare r_x with
Part D on the same seeds (s42–46), paired.

**Other.** Five pairs suffices for ≥ 0.2 effects; report SDs, since detached arms may vary more. A both-detached
(decay-only) arm is not worth 39 GPU-h: it says nothing about the intact learner. AD is clean (the actor still learns
the value effect; v̂ is already KL-dominated). The re-warm is matched. Compute stays at 60 runs (≈ 78 GPU-h).
