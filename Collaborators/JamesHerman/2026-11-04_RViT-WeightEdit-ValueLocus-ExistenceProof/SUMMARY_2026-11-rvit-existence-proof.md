---
type: experiment
status: complete
updated: 2026-11-06
dates: 2026-11-04 → 2026-11-06
architecture: original
summary: Existence proof against "the architecture forces value onto the readout" (L&M SENS-C table). A single rank-1, position-gated edit of the LSTM's own output-gate weights gives an L&M-like value effect through a gain on one location's code (d′ 2.70 → 2.16 there, other location and criterion unchanged, information intact, amplitude ×0.54). But training never puts value there — with the readout frozen there is no sensitivity value effect (value term +0.011 [−0.010, +0.033]; free actor +0.888) — and when handed the edit, the learner undoes ~60% of it at the same rate whether it helps or hurts reward (retained 0.43 vs 0.42) while the actor builds the value effect. Part E (6 Nov): the edit is not decayed but disarmed (token projection re-maps its position trigger; gate offset to ~30%), and either loss alone does it (retention 0.38/0.29 actor-only without decay, 0.45/0.46 critic-only, 0.46/0.42 intact).
sources: [analyses/2026-11-04_rvit-existence-proof/README.md, documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md, documents/plans/RVIT_EXISTENCE_PROOF_PLAN_review.md]
---
# Existence proof: can value act on RViT's per-location code? — *is "value acts on the readout" forced by the architecture, or chosen by learning?*

**Answer (5 Nov).** Chosen by learning.
- **The architecture can carry value in the code.** One hand-written, rank-1, position-gated edit of the LSTM's
  output-gate weights at the low-value location gives a graded, location-specific value effect. At mild strength it is
  L&M-like: d′ 2.70 → 2.16 there, the other location 2.65 → 2.68, criterion unchanged. It works as a gain on that
  location's code (change amplitude ×0.54), with the change information intact (tested-slot loss 0.03). Stronger, it also
  removes information (loss 0.15 [0.11, 0.20]).
- **Training does not put it there.** With the actor frozen, 50k episodes give no sensitivity value effect. The
  congruent/reversed value term is +0.011 [−0.010, +0.033], against +0.888 [+0.861, +0.915] with the actor free. The
  frozen learner did move the criterion through H (−0.29), so it was not stuck.
- **Training does not keep it when given it.** From the edited models, the standard learner undoes about 60% of the
  amplitude effect within 50k episodes, and all of the k = 2 information loss. The rate is the same for a value-consistent
  edit (0.43 retained) and a value-opposing one (0.42). Meanwhile the actor builds FREE's value effect (DiD d′ +0.91).

## Design
- **Architecture:** [[rvit-original]], zk tree; `zk/ep_train.py` wraps `zk/train_orig.py` unchanged (frozen actor,
  two-group Adam with q/k decay 0, θ frozen). Task [[task-lm]], SENS-C table ([[2026-10-lm-sens-recenter]]).
- **Start models:** the which-loss reward ×10 models ([[2026-10-which-loss]]; 10 counterphased pairs, s42–51): no
  value effect, equal encoding. Each got a 5k critic-only warm-up with a new head ([[critic-warmup-finetune]]).
- **Part C (hand edits, no training; 20 models):** a rank-1 edit of the LSTM input→gate weights along the low-value
  location's position direction. C-O is the output gate (a gain on that location's memory output); C-F is the forget
  gate. Strength k is in pre-activation SDs (0.5–8). A first pass with a leaky direction (about 0.75 of the offset on the
  other stimulus slot) was discarded.
- **Part A (50k fine-tunes, 10 pairs each):**
  - FREE and FREE-REV (standard learner; congruent and reversed value assignment);
  - FROZ and FROZ-REV (actor frozen);
  - FROZ-NULL (frozen, start table kept).
  - The reversed arms cancel the start actor's existing gain asymmetry (the "value term").
- **Part D (replaced the planned release-from-FROZ, which was moot):** the C-O edit written into W, a critic re-warm, then
  50k of the standard learner. EDREL1 (k 1 at the low-value location), EDREL2 (k 2; 5 pairs) and EDREL1H (k 1 at the
  *high*-value location: the inertia control).
- **Measures:**
  - the 18° assay;
  - `lm_readout_probe` (tested-slot AUC, pooled decoder, P9 slope, attention);
  - whole-state AUC and fixed-axis amplitude (`zk/ep_geom.py`);
  - the 2×2 actor × transformer hybrid (`zk/ep_hybrid.py`).
- **Plan and reviews:** two independent reviews ("revise before running"; addendum "run Part D but add a control").
  CRC jobs: 4664612 / 4665127 (Part C), 4664614 / 4664615 / 4673533 / 4673535 (Part A), 4671962 / 4671963 (Part D),
  4696871 / 4696758 / 4700032 (Part E).

## Results
- **Part C** (`analyses/2026-11-04_rvit-existence-proof/README.md` §2):

  | edit | d′ at L | d′ at the other location | information loss | amplitude loss (pooled) |
  |---|---|---|---|---|
  | C-O k 1 | 2.16 | 2.68 | 0.03 [−0.01, 0.08] | 0.46 |
  | C-O k 2 | 0.89 | 2.49 | 0.15 [0.11, 0.20] | 0.77 |

  - At k 1, c at L is unchanged (+0.32); at k 2 it moves +0.46.
  - C-F is not location-specific (k 2: 0.51 vs 1.31).
- **Part A** (§3), DiD d′ at 50k:

  | arm | DiD d′ | transformer share of the change |
  |---|---|---|
  | FREE | +0.90 | −0.02 [−0.04, 0.00] |
  | FREE-REV | +0.87 | −0.08 |
  | FROZ | +0.04 | 1 by construction |
  | FROZ-REV | −0.02 | 1 by construction |
  | FROZ-NULL | −0.18 | 1 by construction |

  - Value terms: FREE +0.888, FROZ +0.011. The FROZ trajectory is flat (10k / 25k / 50k).
  - Encoding, amplitude and attention are equal in every arm. FREE's P9 slope is 4.74 / 0.90 (the from-scratch pattern).
- **Part D** (§4), amplitude retained beyond FREE and the end-state transformer share:

  | arm | amplitude retained (r) | transformer share at the end |
  |---|---|---|
  | EDREL1 | 0.43 [0.41, 0.45] | 0.06 |
  | EDREL2 | 0.37 | 0.10 |
  | EDREL1H | 0.42 | −0.03 |

  - EDREL2's information loss is fully restored (r −0.05).
  - EDREL1H's behavioral cost is gone within 1k episodes (DiD d′ −0.51 → −0.08).
- **Measurement point:** a gain placed in H is invisible to the tested-slot AUC (P4 would call C-O k 1 "readout") and to
  the P9 slope at L (2.09 → 2.22). The pooled separation and the fixed-axis amplitude see it. The trained SENS-C models
  have equal pooled separation (1.85 vs 1.82), so they do not use the gain route.

- **Part E, what erodes the edit (6 Nov; README §5):**
  - The edited weights stay 99.5% intact, as decay alone predicts (99.8%), so the edit is not undone. It is disarmed: the
    token projection re-maps the position signal that triggers the edit, so its realized gate offset falls to 0.32 / 0.29
    of the edit value (unedited controls 0.89 / 0.93).
  - Reverting the token projection alone restores 82–98% of the edit's effect, and grafting it reproduces 94–98% of the
    compensation; every other trunk group carries ≤ 7%.
  - Causal arms (5 pairs; retention at the low / high location):

    | arm | retention |
    |---|---|
    | intact | 0.46 / 0.42 |
    | critic detached, no decay | 0.38 / 0.29 |
    | actor detached | 0.45 / 0.46 |

  - Each loss alone is enough, and decay is not needed.
  - The intact learner's equal rate is set by the critic. Without it the actor removes the costly edit slightly faster
    (low − high +0.10 [−0.04, +0.24]).
  - The per-loss gradient attribution failed its own validity check (wrong sign, ×190) and is not read.

## Caveats and corrections
- Part C is a construction; Parts A and D are fine-tunes of 50k (not from scratch). FREE reproduces the from-scratch
  locus.
- The Part D amplitude verdict is "partial" by the planned rule (r ≈ 0.4); the direction and its independence from value
  are the result.
- The force that undoes the edit is inferred (the critic's symmetric KL, by [[2026-10-which-loss]] Phase 0), not tested.
- The first-order Jacobian decomposition and the II-gain / II-rot cosine cut were not usable (uncalibrated); they are
  reported but not interpreted.
- **Operational:** replacing `run_ep.sh` on CRC while jobs ran ended those scripts silently after training. Their assays
  were rerun from the saved checkpoints.

## Where it lives
- **Analysis:** `analyses/2026-11-04_rvit-existence-proof/` (README, `stats/ep_stats.json`, figures
  `figs/fig_partC|partA|locus|partD.{png,pdf}`, `data/runs/`).
- **Code:** `tree_zk/zk/ep_{train,hybrid,geom,edit,make_edit,stats,figs}.py` and `zk/tests/test_ep_train.py`; slurm
  `run_ep.sh`, `ep_cells.slurm`, `ep_edit.slurm`. CRC runs are in `rvit_zk/runs/existence/`.
- **Plan:** `documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md` rev 3, with its review file.

## Bears on
- [[value-acts-on-readout]] **refines** it (Part E adds: the trunk's gradients, critic's or actor's, each remove value put into the code, so the readout locus is a property of shared-trunk gradient learning, not of one magnitude-free loss): the readout locus is a property of the learning rule, not a limit of the
  architecture. A one-edit gain in the code is possible, the learner does not find it, and it removes one that is
  handed to it. The claim's evidence must cite pooled amplitude as well as the AUC.
- [[2026-10-which-loss]] is the mechanism candidate (the critic's KL); [[2026-11-attention-code-clamp]] is the
  companion analysis on attention vs the code; [[2026-10-13-nmi-scope-decisions]] entry 15.

<!-- backlinks:start -->
## Linked from

- [[2026-10-13-nmi-scope-decisions]]
- [[value-acts-on-readout]]
<!-- backlinks:end -->
