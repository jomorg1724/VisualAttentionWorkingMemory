# Factorized frozen Krauzlis diagnostic — selected2297

Exploratory reused-test follow-up; not new independent confirmation.

- event, native BA: final 0.483 [0.467, 0.497]; external phase 0.556 [0.482, 0.638].
- side, native BA: final 0.517 [0.431, 0.602]; external phase 0.489 [0.413, 0.571].
- joint, native BA: final 0.349 [0.290, 0.411]; external phase 0.343 [0.278, 0.410].
- cue, native BA: final 0.522 [0.448, 0.598]; external phase 1.000 [1.000, 1.000].
- label, native BA: final 0.479 [0.422, 0.537]; external phase 0.527 [0.451, 0.601].
- calibrated, native BA: final 0.500 [0.500, 0.500]; external phase 0.491 [0.450, 0.537].

300 component-fit and125 disjoint calibrator-fit training groups;100 validation and175 reused-test physical groups. Operational comparator sees predicted cue/event/side only; no in-sample component outputs in calibration. Matched temporal capacity, train-only scaling, validation-only selection,500 group bootstraps, shuffled-pair control and independent fitted-model replay. Original untouched native episodes scored separately from350 counterfactual-pair rows. No neural extraction, training, cloud or stimulus change. Probe failure is not erasure; oracle substitutions are nondeployable diagnostics.

[Full report and artifacts](../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/FactorizedDiagnostic/REPORT.md).

## Decision: no operational rescue; localization remains the clearest measured limitation

The stored-phase probe reads the cue perfectly, but cannot reliably localize the changed patch: side BA **0.489** [0.413,0.571], AUC **0.514** [0.420,0.609] on149 original event episodes. Any-event detection is also weak: BA **0.556** [0.482,0.638], AUC **0.573** [0.448,0.702]. It declares change on137/149 events **and21/26 catches**; high raw accuracy is therefore not strong event discrimination. Final-only event/side AUCs are0.473/0.502. Neither route establishes a competent event representation under these probes.

Supplying the **true cue alone** does not rescue target reporting (phase hard BA0.488; cue was already decoded perfectly). Supplying **true changed side on event trials**, while retaining predicted cue and predicted event detection, raises phase native BA to **0.874** [0.819,0.922]; supplying **true event alone** gives0.576 [0.506,0.648]. Supplying both true event and true side with predicted cue gives1.000. These **nondeployable substitutions** identify localization as a major bottleneck *in this analysis pipeline*, with residual catch/event detection errors. They do not identify a unique causal lesion in the frozen neural model. The relational rule in this oracle calculation is supplied by the analyst, not learned native binding.

With **predicted inputs only**, the learned factorized calibrator remains at native BA **0.500 final /0.491 phase** (AUC0.535/0.514); final predicts all negatives, phase produces only14 positive reports. Relative to the direct label-only probe, native BA gains are+0.021 [-0.037,+0.078] final and−0.035 [-0.119,+0.046] phase. On the paired cue challenge, phase hard composition flips157/175 pairs but gets both answers correct on only67/149 changed-event pairs: cue sensitivity is not correct evidence binding. Learned phase calibration flips33 pairs and gets both correct on17/149 changed pairs. The native head still flips none and predicts all positives (native BA0.500, paired BA0.500).

The compressed final cue decoder here is weaker than the prior full-memory linear decoder (paired BA0.526 here versus0.663 previously). This follow-up uses300 rather than425 fit groups and96 projected coordinates expanded quadratically rather than the full3136-dimensional linear state. **Do not reinterpret this as new evidence that the cue was erased.** Final-only oracle event+side with predicted cue reaches only0.610 native BA, so cue recovery also limits this particular compressed final pipeline. No extra probe was chosen after viewing these reused test results.

Bootstrap intervals of[1,1] for phase cue and ideal truth substitutions are empirical resampling degeneracy, not certainty. The finite-sample Wilson95% lower bound for175/175 independently correct native episodes is recorded in `finite_sample_bounds.json`; paired rows do not double the sample. Multiple comparisons and reused-test exploration preclude new independent confirmation.
