# Factorized frozen diagnostic — selected2297

**Exploratory reused-test follow-up, not a new independent confirmation.** No neural extraction, deployed training, stimulus changes or cloud work.

## Decision: no operational rescue; localization remains the clearest measured limitation

The stored-phase probe reads the cue perfectly, but cannot reliably localize the changed patch: side BA **0.489** [0.413,0.571], AUC **0.514** [0.420,0.609] on149 original event episodes. Any-event detection is also weak: BA **0.556** [0.482,0.638], AUC **0.573** [0.448,0.702]. It declares change on137/149 events **and21/26 catches**; high raw accuracy is therefore not strong event discrimination. Final-only event/side AUCs are0.473/0.502. Neither route establishes a competent event representation under these probes.

Supplying the **true cue alone** does not rescue target reporting (phase hard BA0.488; cue was already decoded perfectly). Supplying **true changed side on event trials**, while retaining predicted cue and predicted event detection, raises phase native BA to **0.874** [0.819,0.922]; supplying **true event alone** gives0.576 [0.506,0.648]. Supplying both true event and true side with predicted cue gives1.000. These **nondeployable substitutions** identify localization as a major bottleneck *in this analysis pipeline*, with residual catch/event detection errors. They do not identify a unique causal lesion in the frozen neural model. The relational rule in this oracle calculation is supplied by the analyst, not learned native binding.

With **predicted inputs only**, the learned factorized calibrator remains at native BA **0.500 final /0.491 phase** (AUC0.535/0.514); final predicts all negatives, phase produces only14 positive reports. Relative to the direct label-only probe, native BA gains are+0.021 [-0.037,+0.078] final and−0.035 [-0.119,+0.046] phase. On the paired cue challenge, phase hard composition flips157/175 pairs but gets both answers correct on only67/149 changed-event pairs: cue sensitivity is not correct evidence binding. Learned phase calibration flips33 pairs and gets both correct on17/149 changed pairs. The native head still flips none and predicts all positives (native BA0.500, paired BA0.500).

The compressed final cue decoder here is weaker than the prior full-memory linear decoder (paired BA0.526 here versus0.663 previously). This follow-up uses300 rather than425 fit groups and96 projected coordinates expanded quadratically rather than the full3136-dimensional linear state. **Do not reinterpret this as new evidence that the cue was erased.** Final-only oracle event+side with predicted cue reaches only0.610 native BA, so cue recovery also limits this particular compressed final pipeline. No extra probe was chosen after viewing these reused test results.

Bootstrap intervals of[1,1] for phase cue and ideal truth substitutions are empirical resampling degeneracy, not certainty. The finite-sample Wilson95% lower bound for175/175 independently correct native episodes is recorded in `finite_sample_bounds.json`; paired rows do not double the sample. Multiple comparisons and reused-test exploration preclude new independent confirmation.

## Measured accessibility

Balanced accuracy [95% independent-group bootstrap interval]. Native = only original untouched episodes; paired = both cue variants, not twice the independent sample.

| Target / comparator | Final native | Phase native | Final paired | Phase paired |
|---|---:|---:|---:|---:|
| event | 0.483 [0.467, 0.497] | 0.556 [0.482, 0.638] | 0.483 [0.467, 0.497] | 0.556 [0.482, 0.638] |
| side | 0.517 [0.431, 0.602] | 0.489 [0.413, 0.571] | 0.520 [0.432, 0.605] | 0.486 [0.410, 0.568] |
| joint | 0.349 [0.290, 0.411] | 0.343 [0.278, 0.410] | 0.349 [0.290, 0.411] | 0.345 [0.280, 0.411] |
| cue | 0.522 [0.448, 0.598] | 1.000 [1.000, 1.000] | 0.526 [0.511, 0.543] | 1.000 [1.000, 1.000] |
| label | 0.479 [0.422, 0.537] | 0.527 [0.451, 0.601] | 0.500 [0.480, 0.520] | 0.493 [0.432, 0.553] |
| hard | 0.472 [0.394, 0.548] | 0.488 [0.412, 0.563] | 0.478 [0.443, 0.510] | 0.502 [0.438, 0.571] |
| soft | 0.467 [0.409, 0.520] | 0.489 [0.417, 0.559] | 0.482 [0.453, 0.508] | 0.490 [0.430, 0.553] |
| calibrated | 0.500 [0.500, 0.500] | 0.491 [0.450, 0.537] | 0.500 [0.500, 0.500] | 0.517 [0.484, 0.553] |
| shuffled_label | 0.501 [0.444, 0.564] | 0.451 [0.377, 0.523] | 0.498 [0.474, 0.521] | 0.460 [0.412, 0.507] |

Event: catch versus any physical change; side: left versus right **conditional on a real event** (evaluation/training subset only, never an operational input). Joint: left/right/catch. Label: target-only report, not any-change detection. Native n175 (97 target,52 foil,26 catch); event-side n149. Paired n350 rows/175 groups, event-side298 rows/149 groups. AUC, confusion, exact denominators and paired-difference intervals are in metrics.json.

## Diagnostic oracle substitutions — NOT deployable

Hard supplied relation: event AND (changed side = cue). One truth substitution at a time; these are diagnostic ideal-component ceilings, not operational scores or guaranteed monotonic upper bounds. Side substitution applies only to true events; predicted side remains on catches. No truth enters hard/soft/calibrated operational inference.

| Substitution | Final native | Phase native | Final paired | Phase paired |
|---|---:|---:|---:|---:|
| oracle_cue | 0.487 [0.416, 0.558] | 0.488 [0.412, 0.563] | 0.512 [0.440, 0.583] | 0.502 [0.438, 0.571] |
| oracle_event | 0.567 [0.498, 0.638] | 0.576 [0.506, 0.648] | 0.554 [0.527, 0.580] | 0.553 [0.484, 0.627] |
| oracle_side_on_events | 0.515 [0.444, 0.594] | 0.874 [0.819, 0.922] | 0.512 [0.480, 0.543] | 0.907 [0.879, 0.938] |
| oracle_event_side | 0.610 [0.543, 0.686] | 1.000 [1.000, 1.000] | 0.588 [0.563, 0.616] | 1.000 [1.000, 1.000] |
| oracle_all | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## Paired cue challenge

| Operational output | Flips /175 | Both correct /175 | Both correct event /149 |
|---|---:|---:|---:|
| native/label | 0 | 0 | 0 |
| final/label | 0 | 21 | 0 |
| phases/label | 106 | 53 | 43 |
| final/hard | 10 | 15 | 4 |
| phases/hard | 157 | 72 | 67 |
| final/calibrated | 0 | 26 | 0 |
| phases/calibrated | 33 | 38 | 17 |

## Protocol, interpretation and limitations

Split contracts verified before fitting:425train/100validation/175test independent physical groups with two matched cue variants. Fixed RNG splits training into300 component/direct-probe groups and125 disjoint calibrator groups. Every calibrator training prediction is from a component decoder that never saw that group (including scaling). Components are not refitted afterward. Validation-only alpha selection from1/100/10000, first ties; fixed0.5 decisions for hard/soft. Label-only and group-shuffled controls use the same300 groups. No test-guided choice or sweep.

Final and external phase accesses each use the same4752 quadratic features:96 projected coordinates plus upper-triangle products. Projection is fixed/reused, not supervised. Final uses three fixed permutations of final memory; phase uses cue/baseline/post memory. Same coefficients, fit groups and alpha grids across access pairs. Factorized route has extra auxiliary supervision, three component decoders and seven calibration inputs, unlike the direct label probe; its capacity is matched across temporal access, not claimed equal to direct label-only. Raw ridge class1 scores are clipped to[0,1] for composition and are not claimed calibrated probabilities. Learned calibration uses seven multilinear terms and native target labels on separate groups.

Phase access is external diagnostic memory unavailable to the deployed head. Truth never selects a patch or feature. Input-only final prediction was invariant with all earlier arrays removed and independently poisoned with NaNs, including every component and calibrated output. Checkpoint/source/extraction/feature hashes matched original receipts; metadata hashes recorded. Saved fits independently reconstruct all predictions and all score summaries are recomputed.500 independent-group resamples retain cue pairs; intervals are descriptive, unadjusted for multiple exploratory comparisons and conditional on fixed learned probes. Natural-frequency metrics use only original variant0; counterfactual cue-pair frequencies are not native.

Probe failure is not erasure. Feature projection, small catch counts, ridge capacity, sample size, and auxiliary teaching can limit decoding. Success shows accessible/composable information, not native learned use, a causal lesion, biological attention or a validated architecture remedy. All oracle results are nondeployable. One selected checkpoint only; classical validity benefits, causal inhibition/microstimulation and response timing remain unmeasured.

## Reproduction

Use `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python replay.py` for no-fit replay within the recorded deadline. `factorized.py` refuses any existing budget; re-fitting requires a separately authorized empty output and new cap. `test_contract.py` checks input-only final access and relation semantics without fitting. Artifacts: protocol/input_receipt/split_contract/budget/selection/test_freeze/fitted_models/projection/test_predictions/scoring_inputs/bootstrap_scores/metrics/verification/completion, plus these scripts. Original features remain in sibling FrozenDiagnostic/. New nonrenewable600-second cap starts at first fit; at most two CPU numerical threads, no main-model optimizer steps.

## Final numerical audit

All14 saved ridge fits passed exact fit-group scaler reconstruction, selected validation-score replay and regularized normal-equation checks (maximum relative residual3.54e-10); `audit_fits.py` also saved the disjoint-group calibration inputs. Independent held-out prediction replay error was0.0. Finite-sample Wilson95% accuracy lower bound at175/175 is0.9785; zero flips upper bound is0.0215. These are group/episode accuracy bounds, not BA intervals. Original features/checkpoint remained untouched.
