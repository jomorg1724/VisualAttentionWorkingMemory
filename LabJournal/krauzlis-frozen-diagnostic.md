# Native-angle Krauzlis frozen diagnostic — selected2297

## Decision and measured diagnosis
**No go for a claim that a simple replacement readout rescues this checkpoint. Inconclusive about the unique failing mechanism.** The early cue is not wholly erased: full ConvGRU memory decodes it perfectly after the five-frame gap and above chance at report. Coarse physical direction is also accessible, with mean angular errors larger than the26/28° changes; correlated errors could still cancel in a comparator, so this is not proof that event information is absent. Physical changed-patch decoding is weak, and neither final-only nor capacity-matched stored-phase label probes establish correct cue-conditioned report accessibility. This favors investigating fine-motion comparison/binding rather than asserting complete cue loss; it does not identify a causal lesion or justify altering task teaching.

Native fresh n175: BA0.500, AUC0.622; all positive. Memory cue BA: gap1.000, baseline0.831, post0.709, final0.663. Memory direction error41–46°; native changes26/28°. Final-memory label BA0.513/AUC0.490; matched phase-linear BA0.488, phase-quadratic0.501; no reliable paired temporal-access gain.

425/100/175 independent train/validation/test groups, two cue variants each,71 frozen ridge probes; grouped held-out uncertainty and shuffled-label controls. Native/wrapper logits and renderer/RNG parity passed; checkpoint/tensor hashes unchanged, zero deployed optimizer updates, no cloud access.

Full artifacts: `SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/FrozenDiagnostic/REPORT.md`. One nonrenewable1800s cap covered profile, extraction, fitting and report; core run435.53s, final audit elapsed in `final_audit_receipt.json`.
