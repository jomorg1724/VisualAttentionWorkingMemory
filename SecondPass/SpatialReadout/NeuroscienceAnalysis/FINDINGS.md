# Frozen neuroscience findings — checkpoint35039

Executed 14,336 scored trial presentations; 128 independent calibration trials and 52 map trial presentations. This is an exploratory frozen-observer analysis, not retraining or a later final test.

## Native competence
- chromatic_increment: D0 BA 1.000 (n=128)
- contour: D0 BA 0.984 (n=128)
- contrast: D0 BA 1.000 (n=128)
- motion_direction: D0 BA 1.000 (n=128)
- natural_spectrum: D0 BA 1.000 (n=128)
- orientation: D0 BA 1.000 (n=128)
- orientation_cued: D0 BA 1.000 (n=128), D4 BA 1.000 (n=128), D12 BA 1.000 (n=128), D24 BA 1.000 (n=128)
- orientation_ring: D0 BA 1.000 (n=128)
- spatial_binding: D0 BA 1.000 (n=128), D4 BA 1.000 (n=128), D12 BA 1.000 (n=128), D24 BA 1.000 (n=128)
- spatial_frequency: D0 BA 1.000 (n=128)
- Supplemental ring delays are OOD (training catalog is D0 only): D4 BA 0.672, D12 BA 0.648, D24 BA 0.641. Raw group=native means native renderer, not trained support; condition_summary.csv explicitly identifies OOD support.

## Signed orientation and matched cues
- D12 magnitude0°: accuracy 0.969, positive fraction 0.031, n=128. Analysis-only OOD magnitude.
- D12 magnitude3°: accuracy 0.547, positive fraction 0.062, n=128. Analysis-only OOD magnitude.
- D12 magnitude6°: accuracy 0.688, positive fraction 0.188, n=128. Analysis-only OOD magnitude.
- D12 magnitude10°: accuracy 0.938, positive fraction 0.438, n=128. Analysis-only OOD magnitude.
- D12 magnitude15°: accuracy 1.000, positive fraction 0.500, n=128. Native magnitude.
- D12 magnitude30°: accuracy 1.000, positive fraction 0.500, n=128. Native magnitude.
- D12 magnitude45°: accuracy 1.000, positive fraction 0.500, n=128. Native magnitude.
- Matched cue relocation changes the required report target; all pre-overlay scenes and rotations are physically matched. It is not classical valid/invalid cueing. See paired table and Figure2.
- The historical PsychOrientationStream was NOT bit-identical: an extra permutation changes RNG draws. The new sweep passed exact native pixel/label parity before magnitude overrides.

## Allocation and memory
- Actual trial/time/scale/head beta, mean8-row alpha, signed coefficients and separately computed absolute summaries are retained in NPZ. Final-read×all-source and source2×all-read axes are distinct. Reconstruction was tested against real pre-output-convolution reads.
- All10 tasks have representative maps; native cued orientation and binding D12 additionally have trial-bootstrap allocation summaries. No spatial cue comparator is assigned to sensory tasks.
- Binding retrocue arrives after retention: eventual-target differences before query cannot be called anticipatory selection. First2 blanks are frame-stack overlap.

- orientation_cued_D12 beta25 head0, frame2: target−mean3foils +0.01862 [-0.01358,+0.04695], n=16. This is a gate contrast, not final-decision attribution.
- orientation_cued_D12 beta25 head0, frame14: target−mean3foils +0.00000 [+0.00000,+0.00000], n=16. This is a gate contrast, not final-decision attribution.
- orientation_cued_D12 beta25 head1, frame2: target−mean3foils +0.02227 [-0.00422,+0.04907], n=16. This is a gate contrast, not final-decision attribution.
- orientation_cued_D12 beta25 head1, frame14: target−mean3foils +0.00000 [+0.00000,+0.00000], n=16. This is a gate contrast, not final-decision attribution.
- spatial_binding_D12 beta25 head0, frame2: target−mean3foils -0.00750 [-0.06091,+0.04839], n=16. This is a gate contrast, not final-decision attribution.
- spatial_binding_D12 beta25 head0, frame14: target−mean3foils +0.00000 [+0.00000,+0.00000], n=16. This is a gate contrast, not final-decision attribution.
- spatial_binding_D12 beta25 head1, frame2: target−mean3foils -0.00564 [-0.05668,+0.04320], n=16. This is a gate contrast, not final-decision attribution.
- spatial_binding_D12 beta25 head1, frame14: target−mean3foils +0.00000 [+0.00000,+0.00000], n=16. This is a gate contrast, not final-decision attribution.

## Paired functional interventions
- orientation_cued / stimulate_cued_probe_-1.0_feature: Δaccuracy -0.0625 [-0.1406, +0.0000], Δpositive +0.0000, Δd′ -0.9195, Δcriterion -0.2506; n=64.
- orientation_cued / stimulate_cued_probe_-0.5_feature: Δaccuracy -0.0469 [-0.1094, +0.0000], Δpositive -0.0156, Δd′ -0.6719, Δcriterion -0.1268; n=64.
- orientation_cued / stimulate_cued_retention_1.0_feature: Δaccuracy -0.0469 [-0.1094, +0.0000], Δpositive -0.0469, Δd′ -0.3072, Δcriterion +0.1536; n=64.
- orientation_cued / inhibit_cued_probe_1.0: Δaccuracy -0.0312 [-0.0781, +0.0000], Δpositive +0.0000, Δd′ -0.5700, Δcriterion -0.1778; n=64.
- orientation_cued / stimulate_cued_encoding_0.5_feature: Δaccuracy -0.0312 [-0.0781, +0.0000], Δpositive -0.0312, Δd′ -0.2091, Δcriterion +0.1046; n=64.
- orientation_cued / stimulate_cued_encoding_1.0_feature: Δaccuracy -0.0312 [-0.0781, +0.0000], Δpositive -0.0312, Δd′ -0.2091, Δcriterion +0.1046; n=64.
- orientation_cued / stimulate_cued_probe_1.0_random: Δaccuracy -0.0312 [-0.0781, +0.0000], Δpositive -0.0312, Δd′ -0.2091, Δcriterion +0.1046; n=64.
- orientation_cued / inhibit_cued_probe_0.5: Δaccuracy -0.0156 [-0.0469, +0.0000], Δpositive -0.0156, Δd′ -0.1072, Δcriterion +0.0536; n=64.
- Localizer held-out r=0.8177; feature validation passed=True. RMS=0.4574. Direction represents absolute sin2θ preference, NOT task-aligned change preference.
- Interventions touch only post-output finest KDA emissions. Same-module stored state is intact; downstream/coarser recurrence may carry effects. Feature-map support is not an isolated retinal lesion because convolutions/GroupNorm couple locations.
- Full effect table includes all tested doses, epochs, relative sites and random controls. The list above is exploratory ranking, not multiplicity-corrected significance.

## Interpretation and unmeasured comparisons
- No reaction-time policy, calibrated monkey current, neuronal excitation/inhibition identity or biological equivalence is claimed. Signed activation suppression is a functional perturbation.
- No classical cue-validity manipulation exists for this report rule. Foil rotations remain coupled by the native multiset; a fully independent factorial foil sweep and fitted evidence weights were not measured.
- Small causal bins (64TOTAL per condition before any profile reduction) limit psychometric shifts; no threshold/lapse or memory-decay fit is asserted. Trial uncertainty is not between-checkpoint generality.
- Image recognition is partial and omitted from solved-task scope. The two failing cued-motion tasks are outside this study. Sensory motion competence does not establish cued-motion competence.
- Allocation-versus-magnitude and matched-cue allocation contrasts were not collected: allocation maps use native D0 representatives and native D12 primary trials. No magnitude-dependent allocation mechanism is inferred.
- Causal epoch coverage is the pinned finite grid; the source CSV is authoritative. Binding random-direction stimulation was not included; cued orientation has those controls.

## Verification
- Main model unchanged: True; checkpoint bytes/hash unchanged: True.
- Saved-logit argmax, correctness and scored count replay passed for all14336rows.
- 34 figures, each PNG/SVG/PDF; offline trial/frame viewer; raw arrays and source CSVs.
- Budget origin/deadline: 1790690444.1326132 / 1790694044.1326132; rendering complete elapsed 1422.1s.
- Execution status: measured; errors: [].

## Final interpretation and audit additions
- Native signed-cue orientation and binding remain perfect at D0/4/12/24 in this128-per-cell sample; native ring D0 is perfect. Contour is126/128. The ring long-delay results are supplemental OOD, not part of trained-condition competence.
- At15/30/45 degrees, all128 matched cue pairs have both requested reports correct;80/128 pairs change the correct class and the observer follows that reassignment. This supports cue-conditioned evidence use, not a cue-validity benefit. The −.25 aggregate positive-response shift reflects changed report-label frequencies. Loglinear-corrected d-prime/criterion differences in the matched-cue CSV can arise solely from changed class denominators at ceiling and are not evidence of altered sensitivity/bias.
- The finest beta target-minus-foil contrast is exactly0 at late pure blank frame14 for both heads and both primary tasks. At encoding, cued-orientation beta contrasts are+.0186 and+.0223, with95% intervals spanning0. This does not show sustained finest-gate spatial prioritization. Finest gates depend on the current frame stack, not directly on that KDA module's stored state; flat gates do not imply absent memory. Signed source2-to-final coefficients also have target-minus-foil intervals spanning0 at all three scales/heads (16native D12 trials).
- Cued-orientation choice flips occur in16/30 cued-site,2/30 foil-site and0/30 background-site conditions. This is a descriptive count across dependent conditions, not30independent experiments or a significance test. Largest accuracy loss is4/64trials; no universal enhancement or multiplicity-corrected causal-selection claim is justified.
- Binding has0/64choice flips in each of24active intervention conditions. Each cell's95% Wilson upper flip-probability bound is5.6624%; zero-width paired bootstrap bars do not prove invariance. Confidence does move slightly in several cells; the table preserves continuous probability effects.
- Emission-substrate verification measured an unchanged finest KDA state(max difference0), a changed encoding emission(max difference2.3291), and downstream final-state propagation(max difference.2568). CPU replay agrees with saved MPS logits within3.46e-6.
- Full accelerator exposure is14,556presentations /210,438frames, including profiling, parity, calibration and repeated map extraction;14,336are scored experimental presentations. These are not14,556independent base scenes. Additional post-run CPU verification used4presentations and is recorded separately.
- Inference completed in147.55seconds and its worker is released(PID65967 is an unreaped defunct entry, not running). Two plotting failures were repaired using the existing saved data (floating-point Wilson error-bar roundoff; duplicate summary-key merge); no inference restart or deadline renewal occurred. The browser tool could not use the non-Chromium default profile; isolated headless Chrome successfully rendered the actual file://viewer, populated trial3/frame14 and emitted a screenshot/DOM, then needed bounded shutdown termination. This is disclosed inverification.json.
- Remaining omissions are unchanged: no independently factorialized foil sweep, no magnitude-conditioned/matched-cue allocation maps, no binding random-direction pulses, no recognition supplement; classical cue validity and reaction time are undefined for this protocol.
