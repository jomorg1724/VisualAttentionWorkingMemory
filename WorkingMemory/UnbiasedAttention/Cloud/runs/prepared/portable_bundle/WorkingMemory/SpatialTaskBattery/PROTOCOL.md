# Five-task spatial attention and memory battery

Status: **authorized for implementation/training; no results claimed by this protocol**. Defined 2026-09-14 UTC after the user's request for new spatial tasks and unbiased attention. This file defines stimuli, labels and reporting for the new battery. The companion continuation comparison on the old three-task battery has its own executable configuration; old and new task scores must not be treated as paired measurements.

The researcher implements generators and runs experiments. Training uses the authorized finite cloud allocation and profiled exposure recorded by the launcher. This protocol introduces no extra GPU run, validation gate or budget extension.

## Shared interface and task matrix

Each frame is RGB100×100, tensor `[3,100,100]`; an episode is `[T,3,100,100]`. Only rendered images enter the network. Labels, target locations, signs, directions, event indices and source identities are metadata for supervision and analysis, never privileged model inputs. Structural batching of unequal lengths must not make padded frames into extra sensory evidence.

Four-region tasks use centers `(27,27),(73,27),(27,73),(73,73)` and localized Gabor support radius12pixels and motion aperture radius11.5pixels, clear of the existing reserved corner glyphs. Region ordering is top-left, top-right, bottom-left, bottom-right. Rendered task/phase glyphs reuse the existing visual vocabulary where applicable; the supplied task key selects a supervised output head, as in previous experiments. It is not inferred from unrestricted language.

| Task key | Output | Principal conditions | Cue meaning |
|---|---|---|---|
| `orientation_cued` | Binary: relevant signed target rotation occurred | D0/4/12/24 | Spatial ± glyph in precue and both sample frames |
| `motion_duration_cued` | Four cardinal direction classes | Eight target transitions; D0/4/12/24 | Target ring throughout precue/reference/motion; no direction-sign filter |
| `spatial_binding` | Binary: queried location was reassigned | D0/4/12/24 | Retrospective target-location query after the delay |
| `image_recognition` | Binary: exact probe image belongs to the displayed list | List length0/4/12/24; probe repeats3/4/5 | Generic report/phase cue, no answer or item-identity cue |
| `krauzlis_cued_motion` | Binary: mean direction changed in the precued patch |12/20/28 baseline transitions, then8post-event transitions | Brief ring marks target; foil changes must be ignored |

Here D is a count of inserted blank frames, not a calibrated biological duration. Only the Krauzlis adaptation below explicitly defines seconds per rendered frame. All phases, including blanks, go through the ordinary model; no phase-oracle update gate is supplied.

## 1. Spatially cued signed orientation change

Use four independently oriented Gabors. The brief precue places the existing plus/minus glyph at the selected region: the horizontal minus stroke means negative rotation is relevant; the added vertical stroke makes plus, meaning positive rotation is relevant. Reuse the glyph geometry/sign convention from `WorkingMemory.stimuli.visual_cues`, translated to cue the selected region. This is **not** the old recall identity glyph.

The user confirmed a **single sample-to-probe signed change**, not the older cumulative-threshold walk. For target regionq and signc∈{−1,+1}, with signed shortest axial rotationΔθq, the label is

$$y=\mathbf1[c\,\Delta\theta_q>0].$$

Positive examples rotate the target in the sign indicated by the cue. Negative examples either leave the target orientation unchanged or rotate it oppositely, equally often. Target/sign/label are counterbalanced. Rotation magnitudes are drawn from15°,30°,45°; initial axial orientations are uniform on[0,π). The smaller localized raster and these magnitudes are declared new-task choices, not the previous full-field orientation task unchanged.

Prevent global-change shortcuts: first generate the four-region signed-change multiset `[-1,0,+1,random(-1,0,+1)]` independently of label. Assign an aligned entry to the target on positives, and zero/opposite to the target on negatives; shuffle remaining entries onto foils. Use the same magnitude law for target and foils. Thus the label cannot be decoded from whether anything changed or the global histogram of signed changes. Fresh Gabor phases/noise remove literal raster matching without altering orientation labels.

Timing: cue1frame; four-Gabor sample2frames; D blank frames; probe1frame carrying the ordinary report-phase marker. Total `D+4`frames. The sign/location glyph is visible in the precue and both sample frames, then disappears for the delay and probe. This is the pinned implemented timing; do not describe it as a cue-only flash. Record actual angles, signed deltas, cue region and rendered changes in analysis metadata.

## 2. Spatially cued motion-duration judgment

Four independent random-dot patches occupy the four regions. A spatial ring identifies the target in the precue, reference and all eight moving frames (indices0–9), then disappears during blanks and report. It supplies **location only**; do not apply an arbitrary ±direction relevance filter. Reuse the existing four cardinal class order from `DIRECTION_VECTORS` and duration oracle.

Each patch has one reference position frame followed by eight motion transitions. For the target's direction sequence`d1…d8`, report the unique winner

$$y=\arg\max_{k\in\{0,1,2,3\}}\sum_{t=1}^{8}\mathbf1[d_t=k].$$

Generate class-balanced target schedules with no tied winner. Foils have independent schedules drawn by the same law and must not predict the target answer. Balance target locations. The localized renderer uses32dots per patch, radius11.5pixels, and a shared per-trial displacement chosen from.8/1.2/1.6pixels per transition. It randomly replaces16dots per patch per transition, also replacing boundary exits. These are disclosed localized adaptations of the existing duration renderer, separate from the Krauzlis lifetime rule. Do not use the separate Krauzlis direction-change distribution here or claim that this duration task reproduces that experiment.

Timing: location cue1frame; dot reference1frame;8transition frames; D blank frames; report1frame. Total `D+11`frames. The response is the target patch's greatest **total duration**, not its final direction, net displacement, or the winner pooled across patches. Report target winner, counts, final direction and count margin as metadata for stratified analysis.

## 3. Four-location orientation binding with a retrospective query

Show four distinguishable orientations at the four region centers, with random assignments. Use an axial inventory obtained from a random base orientation plus offsets0°,45°,90°,135° modulo180°, then randomly permute its assignment to locations. The regular inventory is an engineering simplification and remains identical across the two labels.

After the sample and delay, reveal an unpredictable target-location query. On positive trials exchange the target orientation with one randomly chosen foil. On negative trials exchange two of the three foils while preserving the target. **Exactly one pair exchanges on every trial:** both classes contain two changed locations and exactly the same orientation inventory. Do not add identity/no-swap negatives to the primary mixture, since they would make global changed-location count partly predictive.

The label is target reassignment, not “anything changed.” Draw the queried region uniformly, independently of sample identity. Counterbalance sample permutations, query region and swapped pairs. Use fresh phases/noise at probe. A probe-only orientation, total inventory or total number of changed sites cannot determine the label. Retrospective query means any of the four locations may later matter; nevertheless, task performance alone does not prove a literal four-slot biological store.

Timing: instruction1frame; sample2frames; D blank frames; ring query1frame; full-array probe1frame carrying report marker. Total `D+5`frames. **The target is not precued for this task.** This differs intentionally from orientation and motion selection; it tests retrieval of a feature-location association after uncertainty about which location will matter.

## 4. Exact-image list recognition

Use the already downloaded BSDS500 photographs, keeping original source identities disjoint between train/validation/test:200/100/200. The existing provenance manifest records files, splits and hashes. This is a new recognition task using those photographs, not the earlier spectral-discrimination objective or a segmentation benchmark. Preserve source attribution and make no new license claim.

List length L is0,4,12or24. All L list images appear **consecutively, one frame per image**, followed by **three blank frames total** after the complete list. There are no interleaved blanks between study images. No source identity repeats within a list. The probe is then shown for K=3,4or5 consecutive **identical** frames. Choose K independently of label/list length. Classification occurs after the final probe repeat; do not secretly convert repetitions into new independent images or average their training labels as independent episodes.

For L>0, balance positive and negative membership. A positive probe uses the exact stored item raster from a uniformly chosen list position. A negative probe comes from a source identity outside the current list, from the same official split and identical preprocessing law. Use a canonical central square crop and bicubic downsampling to100×100RGB, fixed per source image. Apply no glyph overlays to study/probe frames; positive probes are bit-identical to their study raster, and probe repeats are also bit-identical. Report serial position and image age. No item index, list length number or answer is supplied as an oracle token.

For L=0 every probe is negative. Report **specificity/false-positive rate** for that condition; binary BA and AUC are undefined with only one class and must be omitted from checkpoint selection, not assigned artificial chance/perfect values. Keep L=0 separate from nonempty-list aggregates.

Timing: instruction1frame; L consecutive study-image frames; three blank frames after the list; K probe repeats. Total `L+4+K`frames. This corrects the initially proposed `1+4L+K` interleaved-blank timeline before production; use the implemented consecutive-list law. Each underlying episode counts once, regardless of list length/repetition count. Repeated source photos across episodes are not independent new photographs. Final source-aware uncertainty must acknowledge the finite photograph pool and list-level dependence.

## 5. Arcizet–Krauzlis target-versus-foil motion change

Use **Arcizet and Krauzlis (2018)** as the single primary recipe, not an unlabeled blend with Zénon2012 or coherence-pulse studies. The target is a flashed ring-cued patch, and the foil is diametrically opposite. The response means target direction changed; a foil event and a catch trial require a negative response. [Primary-source facts and boundaries](SOURCES.md).

The following are **our explicit100×100and short-sequence adaptations**. They are not a replication of the original trial durations or joystick response-time distribution.

### Raster and motion

Set centers `(20,50)` and `(80,50)`, with central fixation at `(50,50)`. Use2.5pixels/degree: eccentricity30pixels=12°, aperture radius8.125pixels=3.25°. The two baseline patch means differ by90°. Draw one global mean angle uniformly, then offset the other patch by90°. Target side is balanced and chosen independently of change type.

Use16dots per patch as a deliberate raster/compute adaptation. The primary density's time denominator is ambiguous for instantaneous count; do **not** claim that16is a derived replication of its printed density. Use subpixel bilinear splatting: constant RGB background.5, additive dot density contrast.48 with clipping to[0,1], fixation cross.1 and cue ring.95. These are digital values, not calibrated cd/m². Render fractional positions with bilinear splatting or a fixed smooth kernel; rounding0.375pixel steps into misleading immobility is inappropriate.

Render at a defined100Hz (`Δt=.01s`). Speed15°/s converts dimensionally to

$$15\;\mathrm{deg/s}\times.01\;\mathrm{s/frame}\times2.5\;\mathrm{px/deg}=.375\;\mathrm{px/frame}.$$

Each dot gets a Gaussian direction offset with SD16°, sampled independently on birth and held for that lifetime. This is angular dispersion, **not a coherent-dot percentage**. Lifetime is10frames; initialize staggered ages to avoid a global renewal flash. At death or aperture exit, regenerate within the disk independently of event type. Preserve positions and noise offsets when the mean direction changes so the event is a velocity change rather than a new dot cloud. The exact birth/boundary convention is our disclosed implementation choice.

### Event mixture and label

Use target-change57%, foil-only-change29%, and catch14%, retaining the recipe's mixture. On event trials change the selected patch mean by±26°or±28°, balanced over magnitude and CW/CCW. Do not change both patches. Catch trials have a sampled virtual event index with the same duration law. No visual marker announces the event. Render enough consecutive displacement evidence both before and after the event.

Only target-change is positive; foil and catch are negative. Report hit rate, foil false-alarm rate and catch false-positive rate separately, alongside binary BA/AUC and event counts. Overall accuracy at this mixture is not a substitute for those components. Do not make the cue color/shape depend on event outcome.

### Compressed timing

Use2cue frames (20ms),5fixation-only frames (50ms), then one dot-reference frame. Choose `Npre∈{12,20,28}` unchanged-direction baseline transitions; then8post-event transitions; finally one binary report frame. Total `17+Npre`frames =29/37/45. At100Hz the pre-event motion evidence spans120/200/280ms and post-event evidence80ms. Frame0–1cue,2–6blank,7reference,8through`7+Npre`baseline transitions, `8+Npre`through`15+Npre`post-event transitions, `16+Npre`report.

Cue and pre-motion blank intervals are compressed tenfold relative to the selected source; baseline and post-event durations are shortened separately. The15°/s conversion and10-frame lifetime remain tied to the stated100Hz clock. Do not call100Hz frames “100ms each.” The source's initial joystick/fixation settling and continuous response window are omitted. We score a final binary report after all evidence, so this implementation does not measure biological reaction time, sustained withholding or anticipatory false alarms.

## Training and evaluation allocation

The new arm learns all five tasks; do not expect new cues, four-region configurations or image-list rules to generalize without training. Mix all five family losses in each effective gradient update with equal family weight: five microbatches of eight episodes, one from each family; average the five mean losses and take one clipped Adam step. The target is4,000updates ×40episodes=160,000episodes, subject to a lower feasible exposure pinned from profiling before production under the shared finite cloud cap. Record actual per-family episodes, update counts and frame counts. Do not infer a fair comparison from wall time alone.

Within orientation, motion-duration and binding, choose D0/4/12/24 equally. Recognition chooses L0/4/12/24 equally, K3/4/5 equally, with balanced labels conditional on L>0. Krauzlis chooses baseline12/20/28 equally, with the event mixture above. Use fresh checkpointable family-local streams; corresponding evaluation bases can be paired across delay where the exact scene/task law permits. Document any adaptation in executable config before production; do not add extra conditions or a curriculum silently.

Pinned validation matrix:12spatial task×delay cells,12recognition cells (four list lengths×three probe-repeat counts), and3Krauzlis baseline-length cells, totaling27conditions. Use64validation/256final episodes per non-Krauzlis condition and100validation/400final episodes per Krauzlis condition. The renderer uses a shuffled200-trial event cycle; a100-example slice need not have exact57/29/14proportions. Report actual target/foil/catch counts rather than claiming exact proportions for every slice. Fresh validation/test seeds and actual counts remain in the launch configuration. These counts do not authorize exceeding the finite cap.

Selection first maximizes the minimum chance-normalized family BA, then the equal-family mean AUC, with earlier ties. For each family first average its declared condition scores equally; normalize BA as `(BA−chance)/(1−chance)`, using chance.25for four-class motion and.5for binary tasks. Omit all three recognitionL0conditions from BA/AUC selection and report their specificity separately. This gives the five families equal standing despite differing condition counts. Report every cell, selected and terminal checkpoint, all class-specific failures and learning curves. The old three-task companion study retains its own task law and selection; cross-battery scores are not paired evidence of an architecture effect.

No full GS6 or biological-capacity claim follows from acquiring this battery. The immediate question is whether unforced attention can learn useful selection and retention across these concrete tasks.
