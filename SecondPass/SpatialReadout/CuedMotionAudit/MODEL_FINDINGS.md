# Frozen current-checkpoint diagnosis: cued motion

## What the current observer actually does

**Duration has weak but measurable cue-dependent information; it is not simply cue-blind or defeated only by the extra delay. Krauzlis is dominated by an all-positive report policy, with little useful target-specific change evidence in the deployed scores. Neither failure is uniquely localized to a neural module.**

The separate image-only audit establishes usable native pixel information on independent draws: duration 128/128 correct; Krauzlis B20 BA .887903, AUC .956956. That observer has known geometry/timing and external full-history storage; its success is not evidence that this network has learned those operations. See `PIXEL_FINDINGS.md:9–27,88–109`.

### Frozen identity and execution

- Exact checkpoint: `/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_continuation_01/artifacts/checkpoint_035039.pt`; actual `state.step=35039`.
- SHA256: `93762f7968a29092b8acce7ea9632937a23965160822fe98bda9b3e5844531e7`, checked before loading with `weights_only=False` and after each inference run.
- Native `SpatialReadout(task_classes())`, strict complete state load; eval mode; every parameter `requires_grad=False`. **Zero optimizer creation, training updates, backward calls, or probe fits.**
- One local MPS worker at a time, CPU/inter-op threads 1. Original nonrenewable 1200s wall budget persisted **before** first accelerator operation. Primary inference/results completed at 17.64s; supplement completed at 184.83s from that same origin; CPU raw-result verification completed at 252.31s. Supplement reused, never renewed, the original deadline. Report finalization is recorded in `model_final_receipt.json`.
- All model state tensors remained bitwise unchanged. Checkpoint and audited production-source hashes unchanged. Real native `model.forward` versus timestep wrapper: maximum absolute final-logit error **0**. The second Krauzlis native run reproduced all 200 primary final-logit vectors exactly.
- No cloud/API/SSH action, training change, deployed architecture/stimulus edit, or pixel-worker file edit was performed.

## 1. Cue selection: detectable for duration, not useful for Krauzlis

All observations below are one frozen checkpoint, not training-seed variability.

| Task / condition | N | Balanced accuracy | Macro OVR AUC | Predicted class counts |
|---|---:|---:|---:|---|
| Duration D0 native | 128 | .343750 | .628906 | [23,48,29,28] |
| Duration D24, identical nonblank evidence | 128 | .359375 | .625651 | [26,36,17,49] |
| Duration D0, discordant-winner cue retarget | 128 | .359695 | .577161 | [17,46,31,34] |
| Krauzlis B20 native | 200 | .500000 | .530702 | [0,200] |
| Krauzlis B20, left/right cue swap | 200 | .500000 | .493686 | [0,200] |
| Native sensory motion anchor | 32 | 1.000000 | 1.000000 | [8,8,8,8] |

**Duration intervention:** all 128 retargets change the correct winner. The edit rerenders only the ring via native `local_cue` using captured underlying rasters, rather than erasing ring pixels that might contain evidence. Every actual dot pixel is verified preserved; all noncue frames are exact. Predictions change on **89/128** pairs; **15/128 pairs have both answers correct**. The newly-correct versus formerly-correct class margin shifts **+0.17081 logits**, paired bootstrap95 **[+0.08171,+0.25833]**; its sign is appropriate on 86/128 pairs. Mean absolute probability change over all class coordinates is .06657. This is a real causal effect of cue pixels in the appropriate average label direction, not reliable selection/integration competence or an attention-map measurement.

**Krauzlis intervention:** only cue frames0/1 are changed; the entire fixation gap, reference, moving movie and report remain pixel-exact. Target↔foil labels are recomputed; catches remain negative. Labels change on **172/200** pairs, but **0/200 predictions change**, and **0/172 changed-label pairs are both correct**. Newly-correct versus formerly-correct margin shift is only **+.00325 logits**, bootstrap95 **[−.00643,+.01277]**. Mean absolute P(change) change is .01057, so “cue has literally no effect anywhere” would be false. The effect is not useful target-specific decision control here.

Native B20 is exactly **114 target / 58 foil / 28 catch**. Hits, foil false reports and catch false positives are all **100%**. Native accuracy57% versus swapped29% simply follows positive prevalence; **the −28-point accuracy difference is not a cue-validity benefit**. The ring is a target instruction, not a probabilistic validity cue. Native P(change) averages .57798, compatible with prevalence-dominated responding, but this does not prove a pure stored-prior mechanism.

Native final-score AUC bootstrap95: duration D0 **[.56714,.69133]**, D24 **[.56067,.69154]**; Krauzlis B20 **[.44723,.61087]**. Duration's weak information should not be collapsed to “exactly chance.” These are exploratory trial resamples, not multiplicity-corrected confirmatory intervals.

## 2. Delay does not explain the duration failure by itself

All128 D0/D24 pairs have **bit-identical nonblank frames and labels**. D24 changes69/128 argmax reports and moves mean absolute class probability by .04199, but the paired accuracy difference is **+1.5625 percentage points**, bootstrap95 **[−8.59375,+11.71875]**. There is no demonstrated mean decrement from the added24 blanks on this panel; there is already poor D0 competence.

D0 is not memory-free: eight motion transitions must be integrated. Stack3 exposes only the final two moving rasters plus report at D0's final readout, not the complete schedule. In D24, the first two nominal blank updates retain old motion rasters in their stack; subsequent ignore updates do not. Therefore this comparison addresses extra retention cost, not whether recurrence/memory is needed at all.

## 3. Temporal readout: no rescue before the report frame

The unchanged trained head/readout was applied at each actual recurrent timestep. **Earlier readouts are exploratory/OOD**: supervision/deployment uses the final state and the native report phase. Their failure is not proof that information is absent from the full state.

| Condition | Last moving BA / AUC | Final report BA / AUC | Paired accuracy change (report−moving) |
|---|---|---|---|
| Duration D0 | .257813 / .561605 | .343750 / .628906 | +8.59pp [−1.56,+18.75] |
| Duration D24 | .257813 / .561605 | .359375 / .625651 | +10.16pp [−.78,+21.09] |
| Krauzlis B20 | .500000 / .501836 | .500000 / .530702 | 0pp |

For Krauzlis, all200 trials are already predicted positive at pre-event t27, first-postevent t28, last-moving t35 and report t36. Corresponding native AUCs are **.50908,.52519,.50184,.53070**. Report onset raises mean P(change) by **.00490**, paired95 **[.00274,.00696]**, without a single decision change. Thus a hidden high-performing earlier **deployed readout** destroyed exclusively by the report frame is not supported.

Duration final predictions agree with true **first**, **last**, and **count-winner** target directions at **22.656%,22.656%,34.375%**, respectively; mean agreement with each uncued patch's winner is **24.740%**. On the92 trials where last direction differs from the duration winner, predictions match the last direction20.652% versus winner36.957%. Cued-minus-mean-uncued winner agreement is +9.635pp, paired95 [.521,19.271]. A simple final-direction/recency shortcut does not explain these reports. `_motion_schedule` independently fixes first/last before rejection-sampling a unique count winner; endpoint direction must not be assumed to track the answer.

At D0 report, descriptive accuracy rises across speed .8/1.2/1.6 pixels: **30.30% (n33),32.00% (n50),40.00% (n45)**. Count-margin1 accuracy is29.76% (n84), margin2 46.875% (n32); higher margins have only n8/n4 and are not stable estimates. These are small, nonrandomized strata, not a causal speed/margin curve. Full strata, confusion matrices, class coverage and Wilson intervals are saved.

## 4. Actual evidence interventions

### Krauzlis native-law event removal

For each of the original200 B20 scenes, reconstruct the same RNG state immediately before native `_krauzlis`, substitute a catch event, and rerender with **the unchanged native law**. Every pre-event input frame is exact;28 original catches are wholly identical shams. Boundary/lifetime-reset trajectories can diverge after an actual direction intervention, appropriately, so these are not claimed to differ only in an abstract angle variable.

Removing the event produces **200/200 positive reports**, hence zero specificity on all-negative counterparts and **0/200 argmax changes**. Positive-class BA/AUC are undefined for that all-negative arm and are saved as null, not fabricated as a two-class metric.

Paired P(change), **event-present minus matched event-removed**:

| Original event | N | Last-moving difference [bootstrap95] | Final-report difference [bootstrap95] |
|---|---:|---|---|
| Target | 114 | +.003163 [.000395,.006117] | +.001414 [.000004,.002821] |
| Foil | 58 | −.002518 [−.007779,.002181] | +.002228 [.000201,.004679] |
| Catch sham | 28 | exactly0 | exactly0 |

The input event can weakly influence scores, so complete sensory blindness is too strong. At report, both target and foil changes raise P(change) slightly and all reports remain positive. These tiny exploratory effects do **not** establish useful target/foil separation or precisely identify where the computation fails. Confidence intervals are unadjusted across multiple epochs/groups.

### Krauzlis baseline length

Independent native B12 and B28 panels, n100 each with exact57/29/14 event mixtures, also produce all-positive reports: **BA .5** at both; AUC **.513668/.560996**. These are different scenes, not paired baseline-extension interventions. Merely shortening the baseline within native training lengths does not restore categorical performance.

### Duration temporal reversal: explicitly OOD

Reverse the nine reference/moving rasters, keep cue target and report fixed, and transform all direction labels by opposite direction (+2 modulo4), including the full schedule and counts. Surviving-dot displacements reverse, but reverse dot replacement is **not the native process**. n128: BA **.242188**, AUC **.506429**;81/128 predictions change,8/128 pairs are both correct. Accuracy change versus native is −10.156pp, paired95 [−21.875,+.781]. This shows sensitivity to temporal input order but no robust opposite-direction solution; the OOD failure cannot establish the native mechanism.

## 5. Strongest supported localization, and what is not identified

1. **Not gross absence of recoverable pixels:** independent pixel observer successes, with its explicit privileges, show usable evidence on audited native draws.
2. **Not duration extra-delay-only failure:** native D0 already poor, with no paired D24 decrement established here.
3. **Not complete duration cue blindness:** cue relocation has a correctly directed average logit-margin effect. Duration retains some information about the cued count winner but is far from solving selection plus fine-motion integration.
4. **Not a demonstrated report-frame-only collapse:** no competent earlier deployed-head report was found. This does not rule out information accessible to a separately trained readout.
5. **Krauzlis failure is expressed before report as weak useful target-specific evidence and a positive-biased final decision policy**, not merely one bad accuracy threshold hiding a strong ranking (native AUC≈.53). Native event removal, cue swap and B12/B28 do not rescue reports. Small score responses are present, so “encoder entirely blind” is unsupported.
6. **Not an untouched task head:** serialized Adam state, mapped by saved `optimizer_names`, records **2696 duration-head steps** and **2695 Krauzlis-head steps**, both weight and bias, with nonzero moments. Constructor parameters are trainable before diagnostic freezing. A state_dict does not preserve historical `requires_grad` flags; optimizer state is supporting evidence of updates, not proof of adequate exposure or successful optimization.
7. **The 32/32 native sensory anchor only supports its learned sensory domain.** It has different stimulus geometry/speed/density and does not certify the tiny cued patches. No crop/recenter transfer or broad fitted state probe was necessary to distinguish the above observations.

**Still unresolved:** fine-motion encoding versus cue retention/selection versus temporal counting/baseline comparison versus downstream readout/optimization may jointly contribute. The negatives do not uniquely identify one of them, reject this architecture family, or establish that more unchanged training cannot help. No spatial attention map, internal inhibition/microstimulation, reaction-time measurement or biological equivalence is claimed.

## Reproducibility, counts and sources

Primary seeds: duration uses eight16-trial native-balanced blocks with seeds92635039+{0,16,…,112}; Krauzlis continuous native stream92735039; sensory92835039. Timing-only first8 use disjoint93635039/93735039 seeds and are excluded from inference panels. B12/B28 seeds92935039+baseline. D0/retarget/D24 share trial IDs, while each condition has unique independent-scene IDs. Retargeting deliberately favors discordant winners and is not the natural cue-target sampling distribution.

- `model_trials.jsonl`: all original timestep logits, labels and native metadata (including per-patch schedules/counts, cue, motion speed, event time/location/type, trial IDs and verification flags).
- `model_evidence_trials.jsonl`: all reversal, matched event-removal/sham, exact replication and B12/B28 records.
- `model_summary.json`, `model_evidence_summary.json`: complete condition, paired and temporal results.
- `model_verification.json`: primary summary exactly recomputed from disk; all counts, unique trial IDs, task labels, replication logits and checkpoint/source hashes checked; bootstrap uncertainty.
- `model_checkpoint_inspection.json`, `model_optimizer_head_states.json`: checkpoint schema/state/config and serialized per-head optimizer receipts.
- `model_budget.json`, `model_sample_plan.json`, `model_completion.json`, `model_evidence_budget.json`, `model_final_receipt.json`: immutable-origin resource budget, cheap first-eight estimate, full-size production selection and completion.
- Code: `model_diagnostic.py`, `model_evidence.py`, `model_verify.py`; CPU acceptance tests: `model_tests.py`, `model_evidence_tests.py`. No renderer/model production files changed. All raw logits suffice to recompute scores without storing giant image tensors; seed and unchanged source provenance support raster regeneration.

CPU-only verification from repository root:

```sh
PYTHONPATH=. /Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python -m SecondPass.SpatialReadout.CuedMotionAudit.model_verify
```

Do not rerun producers over the recorded files or renew the expired original budget. Fresh reproduction requires a separate explicitly authorized output namespace/budget. Accuracy Wilson intervals and episode-paired bootstrap intervals are conditional on this single checkpoint; schedules/transitions within a trial are not counted as independent replicates. Some small speed/margin strata lack classes; their saved generic balanced_accuracy is represented-class macro recall, not full four-class performance.

Load-bearing source references:

- `WorkingMemory/SpatialTaskBattery/stimuli.py:34–43,97–111`: duration rings, native dots, unique-winner labels, delay/report placement; `:69–78`: native sampling.
- Same `:121–142`: Krauzlis geometry, bilinear rendering, cue/gap, native event law and report semantics.
- `WorkingMemory/stimuli.py:84–89,98–111`: oracle and first/last-constrained duration schedule.
- `WorkingMemory/PlainBaseline/accum.py:40–68`: CNN scales, centering, stack3 and encoder recurrence.
- `SecondPass/SpatialReadout/model.py:29–51`: complete native recurrence and final-state-only readout.
- `ANALYSIS_SOP.md:20,37–49,98–104,156–165`: target-specific semantics, frozen provenance, blank-stack leakage, pairing and inference limits.
