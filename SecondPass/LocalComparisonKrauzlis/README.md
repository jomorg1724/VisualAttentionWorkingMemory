# Local temporal comparisons with shared spatial attention

October5,2026: user approved the targeted successor after frozen-model failure diagnostics. A fresh native Krauzlis model preserves per-location motion comparisons before spatial pooling, routes these through the same single attention map as appearance, and keeps the selected comparison available to the final decision. Previous checkpoints and experiments remain preserved.

## Motivation and scope

The predecessor's cue was100% decodable from final spatial memory. Its earlier motion features supported81.4% local-change balanced accuracy, while a fresh decoder on its final CLS achieved52.9%. Composing a cue decoder with a local motion decoder achieved82.9% on600 fresh movies, but used privileged event windows and local-change supervision. Those are diagnostic results, not this model's expected performance. See `../../LabJournal/shared-attention-krauzlis.md`.

This candidate keeps the native task, original sensory front end, appearance CNN, width64,10x10 spatial grid, one CLS-query attention head, spatial GRU and CLS GRU. It adds an explicit comparison component to the attention values and a transparent temporal trace for those selected values. The final decision no longer has to recover all motion comparison information from the compressed CLS alone.

## Causal local comparison

Let p[t,l,i,k] be the learned displacement population over81 candidate offsets, for causal lag l in{1,2,4}, at location i. Let w[t,l,i] be the original pixel-derived correlation reliability. At EVERY frame, form reliability-weighted averages:

    observation[t] = sum(w[u] p[u], u=t-3..t) / sum(w[u], u=t-3..t)
    reference[t]   = sum(w[u] p[u], u=t-15..t-8) / sum(w[u], u=t-15..t-8)

There are four intervening frames. The observation window has4frames; the reference window has8. Startup has zero available mass rather than invented history. A1e-6 denominator floor handles empty locations; confidence is sqrt(mean_w_reference * mean_w_observation). Both averages are per location and lag. These are causal finite temporal filters, not externally selected trial epochs. The model receives no frame-of-change, baseline-duration, cue-side, true direction, or displacement target.

For each location, construct735 explicit channels: reference, observation, absolute population difference (3*3*81channels, scaled10 and multiplied by confidence), plus lagwise L1 difference and cosine distance (6channels, also confidence-weighted). The comparison occurs BEFORE different locations are pooled.

The existing shared attention alpha[t,i] routes both feature families:

    selected_visual[t] = sum_i alpha[t,i] * visual_value[t,i]
    selected_comparison[t] = sum_i alpha[t,i] * comparison[t,i]
    spatial_state[t] = SpatialGRU(100*alpha[t,i]*visual_value[t,i], spatial_state[t-1])
    cls[t] = CLSGRU(selected_visual[t], cls[t-1])
    comparison_trace[t] = .8*comparison_trace[t-1] + .2*selected_comparison[t]
    logits = FFN(concat(final_cls, final_comparison_trace))

The FFN is799→64GELU→2. The735comparison channels are preserved through selection and the fixed trace; they are not first compressed into the64D CLS. There is no ungated motion decision route, no second attention map, and no supplied target-side selection. The existing visual spatial memory and CLS remain trainable. Optional attention-score interventions act on this same alpha and therefore affect both streams.

The fixed temporal lengths and trace decay are disclosed architectural assumptions. They are not event inputs, but robustness to other sampling rates and much shorter/longer changes remains untested. Shared-map routing is built into the model; useful learned cue-dependent allocation still must be demonstrated behaviorally.

## Training

Fresh378675parameters/43tensors, no inherited model/optimizer or probe weights. Same native renderer: cue, two apertures,16dots each, direction variability, lifetime resets,.375px/frame,26/28degree changes,B12/B20/B28,29/37/45frames. Terminal CE+.05pixel-derived causal prediction loss; no local-change labels, direction supervision, or privileged diagnostic decoders are transferred.

Fresh train/val/test namespaces196101/196102/196103, initialization196211, shuffle196212. FP32Adam3e-4,no clipping,batch32/micro4,full temporal backpropagation. Pools of1000movies are shuffled for2epochs (66updates) then refreshed. Requested6600updates, prospectively reduced to complete66update pools by the measured local profile.

One new finite14400second local cap covers discarded tiny-fit/profile weights, production, all validation, final test and reporting. The supervisor uses an independent deadline guard and exclusive GPU lock; no cloud, automatic extension or second training arm. Tiny-fit at most128updates uses8training movies and is not evidence of generalization. Production starts fresh afterward. Source is frozen before activation.

Validation is now inside the main trainer at66, EVERY100updates, and terminal,100fresh fixed movies per duration (300total). Minimum mean validation CE selects best.pt. The profile explicitly budgets this increased validation frequency. Final selected-model test is1000fresh movies per duration (3000total), fixed terminal argmax, no test calibration. Only best.pt/latest.pt retained, with complete optimizer/RNG state; no external evaluator is needed.

## Verification and results

CPU_CHECKS.json records correct reference/observation window arithmetic, causal prefix behavior, normalized single attention, zero-bias identity,735nonzero comparison-memory channels, all43finite gradients/Adam states, and all learned tensors changing after two updates. These are implementation checks only.

Actual profile, live progress and independent saved-optimizer verification belong in `LocalRuntime/run01`. No training success is claimed by this implementation or its tiny-fit check.

Actual CUDA profile pins2244updates (34completepools,68000presentations,34000unique movies). Median longest-duration effective-batch update4.180s; validation cost0.0674s/movie. Original finite deadline12:29:23AM PDT October6 remains unchanged. Production worker18300 starts with fresh weights; no held-out result yet.
