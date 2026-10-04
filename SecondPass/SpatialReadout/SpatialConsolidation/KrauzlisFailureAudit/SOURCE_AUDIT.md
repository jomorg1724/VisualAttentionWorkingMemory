# Krauzlis failure: read-only source and CPU audit

## Bottom line

**The failure is genuine poor target-change discrimination, not an established broken renderer or frozen training path.** The strongest current explanation is failure to acquire the composition of fine-motion encoding, early spatial-cue retention, and baseline/postevent comparison from a final binary loss. Source inspection identifies these demands but does **not** localize the failing trained representation. Increasing change to ±90° removes the small-angle explanation as a sufficient explanation; it does not remove subpixel motion, cue memory, or temporal comparison.

This audit read `ANALYSIS_SOP.md`, task-suite documentation, exact production sources, existing CuedMotionAudit results and local runtime receipts. No checkpoint was loaded; no model inference, backward pass, optimizer, training, accelerator, network, SSH or cloud action was performed. Only this report was authored. Existing artifacts were preserved. Proposed diagnostics below are tests, not permission to execute them or alter task teaching.

## 1. Evidence identities and boundaries

- **Local native-angle fresh run:** actual saved `report.json` and `test_{selected,terminal}.json` under `/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run` confirm terminal step4595, selected step2297, no failure, complete final evaluations. Selected mean BA .500000/AUC .534178; terminal BA .500000/AUC .558973. Journal documents all-positive reports, including all foil/catch trials: `LabJournal/krauzlis-fresh-attempt02.md:5–20`. These are two roles from one seed, not independent replications; do not select the terminal retrospectively by test AUC.
- **Fresh02 ±90 cloud arm:** user/parent supplies completed7737, selected BA .50/AUC≈.482 and terminal AUC≈.495, with final artifacts on a stopped pod. **This audit cannot independently verify those final results from the off-pod receipts found.** In `/Users/jonathanmorgan/VAWMRuntime/cloud_krauzlis90_fresh02`, `remote_status.json:3–26` records only step21; `:35–67` records the planned7737 updates, not completion. `status.json:24–37` is an earlier step15 snapshot; `provider_status_verified.json:2–19` has empty completion/evaluation and an armed guard. Their old RUNNING fields are not current pod state. No restart or retrieval was attempted.
- **Useful fresh02 production evidence:** `production_verified.json:13–118` records persisted step3/96 episodes, fresh empty initial Adam, 52 changed parameters and changes in CNN, all three KDAs, spatial projection/ConvGRU, terminal transformer, dense readout and active head; inactive heads unchanged. This rejects a wholly inert startup, not later vanishing useful gradients. Verifier checks named Adam step/finite state for every active parameter: `SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisOnly/worker.py:94–112`.
- Nine inspected source files were byte-equal to their copies in runtime `package/deployment_3b03e9c28947/repo`: native renderer, imported `WorkingMemory/stimuli.py`, encoder, KDA, spatial readout, consolidation, ±90 renderer/worker, and joint update worker. This supports the packaged-source trace, not recovery of final model state.

## 2. Exact task, pixels and timing

Primary source: `WorkingMemory/SpatialTaskBattery/stimuli.py:69–73,121–150`.

**Labels:** a shuffled 100-trial cycle contains 57 target changes (`1`), 29 foil changes (`0`), 14 catches (`0`). Side assignment alternates across cycles, balancing each event category across 200 trials. The ring is an instruction defining the relevant patch, **not cue validity**. Foil reports are errors in this task; do not import the reference paper's any-change semantics (`ANALYSIS_SOP.md:20,41–45`). Metadata is discarded by the update function before the model call.

**Timeline, zero-based:** ring at0–1; fixation-only2–6; initial dots7; baseline motion8 through B+7; event first affects frameB+8; eight postevent transitions end at B+15; fixation-only reportB+16. Total B+17 frames (29/37/45). B12/B20/B28 change baseline duration, not blank retention delay. The nominal clock is100Hz and compressed timing is disclosed. No external event marker or trial metadata enters the model (`stimuli.py:134–142`).

**Photometry:** RGB channels are identical, background .5, fixation .1, ring .95. Two patches at(20,50)/(80,50), radius8.125px; 16 dots each. Dots move .375px/update with Gaussian16° persistent offsets, lifetime10 updates and boundary/lifetime replacement. Continuous positions are **bilinearly splatted**, four weights summing to one per dot, amplitude .48, followed by clipping to[0,1] (`:124–139`). Isolated-dot mass is preserved before clipping; overlaps can saturate. This is not nearest-pixel rounding that erases subpixel motion. Intensity changes due to ordinary motion/resets must not be mislabeled isolated event signal.

Native directions start90° apart with random absolute direction. Event changes only target or foil mean by ±26/28°; catches change neither. An ideal surviving dot's event-versus-no-event single-step displacement difference is .168713/.181441px, versus .530330px at90° (computed as `2*.375*sin(angle/2)`). Even the enlarged event is not a large translated object.

**±90 implementation:** `SecondPass/SpatialReadout/SpatialConsolidation/Krauzlis90Fresh/stimuli.py:124` retains the original magnitude RNG draw then overrides `magnitude=90`. Everything else is source-identical. The wrapper instantiates this renderer for training and evaluation (`Krauzlis90Fresh/worker.py:42–49,75–83`). Native and variant are matched before an event, but resets/RNG trajectories may legitimately diverge after it; ordinary separate-cell/run datasets are not paired.

At ±90 an event makes the two mean directions parallel or antiparallel, whichever patch changes. That potentially simplifies **any-event** detection, not target/foil discrimination. The early ring still determines the label. A new CPU test constructed **36 target/foil opposite-label pairs** (three B conditions ×12 seeds): after switching ring target and event category so the physical changed patch remains the same, **all frames2 onward were bit-identical**, and cue frames differed. This proves that even the90° movie alone cannot uniquely determine target versus foil on these matched cases.

## 3. Gradient, loss and readout trace

- `WorkingMemory/PlainBaseline/accum.py:43–68`: center at.5; causal stack3 with zero padding; strided convolution/GroupNorm/ReLU maps50→25→13→7; KDA insertion at25/13/7. No implicit extra presented frame or metadata input.
- `PreAttentiveVision/TemporalIntegration/accumulators.py:19–29,38–66`: differentiable KDA decay/correction/read; normalized keys/queries; learned alpha/beta initialized .9/.5. No detach. Site-local state is not independent whole-network locality: convolutions and GroupNorm mix information/statistics.
- `SecondPass/SpatialReadout/model.py:21–47`: recurrent spatial projection160→64 and ConvGRU, explicit zero initial state each sequence, full history graph retained. The GRU has no feedback into earlier KDAs.
- `SpatialConsolidation/model.py:36–56`: **only final64×7×7 memory** becomes49 spatial tokens plus positional encoding; four-head spatial attention and gated FFN; dense3136→256/ReLU and task head. The transformer does not attend to all historical timesteps and cannot directly recover discarded baseline/cue frames.
- At the Krauzlis report, stack3 contains the final two postevent dot frames plus fixation. Neither baseline nor cue is directly present. Even B12 therefore requires recurrent cue/baseline information; cue pixels have left the raw stack before the first dot frame.
- `SecondPass/JointTraining/worker.py:51–75`: ordinary **unweighted final-report cross-entropy**, microbatch/effective scaling, backward each microbatch, one Adam step, finite checks and no clipping. No auxiliary cue/direction/event loss. Fresh session initializes all parameters trainable and fresh Adam1e-4 (`FreshRun/worker.py:48–60`); cloud clone binds this actual update (`SpatialComparisonReadout/CloudRun/worker.py:65–67`). Reuse of functions does not mean inherited weights.

With no useful conditional evidence, expected CE is minimized at P(target)=.57: loss .683315 and logit .281851 (computed). Argmax then always reports target: accuracy57%, BA50%, hits100%, foil/catch false reports100%. This explains why collapse is attractive and BA is insufficient, **not why conditional evidence failed to emerge**. AUC near .5 argues against mere threshold miscalibration hiding a strong discriminator.

**Proven incidental code defect, not a causal training bug:** generic `parameter_group` recognizes `gru.` but not `spatial_gru.`, `spatial_input.`, `consolidation.` or `readout.` and labels these `inactive_heads` (`JointTraining/worker.py:43–48`). It misnames grouped diagnostics; it does not select optimizer parameters or suppress their gradients. The dedicated persisted verifier uses correct module prefixes. No causal renderer/loss/detach bug was established in this audit.

## 4. What the prior audit rules out—and does not

`SecondPass/SpatialReadout/CuedMotionAudit/PIXEL_FINDINGS.md:5,23–27,62–107` reports native B20 image-only observer BA .887903/AUC .956956 on200 held-out trials, cue decoding200/200. Known geometry, known event boundary and externally stored full history are substantial privileges. This establishes usable native pixels, not neural learnability or a ±90 result. Measured full-frame temporal difference is much weaker than duration; fixed pooling attenuation is **not measurement of trained CNN information loss**.

`CuedMotionAudit/MODEL_FINDINGS.md:11–15,27–35,45–55,63–97` concerns **different checkpoint35039, ConvGRU without terminal transformer**, not either fresh single-task run. It found all-positive Krauzlis reports, weak AUC≈.531, no useful cue-swap decision control, tiny score effects of event removal, and no earlier competent deployed readout destroyed exclusively by report onset. These findings guide tests, not automatic attribution to the new models.

Do not say all other tasks succeed. That checkpoint's duration already failed at D0 (BA .34375), with no demonstrated added-D24 decrement. User/parent additionally identifies full-suite terminal-transformer duration failure versus a later long-trained ConvGRU result; this audit did not retrieve those comparison final artifacts. Architecture, exposure and lineage differ, so neither sensory success nor later duration success isolates the Krauzlis cause.

## 5. Ranked falsifiable hypotheses (not diagnoses)

| Rank | Hypothesis and why | Discriminating observation, retaining native task |
|---|---|---|
| 1 | **Fine-motion/baseline-comparison features are not acquired.** Sparse subpixel evidence and resets precede aggressive spatial reduction; enlarged angles retain those requirements. | On independent frozen-checkpoint episodes, test actual pixel-derived motion versus early CNN/KDA direction information separately pre/post event. Early direction failure favors encoding; strong phase-specific direction but weak change information favors comparison. Fixed pooling alone cannot decide. |
| 2 | **Early cue retention/target binding fails.** Cue vanishes before dots;90° events remain target/foil ambiguous without it. | Matched cue-swapped movies with recomputed labels, plus held-out cue decoding through gap/baseline/postevent. Preserved cue and per-patch event information but absent correct cue-conditioned margins points downstream; absent cue retention supports this hypothesis. |
| 3 | **Final-state compression/readout or optimization fails to use available information.** Only terminal spatial state is classified; CE supports an easy .57 prior. | Compare independent frozen decoders of final versus phase-separated states and trained-head margins; success only with stored earlier states implicates retention/compression, success at final state with failed native head implicates readout/optimization. Fit/test separation and feature-count controls are essential. |
| 4 | **Task-relevant long-range credit is weak despite globally nonzero gradients.** Early step3 weight movement proves connectivity, not cue/event-specific learning at completion. | Once final artifacts are legitimately accessible, inspect actual gate statistics, cue/phase-specific sensitivities and temporal gradient norms under a separately authorized diagnostic. Nonzero total gradients alone do not falsify this. |

These mechanisms can coexist. None justifies silently changing cues, labels, curriculum, auxiliary teaching or restarting the pod. Neural allocation, inhibition, microstimulation and causal module localization remain **not measured** here.

## CPU verification receipt

Two short CPU-only commands, numerical threads capped at2, bytecode/cache writes disabled; no optimizer/model constructed:

1. `.../recurrent_transformer_cpu_env/bin/python -m pytest -q -p no:cacheprovider SecondPass/SpatialReadout/SpatialConsolidation/Krauzlis90Fresh/test_variant.py` → **1 passed in1.34s**. Test body (`test_variant.py:9–39`) checks36 native/variant matched cases, timing, metadata, catch equality, signed90° rotation and exact one-edit source identity.
2. Direct native-law ±90 cue-swap invariant over B12/B20/B28 and seeds0–11 → **36/36 exact same-movie/opposite-label pairs**, .224652s measured loop, torch threads2, no model load, zero updates. For each seed: render target event with target=`seed%2`; rerender same RNG seed as foil with target=`1-seed%2`; assert arrays equal from frame2 and unequal in first two frames.

Both were bounded by120s command timeouts. No production source or existing result was modified. Final cloud trained-state diagnosis remains blocked on authorized artifact access, not on a need to train again.
