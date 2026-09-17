# Motion recovery without discarding orientation retention

Design prepared from the user's request to evaluate training time and a residual from accumulator states to deeper layers. This note proposes two staged comparisons. No training, profiling, cloud resource or new model was launched for this design. Existing checkpoints and diagnostic results remain unchanged.

Execution update,2026-09-13: the user subsequently authorized training locally first and concurrent RunPod work. Stage1 is now implemented in WorkingMemory/TrainingExposure: control_10 runs locally, followed by concurrent focused_50 on Palladio pod sm1kbctuhqfpo5. Both retain the fixed4000-update/32000-episode addition from attention8400. The local cap is14400seconds from its first profile; the cloud cap is four hours from creation. These explicit placement instructions supersede the local-only proposed envelope below. The original sampler is family-local, not delay-cell-local: common family evidence prefixes are preserved, while schedules can assign different delays to those examples. Fresh validation/test seeds are53973001/54973001. Actual launch/progress/cleanup comes from TrainingExposure receipts. No residual arm is part of this Stage1 launch.

## Evidence motivating the tests

The selected pre-update attention8400 model achieves 79.30% delayed single-orientation BA, compared with 58.40% for equally exposed ordinary continuation. Excluding memory attention keys/values only during the 24 blank frames reduces its score to 50.00%; excluding the same source during other phases produces no comparable drop. The intervention also renormalizes visual attention, so it establishes dependence on delay-period routing without uniquely separating recirculation from sensory rejection.

Motion D0/D24 BA is 39.65/35.35%. During moving frames, excluding memory-source attention improves these to 42.77/38.48%, and routing the original sensory field directly into memory_input improves them to 43.55/41.21%. These acute interventions show some execution-time interference but are not trained alternative models.

Separately, validation-fitted class offsets improve immediate motion from 39.65 to 52.54%; the ordinary continuation also improves, from 50.78 to 56.64%. Calibration and routing gains are separate and cannot be added. No calibration was installed in a deployed checkpoint. Motion problems also existed before attention in the spatial experiment.

The completed training devoted only 400 of 4000 updates to motion: 1600 examples at each motion delay. Checkpoint selection ignored motion. Thus more total training, motion allocation, decision fitting, and the information route are distinct hypotheses.

## Experiment 1: more training versus more motion exposure

Continue two arms from the exact same selected attention8400 checkpoint, with no architecture change. Restore all compatible Adam, sampler and RNG state. Keep existing tasks, cues, labels, losses, full-sequence BPTT, batch8, clipping1, normalization and LR policy. In particular, the ordinary motion/orientation heads currently inherit 3e-5 LR; memory/attention/comparator and the binding head use 3e-4. Do not silently change those rates during an exposure test.

| Proposed arm | Motion updates per 80 | Each of eight orientation/binding cells per 80 | Added updates | Added examples |
|---|---:|---:|---:|---:|
| Existing schedule | 4 per motion delay (8 total) | 9 | 4000 | 32000 |
| Motion-emphasized schedule | 20 per motion delay (40 total) | 5 | 4000 | 32000 |

The first arm gets 1600 additional examples per motion delay and 3600 per primary cell. The second gets 8000 per motion delay and 2000 per primary cell. Randomize within each complete 80-update cycle and retain task-local streams so common per-task prefixes match. Different schedules are deliberately not identical global episode order.

Evaluate at 800,1600,2400,3200,4000 additional updates. Plot per-task learning curves against both total updates and examples from that task. At focused800 versus control4000, each has seen 1600 added examples per motion delay; other exposure differs, so this is informative context, not complete causal isolation.

Interpretation:

- Existing schedule versus the unchanged8400 baseline tests whether an additional fixed duration of the same optimization policy helps. A flat curve only addresses this finite horizon.
- Focused versus existing schedule at equal updates tests training allocation. It does not isolate pure training time because nonmotion exposure and the task mixture change.
- Improvement accompanied by orientation/binding regression is a measured tradeoff, not a general upgrade.
- If both plateau, the tested optimization horizon is insufficient to establish an architectural capacity limit. Existing readout-bias evidence remains relevant.

Proposed initial resource envelope: one sequential local GPU worker, a finite 14400-second cap for both arms including profiling and evaluation, with exposure fixed before production if 4000 updates per arm does not fit. This is a proposed new envelope, not reuse of a closed allowance or an active launch. No RunPod restart is included.

## Experiment 2: a shorter path for temporal features

Use one explicit residual placement, tested against a no-residual continuation from the SAME checkpoint under the SAME chosen schedule. Choose that shared checkpoint using Experiment1 validation only; if no candidate meets the preservation screen, retain8400 as the parent. Do not compare a newly trained residual model only with an untrained parent.

The architecture contains two different kinds of recurrent state:

1. Sensory fast/slow traces, at each of three CNN scales: f_i,s_i in B x32 x h_i x w_i, with sizes50x50,25x25,13x13. Their retention coefficients are .25 and .75.
2. E/I working-memory firing rates and adaptation R,A, each B x64 x13 x13. R already projects residually into the classifier through pooled features and memory_output. Reading A would be a different intervention and is not proposed here.

The existing sensory route is:

    fast/slow traces and fixed energy features
      -> learned 72-to-32 emission projection at each scale
      -> merge with current features, local 64-to-32 conv/normalization/activation
      -> spatial pooling and 96-to-64 fusion
      -> H (B,64,13,13)
      -> joint sensory/memory attention
      -> existing E/I working-memory update

Expose the already computed pre-emission feature bank:

    X_i = concat[(f_i+s_i)/2, f_i-s_i, E_i]     # 32+32+8=72 channels
    Q_t = concat_i Pool13(X_i)                 # (B,216,13,13)
    DeltaH_t = Conv1x1_216_to_64(LN216(Q_t))   # (B,64,13,13)
    H_att_t = H_t + DeltaH_t
    U_t = Attention(queries=R_previous, keys/values=[H_att_t,R_previous])

LN216 is non-affine channel LayerNorm at each location (epsilon1e-5), only on the new branch. The projection is bias-free and zero-initialized: 216x64=13824 new parameters, about2.5% of the current model, no additional persistent state. Initial logits match the original model. The new projection receives gradients immediately; gradients through its input branch start at zero and appear after its first update, while the original pathway continues to train normally.

Only the visual source of attention uses H_att. Existing direct sensory and comparator outputs keep the original H. Attention remains the sole external drive into the existing E/I memory input. The shortcut runs on every frame, with no task-specific switch, blank detector or phase oracle. Use a versioned migration with compatible old weights/Adam preserved and new projection moments initialized fresh.

This bypasses learned emission/local/fusion transformations; it does not bypass the CNN, fixed energy equations, spatial pooling or attention. Success could reflect better information access, added capacity or easier optimization. It would not alone prove that the old layers erased information. A matched-capacity control can follow if that narrower causal claim becomes important; it is not part of the initial pair.

A residual to the final classifier is less suitable for the first test: it may help immediate motion without teaching working memory to retain it. With identical blank inputs, history deviations in the fast and slow traces contract by .25^24 and .75^24 (approximately3.6e-15 and.0010). Downstream normalization can amplify small signals, so this is not proof of zero information, but these traces are not a guaranteed long-delay store. The proposed pre-attention route lets useful temporal features enter the established working memory.

Run the residual comparison only after inspecting the exposure comparison, under a separately recorded finite budget. No automatic second stage or cloud restart follows from this note.

## Preserve capabilities rather than hide regressions

Report all ten trained cells separately: single orientation and binding at D0/4/12/24, plus motionD0/D24. Also retain the four held-out-location binding evaluations. Report BA, AUC, confusion and class recalls; duration winner margin remains descriptive. Always include the unchanged parent on identical evaluation examples. Existing calibration results are a separate diagnostic reference; do not silently substitute calibrated scores into one arm.

Proposed validation rule: an eligible checkpoint must show no greater than2 percentage-point BA loss from its paired parent in any of the ten trained-task validation cells; held-out locations remain final-evaluation-only. Among eligible checkpoints select the best minimum motionD0/D24 BA, then their mean AUC, with earlier ties. Include the unchanged parent as a fallback. This is a screening rule; it is not evidence of statistical equivalence. Every smaller decline is still displayed.

Use fresh validation and final evaluation draws for these new training comparisons; the earlier repeatedly examined test set remains diagnostic context. Pair arms and delays, preserve independent four-case binding blocks, and select without final-test feedback. Initial512-per-cell final evaluation is a useful research screen. For claims of preservation, report paired simultaneous one-sided uncertainty relative to a predeclared practical2pp margin; a bound crossing that margin means unresolved preservation, even if the observed mean passes. Do not manufacture certainty or automatically expand evaluation until it passes. Report uncertainty for model-to-model motion gains as well.

A motion gain with a clear orientation/binding loss remains a candidate requiring investigation. Healthy runs finish their fixed allocation; this rule governs interpretation and promotion, not repeated early stopping or launch gates. The existing79.3% long-delay orientation checkpoint stays preserved.

## Literature and scope of biological interpretation

- [GradNorm, Chen et al., ICML2018](https://proceedings.mlr.press/v80/chen18a.html) motivates examining task imbalance and unequal learning progress. We are testing fixed schedules first; we are not claiming that GradNorm is already implemented or needed.
- [Gradient Surgery, Yu et al., NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) and [Recon, ICLR2023](https://arxiv.org/abs/2302.11289) motivate treating conflicting task optimization as a hypothesis. No gradient-surgery method or gradient-conflict campaign is added to these comparisons.
- [Identity Mappings in Deep Residual Networks, He et al., ECCV2016](https://arxiv.org/abs/1603.05027) supports direct signal/gradient routes as an optimization idea. Our projected shortcut is an analogy, not an identity connection across every transformation or proof of feature preservation.
- [Ponce, Lomber and Born, Nature Neuroscience2008](https://www.nature.com/articles/nn2039) studied direct V1-to-MT and indirect V1-to-V2/V3-to-MT pathways; reversible inactivation affected disparity tuning more strongly than direction tuning. Parallel shorter and longer visual routes are biologically motivated. It does not validate our precise216-to-64 projection, LayerNorm or attention circuit.

Source files inspected: WorkingMemory/RecurrentComparison/model.py; WorkingMemory/PreUpdateAttention/model.py and remote_sweep.py; WorkingMemory/model.py; PreAttentiveVision/TemporalIntegration/accumulators.py; completed MechanismDiagnostic and MotionAudit reports.
