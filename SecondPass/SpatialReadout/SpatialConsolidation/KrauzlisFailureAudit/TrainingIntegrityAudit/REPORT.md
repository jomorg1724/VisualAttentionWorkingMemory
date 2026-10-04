# Executable training-integrity audit

## Outcome
No objective-sign, label-index, missing-optimizer-parameter, detached-history, or microbatch-scaling bug was reproduced on the native CPU fixtures. This is **not proof of task acquisition**, nor a diagnosis of the historical chance results. No production code or main trained weights were changed; no cloud, checkpoint loads, or accelerator operations occurred.

**Actionable positive-control warning:** the CNN's very first Adam1e-4 step on a balanced two-episode native batch increases CE. A tenth of that *same step vector* decreases CE. Do not interpret a one-step rise as a broken backward pass, and do not count majority-only fitting as task learning. This does not establish that production batch32 LR was wrong. Dense's corresponding step decreases CE.

## Executed scope and identity
Each arm imports its own `cloud_krauzlis90_{fresh02,dense01}/clean_extract/repo` in a separate process, uses the exact deployed Session constructor and cloned cloud `update`, and checks every imported deployment file against that extraction's deployment manifest. CNN: 27 files; dense: 28; zero missing or mismatched hashes. PyTorch2.8.0, CPU, one intra-op and one inter-op thread. All neural tests use native B12 **29-frame 100x100 RGB** episodes, including cue, gap, motion and report; no resize or shortened sequence. Final fixture has labels [0,1]. Each run has an85-second alarm. Several short harness iterations remained well below the180-second compute limit; final execution times below exclude imports. Independent of the other child's positive-control training.

|Arm|Final audit seconds|Max micro/full gradient error|Max float64 finite-difference relative error|Active tensors changed|Initial CE|Full Adam step CE|Tenth same-step CE|
|---|---:|---:|---:|---:|---:|---:|---:|
|cnn|4.079|5.29e-07|8.15e-07|52|0.709849656|0.743319392|0.693810880|
|dense|3.164|2.65e-08|2.25e-08|22|0.693210244|0.690947831|0.692955554|

## Proven executable passes
- Renderer causal checks span both target sides and B12/B20/B28. Target/foil/catch counterfactuals share nuisance seed; no pixel divergence before event onset, identical final report raster, direction mean changes by90degrees only in the selected event patch. Correct first changed frames20/28/36.
- Stronger cue intervention: identical physical motion with opposite cue targets produces opposite target/foil labels; **every frame after the two cue frames is bit-identical**. Correct label depends on the cue, not just detecting any change.
-100 sequential native episodes have exactly57 target,29 foil,14 catch, and every integer target agrees with actual changed-patch/target relationship. Float32 image range[0,1], int64 labels; unique trial IDs.
- Adapter images and labels equal direct native renderer. Two-at-once equals sequential1+1, including metadata; permutation preserves aligned loss/logits; stream snapshot/resume exact.
- All nine split/cell seeds are distinct; sampled train/val/test movies differ. CNN and dense intentionally share train/val namespaces, while dense final test uses a new namespace.
- Stack3 is ordered oldest-to-newest with subtraction0.5 exactly once and zero centered left padding. Only images and task string enter the forward call; labels and metadata remain in loss/scoring.
- Actual Session optimizer covers every model parameter exactly; CNN's24 unused-head tensors are intentionally inactive, while all52 active CNN tensors and all22 dense tensors receive nonzero finite gradients and change after Adam.
- Native full-sequence loss has nonzero input gradients at **every frame including initial cue**. This excludes total temporal detachment on the exercised path; it does not imply adequate usable cue retention.
- Exact deployed microbatch update (2effective/1micro) equals full-batch unweighted mean CE and gradients to floating-point tolerance. Optimizer step is suppressed only during this comparison; then that real Adam step is executed once.
- Selected sensory, recurrent-gate, and classifier derivatives agree with central finite differences. Float64 audit resolves tiny recurrent derivatives that were below practical fp32 difference precision; training/accumulation tests remain fp32.
- No cross-call recurrent state leakage: logits identical after an unrelated intervening sequence. Batch permutation logits agree. Perfect class0/class1 probabilities give BA=AUC=1 in the exact deployed scorer.

## Actual incidental defects / qualifications
1. **Diagnostic grouping bug, not a training bug:** inherited `parameter_group` recognizes old module prefixes only. Dense encoder/KDA/GRU/readout and CNN spatial/consolidation/readout tensors can be reported under `inactive_heads`. The full list is in each JSON. They demonstrably receive gradients, belong to Adam, and change. Do not use that diagnostic label as evidence of freezing. No fix made.
2. **Shared renderer RNG couples patch nuisances:** changing one patch's trajectory alters boundary resets and RNG consumption, which can change pixels in the other patch in matched-seed counterfactuals (even at onset for some fixtures). Means/labels are correct. Consequently a seed-matched target-vs-catch intervention is not guaranteed to preserve every uncued pixel. The successful cue-only intervention avoids this confound. This is not evidence explaining chance learning and does not authorize renderer changes.
3. **Tiny-batch CNN Adam overshoot:** direction is locally descending, full step overshoots on this fixture; causal relevance to historical batch32 training is unproven. No production LR change recommended from this alone.

## Uncovered paths / limits
- No historical CUDA numerical execution, kernel/version equivalence, checkpoint replay, resumed optimizer validation, deployment-on-pod identity beyond local packaged source, or reproduction of7737-step chance behavior.
- Neural numerical checks cover native B12 only; B20/B28 renderer checks do not validate those longer neural gradients. Presence of early input gradients does not certify every internal state edge or good conditioning.
- Microbatch equality is tested at2/1, not production32/4; unequal microbatch sizes are unsupported by the deployed divisibility assertion.
- Split seed separation plus sampled hash difference is not a proof that no future randomly generated scenes can coincide. No exhaustive statistical leak audit.
- One disposable update and finite differences are gradient/objective sanity, **not learning-to-criterion**. The independent positive-control must provide that evidence.
- Initial audit expectation that the uncued raster would remain fixed under target/catch counterfactuals failed; this was traced to shared RNG resets, documented above, and replaced by correct limited assertions plus the clean cue-only intervention. No production source was modified to pass tests.

## Reproduce / files
`/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python audit.py dense`
then the same command with `cnn`, from this directory (or use absolute script path).

`audit.py`: executable assertions; `cnn_results.json`, `dense_results.json`: exact results, frame gradient norms, source hashes, finite differences and classifications; `cnn.log`, `dense.log`: complete final stdout. Empty `*_scratch/` directories are constructor artifacts, not checkpoints. Existing dirty repository files were preserved.
