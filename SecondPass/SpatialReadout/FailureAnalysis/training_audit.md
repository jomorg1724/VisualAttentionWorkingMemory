# Training/model audit: sensory acquisition versus spatial/sequence failure

## Bottom line

**The failures are present during acquisition, not just at held-out evaluation. No broken loss routing, truncated BPTT, accidental freezing, missing task exposure, or within-trial reset was found.** The strongest measured bottleneck is weak, cancelling spatial-task gradient signal in a shared optimizer dominated by sensory tasks. This is a supported optimization diagnosis, **not yet proof that task interference causes the behavioral failure**. Long-delay forgetting cannot explain the whole pattern: ring and binding already fail at D0, where the causal input stack contains the relevant sample, cue and probe.

A concrete bug exists in **gradient diagnostic categorization**, not gradient application: the inherited logger calls the entire new recurrence/readout “inactive_heads.” Recognition’s apparent training improvement is principally acquisition of the empty-list control, not general list membership. There is no evidence justifying another unchanged training extension as the default response.

## Evidence and scope

- Read-only source and artifact audit. No optimizer step, training launch, GPU/MPS access, architecture change, or source modification. CPU probes used one intra-op and one inter-op thread; frozen checkpoint parameters were compared before/after and remained bit-identical. Backward passes were diagnostic only.
- Artifact root `R=/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2`. Read `report.json`, `progress.jsonl`, `test_selected.json`, `test_terminal.json`, and CPU-loaded `terminal.pt`.
- Terminal step **6760**, selected **5486**. Terminal SHA256 independently recomputed: `1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841`. All 24 audited Python files listed in the run’s source-hash manifest matched current repository bytes. File:line citations below therefore refer to the actual production implementation, not a later rewrite.
- Both saved tests contain **35 unique cells / 4,696 trials**, with confusion counts matching every denominator. Continuation progress contains all **5,070 updates, 1691–6760**. No new full behavioral evaluation was run.
- Important lineage correction: parent3393 was **jointly trained on all 13 tasks**, not a sensory-only pretraining run. Checkpoint `state.parent.exposure` records 261 updates / 8,352 episodes **per task**. The ConvGRU branch adds 520 updates / 16,640 episodes per task. Total inherited lineage: 324,896 episodes; 24,992 per task, but only the branch’s 6,760 updates trained the new ConvGRU/readout.

## 1. Verified critical path and ruled-out implementation faults

| Path | Actual implementation and audit result |
|---|---|
| Task/cell sampling | `SecondPass/JointTraining/core.py:8–34` shuffles every task once per 13-update cycle, then cycles each task’s condition queue. `SecondPass/TaskSuite/suite.py:69–94` selects an independent task/cell-local native stream. No task is omitted. |
| Labels and loss | `JointTraining/worker.py:51–64` obtains images and labels from the same batch, routes only the selected task head, computes CE, and accumulates eight microbatch4 means weighted by 4/32. One zero-grad before accumulation; one Adam step after it (`:53,75`). This is the intended batch32 mean, not an eightfold learning-rate error. |
| Balance | Sensory binary microbatches are balanced (`PreAttentiveVision/neuroscience_stimuli.py:262–268`); motion microbatches include all four labels (`:73–79`). Spatial microbatches need not individually balance, but the preserved native queues balance an effective batch32: ring queue8 (`WorkingMemory/PlainBaseline/variants.py:29–32`), signed orientation queue16, other spatial queues8/16 (`WorkingMemory/SpatialTaskBattery/stimuli.py:69–78`). Krauzlis intentionally has the 57/29/14 event mixture; N0 recognition intentionally overrides labels to zero (`:112–120`). |
| Preprocessing | `WorkingMemory/PlainBaseline/accum.py:54–58` subtracts .5, then constructs causal stack3 with centered-zero start padding. There is no future-frame access or variable-length batch padding. |
| CNN/KDA | `accum.py:43–48,59–68`: four strided conv/GN/ReLU blocks, trainable projections, three KDA modules; KDA state passed between timesteps. `PreAttentiveVision/TemporalIntegration/accumulators.py:19–29,51–66` updates differentiable local associative state with learned gates. This is **not** the separate frozen-PAV wrapper in that file. |
| Final recurrence and supervision | `SpatialReadout/model.py:38–51` starts encoder state and final hidden state at `None` once per trial batch, processes **every** timestep, and applies the head only to the last hidden state. No detach, no `no_grad`, no reset in the temporal loop. No intermediate/cue/transition loss. `:21–26` implements the stated write/reset ConvGRU equation. |
| Trainability/optimizer | `SpatialReadout/worker.py:24–28` asserts all parameters trainable fp32. One Adam group at 1e-4, betas (.9,.999), eps1e-8, zero weight decay. No clipping or LR schedule. Inactive heads have no gradient and do not advance their Adam clocks; shared modules do. |
| Migration/resumption | `SpatialReadout/state.py:72–112,129–143` maps weights and Adam by exact name+shape, preserves task/cell/RNG state, and creates new Adam state only for new tensors. `continuation_v2.py:188–197,260–276` preserves actual model/optimizer state rather than remigrating architecture. Terminal Adam clocks confirm shared inherited tensors10153, every head781, new projection/ConvGRU/readout6760. No missing new-parameter optimizer state. |
| Evaluation | `JointTraining/worker.py:97–123` uses the same image-only forward and matching labels, with eval/no-grad and fresh split-local streams. `JointTraining/core.py:37–66` computes per-class recall/BA and AUC; N0 is explicitly excluded from BA/AUC. No train/eval dropout or batch-normalization mismatch: the network uses GroupNorm. |

### Direct frozen-gradient check

Used terminal weights, `SuiteStream('val')` from a fresh stream, first two trials in each listed cell, CE backward, with no optimizer. Ordinary orientation, ring D0, signed orientation D0/D24, duration D0, Krauzlis B12, and binding D0 all had **nonzero gradients in CNN, KDA, spatial input projection, final ConvGRU, readout, and active head**. Early input frames also received nonzero gradients. This directly rules out complete graph disconnection in these exercised paths, not all possible numerical/semantic failures.

The ReLU is not entirely dead: active readout coordinates were 105/109 for ring, 101/97 for signed D0, 95/100 for signed D24, and 60/61 for Krauzlis (of256). Mean write gates across probed sequences were .554–.708; fractions below.01 or above.99 were .011–.064. These small probes do not show wholesale gate saturation. They do not establish competent retention.

## 2. Proven acquisition and allocation problems

### A. Training losses stay at the uninformed solution

Recomputed from the **last20 updates within each cell** in `R/progress.jsonl` (cell-local windows, not identical wall-clock windows):

| Task/cell | Mean training CE | Interpretation |
|---|---:|---|
| ring D0 | .693802 | Essentially binary uninformed CE .693147 |
| signed orientation D0 / D24 | .698427 / .696363 | No acquisition even at D0 |
| duration D0 / D24 | 1.387931 / 1.386756 | Essentially four-class uninformed CE1.386294 |
| binding D0 / D24 | .700064 / .700959 | No acquisition even without inserted blanks |
| Krauzlis B12 / B28 | .688983 / .684533 | Close to class-prior CE .683315 |
| recognition N0 H3/H4/H5 | .029324 / .038273 / .061159 | Empty-list control learned |
| recognition nonempty cells | .699322–.729170 | Membership still approximately uninformed |

Saved terminal confusions agree: Krauzlis predicts positive on **all600 trials**; duration D12/D24 predicts one direction on every trial. Across cells spatial AUC is also approximately chance; this is not merely a bad .5 threshold hiding a generally strong classifier. Exceptions deserve preservation: recognition N4_H3 reaches BA.640625/AUC.633301, not universal absolute failure.

### B. “6,760 updates” is not 6,760 updates per skill

Scheduler balance is by **task**, not by cell, difficulty, or learning need (`core.py:15–24`). Seven sensory tasks get 7/13 of updates. Every branch task gets520 updates, but signed orientation and binding get130 updates per delay; duration gets129–131, Krauzlis173–174, recognition only43–44 per load/hold cell. Ring has520 D0 updates and still fails, so insufficient delay-cell allocation alone cannot explain it.

There is no easy-first acquisition curriculum: delays and loads are mixed from the start. This is a concrete allocation fact, not evidence that a curriculum would necessarily fix it.

### C. Recognition training reward is diluted by an easy negative control

Three of12 conditions are N0, with **4,128/16,640 branch episodes (24.81%)**. They are deliberately included in training, although excluded from selection BA/AUC (`core.py:15–24,149–160`; `stimuli.py:112–120`). Thus low aggregate recognition CE can improve by recognizing that nothing was studied without solving membership. Last50 task-update mean CE falls from approximately .675 to .535 across the continuation, while all recent nonempty-cell losses remain approximately .70–.73. The head also sees an aggregate negative prior, unlike its balanced nonempty evaluation cells. This is an objective/allocation caveat, not a mislabeled-data bug.

## 3. Measured shared-optimization bottleneck; causal extent remains unproven

**Logged evidence:** over the final650 updates (50 per task), the seven sensory tasks contribute **98.15% of the sum of squared logged whole-model gradient norms**. Last50 per-task mean gradient norms include motion15.16, natural spectrum8.59 and chromatic7.43, versus ring1.01, signed orientation.93 and duration.62. All use the same shared Adam and LR. This is not equal influence despite equal task counts. Important qualification: logged norms include each active head, and Adam rescales coordinates; the98.15% figure is **not** a measured share of parameter displacement or proof of interference.

**More direct CPU evidence:** first32 validation trials per task, accumulated as eight microbatch4 gradients at the same terminal checkpoint, excluding heads from the vector:

| Task | Shared mean-gradient L2 | Mean microbatch-gradient L2 | Ratio |
|---|---:|---:|---:|
| ordinary orientation | 15.9803 | 19.2905 | .8284 |
| ordinary motion | 3.4170 | 8.1852 | .4175 |
| ring D0 | .7626 | 2.9985 | .2543 |
| signed orientation D0 | .8248 | 3.8471 | .2144 |

Spatial microbatch gradients therefore **cancel substantially** in the balanced aggregate; they are not absent. Shared-gradient cosine ordinary-orientation versus signed-orientation was **−.2381** on these fixed draws, versus **+.1972** for ordinary-orientation versus ring. Motion versus ring was −.0146; ring versus signed was −.0085. This measures one local conflict, **not universal sensory–spatial antagonism** or proof that conflicts caused the training trajectory.

Mechanism supported by code: one task supplies each optimizer step (`worker.py:46–48`); shared Adam moments mix all tasks, while task-specific heads only update once per cycle. Sensory success can coexist with poor spatial acquisition when its feature/useful-gradient directions dominate shared adaptation. Discriminating test: repeat balanced shared-gradient and predicted Adam-step alignment over independent batches and earlier checkpoints, reporting per-block norms/cosines and the effect on other tasks’ frozen losses. This requires no optimizer step. A causal training comparison, if later authorized, must hold architecture/exposure fixed while varying only the task-update policy—not silently change several mechanisms.

## 4. Why sensory success does not demonstrate the missing computation

- **Two-frame sensory tasks can compare the raw frames in one final input stack.** The last stack is `[zero,frame0,frame1]` (`accum.py:54–58`). Their success does not show that long-lived KDA/ConvGRU storage was necessary.
- **Ring D0 also has direct access to both cued samples and the probe:** its four-frame timeline is cue,sample,sample,probe (`variants.py:42–58`), so the final stack includes both samples, with their rings, and the probe. Binding D0’s final stack similarly includes the last sample, retrocue and probe (`stimuli.py:92–96`). Chance performance here localizes part of the problem to extracting/routing the correct local comparison, rather than only losing it across a long blank.
- Sensory ordinary orientation is a large central Gabor (envelope sigma19; cycles/image5–10, `neuroscience_stimuli.py:116–121,155–163`); ring/signed Gabors are small peripheral patches (sigma4.5; wavelength5–7 pixels, independent phases and framewise wavelength/amplitude, `variants.py:34–40`; `stimuli.py:79–91`). High central sensory performance does **not** establish local peripheral rotation discriminability. Likewise ordinary motion differs in aperture, dot rendering and displacement from duration/Krauzlis (`neuroscience_stimuli.py:23–29,61–90`; `stimuli.py:104–111,121–142`).
- All tasks share the same image-only representation until the selected head (`model.py:49–51`). Cue-based selection must be learned from image pixels; task ID does not provide target location, sign, phase, or top-down recurrent control. Flattening retains coordinates, but does not automatically implement the location×change or sign×location×change rule. The heads are separate, so this is **not** a direct collision of incompatible label meanings in one head.
- Long tasks add real retention demands. Three recognition blanks flush study images from stack3; by the last ofH≥3 identical probes, the raw stack contains only probes (`stimuli.py:112–120`). Every repeated probe continues updating all recurrent states before the sole loss. Probe-driven overwriting is a specific hypothesis, not a measured diagnosis here.
- Full BPTT is not uniformly effective credit assignment. On two frozen signed-D24 trials, last-sample input-gradient L2 was .02322/.01397 versus probe.28853/.23757, ratios .0805/.0588. The sample gradient survives but is attenuated. With D0 already failing, attenuation is a possible aggravator, not the universal root cause. KDA state dynamics are differentiable and learned; initialized .9 retention is not a fixed trained decay law (`accumulators.py:42–64`).

**Discriminating tests:** (1) frozen matched local-patch orientation probes, moving otherwise identical stimuli from center to each task position; (2) cue swaps while preserving samples/probes, scoring changes against recomputed target labels; (3) pre-probe versus successive-probe content decodability and paired frozen readouts; (4) per-time/per-region Jacobians separating cue, target and foils. These distinguish sensory transfer failure, failed cue use, and memory/readout loss without assuming they are the same mechanism. These extended tests were not run in this audit.

## 5. Concrete logger bug and migration caveat

**Proven diagnostic bug:** `SecondPass/JointTraining/worker.py:43–48` recognizes `blocks/feat/proj`, `acc`, `gru`, and the active head, and sends everything else to `inactive_heads`. New `spatial_input.*`, `spatial_gru.*`, and `readout.*` therefore get mislabeled as inactive heads—**1,034,752 parameters**. `:82–88` consequently gives misleading component diagnostics if enabled for SpatialReadout. Production SpatialReadout does not request these detailed diagnostics (`SpatialReadout/worker.py:46–48`), so its saved global `grad_norm` remains valid; the bug did **not** freeze these tensors or alter Adam. Do not infer “GRU gradient zero” from this inherited grouping. No code was changed.

**Proven migration fact, unproven causal problem:** `SpatialReadout/state.py:90–112` carries old task-head weights and Adam moments while replacing the entire feature-to-head path with new projection/ConvGRU/readout. Matching head shape does not preserve its old feature-coordinate meaning. The old global-GRU output was signed; the new head input is final-readout ReLU (`accum.py:69–75`; `model.py:49–51`). This is a representation change with a temporarily stale head/optimizer interface, not an invalid checkpoint mapping. It plausibly affects initial acquisition; after6,760 branch updates it is not established as the continuing cause. Test it using existing migration/early checkpoints and frozen feature/logit diagnostics before proposing reset/retraining.

## Priority conclusion

1. **Do not spend the next analysis trying to repair nonexistent detach/reset/head-routing bugs.** The exercised gradient paths work, and all tasks received updates.
2. **Start with D0 local comparison and cue-use competence, plus replicated shared-gradient/Adam interference diagnostics.** D0 failure rules out a retention-only explanation and the current evidence shows an actual low-signal optimization bottleneck.
3. **Keep recognition controls and nonempty acquisition separate.** Its falling aggregate loss is otherwise misleading.
4. **Treat long-horizon forgetting, cross-task competition, transfer mismatch and warm-start interface mismatch as distinct falsifiable explanations.** This audit identifies concrete evidence and constraints but does not claim a single proven causal explanation or recommend an architecture replacement.

### Probe reproducibility

Interpreter: `/private/tmp/vawm-task-suite-venv/bin/python`; repository root as cwd; `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`; `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`. Instantiate `SpatialReadout(task_classes())`, load `torch.load(R/'terminal.pt',map_location='cpu')['model']`, call `eval()`. Use a fresh `SuiteStream('val')` for each of the two probe families: first2 trials/cell for timestep gradients and gates; first32 trials/task as eight4-trial microbatches for shared-gradient vectors. CE backward only; clear gradients between microbatches and average concatenated gradients from names not beginning `heads.`. No optimizer constructed or stepped. For readout/gate measurements use forward hooks on `readout` and `spatial_gru.gates`; compare all model state tensors to pre-probe clones afterward. The reported probes are small deterministic diagnostics, not new generalization estimates.
