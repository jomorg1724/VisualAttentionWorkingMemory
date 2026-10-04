# Learned spatial comparison before compression: two-arm acquisition test

User authorization: "please do this. Remember the goals of the architecture and research though. Dont get lost in your 1000s of layers of checks either" after recommending one learned spatial comparison/readout change versus ordinary continuation, with unchanged teaching.

## Research decision and scope

Test whether adding a small, learned local memory/current-evidence comparison before global compression improves acquisition of cued orientation and binding. This is an end-to-end architecture development experiment within the visual-attention/working-memory program, not a probe-score optimization exercise or proof of biological attention. Preserve three KDA stages and spatial ConvGRU. Do not implement a whole Guided Search system, diffuser, new working-memory store or handcoded task solver. All learned weights train at the same inherited LR. No auxiliary angle/cue targets, privileged phases/locations, fixed circular relations, oracle attention, frozen encoder or new teaching/curriculum.

One candidate and one ordinary continuation control. All13 tasks/35 primary conditions, existing native stimuli/photometry/lengths/cues/labels/losses and equal-task scheduling unchanged. Source identical trained terminal6760 (`/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt`, SHA256 `1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841`). Existing parent was jointly trained on all13 tasks. Preserve all historical models; no claim of fresh-weight learnability or causal biological validation.

## Single architecture delta

At each timestep, preserve the existing computation:
- s_t = existing spatial_input(encoded current frame/history),64x7x7.
- h_t = existing spatial_gru(s_t,h_previous),64x7x7.

Candidate adds a shared per-location learned comparator:
- c_t = Conv1x1(64->64)(ReLU(Conv1x1(128->64)(concat[h_previous,s_t]))).
- z_t = h_t + c_t.
- Existing final flatten -> readout256/ReLU -> task head uses z_t rather than h_t.

The existing recurrent state carried forward remains h_t, NOT z_t: no added feedback, state gate or separate memory. Initial h_previous is native zeros. Ordered concat distinguishes prior memory from current sensory; all49 locations share weights with no object centers/cue metadata. Apply the same causal computation for every frame, no sample/query/report oracle. The task loss still uses the native final report only. This is one residual comparison package, not proof that comparison rather than added capacity causes an effect.

Initialize first1x1 using ordinary PyTorch initialization and final1x1 weights/bias to zero, preserving exact parent output at migration. New branch has12,416 parameters. Zero final layer briefly blocks first-layer gradient on the very first update; verify it learns after subsequent updates, do not mislabel as freezing. No learned gate, new normalization, extra task-specific parameters or LR change. Preserve inherited readout/heads rather than randomizing trained sensory decisions.

## Fair continuation

Both arms inherit the SAME model weights, compatible named Adam state, task/condition queues, native streams and RNG from terminal6760. New comparator parameters alone get empty Adam state; isolate their initialization RNG so next training draws match. Adam1e-4, same betas/epsilon/weight decay, effective batch32/micro4, no clipping, fp32 full BPTT, all parameters trainable. Common per-task prefixes and cell sequence must match between arms. Carry lineage/exposure/optimizer time honestly; new experiment counters may be additional updates, but do not reset learned Adam steps.

Target2600 additional updates EACH arm =83,200 episodes/arm =200 updates and6,400 episodes/task/arm; cumulative ConvGRU-lineage step9360. One local MPS worker, sequential arms, CPU threads≤2. Run control first, candidate second, with durable external supervisor owning the automatic sequential transition. Same total exposure, no extra arm.

## Budget, selection, reporting

One new finite86,400-second (24h) cap for BOTH arms begins before the first accelerator profile, includes profiles/baseline/training/evaluation/reporting. No renewal. Prior production costs imply roughly14.4 combined optimizer hours before overhead. Use existing complete per-cell train/eval timings plus one brief real-launcher candidate/control timing check (representative short/long frames; at most one13-task cycle per arm, disposable and never inherited). Pin equal feasible exposure BEFORE production; if2600 cannot fit with margin, reduce both in13-update cycles and disclose actual exposure. Do not cut a healthy arm unequally or alter teaching to chase results.

One fresh shared full-suite baseline validation (candidate/control identical at migration), midpoint and terminal full-suite validation for each arm; no additional peeks/campaign. New fresh validation namespace shared across arms and separate fresh paired final namespace, different from all prior diagnostic/final streams. Validation selection prospectively prioritizes equal-cell mean AUC over orientation_cued and spatial_binding across all8 conditions, then existing equal-task suite mean AUC, earlier ties. This explicitly tests the targeted acquisition question; do not reuse old selection histories as if same criterion. Parent baseline eligible for each arm. Report all13 tasks/all35 cells, empty recognition specificity separately, Krauzlis target/foil/catch. Do not use aggregate selection to hide regressions.

Final selected AND terminal comparison on identical fresh examples across arms,128/cell and200/Krauzlis as inherited, with checkpoint deduplication when identical. Primary outcomes: cued orientation/binding per-delay BA/AUC and paired candidate-control differences. Provisional practical signal: >=5pp mean BA gain over the8 target cells with positive paired uncertainty support; every other task's regression visible. Treat one-parent/one-run result as pilot evidence, not a universal mechanism or mature architecture. No post-test model selection or automatic further runs.

## Minimal launch-critical work

Reuse existing evaluator, streams, state mapping and verified Interactive launchd/runtime-copy approach. New files only in this experiment plus necessary scoped source copy into a separately named runtime. Do NOT rewrite framework or alter old source manifests. One focused preflight: migration/initial-logit equality, matching next draws/queues, finite gradients with new comparator updates, and budget feasibility through actual launcher/QoS. No broad test suite, repeated review gates, full reprofiling or extra diagnostics. One researcher is sole launch owner and may proceed directly after those checks; do not wait for parent permission polling.

Use one-shot launchd, KeepAlive=false, absolute cap supervisor, same verified Interactive scheduling and worker lock. Runtime outside Desktop to avoid already-known TCC issue. Do not stop unrelated processes. Start production promptly; return when a post-update checkpoint and advancing progress are persisted, including exact run directory, launch label/PIDs, step, pinned allocation/deadline and automatic report location. A live process/profile update alone is not training. Parent verifies saved production state and updates journals, without rerunning the preflight. Preserve old runs; no cloud.
