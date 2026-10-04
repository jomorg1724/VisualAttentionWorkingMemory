> Historical snapshot through October 4, 2026. Earlier present-tense launch statements are not current status. Relative links have been adjusted for this archive location.

**October4 — angular contrastive CNN completed: simplified comparison solved.** FreshCNN25024updates/782000presentations/391000unique;3072fresh continuous-change tests,100%BA/AUC1.0 allthree speeds ×26/28°; explicit angular supervision. Best19000/latest25024 CPUverified64Adam states. No active local/cloud training. Original nativeKrauzlis task remains untested for this model. [Journal](../angular-contrastive-motion.md).

**October4 — angular contrastive CNN now TRAINING locally.** Fresh wholemodel/MPS, all64Adam states verified at saved1. Target25,024updates, original new8h launch cap; no other model interrupted, predecessorFFN completed. [Journal](../angular-contrastive-motion.md).

**October4 — fresh angular contrastive CNN QUEUED locally.** Threeframes →CNN →128Dunit representation; springloss with targetdistance proportional to circular direction change. Allweights fresh, CPUcheck passed; no training/budget started. [Journal](../angular-contrastive-motion.md).

**October4 — frozen predictive encoder + FFN completed at chance.** All10,240updates/640,000presentations/320,000unique, 23minutes. Fresh3072-trial test BA50%, AUC0.49519, CE0.693165; predicted no-change throughout. Best5632/latest10240 verified, encoder remained frozen; no active training. [Journal](../predictive-motion-change.md).

**October4 — predictive encoder + simple FFN running locally.** Frozenencoder50,000, concat512+512 →256→64→2, balanced26/28° same/change continuous-dot clips. Target10,240 headupdates, batch64, 2epochs/1000freshtrials; no cloud. [Journal](../predictive-motion-change.md).

# Visual Attention and Working Memory — lab journal

**October3 — Variational motion predictor built and initiallocalpilotcompleted.** ThreeorderedRGBframes→residualCNN→one512DGaussianvector→fourthframeCNNdecoder; all15,463,363parametersfresh/trainable. Newconstant-velocityfullfieldpersistentdots, uniformdirectionacrosssamples andspeeds.375/1/2px, nochanges/cues/labels. Support-balancedprediction+KLwarmup,2epochs/1000samples, batch32micro4 FP32Adam1e-4/noclip. Prospective20mincap pinned256updates/8000presentations/4000unique; completedearly7:37PM, best/latest256CPUverified/138Adamstates. FinalmeanpredictionMSE.004809 versuscopylast.003839/gray.005469; imageslargelyflat/blurry, motionlearningnotdemonstrated. Cloudweightedepoch2finished2310/70000/35000unique, selectedfinalBA50%/AUC.503969; allartifactsretrievedandpoddeleted. NoactiveGPU/cloudrun orautonomouscontinuation. [Predictivejournal](../variational-motion-prediction.md).

**October3 — Weighted-mean CNN–RViT epoch2 TRAINING on the existing A40.** Saved3/96 CPUverified/all92Adam states and tensors changed; live5. Fresh wholemodel/Adam/RNG/streams, solechange10→2epochs per1000trials. Pinned2310updates/70000presentations/35000unique trials (35pools), FP32/fullBPTT32micro4/Adam1e-4/noclip. Original8h/$5deadline unchanged (hard9:58PM PDT); guard/mirror/retrieval/delete armed. Oldcloud cancelled1304/39544, best/latest retrieved; local×10 continues. [Journal](../weighted-mean-rvit-epoch2.md). No validation yet.

[Local RViT input×10 experiment](../vae-rvit-input10.md) — original local classifier stopped227; new scaled-input trial profiling.

- [Weighted-mean CNN → RViT](../weighted-mean-rvit.md): fresh weightedRGB encoder/token recurrent learner, automatically queued after the activeVAE–RViT.

- [VAE encoder → RViT](../vae-encoder-rvit.md): trained deterministic three-frame spatial means,169×256tokens, fresh recurrent response learner; new local training requested.

- [Three-frame VAE](../krauzlis-three-frame-vae.md): fresh local reconstruction pretraining with spatial256x13x13 latent, replacing weighted CNN-GRU.

- [Weighted-mean CNN–GRU](../krauzlis-weighted-mean-conv-gru.md): fresh local MPS training, fixed three-frame averaging and ordinary GRU on the same no-cue single-stimulus task; cloud queue disabled.

- [Single-stimulus change detection](../krauzlis-single-stimulus.md): no cue or competing patch, exact original sequence timing, unchanged conv RViT.


- [Random-frame gradients and direct memory carry](../krauzlis-random-frame-gradients.md): sampled temporal updates, suffix Jacobians and cloud launch evidence.


**2026-10-03 UTC — [Loss and temporal-gradient diagnosis](../krauzlis-training-path-audit.md).**
Objective/decoding/accumulation checks passed; fullnative-trial early gradients
are extremely small in bothRViTs despite all learned parameters updating.
KDA's early gradient remains substantial on the tested trial. Current jobs unchanged.

**2026-10-03 UTC — [Structured-motion RViT](../krauzlis-simoncelli-heeger.md) TRAINING locally; [technical architecture/proposal PDF](../../SecondPass/StructuredMotionRViT/TechnicalDocument/architecture_proposal.pdf) complete.**
SavedproductionAdam independently verified, fresh92learnedtensors;660updates/
20000presentations/2000unique pinned. OldlocalCNN-GRU stopped/saved; cloudcaps unchanged.

**2026-10-03 UTC — [Simoncelli–Heeger motion front end](../krauzlis-simoncelli-heeger.md) implemented, untrained.**
Uses the actual NeurIPS 2024 author code with documented causal/task adaptations;
three focused CPU checks passed. Existing jobs continue unchanged.

**2026-10-03 UTC — Online RViT cancelled; fresh replay RViT TRAINING.**
User requests1000 generated movies reused10 shuffledepochs before replacement.
Old RViT stopped2075/66400;89filesverified and terminal92AdamstatesCPU-reloaded.
SameA40/original05:16:04UTC harddeadline, no caprenewal; KDA/localGRUcontinue.
[Replay experiment](../krauzlis-rvit-replay.md) · [Run evidence](../../SecondPass/TwoFrameRViTReplay/RUN_STATUS.md).
Production checkpoint 3 / 96 presentations CPU verified: all 92 Adam states and parameters advanced. Live update 15 / 480 presentations, first pool and epoch. Seven complete pools pinned: 2,310 updates / 70,000 presentations / 7,000 unique movies.


**2026-10-02 — Two-frame RViT TRAINING on a second A40; faster validations enabled.** [Run](../../SecondPass/TwoFrameRViT/RUN_STATUS.md). Explicit parallel launch beside16headKDA, fresh7,270,290params/92tensors; checkpoint3/96episodes downloaded/hashverified/CPUreloaded, all92params/Adamstatesadvanced. Nativeprofilepinsfull4216updates/134912episodes,18costed validationlooks100,250,500,...terminal at100trials/cell; pairedfresh200/cellfinals. Pod7f27p6jxpitihn,NEW8h/$5cap21:16:04UTC→05:16:04UTC /October2 10:16:04PM PDT, independentguard/mirror. Native teaching unchanged. CPU-only earlysnapshotsof existingKDA/CNN-GRU running without changingtraining/selection; labelthosevalidationonly. Existingrunscontinue,3layercancelledpodstaysdeleted.

**2026-10-02 — Fresh16-head single-layer KDA TRAINING, optimizer verified; three-layer cancelled.** [Run evidence](../../SecondPass/SequenceKDA16/RUN_STATUS.md). Userrequestedkillthree-layerandstartqueuedmodel. NewA40podpa0ko8f2qirisy; nativeprofilepins4216updates/134912episodes,FP32fullBPTT/effective32micro4. Actualcheckpoint3/96episodes downloaded/hashverified/CPUreloaded,all27params/Adamstatesadvanced,freshconstructor/emptyinitialAdam/nativeRNGstreams verified; no profile/predecessorinheritance. Independentguard/mirroractive,NEW8h/$5cap20:33:14UTC→04:33:14UTC /October2 9:33:14PM PDT. Oldthree-layerstopped2615/83680;71artifactsand55stateAdamcheckpointverifiedbeforepod5hnvb87npqpqb4stopped/deleted,nofinaltests. LocalCNN-GRUcontinues;RViTuntrained. Earlierlaunchpending/waitingnotesarehistorical.

**2026-10-02 — Three-layer KDA CANCELLED; queued16headKDA launched on newA40.** Userrequestedkillandreplacement. Three-layerstopped2615/4216updates,83680episodes; all71artifactsretrieved/hashverified,terminalCPUreloaded55Adamstates,andpod5hnvb87npqpqb4stopped/deleted. Onlycompletedvalidation2108gave50%BAallconditions/meanAUC0.499388; nofinaltests. Newfreshsingle-layer16headKDApodpa0ko8f2qirisy launched underNEW8h/$5,cap20:33:14UTC→04:33:14UTC (October2 9:33:14PM PDT),nativeCUDAprofile/pin inprogress, productionAdamproofpending. LocalCNN-GRUcontinues;RViTimplementationonly. [Cancelledrun](../../SecondPass/SequenceKDA3/RUN_STATUS.md) · [Newrun](../../SecondPass/SequenceKDA16/RUN_STATUS.md).

**2026-10-02 — Two-frame CNN / visual-query RViT IMPLEMENTED, untrained.** [Architecture](../../SecondPass/TwoFrameRViT/README.md), [journal](../krauzlis-two-frame-rvit.md). Ordered previous/current RGB CNN → 13×13×256 tokens; one shared recurrent block with independent visual self-attention and previous-memory cross-attention, both queried by current X. Per-token 256→16, flatten2704→FFN→2. 7,270,290 fresh trainable parameters/92tensors, full temporal gradients and CNN checkpointing. Three focused CPU checks plus complete native45-frame B28 forward passed; no GPU/profile/training launched or new compute allowance. Existing cloud/local runs and16headKDAqueue preserved.

**2026-10-02 — Single KDA with16full-width heads IMPLEMENTED and AUTOMATICALLYQUEUED.** [Run status](../../SecondPass/SequenceKDA16/RUN_STATUS.md), [journal](../krauzlis-sequence-kda16-heads.md). OneKDA/terminalCLS,16×64key/value heads,737,170freshparameters/27tensors,8×originalKDAstate. Userselectednew8h/$5cloudrunafterthree-layerpod5hnvb87npqpqb4 completes, artifacts/finalcheckpointsverify and deletionconfirms. Parentqueue63567/PPID1active; no new rental/profile/optimizer/budgetstart. NativeGPUprofilewillpinfeasibleexposure,target4216,Adam1e-4/effective32micro4/FP32fullBPTT/native teachingunchanged. Sevenfocusedchecks passed. Cloudthree-layer and localCNN–GRUcontinue unchanged.

**2026-10-02 — Delayed-frame CNN/standardGRU LOCALTRAINING, optimizer verified.** [Live run](../../SecondPass/DelayedFrameGRU/RUN_STATUS.md), [journal](../krauzlis-delayed-frame-gru.md). Fresh21,293,770parameter model,AppleM4Max36GiB/MPS,Adam1e-4/no clipping,fullBPTT/FP32,effective32/micro1. Nativeprofilepinned2,631updates/84,192episodes beforeproduction;877updates/28,064episodes eachB12/B20/B28. Checkpoint3/96episodes CPUverified,all86Adamstatesadvanced;launchdowner54824/PPID1+guard54822. HardcapOctober2 7:57:51PM PDT; no renewal. Cloudthree-layerKDAcontinues unchanged. Localvalidation/finals pending; earlieruntrainedcandidatenotes superseded.

**2026-10-02 — Delayed-frame CNN/standardGRU candidate implemented while THREE-layerKDA training continues.** [Design and source](../../SecondPass/DelayedFrameGRU/README.md), [journal](../krauzlis-delayed-frame-gru.md). Two independently trainable residualCNNs, all24convolutions stride1/full100x100/no pooling; current256+previous256→standardGRU512→256→binaryhead.21,293,770fresh parameters/86tensors; five focused CPU checks passed. Native teaching unchanged, explicit causal delay/fullgradients. **Candidate not trained; no new cloud job/budget.** The KDApod/guard/mirror/deadline below remain active.

**2026-10-02 — Fresh THREE-layer whole-sequence KDA training on RunPod, optimizer verified.** [Run evidence](../../SecondPass/SequenceKDA3/RUN_STATUS.md) and [depth comparison journal](../krauzlis-sequence-kda3-depth.md): A40pod`5hnvb87npqpqb4`,324488fresh trainable parameters,3pre-LNresidualKDA blocks/terminalCLS. Native tasks and Adam settings unchanged. SteadyGPUprofilepinsfull4216updates/134912episodes matching one-layer exposure. Checkpoint3/96episodes downloaded/hashverified/CPUreloaded; all55namedAdam states advanced. Guard/mirror active. Originalcap18:20:05UTC→02:20:05UTC /October2 7:20:05PM PDT preserved through initial unstarted-pod replacement; no renewal. Validation/finaltests pending. One-layer completion below remains the baseline.

**2026-10-02 — Single layer whole sequence KDA completed; task not acquired.** [Final results and artifacts](../../SecondPass/SequenceKDA/RUN_STATUS.md): all4,216updates/134,912episodes completed from fresh weights. Validation selected2,108; selected and terminal4,216 both give50% BA in B12/B20/B28 on200 fresh paired tests/condition, with all-positive decisions (57% raw accuracy,100% foil/catch false positives). Selected/terminal mean test AUC0.500102/0.502422. All45 manifest files retrieved and verified; both checkpoints CPU-reloaded with27 active Adam states. A40pod`8gamd8ems1pa0n` stopped and deleted after verified retrieval at17:56:01UTC. No renewed cap or further training. Earlier live entries are historical snapshots.

**2026-10-02 — Single layer sequence KDA training launched.** [Run and evidence](../../SecondPass/SequenceKDA/RUN_STATUS.md): one fresh A40, exactlyoneKDA/CLS, native Krauzlis teaching unchanged; GPUprofile pinned4216updates/134912episodes inside8h/$5. Checkpoint3/96episodes downloaded and CPUreloaded with all27Adam states advanced. Automatic mirror/guard active; final results pending.

**2026-10-02 — Single layer whole sequence KDA implementation and cloud launch authorized.** [Design and implementation](../krauzlis-sequence-kda-design.md): direct RGB patches, spatial/time positions, exactly one global KDA and terminal CLS classifier, whole model fresh, native teaching preserved. FP32 full-sequence output/state/gradient parity passed; native CPU checkpoint3 verifies96episodes and all27Adam states advanced. Deployment preparation is in progress; no cloud production claimed yet.

**Probe adequacy completed — fresh600/150/300 grouped splits.** [Results](../krauzlis-probe-adequacy-diagnostic.md): matched raw-pixel ridge side BA0.500–0.512 despite endpoint optical-flow0.740; trained/random CNN probes remain near chance, removing projection does not rescue them. This probe family is not validated as an information-loss assay; no neural erasure or temporal-comparison lesion established. All72 fits/79 metric records replayed exactly, both frozen model identities preserved, original1800s cap retained. No training/task/cloud change.

**Upstream frozen selected2297 diagnostic completed on300 fresh native groups.** [Results](../krauzlis-upstream-motion-diagnostic.md): full-history pixel changed-side BA0.965, endpoint pixel0.760, direct CNN/KDA pre/post probes near0.50; no matched temporal-access rescue. Exact source/checkpoint parity and saved-prediction replay passed. No deployed training/task/cloud changes; original1200-second cap preserved.

**Frozen selected2297 factorized follow-up completed — exploratory reused-test.** [Results](../krauzlis-factorized-diagnostic.md). Original native n175, paired challenge175 groups. Disjoint component/calibrator fitting; matched final/external-phase access; fitted-model replay and grouped uncertainty. No model/stimulus/cloud changes.

**2026-10-01 — Local fresh Krauzlis run completed; cloud attempt02 launcher prepared, not yet launched by researcher.** [Evidence and protocol](../krauzlis-fresh-attempt02.md). Local saved report:4595updates/147040episodes; validation-selected2297 final BA0.500000/AUC0.534178, terminal4595 BA0.500000/AUC0.558973, all-positive decisions in all three conditions. This supersedes the local live snapshot below. Explicit NEW8h/$5 cloud retry remains wholly fresh; no checkpoint inheritance. Real CPU checkpoint3 now produces native verification and actual launcher readiness; source/guard/profile-handoff suites passed. Parent owns rental; no cloud calls/GPU work in this preparation. Prior logs/artifacts preserved.


**2026-09-24 05:47 UTC — [Final ConvGRU review-ready](../spatial-readout-convgru.md):**78 final CPU tests;35/35 disposable MPS cells profiled and full-state checkpoint independently verified. Pinned1690 added updates /54080episodes,5.050 projected optimizer hours, unchanged13:33:53.442736 UTC hard deadline. Parent independent review and production launch are next; no production is claimed. [Exact files/config receipt](../../SecondPass/SpatialReadout/REVIEW_READY.json).

**2026-09-24 — [Final spatial ConvGRU after KDA](../spatial-readout-convgru.md): implementation and78 CPU tests passed;35/35 renderer cells verified.** The new branch warm-starts verified global-GRU3393, keeps all three KDA modules and compresses only after final spatial recurrence. Disposable MPS profiling is active under one immutable eight-hour cap; production awaits parent source/config review. A real launchd ownership/cap probe passed using the verified local runtime outside protected Desktop. The old v4 run is stopped; older live wording below is historical.

**V4 RECOVERED / RUNNING — verified 2026-09-23 08:14:42 UTC.** After normal v3 completion and the preserved pre-activation queue failure, explicit recovery passed all predecessor and unchanged feasibility gates. Independent CPU readback verified exact model/Adam/scheduler/stream/RNG migration from terminal 2860, then checkpoint 2861 with 42 changed model tensors and 42 advanced Adam states; persisted progress reached **2870 / 91,840 episodes**. Parent owns handed-off supervisor **`proc_b345f6fe2730` / PID15739**, sole MPS worker **15763**. The new cap began **08:13:26.181166 UTC** and expires **16:13:26.181166 UTC**, with no renewal. Target remains exactly **2145 added updates / 68,640 episodes**, endpoint **5005 / 160,160**. **66 CPU tests passed**; final evaluations remain pending. Monitor run `recovery_result.json`, `live_status.json` and `latest_checkpoint.json`; original `queue_result.json` deliberately preserves the blocked attempt. [Recovery details](../../SecondPass/JointTraining/RECOVERY_V4.md). All earlier statuses below are historical snapshots.

**New authorization / QUEUED, 2026-09-23 07:46 UTC:** another **2,145 updates / 68,640 episodes**, reaching **5,005 / 160,160**, is durably queued after v3 finishes validation2860, both complete final evaluations and normal OS exit. The unchanged recipe adds 165 updates / 5,280 episodes per task. v3 is still the sole MPS worker (PID51433), last verified in `final_test_terminal` at step2860; it was not interrupted. The new eight-hour cap has **not started**. Parent owns queue **`proc_af5aedcba7d9` / PID11016**, transferred with completion delivery; queue expiry is **09:13:50.062308 UTC**. Receipt and exact-state/budget gates fail closed. No v4 optimizer progress is claimed. CPU tests: **49 passed**, compileall passed. See [v4 protocol](../../SecondPass/JointTraining/AMENDMENT_V4.md) and [queue handoff](../../SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/handoff.json). Earlier running/completed descriptions below are historical snapshots.


**Technical report, 2026-09-23:** [Joint KDA architecture and proposed microstimulation](../joint-kda-architecture-microstimulation.md) — ten-page [PDF](../../SecondPass/JointTraining/TechnicalReport/architecture_microstimulation.pdf), source-audited comparison to Morgan, Albanna & Herman, and a frozen preliminary validation snapshot through step2314. This is documentation and protocol design, not a new intervention or training launch.

**New authorization / running, 2026-09-23:** the user explicitly requested “go ahead and set up a 8 hour training run for more trsining and more updates”. The versioned v3 continuation resumes **terminal 715**, not historical selected 117. It pins **2,145 additional updates / 68,640 additional episodes** (5,280/task), reaching **2,860 total updates / 91,520 episodes** (7,040/task). One local MPS worker; no architecture, loss, stimuli, sampling, optimizer or batch changes.

Verified live update733 and checkpoint728; hard deadline **2026-09-23 09:08:50.062308 UTC**. Parent owns guardian `proc_5cbdb07600dd` (sole MPS worker51433). Details and automatic final-report location: [joint-training record](../joint-suite-training.md). Earlier completion/no-further-authorization statements below are historical and superseded by this explicit new allowance.


[Next-agent handoff](../../HANDOFF.md) ·
[paper-writing research handoff](../../PAPER_HANDOFF.md).

This is the project's research wiki: what we built, what actually happened, why the next experiment followed, and what remains uncertain. It covers extant work after the user-authorized repository reset, through the dated snapshot below. Deleted pre-reset projects and plans have not been recovered.

**Current state (2026-09-22):** the second-pass KDA model has demonstrated
orientation-family acquisition and retention; ConvGRU ties its delayed scores.
The [unified task suite](../../SecondPass/TaskSuite/README.md) assembles 13
tasks in 35 primary cells. The user-authorized [fresh local joint-training
run](../joint-suite-training.md) **completed** 715 updates / 22,880 episodes
(1,760/task) within the original four-hour cap. Both checkpoints completed
35-cell final tests. Terminal 715 learned several sensory tasks strongly,
but motion and spatial/sequence tasks remained near chance. The
minimum-task-first validation rule selected earlier checkpoint 117 instead;
[completed results](../../SecondPass/JointTraining/RESULTS.md) report both without
retrospectively changing selection. This is not full-suite convergence.
The [v2 amendment](../../SecondPass/JointTraining/AMENDMENT_V2.md) preserves the
allocation correction and progress-preserving handover record. No additional
training has been launched.
See the [assembly record](../task-suite-assembly.md) and
[current status](../CURRENT_STATUS.md). Older live labels in the experiment
catalogue are historical snapshots; the September 16 reset closed that lineage.

## Start here

- [Current state, capabilities and unresolved failures](../CURRENT_STATUS.md)
- [Chronology: observations, decisions and corrections](../CHRONOLOGY.md)
- [Current architecture: tensors, equations and what learns](../ARCHITECTURE.md)
- [Tasks, timing, datasets and metric definitions](../TASKS_AND_METRICS.md)
- [Open questions and what evidence would answer them](../OPEN_QUESTIONS.md)
- [Research foundations and limits of biological claims](../RESEARCH_FOUNDATIONS.md)
- [Completed local frozen-readout diagnostic](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
- [How to update this journal](../MAINTENANCE.md) and [new experiment template](../EXPERIMENT_TEMPLATE.md)
- [Source inventory and hashes](../evidence_manifest.json)

## Experiment catalogue

| ID | Experiment | Status at journal entry |
|---|---|---|
| 01 | [Five convolutional encoders: the first corrected sensory screen](../experiments/01-pav-encoder-screen.md) | Complete |
| 02 | [Can complementary encoder errors be combined cheaply?](../experiments/02-saved-ensemble-audit.md) | Complete |
| 03 | [Gabor and late-SE additions did not solve contour](../experiments/03-hybrid-continuation.md) | Complete |
| 04 | [Contour succeeds when it receives enough training allocation](../experiments/04-contour-task-allocation.md) | Complete |
| 05 | [One frame at a time: opponent traces beat the tested KDA and ConvGRU fits](../experiments/05-causal-temporal-accumulators.md) | Complete |
| 06 | [The broad sequence battery did not isolate memory capacity](../experiments/06-sequence-battery.md) | Complete |
| 07 | [Focused recurrent memories learn the rules; LSTM leads on motion duration](../experiments/07-lstm-versus-ei.md) | Complete |
| 08 | [Ordering sensitivity: the E/I model benefits from later winning evidence](../experiments/08-recency-diagnostic.md) | Complete |
| 09 | [Earlier motion evidence remains in the firing-rate population](../experiments/09-state-accessibility.md) | Complete |
| 10 | [An existing output-only refit recovers much of motion performance](../experiments/10-readout-refit.md) | Complete |
| 11 | [Retaining a motion judgment improves; delayed orientation comparison remains weak](../experiments/11-retention-learning.md) | Complete |
| 12 | [Orientation survives the delay but the ordinary comparison fails](../experiments/12-orientation-accessibility.md) | Complete |
| 13 | [Keeping a spatial field helps binding but does not solve every task](../experiments/13-spatial-memory.md) | Complete |
| 14 | [Additive controller feedback gives a narrow gain, not a maintenance explanation](../experiments/14-selective-maintenance.md) | Complete |
| 15 | [Joint attention substantially improves delayed orientation, with a motion cost](../experiments/15-preupdate-attention.md) | Complete |
| 15a | [The orientation improvement depends on memory-source routing during blanks](../experiments/15a-attention-mechanism.md) | Complete |
| 15b | [Motion errors include decision bias and predate attention](../experiments/15b-motion-audit.md) | Complete |
| 16 | [Training exposure:10% versus50% motion](../experiments/16-training-exposure.md) | Complete |
| 17 | [Frozen attention maps and scene overlays](../experiments/17-attention-maps.md) | Corrected per-frame viewer complete |
| 18 | [Bias-free attention: old-task control and new spatial battery](../experiments/18-unbiased-attention-spatial-battery.md) | Old arm complete; new battery cancelled |
| 19 | [Five-task acquisition with original biases restored](../experiments/19-biased-spatial-battery.md) | User-stopped partial; retrieved and pod deleted |
| 20 | [Focused cued motion-duration continuation](../experiments/20-single-task-motion.md) | Complete; no task-acquisition gain |
| 21 | [Prospective sensory-conditioned attention queries](../experiments/21-prospective-query.md) | User-stopped partial; retrieved and pod deleted |
| 22 | [Spatially preserving task-conditioned priority readout](../experiments/22-spatial-priority-readout.md) | Scratch cloud training live; complementary frozen-core diagnostic complete |
| 23 | [Dual pre/post attention with exclusive priority-map decoding](../experiments/23-dual-attention-priority.md) | Independent scratch cloud training live |
| 24 | [AV-context v2: five combined changes, local scratch run](../experiments/24-av-context-v2.md) | Local training live; pre-registered gate at 2,400 |
| 25 | [Battery audit: ideal observers, streams, one-step training diagnostics](../experiments/25-battery-audit.md) | completed 2026-09-16 (audit only, no training) |
| 26 | [Plain baseline, rung 1: standard CNN+GRU from scratch, recipe sweep, difficulty ladder](../experiments/26-plain-baseline-rung1.md) | completed 2026-09-17 (orientation family; other families record runs only) |
| 27 | [Accumulator states inside the conv stack vs plain baseline: gate, curriculum, delay ladder](../experiments/27-accumulator-conv-stack.md) | completed 2026-09-17 (RunPod, both lanes, two seeds) |

The useful trajectory is not a sequence of architectures declared permanently good or bad. It contains task acquisition failures, output-use failures, genuine improvements, and regressions that triggered targeted diagnostics. Notably, contour was solved by changing allocation; earlier motion information survived in E/I rates and benefited from output refitting; spatial memory helped binding but harmed motion; pre-update attention improved delayed orientation, and its blank-period routing is functionally important.

## Scope and provenance

The broader project follows Jeremy Wolfe's Guided Search 6.0 as a functional scaffold, omits its diffuser, and provisionally treats activated long-term memory as synaptic weights at the user's request. PAV is one component. The second-pass working model uses a CNN with multiscale spatial KDA states and a GRU readout, not the old E/I pre-update-attention stack. It is not a complete GS6 implementation or a validated biological model.

This wiki adds an explanatory layer over retained code, reports, scores, checkpoints and receipts. It does not launch experiments, change models or manufacture missing evidence. Completed pages link primary local artifacts. Live statuses are explicitly dated; they require refresh rather than being treated as permanent facts.

Initial documentation compiled 2026-09-14T00:21:31.576956+00:00. Scientific claims describe the saved experiments, not independent replication across seeds or general human performance.
