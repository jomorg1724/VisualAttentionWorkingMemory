> Historical chronology snapshot; current status is maintained separately.

**October4 — angular contrastive CNN completed: simplified comparison solved.** FreshCNN25024updates/782000presentations/391000unique;3072fresh continuous-change tests,100%BA/AUC1.0 allthree speeds ×26/28°; explicit angular supervision. Best19000/latest25024 CPUverified64Adam states. No active local/cloud training. Original nativeKrauzlis task remains untested for this model. [Journal](angular-contrastive-motion.md).

**October4 — angular contrastive CNN now TRAINING locally.** Fresh wholemodel/MPS, all64Adam states verified at saved1. Target25,024updates, original new8h launch cap; no other model interrupted, predecessorFFN completed. [Journal](angular-contrastive-motion.md).

**October4 — fresh angular contrastive CNN QUEUED locally.** Threeframes →CNN →128Dunit representation; springloss with targetdistance proportional to circular direction change. Allweights fresh, CPUcheck passed; no training/budget started. [Journal](angular-contrastive-motion.md).

**October4 — frozen predictive encoder + FFN completed at chance.** All10,240updates/640,000presentations/320,000unique, 23minutes. Fresh3072-trial test BA50%, AUC0.49519, CE0.693165; predicted no-change throughout. Best5632/latest10240 verified, encoder remained frozen; no active training. [Journal](predictive-motion-change.md).

**October4 — predictive encoder + simple FFN running locally.** Frozenencoder50,000, concat512+512 →256→64→2, balanced26/28° same/change continuous-dot clips. Target10,240 headupdates, batch64, 2epochs/1000freshtrials; no cloud. [Journal](predictive-motion-change.md).

# How we got here

**October3 — Variational motion predictor built and initiallocalpilotcompleted.** ThreeorderedRGBframes→residualCNN→one512DGaussianvector→fourthframeCNNdecoder; all15,463,363parametersfresh/trainable. Newconstant-velocityfullfieldpersistentdots, uniformdirectionacrosssamples andspeeds.375/1/2px, nochanges/cues/labels. Support-balancedprediction+KLwarmup,2epochs/1000samples, batch32micro4 FP32Adam1e-4/noclip. Prospective20mincap pinned256updates/8000presentations/4000unique; completedearly7:37PM, best/latest256CPUverified/138Adamstates. FinalmeanpredictionMSE.004809 versuscopylast.003839/gray.005469; imageslargelyflat/blurry, motionlearningnotdemonstrated. Cloudweightedepoch2finished2310/70000/35000unique, selectedfinalBA50%/AUC.503969; allartifactsretrievedandpoddeleted. NoactiveGPU/cloudrun orautonomouscontinuation. [Predictivejournal](variational-motion-prediction.md).

**October3,7:09PM PDT snapshot:** localinput×10 completed330/10000, freshfinalBA50%/meanAUC.489996 (all-change), best/latest330 verified/guardexited. Cloud epoch2weightedCNNRViT2099/2310 /63640presentations, last100loss.683817; validation2000BA50%/meanAUC.516252/all-change. Cloudcontinuesunderoriginalcap9:58PM, localstaysfinished.

**October3 — Weighted-mean CNN–RViT epoch2 TRAINING on the existing A40.** Saved3/96 CPUverified/all92Adam states and tensors changed; live5. Fresh wholemodel/Adam/RNG/streams, solechange10→2epochs per1000trials. Pinned2310updates/70000presentations/35000unique trials (35pools), FP32/fullBPTT32micro4/Adam1e-4/noclip. Original8h/$5deadline unchanged (hard9:58PM PDT); guard/mirror/retrieval/delete armed. Oldcloud cancelled1304/39544, best/latest retrieved; local×10 continues. [Journal](weighted-mean-rvit-epoch2.md). No validation yet.

**October3 — Input×10 VAE–RViT TRAINING locally: saved optimizer verified.**
**Production training independently verified:** saved update1 /32 trial presentations; all94 Adam states advanced and all94 learned parameter tensors changed. CPU checkpoint reload verified, input scale10 in checkpoint provenance. Initial production loss0.69538. This is training progress, not evidence of held-out acquisition.
Pinned330updates/10000presentations/1000unique trials; same1000×10schedule; hard10:54PM PDT. Previous local model stopped227/6928; cloud weightedCNN–RViT continues.

## October 3 — Local VAE–RViT input ×10 experiment

User requested stopping the local classifier and repeating training with inputs multiplied by ten. Original local run stopped cleanly at 227 updates / 6,928 trial presentations; best100 and latest227 remain preserved. Cloud weighted-mean CNN–RViT continues independently.

`SecondPass/VAERViTInput10` scales the 169×256 encoded tokens by exactly10 **after** spatial positions and token LayerNorm, immediately before the original recurrent block. Internal query/memory/FFN normalization remains unchanged. Consequently, the current-token residual is scaled while normalized attention queries are mostly scale invariant; this is an input-amplitude experiment, not an attention-temperature change.

Same VAE9900 encoder+mu transfer; fresh recurrent/decoder/positions, Adam, RNG and zero classification counters. All94 learned tensors train. Same seeds and train/validation/test namespaces as the original classifier for matched examples. Same no-cue/single-stimulus movies, final-trial crossentropy, full-sequence gradients, FP32/MPS, Adam1e-4/no clipping, batch32/micro1, twoCPUthreads, 1000 movies reused for10 shuffled epochs. Requested990 updates /30,000 presentations /3,000 unique movies, prospectively limited to feasible complete330-update pools by native profiling. A new finite8h local cap covers profile/training/evaluation/reporting; only best/latest checkpoints.

Focused CPU check passed: exact10×token equality, all94 finite parameter gradients and a nonzero earliest-frame gradient. New immutable runtime `VAWMRuntime/vae_rvit_input10_local01`; launcher/independent guard active. Native profiling is underway; no persisted production progress yet.

- 2026-10-03 15:57 UTC: User replaces local weightedCNNGRU with three-frame VAE. Old model saved/stopped570/17,288. NewVAE9,507,913params/latent256x13x13; no response labels. Native MPS profilepins9,900updates/300,000triplets/30,000unique under8hLOCALcap. Checkpoint3/96 CPUverified126Adamstates/tensors. [Record](krauzlis-three-frame-vae.md).

- 2026-10-03 14:44 UTC: User authorizes fresh weighted CNN–GRU training locally. MPS full native profile pins2,310updates/70,000presentations/7,000unique under new8hLOCALcap ending3:40PM Pacific. Productioncheckpoint3/96 independently CPUverified, all70 Adam states/tensors advanced; batch32micro4fullBPTT. Clouds stay closed. [Run](../SecondPass/WeightedMeanConvGRU/RUN_STATUS.md).

- 2026-10-03 UTC: User kills cloud pods, then holds weighted CNN for tomorrow. Disabled automatic queue4518/wake4519; retrieved/verified66single-stimulus artifacts and checkpoint1700/92Adam states, stopped/deleted exactpodjxbmb44y9wamhl. Provider list empty. Weighted CNN source preserved; no launch today. Local training unchanged.

- 2026-10-03 04:43 UTC: Random-frame RViT completed all 3,960 updates; fresh selected/terminal BA50% all three conditions, AUC0.48828/0.47909. Retrieved/verified final artifacts and deleted exact pod. [Final report](../SecondPass/RandomFrameRViT/FINAL_REPORT.md).

- 2026-10-03 04:13 UTC: User requests a queued weighted three-frame mean CNN–GRU on the same single-stimulus task. Implemented 0.5/0.4/0.1 causal RGB mixing, shared residual CNN, direct spatial flatten to one standard GRU, fresh replay worker. Focused CPU check passed; automatic parent queue4518 waits for current single-stimulus retrieval/deletion, with no current rental or allowance consumption. [Record](krauzlis-weighted-mean-conv-gru.md).

- 2026-10-03 03:31 UTC: Single-stimulus conv RViT production verified; all92learned tensors/Adam states advanced from fresh initialization, full2,310updates/70,000presentations/7,000unique pinned under new8h/$5. Original timing/retained dots unchanged, cue/other patch removed. [Run](krauzlis-single-stimulus.md).

- 2026-10-03 UTC: User requests a cue/competition simplification with exact sequence lengths and original locations. Built one-stimulus projection with unchanged conv RViT/full BPTT, passed focused stimulus checks, and provisioned one guarded A40 under new 8h/$5. [Record](krauzlis-single-stimulus.md).


- 2026-10-03 03:12 UTC: Older RViT replay completed 2,310 updates. Lower training loss (final 100-update mean 0.49881) did not generalize: selected final mean BA 50%, terminal 47.83%; mean AUC 0.4672/0.4922. All 63 final artifacts and both checkpoints verified before pod deletion. [Final report](../SecondPass/TwoFrameRViTReplay/FINAL_REPORT.md).


- 2026-10-03 02:43 UTC: Random-frame-gradient gated motion RViT cloud production verified from fresh weights; 94 Adam states/tensors advanced, analytic buffers unchanged. Pinned 3,960 updates / 120,000 presentations / 12,000 unique movies under new 8h/$5; KDA16 already retrieved/deleted. [Evidence](krauzlis-random-frame-gradients.md).

- 2026-10-03 UTC: User requests a gradient correction, then independently sampled frame updates with live suffix Jacobians, and immediate cloud replacement. Built gated carry plus unbiased timestep sampling; focused proof passed; KDA16 freed its slot by completing/retrieving/deleting. [Record](krauzlis-random-frame-gradients.md).


- 2026-10-03 UTC: User questions objective/decoder/gradients. Targeted CPU check confirms CE and accumulation math but finds severe full-sequence early-gradient attenuation in bothRViTs on a paired29-frame native trial; KDA retains an early gradient. No live changes or extra training. [Record](krauzlis-training-path-audit.md).

- 2026-10-03 UTC: User requests localmotion-energyRViT training beforetechnicalwriteup. CNN-GRU saved2575/82400 and stopped; exactCPUfixedfrontend/MPSlearned execution, freshproductioncheckpoint3/96 independently verified. Pin660updates/20000presentations/2000unique under newlocal8hcap; proposalPDF written afterproductionstart. [Record](krauzlis-simoncelli-heeger.md).

- 2026-10-03 UTC: User requests actual Simoncelli–Heeger motion-energy CNN and paper link. Built five-scale author-derived front end plus fresh RViT task pathway; three CPU checks pass. No new training or cloud action. [Record](krauzlis-simoncelli-heeger.md).

- 2026-10-03 UTC: User cancels online RViT at2075/66400 and requests fresh1000-movie pools reused10epochs. Old89artifacts/92Adamstates verified; replacementusesexistingpod/originaldeadline. [Record](krauzlis-rvit-replay.md).

**2026-10-02 — Two-frame RViT TRAINING on a second A40; faster validations enabled.** [Run](../SecondPass/TwoFrameRViT/RUN_STATUS.md). Explicit parallel launch beside16headKDA, fresh7,270,290params/92tensors; checkpoint3/96episodes downloaded/hashverified/CPUreloaded, all92params/Adamstatesadvanced. Nativeprofilepinsfull4216updates/134912episodes,18costed validationlooks100,250,500,...terminal at100trials/cell; pairedfresh200/cellfinals. Pod7f27p6jxpitihn,NEW8h/$5cap21:16:04UTC→05:16:04UTC /October2 10:16:04PM PDT, independentguard/mirror. Native teaching unchanged. CPU-only earlysnapshotsof existingKDA/CNN-GRU running without changingtraining/selection; labelthosevalidationonly. Existingrunscontinue,3layercancelledpodstaysdeleted.

**2026-10-02 — Fresh16-head single-layer KDA TRAINING, optimizer verified; three-layer cancelled.** [Run evidence](../SecondPass/SequenceKDA16/RUN_STATUS.md). Userrequestedkillthree-layerandstartqueuedmodel. NewA40podpa0ko8f2qirisy; nativeprofilepins4216updates/134912episodes,FP32fullBPTT/effective32micro4. Actualcheckpoint3/96episodes downloaded/hashverified/CPUreloaded,all27params/Adamstatesadvanced,freshconstructor/emptyinitialAdam/nativeRNGstreams verified; no profile/predecessorinheritance. Independentguard/mirroractive,NEW8h/$5cap20:33:14UTC→04:33:14UTC /October2 9:33:14PM PDT. Oldthree-layerstopped2615/83680;71artifactsand55stateAdamcheckpointverifiedbeforepod5hnvb87npqpqb4stopped/deleted,nofinaltests. LocalCNN-GRUcontinues;RViTuntrained. Earlierlaunchpending/waitingnotesarehistorical.

**2026-10-02 — Three-layer KDA CANCELLED; queued16headKDA launched on newA40.** Userrequestedkillandreplacement. Three-layerstopped2615/4216updates,83680episodes; all71artifactsretrieved/hashverified,terminalCPUreloaded55Adamstates,andpod5hnvb87npqpqb4stopped/deleted. Onlycompletedvalidation2108gave50%BAallconditions/meanAUC0.499388; nofinaltests. Newfreshsingle-layer16headKDApodpa0ko8f2qirisy launched underNEW8h/$5,cap20:33:14UTC→04:33:14UTC (October2 9:33:14PM PDT),nativeCUDAprofile/pin inprogress, productionAdamproofpending. LocalCNN-GRUcontinues;RViTimplementationonly. [Cancelledrun](../SecondPass/SequenceKDA3/RUN_STATUS.md) · [Newrun](../SecondPass/SequenceKDA16/RUN_STATUS.md).

**2026-10-02 — Two-frame CNN / visual-query RViT IMPLEMENTED, untrained.** [Architecture](../SecondPass/TwoFrameRViT/README.md), [journal](krauzlis-two-frame-rvit.md). Ordered previous/current RGB CNN → 13×13×256 tokens; one shared recurrent block with independent visual self-attention and previous-memory cross-attention, both queried by current X. Per-token 256→16, flatten2704→FFN→2. 7,270,290 fresh trainable parameters/92tensors, full temporal gradients and CNN checkpointing. Three focused CPU checks plus complete native45-frame B28 forward passed; no GPU/profile/training launched or new compute allowance. Existing cloud/local runs and16headKDAqueue preserved.

**2026-10-02 — Single KDA with16full-width heads IMPLEMENTED and AUTOMATICALLYQUEUED.** [Run status](../SecondPass/SequenceKDA16/RUN_STATUS.md), [journal](krauzlis-sequence-kda16-heads.md). OneKDA/terminalCLS,16×64key/value heads,737,170freshparameters/27tensors,8×originalKDAstate. Userselectednew8h/$5cloudrunafterthree-layerpod5hnvb87npqpqb4 completes, artifacts/finalcheckpointsverify and deletionconfirms. Parentqueue63567/PPID1active; no new rental/profile/optimizer/budgetstart. NativeGPUprofilewillpinfeasibleexposure,target4216,Adam1e-4/effective32micro4/FP32fullBPTT/native teachingunchanged. Sevenfocusedchecks passed. Cloudthree-layer and localCNN–GRUcontinue unchanged.

**2026-10-02 — Delayed-frame CNN/standardGRU LOCALTRAINING, optimizer verified.** [Live run](../SecondPass/DelayedFrameGRU/RUN_STATUS.md), [journal](krauzlis-delayed-frame-gru.md). Fresh21,293,770parameter model,AppleM4Max36GiB/MPS,Adam1e-4/no clipping,fullBPTT/FP32,effective32/micro1. Nativeprofilepinned2,631updates/84,192episodes beforeproduction;877updates/28,064episodes eachB12/B20/B28. Checkpoint3/96episodes CPUverified,all86Adamstatesadvanced;launchdowner54824/PPID1+guard54822. HardcapOctober2 7:57:51PM PDT; no renewal. Cloudthree-layerKDAcontinues unchanged. Localvalidation/finals pending; earlieruntrainedcandidatenotes superseded.

**2026-10-02 — Delayed-frame CNN/standardGRU candidate implemented while THREE-layerKDA training continues.** [Design and source](../SecondPass/DelayedFrameGRU/README.md), [journal](krauzlis-delayed-frame-gru.md). Two independently trainable residualCNNs, all24convolutions stride1/full100x100/no pooling; current256+previous256→standardGRU512→256→binaryhead.21,293,770fresh parameters/86tensors; five focused CPU checks passed. Native teaching unchanged, explicit causal delay/fullgradients. **Candidate not trained; no new cloud job/budget.** The KDApod/guard/mirror/deadline below remain active.

**2026-10-02 — Fresh THREE-layer whole-sequence KDA training on RunPod, optimizer verified.** [Run evidence](../SecondPass/SequenceKDA3/RUN_STATUS.md) and [depth comparison journal](krauzlis-sequence-kda3-depth.md): A40pod`5hnvb87npqpqb4`,324488fresh trainable parameters,3pre-LNresidualKDA blocks/terminalCLS. Native tasks and Adam settings unchanged. SteadyGPUprofilepinsfull4216updates/134912episodes matching one-layer exposure. Checkpoint3/96episodes downloaded/hashverified/CPUreloaded; all55namedAdam states advanced. Guard/mirror active. Originalcap18:20:05UTC→02:20:05UTC /October2 7:20:05PM PDT preserved through initial unstarted-pod replacement; no renewal. Validation/finaltests pending. One-layer completion below remains the baseline.

**2026-10-02 — Single layer whole sequence KDA completed; task not acquired.** [Final results and artifacts](../SecondPass/SequenceKDA/RUN_STATUS.md): all4,216updates/134,912episodes completed from fresh weights. Validation selected2,108; selected and terminal4,216 both give50% BA in B12/B20/B28 on200 fresh paired tests/condition, with all-positive decisions (57% raw accuracy,100% foil/catch false positives). Selected/terminal mean test AUC0.500102/0.502422. All45 manifest files retrieved and verified; both checkpoints CPU-reloaded with27 active Adam states. A40pod`8gamd8ems1pa0n` stopped and deleted after verified retrieval at17:56:01UTC. No renewed cap or further training. Earlier live entries are historical snapshots.

**2026-10-02 — Fresh one-layer sequence KDA training launched and verified.** [Run evidence](../SecondPass/SequenceKDA/RUN_STATUS.md): Palladio A40pod`8gamd8ems1pa0n`, exactlyoneglobalKDA over allrawRGBpatches plus terminalCLS; native26/28degree Krauzlis unchanged. User corrected excessive checking; repeated paid remote CPU suites removed. NativeGPUprofile prospectively pinned4216updates/134912episodes; production reconstructed fresh model/Adam/RNG/streams. Checkpoint3/96episodes downloaded/hashverified/CPUreloaded, all27parameter tensors changed andnamedAdamstep3. Snapshot11updates/352episodes, no performance result yet. Absolute8h/$5deadline00:15:49.999944UTC October3, guard/private status/15secondmirrorverified; no extension or automatic extra exposure.

**2026-10-02 — User requested whole sequence patches with CLS, then specified one KDA layer.** [Recorded design](krauzlis-sequence-kda-design.md) fixes shared RGB patch embedding, spatial/time positions, one global two-head KDA and final CLS classifier. Decay initialization accounts for 100 patch updates per frame. Native task and teaching unchanged; entirely fresh future initialization. Design only, with a proposed later bounded cloud run and no implementation or compute launch.

**Upstream frozen selected2297 diagnostic completed on300 fresh native groups.** [Results](krauzlis-upstream-motion-diagnostic.md): full-history pixel changed-side BA0.965, endpoint pixel0.760, direct CNN/KDA pre/post probes near0.50; no matched temporal-access rescue. All choices frozen before new test generation, grouped uncertainty and exact prediction replay. One missing optional-dependency import recovered before fits under the unchanged1200-second cap. No deployed training/task/cloud changes.

**Frozen selected2297 factorized follow-up completed — exploratory reused-test.** [Results](krauzlis-factorized-diagnostic.md). Original native n175, paired challenge175 groups. Disjoint component/calibrator fitting; matched final/external-phase access; fitted-model replay and grouped uncertainty. No model/stimulus/cloud changes.

**2026-10-01 — Local fresh Krauzlis run completed; cloud attempt02 launcher prepared, not yet launched by researcher.** [Evidence and protocol](krauzlis-fresh-attempt02.md). Local saved report:4595updates/147040episodes; validation-selected2297 final BA0.500000/AUC0.534178, terminal4595 BA0.500000/AUC0.558973, all-positive decisions in all three conditions. This supersedes the local live snapshot below. Explicit NEW8h/$5 cloud retry remains wholly fresh; no checkpoint inheritance. Real CPU checkpoint3 now produces native verification and actual launcher readiness; source/guard/profile-handoff suites passed. Parent owns rental; no cloud calls/GPU work in this preparation. Prior logs/artifacts preserved.


**2026-10-01 00:54 UTC — Local Krauzlis-only fresh terminal-transformer acquisition launched and independently verified.** Explicit new local authorization selected WHOLEMODEL FROM SCRATCH; no predecessor or disposable-profile state inherited. New [KrauzlisOnly adapter](../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisOnly/README.md) reuses native renderer, update loop, evaluator and full-state checkpoints while restricting scheduling/selection to unchanged B12/B20/B28. CPU regression failed first for absent implementation then passed in actual Torch2.8 runtime. Source-only hashed runtime and launchd import/timeout probe verified before accelerator work. New cap began00:52:13.337213UTC and ends08:52:13.337213UTC; profile covered two batch32/micro4 optimizer updates and20 forward episodes per condition, then discarded all state. Prospectively pinned4595updates/147040episodes instead of10000 to fit conservative measured eight-hour allocation; two validation100/cell looks at2297/4595, mean AUC then BA, final paired selected/terminal200/cell with event-subgroup confusion/rates. Launchd supervisor91782/PPID1 owns worker91913, no automatic restart. Independent checkpoint3 SHA256 `909ef571c87435b7b788c01d643382693afeed5063d663b9a6dda03927865dd8` verified77 fresh constructor tensors, empty initial Adam/native streams,52 changed used learned tensors and advanced named Adam/three streams/MPS RNG.37 updates observed, no held-out production scores yet; first13 update timing0.678x matched profile. Run `/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run`; full launch receipt `independent_launch_verification.json`. No cloud operations or lifecycle guesses; prior sources/artifacts retained. The prior cancelled local request itself launched nothing; this is the newly authorized run.


**2026-09-30 05:48 UTC — Whole-model fresh spatial consolidation launched.** User authorized one transformer over the final49 spatial memory tokens, reshaped before unchanged dense readout, no recurrent feedback and no pretraining. [Variant](../SecondPass/SpatialReadout/SpatialConsolidation/README.md):1433396 parameters, one pre-RMSNorm/4-head SDPA/SwiGLU176 block with fixed2D positions. New A40 pod `dwjm8cvx7qaqp5` has immutable12h/$8 cap ending17:36:34UTC; measured preproduction allocation29874updates/955968episodes, not exposure-matched to existing continuation. Verified39 production updates and locally hash-checked/loaded step2 full checkpoint, including advancing Adam. No production validation yet. Independent shutdown and off-pod status verified; automatic local mirroring remains inactive, manual transfer verified. Original ConvGRU unchanged. Runtime `cloud_spatial_consolidation_01` records source package, profile, allocation, initial state and retrieval evidence.

**2026-09-29 — Frozen fresh-lineage checkpoint35039 diagnostics delivered.** [Cued-motion diagnosis](../SecondPass/SpatialReadout/CuedMotionAudit/MODEL_FINDINGS.md): durationD0/D24 BA34.38/35.94%, weak correct-direction cue effect, no established added-delay decrement; KrauzlisB20 BA50%/.531AUC with all-positive reports under cue swapping/event removal. Pixel observers establish usable evidence, not a uniquely localized neural failure. Subsequent user-requested [neuroscience study](../SecondPass/SpatialReadout/NeuroscienceAnalysis/FINDINGS.md) produced14,336 scored presentations,34-page atlas, actual trial/time maps and paired inhibition/stimulation on solved spatial tasks. Native cued orientation/binding perfect in sampled delays; below-native orientation magnitudes reveal graded behavior. Small orientation perturbation effects and null binding flips do not establish monkey-equivalent mechanisms. Parent checked773 hashes, replayed every scored decision and inspected graphics. Pinned grid completed under original3600s allowance with unchanged weights/cloud training; independent foil factorial, matched-cue/magnitude allocation and binding random-pulse controls remain unmeasured. No extra experiment launched.

**2026-09-29 — Explicit longer-run authorization and saved-state transition.** User requested stop/save/latest-checkpoint continuation for a ten-fold longer RunPod run. Fresh ConvGRU stopped at7739; terminal checkpoint and33 artifacts downloaded/hash-verified; no final test on signal stop. Disclosed target104000 additional updates (111739 cumulative), same model/native tasks/Adam/RNG/streams, new36h/$20 cap including setup/evaluation/retrieval. Deliberate same-pod container restart replaced original eight-hour guard; authenticated new guard and private provider status verified through2026-09-30T18:14:46.431188Z. Continuation harness implementation pending; this entry is not a launch claim. Original validation5200 remains selection evidence, not a final test or full-battery success.

**2026-09-29 04:07 UTC — User-requested ConvGRU FROM SCRATCH launched.** Entire CNN/KDA/final-spatial-ConvGRU/readout/heads initialized fresh; direct constructor equality checked against checkpoint zero, empty Adam/native streams and advancing saved state verified. One A40 `5awq67fxgnsu1m`,89updates/2848episodes observed, local checkpoint78 verified. Complete10400update/332800episode allocation pinned from profile under new8h/$5cap ending12:02:21UTC; unchanged13tasks/35conditions and learning recipe, no transferred weights or new curriculum. No validation yet; near-uniform initial losses are not an acquisition result. Original global-GRU remains the historical fresh baseline, earlier warm-start branches remain separate, and cancelled transformer runs stay stopped. Runtime `cloud_convgru_fresh_01`; [implementation](../SecondPass/SpatialReadout/FreshRun/README.md).

**2026-09-29 03:18 UTC — User cancellation and architecture rollback.** Stopped no-CLS pod `7ij62e571pln8w`; provider confirms EXITED. Removed mirror service, verified process absent and locally saved checkpoint2899 digest. Preserve disk/artifacts, disclose continuing storage billing, no restart. Restored original CNN/spatial-KDA/global-GRU as the active architecture, not a transferred trained checkpoint; see [provenance](../SecondPass/ACTIVE_BASELINE.md). User prohibits inherited weights unless specifically requested. Earlier changed-architecture warm starts remain historical evidence only, not proof of from-scratch learning. Cancellation does not establish architecture incapacity or completed evaluation.

**2026-09-29 02:03 UTC — Fresh retry launched and persisted progress verified.** Same no-CLS model/native teaching, all weights/Adam/streams freshly constructed from the recorded seeds, with no state transfer. A40 `7ij62e571pln8w` reached96updates/3072episodes; local automatic mirror verified checkpoint39, remote saved91. Preproduction measured allocation3224updates/103168episodes, first validation1612; deadline05:57:23.958244UTC and$3maximum. Fixed guard publishes status/reason to a private provider record without modifying the running pod, and stops only for verified full retrieval or absolute deadline. Launchd-owned local mirror is active and exercised, not merely configured. New accuracy pending. Prior attempt01 disk remains preserved and its final outcome unresolved.

**2026-09-29 — Opaque shutdown incident and authorized retry.** Attempt01 was provider-stopped at22:32:55UTC,111.63minutes after start and before its four-hour deadline. Account-issued stop is established; completion versus stale-progress guard trigger is not. Saved remote results remain inaccessible due to host recovery capacity; only checkpoint442 verified locally. User requested relaunch without this failure mode. Preparing the unchanged fresh architecture/tasks with a guard limited to verified retrieval or absolute spending cap, provider-readable private status independent of pod disk, and a launchd-owned checkpoint/result mirror. Prior run is preserved as unresolved, not called completed or a model failure.

**2026-09-28 20:47 UTC — User-corrected fresh no-CLS training launched.** User rejected inherited weights and explicitly requested a new pod run. Fresh CNN/KDA/transformer/convdecoder/heads, empty Adam/new streams and RNG verified in persisted production state. A40 `zab3qcxa59uxil`, checkpoint52 already locally verified. Measured preproduction allocation4498updates/143936episodes; original13tasks/35conditions unchanged. New finite4h/$3 maximum, independent stop at2026-09-29T00:41:16.233494Z. No accuracy result yet. [Fresh training implementation](../SecondPass/SpatialRecurrentConvDecoder/README.md).

**2026-09-28 — Recurrent transformer/CLS completed and retrieved.** Full2600 updates/83200episodes; validation selected1300, terminal2600; both fresh finals complete35conditions. Several sensory tasks learned, but spatial/memory tasks remain near chance; selected/terminal motion46.88/25.78%, cued duration23.63/25.78%, binding50.39/52.34%, recognition51.91/50%. Empty specificity100% separately. All40 artifact hashes/sizes verified locally; CPU-only bounded recovery did not restart training. Pod stopped, disk preserved, no automatic continuation. [Results](../SecondPass/SpatialRecurrentTransformer/RESULTS.md).

**2026-09-28 16:03 UTC — Recurrent convolutional transformer with persistent CLS launched and verified.** From recovered terminal9360, retained compatible CNN/KDA/input/head weights and Adam, replaced ConvGRU/comparator/flatten readout with two64-channel/two-head recurrent transformer blocks and CLS-only decisions. All56 fresh tensors demonstrably update; exact migrated state preserved. Live75updates/2400episodes, saved65; checkpoint39 recovered locally. New-architecture profile fits all2600updates/83200episodes within unchanged21:46UTC/$12 cap; all13tasks/35conditions unchanged. New validation/test namespaces; accuracy pending, firstlook1300. Independent cloud stop remains armed; no new local login service installed. [Implementation](../SecondPass/SpatialRecurrentTransformer/README.md).

**2026-09-28 — Comparison run completed/retrieved; recurrent transformer authorized next.** Selected1300 and terminal2600 both completed35-cell held-out tests, all53 published artifact hashes verified locally; parent terminal cumulative9360 preserved. Spatial/memory performance remains near chance despite strong basic sensory discrimination. User now explicitly requests joint sensory/memory attention, recurrent H-residual convolutional transformer updates, and CLS-only decisions. One new candidate is being implemented on the unchanged battery; no new launch yet, $12/21:46:03UTC cap retained. [Completed comparison results](../SecondPass/SpatialComparisonReadout/CloudRun/RESULTS.md).

**2026-09-28 14:26 UTC — Learned spatial-comparison candidate finally reached verified cloud production.** A40 pod `txmfzvhbc9zm3b` deployed under the unchanged attempt deadline21:46:03 UTC and $12 ceiling. A Torch2.8 Adam metadata mismatch was reproduced and fixed by explicitly carrying `decoupled_weight_decay=False`, preserving every inherited optimizer tensor/step/option. Checkpoint26 independently verified832 new episodes, all four comparator tensors learning and advanced native streams; full13-task cycle timing fits the measured plan. Target2600 additional updates/83200episodes from terminal6760, unchanged native13-task/35-condition protocol and no new control arm. Disposable profile discarded; all failed preproduction receipts preserved. No new held-out result yet. Evidence: runtime `cloud_comparison_03/production_verified.json`; current status supersedes earlier failed/stopped snapshots.

**2026-09-28 — Cloud attempt did not reach verified optimizer progress.** Deployment failed after interrupted uploads and provider/network loss. Authenticated account readback confirms `vj0rc2sb7da5m7` is `EXITED`; stopping cause/time and actual charges remain unknown. No restart after expired cap. Code and original checkpoints remain local; stopped disk retained pending recovery. No new model-performance claim.

**2026-09-27 — Cloud placement and training authorized for the new candidate only.** User asked for a smaller suitable RunPod GPU and training start. No3090/A5000 stock; one24GB4090 pod `vj0rc2sb7da5m7` created09:34:36.695UTC at$0.74/GPU-hour plusstorage, finite8h cap through17:34:36.695UTC. Verified SSH/GPU/PyTorch; versioned CUDA launch preparation dispatched. Target2600 updates/83,200episodes from terminal6760 with unchanged13tasks/35cells, inherited Adam/native streams, fullBPTT/fp32; no new teaching or extra control arm. Local cancelled run remains preserved. Training requires separate persisted-progress evidence; this entry records provisioning only.

**2026-09-25 — Matched final-only/time-separated diagnostic completed.** Twelve structured diagnostic fits selected before1024 new test episodes; capacity/supervision matched within each layer/access pair. Whole ConvGRU final-only63.09% orientation/85.74% binding, versus61.33%/87.70% time-separated; no clear primary BA benefit from external temporal access. Early final-only90.82%/100% removes the prior external diagnostic-memory explanation for strong early D0 results. Final25649.80%/62.50% supports investigating decision/representation use, without a causal compression claim. Frozen model/source/prior-artifact hashes unchanged, native wrapper parity exact, final-only invariance verified;70.93s bounded analysis/report. [Report](../SecondPass/SpatialReadout/TemporalAccessDiagnostic/REPORT.md).

**2026-09-25 — Frozen feature/comparator investigation completed.** Terminal6760 native D0 orientation/binding:1024/256/512 independent episodes per task. Cues remain decodable; early local angle information supports a91.60%/100% auxiliary-supervised structured diagnostic, while a label-only comparator remains49.22%/56.25% and is not a clear paired improvement. The structured result has supplied circular relations and time-separated activation access; it does not establish final-state-only readout rescue or recurrent retention. Checkpoint/source hashes unchanged; actual frozen extraction/fits/report360.82s within1800s; no restart after a false/null tracker exit. [Full result](../SecondPass/SpatialReadout/FeatureDiagnostic/REPORT.md).

**2026-09-25 06:37:41 UTC — Triple-length ConvGRU continuation completed.** All5070 additional updates /162240 fresh episodes finished, total6760 /216320. Selected5486 and terminal6760 each have35 unique final cells. Additional optimizer work14.023h; exit0 before unchanged cap. Terminal motion100% and contour82.81% show acquisition gains; spatial tasks still chance. Selected5486 retained under prospective mean-validation-AUC rule, despite terminal's stronger motion. No further run launched. [Record](spatial-readout-convgru.md).

**2026-09-24 12:01:55 UTC — ConvGRU training and final evaluation complete.** After exact-state scheduling repair at124, all1690 planned additional updates /54080 episodes finished; selection1690, all35 final cells verified. Supervisor exit0, no cap trigger, about92minutes before unchanged deadline. Terminal checksum verified at status inspection. Basic sensory tasks mostly strong; contour64.84% BA; motion25%, cued motion25.20%, all spatial binary tasks near50%. The spatial-GRU replacement did not acquire the difficult spatial battery under this exposure. No additional run launched. [Results and caveats](spatial-readout-convgru.md).

**2026-09-24 05:47 UTC — Profiling completed; production not launched.**35 disposable effective-batch32/micro4 updates covered every cell in661.729s. Full-state readback proved exact migration and eight changed fresh tensors with Adam step35.78 final CPU checks passed. Conservative preproduction allocation pinned1690 additional updates /54080episodes, with validation845/1690 and selected/terminal35-cell final tests. Projection5.050optimizer hours; hard deadline13:33:53.442736 UTC, no reset. Untouched final namespace94692763 supersedes the tiny untrained CPU-smoke namespace before config pinning; all old receipts remain. Researcher returns the exact review-ready artifacts so parent can independently review and launch immediately. [Record](spatial-readout-convgru.md).

**2026-09-24 — New architecture branch, not a restart of expired v4.** User authorized final ConvGRU while retaining all three KDA encoder modules. Verified source3393/108576 episodes migrated by exact parameter name+shape with compatible Adam, scheduler, streams and RNG; fresh spatial recurrence/readout and reset selection.78 CPU tests passed, native suite35/35. Actual launchd CPU tests exposed Desktop TCC denial; a byte-verified local runtime outside Desktop passed ownership (parent PID1) and hard-cap child termination without permission changes. Disposable MPS profiling now measures all35 cells at batch32/micro4; new cap epoch1790228033.4427361→1790256833.4427361 never resets. Parent reviews and launches only after the pinned profile/config is returned. No production or scientific score is claimed yet. [Detailed record](spatial-readout-convgru.md).

**V4 RECOVERED / RUNNING — verified 2026-09-23 08:14:42 UTC.** After normal v3 completion and the preserved pre-activation queue failure, explicit recovery passed all predecessor and unchanged feasibility gates. Independent CPU readback verified exact model/Adam/scheduler/stream/RNG migration from terminal 2860, then checkpoint 2861 with 42 changed model tensors and 42 advanced Adam states; persisted progress reached **2870 / 91,840 episodes**. Parent owns handed-off supervisor **`proc_b345f6fe2730` / PID15739**, sole MPS worker **15763**. The new cap began **08:13:26.181166 UTC** and expires **16:13:26.181166 UTC**, with no renewal. Target remains exactly **2145 added updates / 68,640 episodes**, endpoint **5005 / 160,160**. **66 CPU tests passed**; final evaluations remain pending. Monitor run `recovery_result.json`, `live_status.json` and `latest_checkpoint.json`; original `queue_result.json` deliberately preserves the blocked attempt. [Recovery details](../SecondPass/JointTraining/RECOVERY_V4.md). All earlier statuses below are historical snapshots.

**2026-09-23 08:12 UTC — V3 completed; v4 queue blocked before activation:** v3 finished normally at08:04:35, terminal2860 /91520 episodes with both35-cell finals complete and selected2314 retained. At08:04:36 the queue rejected a changed raw process identity; the actual changed string was not saved. A CPU normal-exit zombie transition reproduces this failure mechanism without proving the historical transient. Explicit versioned recovery retains blocked receipts, requires all original PIDs absent and full normal-completion verification, and passes66 CPU tests. All-final-timings feasibility remains28637.609s <=28800s with the original estimator/exposure. No v4 optimizer progress is yet claimed. [Incident record](../SecondPass/JointTraining/RECOVERY_V4.md). Earlier live descriptions below are historical.

**New authorization / QUEUED, 2026-09-23 07:46 UTC:** another **2,145 updates / 68,640 episodes**, reaching **5,005 / 160,160**, is durably queued after v3 finishes validation2860, both complete final evaluations and normal OS exit. The unchanged recipe adds 165 updates / 5,280 episodes per task. v3 is still the sole MPS worker (PID51433), last verified in `final_test_terminal` at step2860; it was not interrupted. The new eight-hour cap has **not started**. Parent owns queue **`proc_af5aedcba7d9` / PID11016**, transferred with completion delivery; queue expiry is **09:13:50.062308 UTC**. Receipt and exact-state/budget gates fail closed. No v4 optimizer progress is claimed. CPU tests: **49 passed**, compileall passed. See [v4 protocol](../SecondPass/JointTraining/AMENDMENT_V4.md) and [queue handoff](../SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/handoff.json). Earlier running/completed descriptions below are historical snapshots.


[Index](README.md) · [Current status](CURRENT_STATUS.md)

The chronology follows the post-reset development sequence. Run-directory dates show the main sensory work on September 12,2026 and memory/attention development through September 13. A run may begin one local date and finish another UTC date; ordering below is based on ancestry and saved reports, not an invented minute-by-minute history.

## 1. Establish useful sensory encoding before choosing a complete system

The user reset the repository because an inherited architecture and its justification had displaced component-wise research. The new scaffold was Guided Search 6.0, with PAV first and other components later. The first binary dot-displacement design was corrected to cardinal random-dot motion with exactly two ordered frames. Its stopped artifacts are historical diagnostics, not final evidence.

[Five encoders](experiments/01-pav-encoder-screen.md) then received the same seven-task exposure. ConvNeXt learned broad sensory distinctions and motion best in that screen, while other candidates were stronger at contour grouping. The next question was therefore whether useful computations could be combined, not whether one model should be permanently mandated.

## 2. Combination ideas did not substitute for acquisition

The [CPU ensemble audit](experiments/02-saved-ensemble-audit.md) found complementary contour errors but a motion cost from averaging. A [three-arm hybrid comparison](experiments/03-hybrid-continuation.md) tested a Gabor branch and late-SE against equally trained continuation. Neither met the contour target; continuation itself improved markedly.

The decisive [allocation experiment](experiments/04-contour-task-allocation.md) changed only which existing task received more updates. Contour improved 67.9%→96.2%, while all seven point accuracies stayed above 95%. This is the project's first clear demonstration that weak endpoint behavior need not require a new architecture. It also exposed a small contrast cost that remains part of the record.

## 3. Replace simultaneous pair access with causal sensory state

The user asked whether motion should be computed from successive frames and an accumulator. The [KDA/ConvGRU/opponent comparison](experiments/05-causal-temporal-accumulators.md) froze the trained encoder and learned new temporal/readout components. Opponent fast/slow traces with fixed quadrature-energy computations reached 98.44–100% across the two-step battery. Reset and reversed-frame tests supported use of history and order.

The user selected this neuroscience-inspired winner for further work. High two-frame accuracy was never evidence of long memory; that became the next measured question.

## 4. A broad sequence battery failed to acquire several new rules

The [sequence battery](experiments/06-sequence-battery.md) added instructions, cues, integration, delays and retrospective probes while unfreezing learned weights. Many rules stayed weak even at minimal delays. Longer-delay chance behavior therefore did not cleanly identify a retention limit. The correction was to focus acquisition and compare explicit recurrent memory, not to claim the sensory model had a measured universal memory capacity.

## 5. Focused recurrent memories acquire tasks; investigate E/I's motion gap

The [LSTM/EI comparison](experiments/07-lstm-versus-ei.md) learned focused motion-duration and orientation tasks after 40,000 episodes per arm. LSTM did better on longer motion integration. A [matched-order diagnostic](experiments/08-recency-diagnostic.md) showed greater E/I benefit from late winning evidence.

An appealing immediate story would have been excessive leak. The [state-accessibility diagnostic](experiments/09-state-accessibility.md) instead found early count information still available in final firing rates and a linear duration-readout rescue. The project consequently deferred slow synapses and [refit the existing outputs](experiments/10-readout-refit.md), improving standard-task motion 70.31%→79.30% without changing upstream computation.

## 6. Retaining a decision and comparing a delayed visual feature separate

[Retention training](experiments/11-retention-learning.md) made motion accuracy approximately flat through 24 blanks. Its remaining 20% error was already present without delay. Orientation comparison, whose answer requires a new probe, remained near chance at 24 blanks.

The [orientation diagnostic](experiments/12-orientation-accessibility.md) found accurate pre-probe angle information despite the deployed comparison failure. A decoder trained at sample time failed later, while a late-trained decoder succeeded. A label-only diagnostic comparator rescued performance. “Blanks erase the feature” was therefore too strong: accessibility and use had to be separated.

## 7. Spatial memory solves a binding task but introduces a major trade-off

The user's proposed 13 × 13 × 64 memory became a [spatial competitor](experiments/13-spatial-memory.md) against dense memory, both with explicit old-memory/current-input comparison. Spatial binding improved strongly, including new location centers. Native single-item D24 did not improve, and motion deteriorated. Binding and native orientation differ in task construction, so high binding is not evidence of higher two-item capacity.

This was a useful component with a known cost, not an accepted universal replacement. The earlier motion-competent model remains preserved.

## 8. Keep teaching fixed while testing learned maintenance

The user rejected changing the target to an angle-supervised delayed report. The implemented [additive controller](experiments/14-selective-maintenance.md) therefore used the same tasks and losses. It produced only a narrow delayed-orientation gain; interrupting its blank-period current did not remove the benefit.

The alternative [pre-update attention](experiments/15-preupdate-attention.md) lets old memory query both current sensory and old-memory values. It improved single D24 to 79.30% versus 58.40% continuation, but hurt immediate motion. A [phase-specific intervention](experiments/15a-attention-mechanism.md) showed that orientation depends on memory-source routing during blanks. Active refresh versus rejection of blank input is still unresolved.

## 9. Preserve the gain and investigate the regression

The [motion audit](experiments/15b-motion-audit.md) found severe class bias, weak task allocation and a motion deficit predating attention. Calibration partially rescued choices but did not solve delayed performance. The user explicitly requested investigation instead of accepting progress that sacrifices another domain.

[Current Stage1](experiments/16-training-exposure.md) compares unchanged 10% motion continuation against 50% motion at equal total additional updates, starting the same attention 8400 checkpoint. Local control starts first and the additional arm uses a concurrent authorized RunPod. Fresh selection checks all trained cells, preventing an aggregate improvement from hiding task regressions. A proposed temporal residual remains Stage2 design only, contingent on these findings.

## Decisions that remain intentionally unresolved

No universal architecture winner, biological validation, formal multi-item capacity or general attentional selection ability has been established. Slow synapses were deferred after recoverability evidence; new angle teaching was rejected; a residual has not been trained. These are scientific decisions and user constraints, not missing implementations to quietly complete.

## Prospective-query intervention launched (2026-09-15T01:04:25.1474724+00:00)

After the five-task original-bias run and task-only motion continuation remained weak on cued motion, the user authorized one architecture-only test from the intact attention8400 parent. [Experiment 21](experiments/21-prospective-query.md) adds a zero-initialized learned scalar current-sensory residual to pre-update queries while preserving keys, values, biases, E/I memory, tasks, losses, optimizer settings, exposure and evaluation. Exact-zero equivalence and nonzero-gradient checks passed.

Fresh RunPod pod `dqi13o2x3qkvos` uses one A100-SXM4-80GB. Setup exposed CRLF and omitted-fixture packaging defects before training; both were corrected under the same deadline with no training exposure or scientific change. Profiling then passed and production advanced beyond step8400. Frozen local diagnostics found cue-dependent routing/output changes but near-chance task behavior, while a split linear probe of local final-moving-frame sensory features remained near chance. This separates detectable cue effects from successful duration-rule use and supplies no evidence for the probed sensory summary. Final training claims await scheduled results; historical control matching ends at step11600.

## Prospective-query run stopped after its first validation (2026-09-15T02:04:33.9212457+00:00)

Step-9200 validation improved orientation and spatial binding but did not improve
the primary cued motion-duration failure; recognition was mixed and worse at
load 24. The user stopped the run at logged step9430 /41,200 episodes. Durable
checkpoint9216 contains32,640 episodes. All29 indexed artifacts were verified,
the A100 pod was deleted, and account spend returned to zero. This is a
user-stopped negative primary result, not a completed 160,000-episode
comparison or final held-out evaluation.

## Spatial-priority readout launched (2026-09-15T03:31Z)

The user authorized an independent attention8400 branch that changes only the
terminal decision readout. Final sensory, E/I-rate and comparator fields stay
aligned at13×13 through a shared1×1/3×3 convolutional stack; each task learns
spatial selection and local class-evidence maps. Original attention and its
source/locality biases remain intact, and prospective gamma is absent.

[Experiment22](experiments/22-spatial-priority-readout.md) passed exact
inheritance/Adam, shape, normalization and gradient checks. A fresh community
RTX4090 RunPod began the fixed160,000-episode run. Profiling projected the
work within its eight-hour deadline; production was verified beyond step8400
with finite metrics and a live45-second monitor. No validation result existed
at this entry, so no performance conclusion is recorded.

## Spatial-priority lineage corrected and relaunched from scratch (2026-09-15T03:53Z)

The user corrected the experiment before its first validation: the complete
trainable architecture, not merely its new readout, must start from scratch.
The inherited attempt was immediately stopped at 320 updates /12,800 episodes;
its RTX4090 pod was deleted and confirmed absent. It produced no validation
and is not architecture evidence.

The corrected `spatial_priority_readout_scratch_v2` bundle contains no parent
checkpoint and loads zero model or Adam tensors. It freshly constructs the
opponent sensory path, original biased `JointAttention`, spatial E/I memory,
comparator and priority-map readout from documented seeds. Scratch
determinism, component identity, gamma/global-pooling absence, normalized maps
and finite gradients passed locally and on the pinned cloud runtime.

A new community RTX3090 pod `ce00y2ooosl7wc` launched the same five-task,
4,000-update /160,000-episode protocol from step0. The three-update profile
averaged3.4266seconds and projected the complete bounded work at about
5.51hours/$1.21. Training was verified active with finite metrics; no
validation existed at this chronology entry. Historical model comparisons are
not initialization-controlled ablations of this scratch run.

## Independent dual-attention priority arm launched (2026-09-15T04:36Z)

[Experiment 23](experiments/23-dual-attention-priority.md) separates
pre-update integration from post-update spatial selection. One width-64 head
produces raw context and the E/I drive; after the unchanged E/I update, another
width-64 head lets updated rates query `C/H/R`. Only terminal post-attention
field `P_T` enters the spatial-priority decoder.

The complete 549,344-parameter model and Adam optimizer start from scratch.
Local and pinned cloud checks verified the requested shapes, finite
forward/backward, nonzero output-loss gradients through both Q/K/V paths,
normalized selection, no comparator/gamma/global bypass, and exact equality
for 119 unchanged scratch-control tensors. Independent RTX 3090 pod
`f7y0zw02f4fzum` began the fixed 4,000-update /160,000-episode run with finite
metrics. The control and concurrently launched comparator pods remain
separate and untouched. No validation conclusion is available at launch.

## September 13 local evening update: cloud allocation arm completes

The 50% motion arm completed and selected 11600. Fresh-test motion improves over parent, but single D12 falls 2.93 pp. Cloud retrieval and deletion are complete. The local 10% control still trains, so the schedule comparison remains incomplete. The user also authorized queued frozen latent-state visualization after the local GPU becomes free. See the dated [current status](CURRENT_STATUS.md) for receipt links and the distinction between earlier validation and new test scores.

## User correction: inspect attention maps first (2026-09-14T00:45:26.831440+00:00)

The user paused further model experiments and the broad latent/probe investigation, requesting condition-averaged attention maps over immediate and remembered-scene references instead. LatentDynamics supervisor 4644 was stopped before extraction; no compute budget was started. The researcher now prepares frozen forward-only attention capture after the existing local run's training/evaluation exits. Both source banks will be shown within each of the two heads; neither head is source-exclusive. The earlier latent proposal remains documented as paused. See [current status](CURRENT_STATUS.md) and the [pause receipt](../WorkingMemory/LatentDynamics/queue_receipt.json).

## Training and requested visualization complete (2026-09-14T00:56:21.960683+00:00)

Both Stage1 arms completed 32,000 additional episodes. Local control selectedterminal 12400 with motion D0/D24 41.02%/41.99% and single D12/D24 96.88%/91.60%; focused selected 11600 has stronger motion but weaker orientation and a D12 regression below parent. No all-domain winner is established. Full parent/selected/terminal tables and uncertainty are in [the final Stage1 entry](experiments/16-training-exposure.md).

After local training/evaluation exited, the narrower [AttentionMaps capture](experiments/17-attention-maps.md) completed 192 paired base episodes /384 presentations in 30.45 seconds using unchanged attention 8400. The viewer now shows both sources for each head, phase/condition means, receiving-query views and labelled scene references. Source-mass differences do not establish spatial focus or remembered content. Broad latent/probe work remains paused; the cloud is deleted and owned workers are absent. No subsequent model experiment has been launched.

## Presentation mismatch corrected rather than hidden (2026-09-14T01:02:13.120024+00:00)

The user rejected the first promoted single-query figure and the phase/time-averaged overview as answers to the requested full-grid temporal view. The initial capture was complete, but its presentation did not meet that request. Cached matrices confirmed strong locality (about 94% joint mass at each query's coordinate, distance penalty about 4.03); the single bright cell therefore did not demonstrate absent attention. Researchers are preparing individual trials at every timestep, retaining earlier averages as explicitly historical summaries and staying within the same capture deadline. See [entry 17](experiments/17-attention-maps.md).

## Every-timestep correction completed (2026-09-14T01:05:05.388626+00:00)

The corrected [AttentionMaps viewer](../WorkingMemory/AttentionMaps/index.html) now displays individual trials at exact frames, all receiving-grid cells for each head/source, and timeline playback without temporal or trial averaging. Frozen capture added 48 movies /912 frame observations in 6.76 seconds within the original deadline, with no added budget or model change. The earlier phase summaries are preserved as historical averages. [Entry 17](experiments/17-attention-maps.md) records both the initial mismatch and the completed correction rather than treating the first delivery as sufficient.


## Explicit attention priors become the next controlled change (2026-09-14T01:40:21.630790+00:00)

Inspecting the actual maps exposed strong local routing under the learned distance penalty. The user authorized removing explicit source and distance biases while preserving trained attention projections, then both continuing old tasks and teaching a spatial battery. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) separates those questions: the old arm uses the prior 32,000-episode control recipe and reused paired evaluation, while the new five-task arm studies acquisition with fresh heads and task streams. A single L40S pod is provisioned under an eight-hour cap; no production training or performance result is yet verified at this snapshot. The protocol preserves the corrected consecutive-image recognition timeline, signed orientation glyph, retrocued binding and one explicitly adapted Arcizet–Krauzlis recipe. Broader latent analyses remain paused.


## Cloud setup replacement without budget renewal (2026-09-14T01:50:22.617728+00:00)

The first L40S host failed CUDA initialization under two runtimes and one restart; it trained zero updates. Failure evidence was retrieved and verified, then the pod stopped/deleted. A replacement L40 48 GB was created at 01:48:02.15 UTC with the original 09:36:25.001 UTC deadline, unchanged source/model/tasks and no allowance renewal. It is in setup, not yet verified training. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) links both receipts.


## Production launch verified (2026-09-14T01:57:21.565752+00:00)

Both profiles passed on the replacement L40 with Torch 1.13.1+cu117, and the full 32,000 old-task / 160,000 new-task episode allocations were pinned. The launch receipt records old-arm global 8605, +205 updates / 1,640 fresh episodes and finite loss; the five-task arm is queued after it. Supervisor and incremental retrieval watcher continue under the original deadline. Profile durations are estimates, not completed work. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md).


## Old-task bias-removal result is negative (2026-09-14T02:40:33.400826+00:00)

Both old-task continuations completed 32,000 episodes. Joint source/locality-bias removal drives terminal binding D24 from control 99.41% to 49.80% and single D24 from 91.60% to 57.62%, while small motion changes have paired intervals including zero. All five validation looks failed; automatic selection falls back to the still-biased parent 8400. This does not erase the negative trained-terminal result. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) preserves all 14 paired cells, uncertainty and the warm-start/reused-test limits. The separately authorized five-task arm continues unchanged toward 160,000 episodes under the original deadline; no additional experiment is implied by the old-arm failure.


## User cancels bias-removed acquisition, then restores the original model (2026-09-14T02:44:04.427099+00:00)

The five-task worker/supervisor stopped at logged global 8550 (+150 updates / 6,000 episodes), with durable checkpoint 8448 and GPU processes empty. The user subsequently asked to restore original biases and launch again. Researchers prepare a separate acquisition run from intact attention 8400, keeping tasks, fresh heads/streams, optimization and the original deadline. No damaged or partial model is the new parent. [Entry 18](experiments/18-unbiased-attention-spatial-battery.md) preserves the cancellation; [entry 19](experiments/19-biased-spatial-battery.md) records the new authorization without claiming launch.


## Original-bias acquisition launch verified (2026-09-14T02:48:44.540830+00:00)

The separate BiasedTraining launch receipt verifies global 8448 (+48 updates / 1,920 episodes), finite loss 0.85955 and full 4,000-update / 160,000-episode exposure. It retains intact attention 8400 source/locality terms and 123 compatible Adam states, with fresh heads identical to the cancelled initialization. The healthy L40 and original deadline are unchanged. [Entry 19](experiments/19-biased-spatial-battery.md) links launch evidence; the bias-removed acquisition stays cancelled and no old-task rerun occurs.


## Uneven joint acquisition motivates one local task-focused branch (2026-09-14T04:42:55.701953+00:00)

The original-bias joint model's validation checkpoint 10000 solves binding but remains near chance on cued motion duration; the cloud worker continues beyond this evaluated checkpoint. The user authorizes a local motion-only continuation from 10000, preserving model/cues/optimizer policy and changing the gradient-producing task allocation. Target 32,000 episodes under a new 7,200-second local cap is pending profiling. This asks whether focused training can acquire the task; it does not predeclare interference. [Entry 20](experiments/20-single-task-motion.md) records the current authorization and provisional validation evidence.


## Motion-only local launch verified (2026-09-14T04:48:12.203271+00:00)

Profiling reduced the proposed 32,000-episode local target before production to 17,600 episodes / 2,200 updates within the unchanged 7,200-second cap. The launch receipt records global 10065, +520 motion episodes, finite loss and restored model/133 Adam states/RNG/family stream/delay scheduler. Full motion CE uses inherited learning rates without compensation; this is focused acquisition, not isolated interference measurement. Cloud joint training remains untouched. [Entry 20](experiments/20-single-task-motion.md) records deadline, evaluation splits and launch evidence.


## Task-focused duration continuation does not acquire the task (2026-09-14T05:45:20.729828+00:00)

The local motion-only run completed all2,200updates/17,600episodes. Selected11760 and terminal12200 remain near25% held-out accuracy acrossD0/4/12/24; all paired gains includezero and AUC is near.5. Finite losses settle nearlog4 with restricted class choices. Failure already atD0 rules out a solely inserted-blank explanation, but does not identify a bug, interference or a capacity limit. All97 artifacts were verified, localworkers exited and the cloud joint run was left unchanged. [Entry20](experiments/20-single-task-motion.md) preserves the negative result and exposure/split distinctions.


## User-requested shutdown completed (2026-09-14 05:52:23 UTC)

The partial original-bias cloud run stopped at logged11923/140,920episodes, with durable11776/135,040episodes; no final held-out evaluation was run. All73 artifacts were retrieved and verified, then root confirmed stopEXITED, deleteHTTP204 and an empty pod list. The completion monitor is paused and no project model workers remain. Last completed validation11600 is preserved as validation, while selected-so-far10000 and durable11776 remain distinct. The local motion-only experiment had already completed. [Entry19 closure](experiments/19-biased-spatial-battery.md) and [cleanup receipt](../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json). No automatic restart or additional experiment is authorized.

## Frozen terminal fields do not yield a spatial-probe motion rescue (2026-09-15T03:58Z)

While the independent scratch cloud run continued untouched, a bounded local
[post-hoc diagnostic](../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
froze motion-only step12200 and trained pooled and spatial-priority probes on
the exact same cached final sensory/rate/comparator tensors. The probes were
capacity matched within70 parameters and used grouped, independently split
base movies across D0/4/12/24.

Held-out BA was23.83% pooled and24.32% spatial, paired difference+0.49pp
(95% CI -1.76 to+2.64), with AUCs0.4805 and0.4832. No delay-level advantage
was reliable. The priority map localized the target region above uniform
overall but did not decode the direction winner. This is evidence against a
simple frozen-final-field readout rescue under the tested probes, not evidence
that relevant information is absent at other times or that end-to-end spatial
readout training must fail.

## 2026-09-15T22:01-07:00 — AV-context v2 local launch

Pre-flight on the local scratch motion-only v1 step-3200 checkpoint measured
1.8e-17 attention mass beyond 3 cells per head and chance-level linear
decoding of the cued direction from H_T, R_T, C_T and H_9. The combined
five-change arm (A opponent channels, B mixed pooling, C wide neutral head 1,
D non-zero selection init, E single learning rate) passed the bit-identity
gate against v1 and started training locally; 6.51 s/update; stopping rule
at validation 2,400. See [experiment 24](experiments/24-av-context-v2.md).

## 2026-09-15T22:35-07:00 — user-requested shutdown of both cloud pods

At the user's instruction both RunPod pods were stopped and deleted: the v1
AV-context comparator resume pod `6mopzgoioemdp2` (logged step ≈6,400, mid
validation 6400) and the dual-attention resume pod `629g1utqkk8non` (logged
step ≈5,464). Stop returned 200, delete 204, subsequent lookup 404, and the
account pod list is empty. **The pre-deletion retrieval failed on both pods**
(the remote inventory probe exited non-zero after the remote workers were
killed; cause not reproducible because the pods are gone), and the shutdown
script deleted anyway. Lost: all remote checkpoints and the full
`metrics.csv` of both arms. Kept locally from the 45-second watchers:
validation summaries, predictions and priority maps through 5600 (AV-context)
and 4800 (dual), the last 256 metric rows of each, and the last aggregate
snapshots. Receipts:
[AV-context](../WorkingMemory/AttentionContextComparator/runs/resume_20260915_183128/user_stop_receipt.json),
[dual](../WorkingMemory/SpatialPriorityReadout/DualAttention/runs/dual_resume_20260915_183345/user_stop_receipt.json).
`WorkingMemory/cloud_shutdown.py` now refuses to delete a pod whose retrieval
failed unless `--force` is passed.

## 2026-09-16T08:15-07:00 — v2 overnight pod stopped as a negative result

Pod `vqpgk21cpi53b6` trained the five-change v2 model unattended overnight to
step 10,621 (424,840 episodes) with all tasks at chance and decaying
gradients; the user called it a failure and had it stopped. Retrieval of all
49 checkpoints and logs was verified before deletion (stop 200, delete 204,
lookup 404). Overnight, all local processes had died at ≈22:40; the local v2
run was resumed in place from checkpoint 256. See
[experiment 24](experiments/24-av-context-v2.md).

## 2026-09-16T21:45-07:00 — battery audit completed, no training

Ideal observers, stream checks and one-step gradient diagnostics
([experiment 25](experiments/25-battery-audit.md)). All five tasks are
recoverable from pixels at D0 (BA 0.95-1.00); streams are correct; the v2
recipe clips every update to 0.36 of its size from the first step and its
trained checkpoint is input-invariant. The handoff's Krauzlis suspicion
(integer rasterisation) was wrong: `_krauzlis` renders sub-pixel dots
bilinearly. Decision: proceed to the plain baseline (ladder rung 1) on the
laptop.

## 2026-09-17T02:20-07:00 — plain baseline rung 1 and 2, orientation family

Experiment 26 completed on the laptop. The specified recipe (Adam 1e-3,
batch 64) collapses to a constant output within 150 updates on all five tasks,
by the same first-step mechanism as the v2 lineage. Ten recipes leave the real
orientation task at chance in 100k episodes, and a centred stack-3 lr 1e-4
run stays at chance through 544k. A within-family ladder shows every two-way
rung is learned to BA 1.000 from scratch, and the real task is learned to
0.998-1.000 within 12.8k episodes from either two-way parent. The same model
continued on mixed delays learns no delay (D4/12/24 at chance, D0 retained).
Decision: a delay ladder next, then the same ladder for the other families.

## 2026-09-17T12:10-07:00 — accumulator-in-the-conv-stack program completed on RunPod

Experiment 27: four arms (plain, convgru, opponent, kda) x two seeds, each
through ring gate, D0 curriculum and a three-stage delay ladder, on one 3090
pod (`hy2m2tjf1awuhf`). ConvGRU and KDA at ceiling at every delay on both
seeds; plain 0.85-0.88; opponent split at D24. The pod idled about 8 h after
completion because the local watcher died with the session; finalised by hand
(stop 200, delete 204, lookup 404). Checkpoints were not retrieved.

## 2026-09-22 — unified suite prepared for later fresh-weight training

The user requested all tasks described in the current-model discussion as one
suite, with training later from fresh weights. [Task-suite assembly](task-suite-assembly.md)
combines the seven sensory tasks, ring-cued orientation, and five spatial tasks
without changing their rendered laws. The catalog has 13 tasks / 35 primary
cells, separate task/condition streams, and explicit single-class recognition
and unequal Krauzlis-event reporting rules. CPU checks passed all cells and
fresh-KDA forward compatibility; eight contract tests passed after adding the
review-identified photo-manifest resume check. No training,
cloud run, checkpoint loading or weight updates occurred. The later joint
training allocation, reporting implementation and compute budget remain to be
specified; the old per-family curriculum is not silently inherited.

## 2026-09-22 — fresh local joint training launched

The user authorized local fresh-weight training across all 13 tasks / 35
conditions. [Joint-training record](joint-suite-training.md) documents one
Apple MPS worker, all learned parameters trainable, one Adam LR and no clipping.
Parent verified production checkpoint 13 by hash and CPU load, checkpoint-zero
empty Adam, and advancing persisted progress (21 updates / 672 episodes at
06:59:57 UTC). The pinned horizon is 156 updates / 4,992 episodes (384/task),
with four planned validation looks and separate final evaluation inside the
original four-hour cap ending 10:47:56 UTC. This small pilot does not establish
sufficient acquisition exposure. No new held-out performance is yet available;
earlier trained checkpoints remain untouched.

## 2026-09-22 — evaluation-heavy allocation corrected within the same cap

The user asked why acquisition was so small and explicitly chose **“Keep the
existing four-hour limit.”** The 156-update plan was an allocation error, not
an evidence-based sufficient horizon: evaluation precision/reserves displaced
optimizer exposure. The [v2 amendment](../SecondPass/JointTraining/AMENDMENT_V2.md)
uses live per-task/per-cell timing to pin **715 cumulative updates / 22,880
episodes / 1,760 per task** without extending epoch **1790074076.5856378**
(10:47:56.585638 UTC) or changing the original origin.

The original worker gracefully finished its boundary at step **117 / 3,744
episodes**, including validation 117. Its terminal checkpoint was suspended,
CPU-loaded and verified before exact-worker termination deferred automatic
finalization. Original worker `-9` / supervisor exit 247 was an intentional
handover, not a scientific failure. The separate continuation preserves model,
Adam, scheduler/condition queues, streams, all RNG, optimizer time, exposure
and completed validation history. Original hashed sources remain unchanged.

An initial pre-optimizer launch falsely identified macOS's `caffeinate` helper
as a second worker because its argv echoed the child command. Receipts were
archived; executable-aware matching fixed the root cause while retaining the
single-worker lock. **23 tests passed**, including exact next-update replay,
budget refusal, CPU continuation, 35-cell final coverage and the guard case.
Successful supervisor **78932** / worker **78937** is parent-owned through
**`proc_e7bdb8db0d3f`** with completion notification. A later handoff readback
found **141 persisted updates / 4,512 episodes**, independently SHA-loaded
checkpoint **130**, all **66** model tensors changed and all **66** Adam
parameter steps advanced since migration; original PIDs were absent and only
one actual joint-training Python worker was present.

Completed validation looks 39/78/117 remain in selection; one new full-cell
look is planned at 715. Final selected and terminal checks retain every cell,
ordinary precision reduced 256→128 and Krauzlis held at 200. N0 specificity and
event/side denominators remain explicit. Potentially forwarded old test draws
are disclosed; no final-test metrics were used for planning/selection and a
fresh test-only namespace is pinned. Cumulative training curves retain the
original 117 updates. Final scores are pending: this is more substantive
acquisition, not a convergence or overnight-equivalence claim.

## 2026-09-22 — joint training completed; selective sensory acquisition

The corrected run completed the pinned **715 updates / 22,880 episodes** and
both 35-cell final tests, ending normally at **10:04:27 UTC**. Total elapsed
time was **3.275 hours**, optimizer time **1.968 hours**, within the original
four-hour cap. Parent hash-verified and CPU-loaded terminal 715. There was no
hard-cap kill, reported run failure, or automatic further training.

[Completed results](../SecondPass/JointTraining/RESULTS.md) show terminal BA:
contrast **100%**, chromatic increment **99.22%**, natural spectral detail
**92.97%**, spatial frequency **63.28%**. Motion and spatial/sequence tasks
remain near chance. The official validation-selected checkpoint is **117**:
the minimum-task-first criterion favored its least-bad task floor despite
terminal mean validation AUC rising to **0.6296** versus **0.5203**. Both
checkpoints are reported, not relabeled after inspecting tests.

Krauzlis responses were always positive for both checkpoints (100% target
hits and 100% foil/catch false positives). Empty recognition specificity was
100%, separate from chance nonempty recognition. This is not full-suite
acquisition or proof of architectural limits. Preserve terminal progress;
prospectively revise the selection objective before any newly authorized
longer training. No new task, curriculum, architecture or run is launched.

## 2026-09-23 — explicit eight-hour terminal715 continuation

**New authorization / running, 2026-09-23:** the user explicitly requested “go ahead and set up a 8 hour training run for more trsining and more updates”. The versioned v3 continuation resumes **terminal 715**, not historical selected 117. It pins **2,145 additional updates / 68,640 additional episodes** (5,280/task), reaching **2,860 total updates / 91,520 episodes** (7,040/task). One local MPS worker; no architecture, loss, stimuli, sampling, optimizer or batch changes.

The new 28,800-second allowance starts **01:08:50.062308 UTC** and ends **09:08:50.062308 UTC on September 23** (epoch **1790154530.062308**), including migration and finalization. No renewal. At **01:12:32 UTC**, optimizer progress was independently read through **733 / 23,456 episodes**; checkpoint **728** was SHA-verified and CPU-loaded. Migration preserved model, all Adam state, sampler/condition queues, native streams and Python/NumPy/CPU/MPS RNG exactly. First production update 716 is retained, not discarded.

Validation-only selection is prospectively **equal-task mean AUC**, then mean chance-normalized task BA, earlier ties. Baseline validation715 is reused; historical winner117 remains historical. Four new all-35-cell looks: **1248, 1781, 2314, 2860** (64 ordinary / 100 Krauzlis). Final selected and terminal tests use 128 ordinary / 200 Krauzlis in fresh final-only namespace **94392763**, deduplicating only identical models. Prior tests were seen; new draws reuse official test source identities, so this remains exploratory—not a wholly untouched population or convergence claim.

Parent owns CPU guardian **`proc_5cbdb07600dd` / PID51946**, attached to live supervisor **51429** and the sole MPS worker **51433**. The first tracking handle falsely reported exit after stdout/stderr redirection; OS identities and real updates disproved a worker stop. The guardian fixes completion delivery without restarting training or moving its deadline. Automatic `report.json`, `REPORT.md`, both final-cell JSONs and cumulative validation curves will be written under `SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v3_8h/`. Earlier completed715 results and source artifacts are retained; all older authorization/live statements below are historical.

## 2026-09-23 — architecture and microstimulation technical report

The user requested approximately ten pages describing the current architecture, proposed microstimulation, comparability to Morgan, Albanna & Herman, expected effects and preliminary metrics. The [report and audit record](joint-kda-architecture-microstimulation.md) document 2,750,324 trainable parameters, three local KDA fields plus global GRU, exact update equations and a paired calibrated perturbation design. Functional causal questions are comparable; spatial allocation clamps, signed KDA perturbations and biological electrical stimulation are not the same manipulated variable.

The frozen **06:48:15 UTC** snapshot includes six complete 35-cell evaluations. Training had reached2647, while latest measured validation2314 had mean AUC0.69374; its spatial group remained near chance. Strong sensory scores, all-positive orientation/Krauzlis decisions and empty-recognition specificity are reported separately. V3 final tests were pending at this cutoff. No new forward evaluation, model changes, training, cloud use or stimulation experiment was performed for the document. Existing run scope and deadline were preserved.

**2026-10-03 19:34 UTC — VAE COMPLETED9900updates; only best9500/latest9900 retained.**
300,000triplet presentations /30,000unique movies, same original allocation/cap. Fresh paired final tests complete (384independentmovies /1152triplets per model). Last100total loss0.00063257/recon0.00060685. Selected held-out balancedrecon0.00064548 versuscopy-middle0.00342465; temporal-differenceMSE0.00005794 versus0.00008455 (31.5%lower). Full-imageMSE0.00005657 approximatelycopy-middle0.00005637. No response model/accuracy tested. ParentCPUverified both full checkpoints and126Adam steps; guard exited normally. [Final report](../SecondPass/ThreeFrameConvVAE/FINAL_REPORT.md) | [Completion proof](../SecondPass/ThreeFrameConvVAE/LocalRuntime/completion_verified.json). No continuation or new run launched.


## Actual production launch

**October3,20:29UTC — VAERViT TRAINING locally; production optimizer independently verified.**
Fresh classification learner initialized only62encoding tensors from latestVAE9900,32new recurrence/readout tensors; all94trainable. ParentCPUverified saved update3 /96trials, all94Adam states advanced and all94parameter tensors changed, freshemptyinitialoptimizer/RNG/streams/zero classification counters. Nativeprofile pins990updates /30,000trial presentations /3,000unique trials, threecomplete1000movie×10epoch pools; requested2310 reduced BEFORE production from measured costs. Validation100/250/500/990 at100/cell; final200/cell, same originalchange/no-change labels57/43 andnative29/37/45frame movies. MPS, FP32/fullsequenceBPTT, batch32/micro1, Adam1e-4/no clipping, twoCPUthreads. Onlybest.pt/latest.pt saves, no profileweight files. Firstthree losses are preliminary (~0.70); no validation yet.

New8hlocalcap October3,1:26:13PM–9:26:13PM PDT (sciencecutoff9:16:13PM), no extension. Productionworker58109/supervisor57312/guard57310; profile process exited andallstate discarded. Runtime`/Users/jonathanmorgan/VAWMRuntime/vae_rvit_local01/run`. BothVAEbest9500/latest9900 remain intact; allcloudclosed. [Production evidence](../SecondPass/VAERViT/LocalRuntime/production_verified.json) | [Architecture](../SecondPass/VAERViT/README.md).

Measured steady batch32 fullBPTT costs bycondition: {"B20": 19.082532041938975, "B28": 23.51351070799865, "B12": 17.799330500070937}. Prospectiveallocation andbudget remain in runtimeconfig. This confirms initialization/optimization, not responseacquisition; heldout results pending.

**October3 — Weighted-mean CNN–RViT QUEUED locally.**
CausalrawRGB .5Xt+.4Xt-1+.1Xt-2 (repeatfirst startup), fresh residualCNN256×13×13 ->169×256tokens ->original shared8headRViT ->256→16tokenreduction/flatten2704FFN2. All7,269,426parameters /92tensors fresh/trainable; no VAE or other checkpoint input. Same no-cue/single-stimulus29/37/45frame task and1000movie×10epochreplay; FP32/fullBPTT/Adam1e-4/no clipping/batch32micro1/CPU2. Newfinite8hLOCALcap starts ONLY after queueactivation, target2310updates pinnedtofull330pools fromfuture nativeprofile; no renewal/cloud. Onlybest.pt/latest.pt.

Sourceonly35file snapshot prepared at`/Users/jonathanmorgan/VAWMRuntime/weighted_mean_rvit_local01/repo`; no newbudget/profile/training. Durableone-shotqueue66646 at`VAWMRuntime/weighted_mean_rvit_queue01`, waitingfor currentVAERViT supervisor/worker exit afterevaluation andglobalGPUlockfree. CPUshortmodel/gradient/filtercheck passed; parentconfirmedqueuealive/waiting/no newbudget andcurrenttrainingcontinuing(update66/2000presentations). QueuekeepsMacawake, neverstopsotherjobs,expires5minafterpriorharddeadline, no automaticretry. [Queue proof](../SecondPass/WeightedMeanRViT/LocalRuntime/queue_verified.json) | [Model](../SecondPass/WeightedMeanRViT/README.md).


## Actual A40 production launch

**October3 — Weighted-mean CNN–RViT TRAINING on RunPod A40; persisted optimizer verified.**
Pod`7q3pqnv51lil36`, .49USD/hour. New8h/$5cap began20:58:54UTC/1:58:54PM PDT, hardstop09:58:54 PM PDT, sciencecutoff10min earlier. TargetFULL2310updates /70,000trial presentations /7,000unique movies prospectivelypinned beforefreshproduction; profile native batch32micro4 B12/B20/B28 3.80/4.55/5.57seconds, peakCUDA452MB. No reducedexposure. Wholefresh7,269,426params/92tensors, alltrainable; exactweightedRGB.5/.4/.1, currentno-cue/single-stimulus task, 1000movie×10epochreplay, FP32mathSDPA/TF32off/fullBPTT/Adam1e-4/noclip.

Savedproduction3/96 locallydownloaded/digest+CPU92Adamstates verified; nativecheckpoint verifier confirmsall92parameter tensors changed from directfreshconstructor, emptyinitialAdam/streams. Profile separate/discarded, no checkpointfiles. Runtime`/workspace/vawm_weighted_mean_rvit_01`, worker430/owner265, independentauthenticatedbillingguard45. Privateproviderstatus, launchdmirror, automaticretrieval/CPUverificationthenSTOP/DELETE armed; Macawakehelper endsaftercleanup. Onlybest.pt/latest.pt retained (localmirror overwriteslatest, no numberedcopies). [Production evidence](../SecondPass/WeightedMeanRViT/CloudRuntime/production_verified.json) | [Downloaded state proof](../SecondPass/WeightedMeanRViT/CloudRuntime/downloaded_checkpoint_verified.json) | [Pinned allocation](../SecondPass/WeightedMeanRViT/CloudRuntime/native_profile_allocation.json).

Localqueue66646cancelled/no localweightedbudget/profile; originalVAERViT Macrun continues (observedupdate123). No extra cloudarm or cap renewal. Currentcloudresults preliminary/no validation yet; no acquisitionclaim.

Platform comparison: CUDA/A40micro4 versuspriorAppleMPSmicro1, same effective32/fulltemporalgradients/FP32/Adam. This newweightedmodel startsfresh, whilecurrentlocalVAERViT haspretrainedencoder; hardware/initialization/exposure differ, so resultsareseparatepilot evidence ratherthancontrolledcausalpretrainingcomparison.
