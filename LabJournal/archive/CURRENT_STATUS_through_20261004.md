> Historical snapshot through October 4, 2026. Earlier present-tense launch statements are not current status. Relative links have been adjusted for this archive location.

**October4 — angular contrastive CNN completed: simplified comparison solved.** FreshCNN25024updates/782000presentations/391000unique;3072fresh continuous-change tests,100%BA/AUC1.0 allthree speeds ×26/28°; explicit angular supervision. Best19000/latest25024 CPUverified64Adam states. No active local/cloud training. Original nativeKrauzlis task remains untested for this model. [Journal](../angular-contrastive-motion.md).

**October4 — angular contrastive CNN now TRAINING locally.** Fresh wholemodel/MPS, all64Adam states verified at saved1. Target25,024updates, original new8h launch cap; no other model interrupted, predecessorFFN completed. [Journal](../angular-contrastive-motion.md).

**October4 — fresh angular contrastive CNN QUEUED locally.** Threeframes →CNN →128Dunit representation; springloss with targetdistance proportional to circular direction change. Allweights fresh, CPUcheck passed; no training/budget started. [Journal](../angular-contrastive-motion.md).

**October4 — frozen predictive encoder + FFN completed at chance.** All10,240updates/640,000presentations/320,000unique, 23minutes. Fresh3072-trial test BA50%, AUC0.49519, CE0.693165; predicted no-change throughout. Best5632/latest10240 verified, encoder remained frozen; no active training. [Journal](../predictive-motion-change.md).

**October4 — predictive encoder + simple FFN running locally.** Frozenencoder50,000, concat512+512 →256→64→2, balanced26/28° same/change continuous-dot clips. Target10,240 headupdates, batch64, 2epochs/1000freshtrials; no cloud. [Journal](../predictive-motion-change.md).

## Latest result: predictive model completed

VariationalMotionPredictor completed50,240 updates/1.57M presentations/785,000 unique samples in7h44m. Fresh768-sequence test balanced MSE0.00029275, 92.4% below copy-last, better at all three speeds. Best50,000/latest50,240 independently checked; no active local or cloud training. Downstream change detection remains untested. See [journal](../variational-motion-prediction.md).

## Local predictive training running

VariationalMotionPredictor resumed from256; target50,240 cumulative updates or8h. Saved Adam progress verified beyond256. See [run journal](../variational-motion-prediction.md). Cloud remains off.

# Current state of the project

**October3 — Variational motion predictor built and initiallocalpilotcompleted.** ThreeorderedRGBframes→residualCNN→one512DGaussianvector→fourthframeCNNdecoder; all15,463,363parametersfresh/trainable. Newconstant-velocityfullfieldpersistentdots, uniformdirectionacrosssamples andspeeds.375/1/2px, nochanges/cues/labels. Support-balancedprediction+KLwarmup,2epochs/1000samples, batch32micro4 FP32Adam1e-4/noclip. Prospective20mincap pinned256updates/8000presentations/4000unique; completedearly7:37PM, best/latest256CPUverified/138Adamstates. FinalmeanpredictionMSE.004809 versuscopylast.003839/gray.005469; imageslargelyflat/blurry, motionlearningnotdemonstrated. Cloudweightedepoch2finished2310/70000/35000unique, selectedfinalBA50%/AUC.503969; allartifactsretrievedandpoddeleted. NoactiveGPU/cloudrun orautonomouscontinuation. [Predictivejournal](../variational-motion-prediction.md).

**October3,7:09PM PDT snapshot:** localinput×10 completed330/10000, freshfinalBA50%/meanAUC.489996 (all-change), best/latest330 verified/guardexited. Cloud epoch2weightedCNNRViT2099/2310 /63640presentations, last100loss.683817; validation2000BA50%/meanAUC.516252/all-change. Cloudcontinuesunderoriginalcap9:58PM, localstaysfinished.

**October3 — Weighted-mean CNN–RViT epoch2 TRAINING on the existing A40.** Saved3/96 CPUverified/all92Adam states and tensors changed; live5. Fresh wholemodel/Adam/RNG/streams, solechange10→2epochs per1000trials. Pinned2310updates/70000presentations/35000unique trials (35pools), FP32/fullBPTT32micro4/Adam1e-4/noclip. Original8h/$5deadline unchanged (hard9:58PM PDT); guard/mirror/retrieval/delete armed. Oldcloud cancelled1304/39544, best/latest retrieved; local×10 continues. [Journal](../weighted-mean-rvit-epoch2.md). No validation yet.

**October3 — Input×10 VAE–RViT TRAINING locally: saved optimizer verified.**
**Production training independently verified:** saved update1 /32 trial presentations; all94 Adam states advanced and all94 learned parameter tensors changed. CPU checkpoint reload verified, input scale10 in checkpoint provenance. Initial production loss0.69538. This is training progress, not evidence of held-out acquisition.
Pinned330updates/10000presentations/1000unique trials; same1000×10schedule; hard10:54PM PDT. Previous local model stopped227/6928; cloud weightedCNN–RViT continues.

Native profile completed; production worker9023/supervisor5300. Before production, measured throughput pinned **330 updates /10,000 trial presentations /1,000 unique movies** (one full 1000-movie×10-epoch block), reduced from requested990 to fit the new8h cap. Validation100/250/330, final200trials per condition. HardstopOctober3,10:54:06PM PDT, science cutoff10:44:06PM. Same original seeds for matched baseline inputs. Production initialization active; saved optimizer proof pending.

## October 3 — Local VAE–RViT input ×10 experiment

User requested stopping the local classifier and repeating training with inputs multiplied by ten. Original local run stopped cleanly at 227 updates / 6,928 trial presentations; best100 and latest227 remain preserved. Cloud weighted-mean CNN–RViT continues independently.

`SecondPass/VAERViTInput10` scales the 169×256 encoded tokens by exactly10 **after** spatial positions and token LayerNorm, immediately before the original recurrent block. Internal query/memory/FFN normalization remains unchanged. Consequently, the current-token residual is scaled while normalized attention queries are mostly scale invariant; this is an input-amplitude experiment, not an attention-temperature change.

Same VAE9900 encoder+mu transfer; fresh recurrent/decoder/positions, Adam, RNG and zero classification counters. All94 learned tensors train. Same seeds and train/validation/test namespaces as the original classifier for matched examples. Same no-cue/single-stimulus movies, final-trial crossentropy, full-sequence gradients, FP32/MPS, Adam1e-4/no clipping, batch32/micro1, twoCPUthreads, 1000 movies reused for10 shuffled epochs. Requested990 updates /30,000 presentations /3,000 unique movies, prospectively limited to feasible complete330-update pools by native profiling. A new finite8h local cap covers profile/training/evaluation/reporting; only best/latest checkpoints.

Focused CPU check passed: exact10×token equality, all94 finite parameter gradients and a nonzero earliest-frame gradient. New immutable runtime `VAWMRuntime/vae_rvit_input10_local01`; launcher/independent guard active. Native profiling is underway; no persisted production progress yet.

**October3 — Weighted-mean CNN–RViT TRAINING on RunPod A40; persisted optimizer verified.**
Pod`7q3pqnv51lil36`, .49USD/hour. New8h/$5cap began20:58:54UTC/1:58:54PM PDT, hardstop09:58:54 PM PDT, sciencecutoff10min earlier. TargetFULL2310updates /70,000trial presentations /7,000unique movies prospectivelypinned beforefreshproduction; profile native batch32micro4 B12/B20/B28 3.80/4.55/5.57seconds, peakCUDA452MB. No reducedexposure. Wholefresh7,269,426params/92tensors, alltrainable; exactweightedRGB.5/.4/.1, currentno-cue/single-stimulus task, 1000movie×10epochreplay, FP32mathSDPA/TF32off/fullBPTT/Adam1e-4/noclip.

Savedproduction3/96 locallydownloaded/digest+CPU92Adamstates verified; nativecheckpoint verifier confirmsall92parameter tensors changed from directfreshconstructor, emptyinitialAdam/streams. Profile separate/discarded, no checkpointfiles. Runtime`/workspace/vawm_weighted_mean_rvit_01`, worker430/owner265, independentauthenticatedbillingguard45. Privateproviderstatus, launchdmirror, automaticretrieval/CPUverificationthenSTOP/DELETE armed; Macawakehelper endsaftercleanup. Onlybest.pt/latest.pt retained (localmirror overwriteslatest, no numberedcopies). [Production evidence](../../SecondPass/WeightedMeanRViT/CloudRuntime/production_verified.json) | [Downloaded state proof](../../SecondPass/WeightedMeanRViT/CloudRuntime/downloaded_checkpoint_verified.json) | [Pinned allocation](../../SecondPass/WeightedMeanRViT/CloudRuntime/native_profile_allocation.json).

Localqueue66646cancelled/no localweightedbudget/profile; originalVAERViT Macrun continues (observedupdate123). No extra cloudarm or cap renewal. Currentcloudresults preliminary/no validation yet; no acquisitionclaim.

**October3 — Weighted-mean RViT moved to CLOUD; local queue cancelled.**
UserexplicitRunPodrequest supersedeslocalqueue. Queue66646stopped, no localbudget started; currentVAE–RViT stayslocal. CUDAadapter andguard/mirror beingprepared; source/model CPUproof complete. NewoneA40/8h/$5cap starts actualpodcreation, target2310prospectivelypinned nativeprofile; allfreshweights/Adam/RNG/streams, FP32/fullBPTT32micro4/Adam1e-4/noclip, same task/schedule. Onlybest/latest, retrieve+verifythenstop/delete; allothercloudjobsclosed. No paidcreation or CUDAproduction yet. LiveproviderquoteA40HIGH/.49hr, providerpodlistempty. [Queue cancellation](../../SecondPass/WeightedMeanRViT/LocalRuntime/queue_cancelled.json).

**October3 — Weighted-mean CNN–RViT QUEUED locally.**
CausalrawRGB .5Xt+.4Xt-1+.1Xt-2 (repeatfirst startup), fresh residualCNN256×13×13 ->169×256tokens ->original shared8headRViT ->256→16tokenreduction/flatten2704FFN2. All7,269,426parameters /92tensors fresh/trainable; no VAE or other checkpoint input. Same no-cue/single-stimulus29/37/45frame task and1000movie×10epochreplay; FP32/fullBPTT/Adam1e-4/no clipping/batch32micro1/CPU2. Newfinite8hLOCALcap starts ONLY after queueactivation, target2310updates pinnedtofull330pools fromfuture nativeprofile; no renewal/cloud. Onlybest.pt/latest.pt.

Sourceonly35file snapshot prepared at`/Users/jonathanmorgan/VAWMRuntime/weighted_mean_rvit_local01/repo`; no newbudget/profile/training. Durableone-shotqueue66646 at`VAWMRuntime/weighted_mean_rvit_queue01`, waitingfor currentVAERViT supervisor/worker exit afterevaluation andglobalGPUlockfree. CPUshortmodel/gradient/filtercheck passed; parentconfirmedqueuealive/waiting/no newbudget andcurrenttrainingcontinuing(update66/2000presentations). QueuekeepsMacawake, neverstopsotherjobs,expires5minafterpriorharddeadline, no automaticretry. [Queue proof](../../SecondPass/WeightedMeanRViT/LocalRuntime/queue_verified.json) | [Model](../../SecondPass/WeightedMeanRViT/README.md).

**October3,20:29UTC — VAERViT TRAINING locally; production optimizer independently verified.**
Fresh classification learner initialized only62encoding tensors from latestVAE9900,32new recurrence/readout tensors; all94trainable. ParentCPUverified saved update3 /96trials, all94Adam states advanced and all94parameter tensors changed, freshemptyinitialoptimizer/RNG/streams/zero classification counters. Nativeprofile pins990updates /30,000trial presentations /3,000unique trials, threecomplete1000movie×10epoch pools; requested2310 reduced BEFORE production from measured costs. Validation100/250/500/990 at100/cell; final200/cell, same originalchange/no-change labels57/43 andnative29/37/45frame movies. MPS, FP32/fullsequenceBPTT, batch32/micro1, Adam1e-4/no clipping, twoCPUthreads. Onlybest.pt/latest.pt saves, no profileweight files. Firstthree losses are preliminary (~0.70); no validation yet.

New8hlocalcap October3,1:26:13PM–9:26:13PM PDT (sciencecutoff9:16:13PM), no extension. Productionworker58109/supervisor57312/guard57310; profile process exited andallstate discarded. Runtime`/Users/jonathanmorgan/VAWMRuntime/vae_rvit_local01/run`. BothVAEbest9500/latest9900 remain intact; allcloudclosed. [Production evidence](../../SecondPass/VAERViT/LocalRuntime/production_verified.json) | [Architecture](../../SecondPass/VAERViT/README.md).

**October 3 — VAE-encoder RViT local profiling ACTIVE; production pending.**
Newrun `/Users/jonathanmorgan/VAWMRuntime/vae_rvit_local01/run`; MPSnativeprofile worker57333/supervisor57312, independentguard57310 armed. New8hcap began1791059173.039737, hardstopOctober 03, 09:26:13 PM PDT; no extension. First fullbatch32update completed;3warm+3steady profile fixes feasible whole330updatepool exposure prospectively. Allprofileweights discarded before separate production; no profilecheckpoints saved. Classification learner transfers ONLY62encoding tensors from latestVAE9900, fresh32core/readout tensors, all94trainable; no optimizer/RNG/streams inherited. No production optimizer progress claimed yet. [Journal](../vae-encoder-rvit.md).

**2026-10-03 19:34 UTC — VAE COMPLETED9900updates; only best9500/latest9900 retained.**
300,000triplet presentations /30,000unique movies, same original allocation/cap. Fresh paired final tests complete (384independentmovies /1152triplets per model). Last100total loss0.00063257/recon0.00060685. Selected held-out balancedrecon0.00064548 versuscopy-middle0.00342465; temporal-differenceMSE0.00005794 versus0.00008455 (31.5%lower). Full-imageMSE0.00005657 approximatelycopy-middle0.00005637. No response model/accuracy tested. ParentCPUverified both full checkpoints and126Adam steps; guard exited normally. [Final report](../../SecondPass/ThreeFrameConvVAE/FINAL_REPORT.md) | [Completion proof](../../SecondPass/ThreeFrameConvVAE/LocalRuntime/completion_verified.json). No continuation or new run launched.

**October 3, 11:59 AM PDT — VAE RESUMED and optimizer progress verified.**
Restored complete update5600 model/Adam/RNG/streams/scheduler. Latest independently CPU verified saved update5700 /172,768triplet presentations, all126Adam states at the saved step; live update5784. Only `resume01/latest.pt` and `resume01/best.pt` exist. Original5600 filename promoted/removed after verification; preserved checkpoint content/state. Startupval5600 balancedrecon0.001003128 initializes available best; historical5500 weights unavailable. Same9900target and original4:49:10PM PDT hardstop, no renewal; cloudclosed. Resume worker15761/guard15751. [Recovery evidence](../../SecondPass/ThreeFrameConvVAE/LocalRuntime/resume_verified.json).

**2026-10-03 — Checkpoint cleanup COMPLETE; VAE STOPPED.**
User authorized removing old experiment checkpoints. Deleted1,518files /57.9GB; disk now109GiB (~117GB) free. Only VAE checkpoint5600 remains, verified complete model+Adam+RNG+streams/scheduler. Code/logs/reports retained; historical checkpoint references are no longer available. VAE failed saving5700 at9:31AM PDT; latest saved5600, no final tests. Restart held; no cap renewal. [Cleanup receipt](../../SecondPass/ThreeFrameConvVAE/LocalRuntime/checkpoint_cleanup.json).

**2026-10-03 15:57 UTC — Three-frame VAE TRAINING locally; production verified.**
Old CNN-GRU saved/stopped570updates/17,288presentations. Fresh VAE ordered3frames
through spatiallatent256x13x13, reconstruction-only; all126Adamstates/tensors
advanced in checkpoint3/96, parentCPUverified. Nativeprofilepins9,900updates/
300,000triplets/30,000unique, batch32micro4, new8hLOCALcap ending4:49PM PDT.
Val100 balancedrecon0.009875 versusgray0.029108/copy-middle0.003412; preliminary
reconstruction still trails copy-middle. Cloud stays closed.
[Record](../krauzlis-three-frame-vae.md).

**2026-10-03 14:44 UTC — Weighted-mean CNN–GRU TRAINING locally; production Adam verified.**
Fresh6,999,474parameter/70tensor model on MPS, same no-cue/no-distractor task.
Native profile pins all2,310updates/70,000presentations/7,000unique, batch32micro4,
1000trials×10epochs, FP32/full BPTT/Adam1e-4/no clipping. Checkpoint3/96 independently
CPU reloaded; all70 parameters/Adam states advanced, fresh constructor/empty Adam/
MPS RNG verified. Early snapshot17/544, losssofar0.68722; validation pending.
Harddeadline October3,3:40:04PM PDT. Previous local run complete; cloud stays closed.
[Record](../krauzlis-weighted-mean-conv-gru.md).

**2026-10-03 UTC — ALL CLOUD PODS DELETED; weighted CNN held for tomorrow.**
User-cancelled single-stimulus RViT, 66 artifacts and checkpoint1700/92Adam states retrieved/verified before exact pod deletion. Provider list verified empty. Weighted CNN queue stopped with no rental/allowance started; implementation preserved. Local job continues under its original cap. [Cancellation receipt](../../SecondPass/SingleStimulusRViT/CloudRuntime/cancellation_retrieval_verified.json).

**2026-10-03 UTC — Cloud jobs cancelled; weighted CNN held for tomorrow.**
User requests killing pods. Next weighted CNN queue is disabled with implementation preserved; no next rental or budget ever started. Current single-stimulus cloud shutdown/retrieval is underway. Random-frame cloud already completed/deleted. Local job was not included in the cloud cancellation.

**2026-10-03 04:43 UTC — Random-frame RViT COMPLETED, fresh finals remain at chance.**
All 3,960 updates / 120,000 presentations / 12,000 unique movies; final 100-update mean loss 0.68455. Selected3500 and terminal3960 both 50% BA in B12/B20/B28, fresh200/cell; mean AUC0.48828/0.47909. Final artifacts/checkpoints verified and pod5us0rp5jwmg5bu deleted. No continuation. [Final report](../../SecondPass/RandomFrameRViT/FINAL_REPORT.md).

**2026-10-03 04:13 UTC — Weighted-mean CNN–GRU AUTOMATICALLY QUEUED.**
Fresh 6,999,474-parameter /70-tensor model: causal pixel mean0.5/0.4/0.1 → shared
CNN → flatten2704 → standard GRU256 → final binary classifier. Same single-stimulus,
no-cue/no-foil task, full BPTT, 1,000-trial pools×10 epochs, target2,310 updates.
Queue4518/PPID1 waits for current single-stimulus pod final retrieval/deletion, then
launches under a new8h/$5 cap starting at creation. Short CPU checks passed.
No new rental or training yet; existing jobs/caps unchanged.
[Record](../krauzlis-weighted-mean-conv-gru.md).

**2026-10-03 03:31 UTC — Single-stimulus no-cue conv RViT TRAINING on A40.**
Whole fresh original conv RViT/full BPTT, exact29/37/45frames and retained original
patch pixels at(20,50)/(80,50); cue and other patch removed. Pin2,310 updates /
70,000 presentations /7,000 unique movies,1,000-trial pools×10epochs,batch32/micro4.
Production checkpoint3/96 CPU reloaded:all92learned parameters/Adam states advanced
from direct fresh constructor/empty Adam. Snapshot6/192,loss0.68173; validation pending.
Pod`jxbmb44y9wamhl`,new8h/$5,hard11:28:02UTC /October3,4:28:02AM Pacific.
Other cloud/local runs continue. [Record](../krauzlis-single-stimulus.md).


**2026-10-03 03:12 UTC — Older cloud RViT replay COMPLETED; fresh tests remain at chance.**
All 2,310 updates / 70,000 presentations / 7,000 unique movies, final 100-update
mean loss 0.49881. Validation selected 1,250: fresh mean BA 50%, AUC 0.4672;
terminal mean BA 47.83%, AUC 0.4922, 200 trials/condition. All 63 final artifacts
verified, both final checkpoints CPU reloaded with 92 Adam states, pod deleted.
No longer continuation or cap renewal. Random-frame cloud/local motion RViTs continue.
[Final report](../../SecondPass/TwoFrameRViTReplay/FINAL_REPORT.md).


**2026-10-03 02:43 UTC — Random-frame-gradient gated motion RViT TRAINING on A40.**
Fresh production checkpoint 3 / 96 presentations CPU verified: all 94 learned tensors
and Adam states advanced; analytic buffers unchanged. Pinned 3,960 updates / 120,000
presentations / 12,000 unique movies, batch 32/micro4, 1,000-trial pools × ten epochs.
One uniform frame per trial, T-scaled temporal parameter derivative, live suffix memory
Jacobians and ordinary decoder CE. Snapshot 25 / 800, loss 0.70018; validation pending.
Pod `5us0rp5jwmg5bu`, new 8h/$5, hard 10:39:56 UTC / October 3, 3:39:56 AM Pacific.
KDA16 completed/retrieved/deleted; existing cloud RViT and local run continue.
[Record](../krauzlis-random-frame-gradients.md).


**2026-10-03 UTC — Targeted objective check passes; RViT temporal gradients severely attenuate on a complete native trial.**
Cue/final input-gradient ratios1.33e-23 cloudRViT1500,3.07e-14 motionRViT100;
KDA4100 ratio0.8795. All learned tensors receive gradients, but that does not rule
out weak early-time learning. One paired29-frameB12 trial/checkpoint, CPU-only;
loss/label/decoder/accumulation checks passed. Jobs/caps unchanged.
[Evidence and limitations](../krauzlis-training-path-audit.md).

**2026-10-03 01:30UTC — Structured-motion RViT fresh local TRAINING.**
CNN-GRU cancelled/saved2575updates/82400trials. Newmotion-front candidate production
checkpoint3/96 CPUreloaded; all92Adamstates/learnedparameters advanced, fixedbuffers
unchanged. Pin660updates/20000presentations/2000unique,1000trials10epochs/pool;
CPUanalytic/MPSlearned, batch32/micro1, fullBPTT. New8h hard09:15:49UTC /October3
2:15:49AM PDT, independentguard; cloudcaps unchanged. No held-out result yet.
[Record](../krauzlis-simoncelli-heeger.md) · [TechnicalproposalPDF](../../SecondPass/StructuredMotionRViT/TechnicalDocument/architecture_proposal.pdf).

**2026-10-03 UTC — Author-derived Simoncelli–Heeger candidate IMPLEMENTED, UNTRAINED.**
Five-scale analytic motion CNN, causal nine-frame windows and fresh RViT fusion/readout.
Three focused CPU checks passed; no training, GPU profile or queue launched.
Current jobs and caps unchanged. [Record](../krauzlis-simoncelli-heeger.md).

**2026-10-03 00:41 UTC — All three runs continue.** RViT replay289/2310,
first pool epoch9/10,8800presentations/1000unique; KDA16heads2970/4216,
95040freshmovies; localCNN-GRU2333/2631,74656freshmovies. Recent100-update mean
losses0.681589/0.683380/0.683749 respectively. Latest validations: RViT250meanAUC
0.463484, KDA2108meanAUC0.537502, GRU1315meanAUC0.537774; BA50%allconditions
for every model. These are live validation snapshots, not final held-out tests.
No failure or cap renewal; independent guards/mirrors remain active.


**2026-10-03 00:33UTC — RViT replay update182, first pool epoch6/10.**
Mean100-update loss0.683764; validation100 BA50% allconditions/meanAUC0.476540,
all-positive decisions. No demonstrated acquisition yet; training continues.
[Replay record](../krauzlis-rvit-replay.md).


**2026-10-03 UTC — Online RViT cancelled; fresh replay RViT TRAINING.**
User requests1000 generated movies reused10 shuffledepochs before replacement.
Old RViT stopped2075/66400;89filesverified and terminal92AdamstatesCPU-reloaded.
SameA40/original05:16:04UTC harddeadline, no caprenewal; KDA/localGRUcontinue.
[Replay experiment](../krauzlis-rvit-replay.md) · [Run evidence](../../SecondPass/TwoFrameRViTReplay/RUN_STATUS.md).
Production checkpoint 3 / 96 presentations CPU verified: all 92 Adam states and parameters advanced. Live update 15 / 480 presentations, first pool and epoch. Seven complete pools pinned: 2,310 updates / 70,000 presentations / 7,000 unique movies.


**2026-10-02 21:31UTC — All three runs continue; first RViT and early GRU validations at chance.** RViT160/4216,16headKDA668/4216,localCNN-GRU1037/2631. RViTvalidation100(100/cell) BA50%allconditions/meanAUC0.442132; earlierKDA400(100/cell) BA50%/AUC0.509996; CPUfrozenGRU900(50/cell) BA50%/AUC0.526615. Thesearelivevalidation snapshots, notfinaltestresults; CPUdiagnostics do notaffectselection. Recent100-updatemeanlosses0.68412/0.68508/0.68282 respectively. No caprenewal orstopconditiontriggered.

**2026-10-02 — Two-frame RViT TRAINING on a second A40; faster validations enabled.** [Run](../../SecondPass/TwoFrameRViT/RUN_STATUS.md). Explicit parallel launch beside16headKDA, fresh7,270,290params/92tensors; checkpoint3/96episodes downloaded/hashverified/CPUreloaded, all92params/Adamstatesadvanced. Nativeprofilepinsfull4216updates/134912episodes,18costed validationlooks100,250,500,...terminal at100trials/cell; pairedfresh200/cellfinals. Pod7f27p6jxpitihn,NEW8h/$5cap21:16:04UTC→05:16:04UTC /October2 10:16:04PM PDT, independentguard/mirror. Native teaching unchanged. CPU-only earlysnapshotsof existingKDA/CNN-GRU running without changingtraining/selection; labelthosevalidationonly. Existingrunscontinue,3layercancelledpodstaysdeleted.

**2026-10-02 — Fresh16-head single-layer KDA TRAINING, optimizer verified; three-layer cancelled.** [Run evidence](../../SecondPass/SequenceKDA16/RUN_STATUS.md). Userrequestedkillthree-layerandstartqueuedmodel. NewA40podpa0ko8f2qirisy; nativeprofilepins4216updates/134912episodes,FP32fullBPTT/effective32micro4. Actualcheckpoint3/96episodes downloaded/hashverified/CPUreloaded,all27params/Adamstatesadvanced,freshconstructor/emptyinitialAdam/nativeRNGstreams verified; no profile/predecessorinheritance. Independentguard/mirroractive,NEW8h/$5cap20:33:14UTC→04:33:14UTC /October2 9:33:14PM PDT. Oldthree-layerstopped2615/83680;71artifactsand55stateAdamcheckpointverifiedbeforepod5hnvb87npqpqb4stopped/deleted,nofinaltests. LocalCNN-GRUcontinues;RViTuntrained. Earlierlaunchpending/waitingnotesarehistorical.

**2026-10-02 — Three-layer KDA CANCELLED; queued16headKDA launched on newA40.** Userrequestedkillandreplacement. Three-layerstopped2615/4216updates,83680episodes; all71artifactsretrieved/hashverified,terminalCPUreloaded55Adamstates,andpod5hnvb87npqpqb4stopped/deleted. Onlycompletedvalidation2108gave50%BAallconditions/meanAUC0.499388; nofinaltests. Newfreshsingle-layer16headKDApodpa0ko8f2qirisy launched underNEW8h/$5,cap20:33:14UTC→04:33:14UTC (October2 9:33:14PM PDT),nativeCUDAprofile/pin inprogress, productionAdamproofpending. LocalCNN-GRUcontinues;RViTimplementationonly. [Cancelledrun](../../SecondPass/SequenceKDA3/RUN_STATUS.md) · [Newrun](../../SecondPass/SequenceKDA16/RUN_STATUS.md).

**2026-10-02 — Two-frame CNN / visual-query RViT IMPLEMENTED, untrained.** [Architecture](../../SecondPass/TwoFrameRViT/README.md), [journal](../krauzlis-two-frame-rvit.md). Ordered previous/current RGB CNN → 13×13×256 tokens; one shared recurrent block with independent visual self-attention and previous-memory cross-attention, both queried by current X. Per-token 256→16, flatten2704→FFN→2. 7,270,290 fresh trainable parameters/92tensors, full temporal gradients and CNN checkpointing. Three focused CPU checks plus complete native45-frame B28 forward passed; no GPU/profile/training launched or new compute allowance. Existing cloud/local runs and16headKDAqueue preserved.

**2026-10-02 — Single KDA with16full-width heads IMPLEMENTED and AUTOMATICALLYQUEUED.** [Run status](../../SecondPass/SequenceKDA16/RUN_STATUS.md), [journal](../krauzlis-sequence-kda16-heads.md). OneKDA/terminalCLS,16×64key/value heads,737,170freshparameters/27tensors,8×originalKDAstate. Userselectednew8h/$5cloudrunafterthree-layerpod5hnvb87npqpqb4 completes, artifacts/finalcheckpointsverify and deletionconfirms. Parentqueue63567/PPID1active; no new rental/profile/optimizer/budgetstart. NativeGPUprofilewillpinfeasibleexposure,target4216,Adam1e-4/effective32micro4/FP32fullBPTT/native teachingunchanged. Sevenfocusedchecks passed. Cloudthree-layer and localCNN–GRUcontinue unchanged.

**2026-10-02 — Delayed-frame CNN/standardGRU LOCALTRAINING, optimizer verified.** [Live run](../../SecondPass/DelayedFrameGRU/RUN_STATUS.md), [journal](../krauzlis-delayed-frame-gru.md). Fresh21,293,770parameter model,AppleM4Max36GiB/MPS,Adam1e-4/no clipping,fullBPTT/FP32,effective32/micro1. Nativeprofilepinned2,631updates/84,192episodes beforeproduction;877updates/28,064episodes eachB12/B20/B28. Checkpoint3/96episodes CPUverified,all86Adamstatesadvanced;launchdowner54824/PPID1+guard54822. HardcapOctober2 7:57:51PM PDT; no renewal. Cloudthree-layerKDAcontinues unchanged. Localvalidation/finals pending; earlieruntrainedcandidatenotes superseded.

**2026-10-02 — Delayed-frame CNN/standardGRU candidate implemented while THREE-layerKDA training continues.** [Design and source](../../SecondPass/DelayedFrameGRU/README.md), [journal](../krauzlis-delayed-frame-gru.md). Two independently trainable residualCNNs, all24convolutions stride1/full100x100/no pooling; current256+previous256→standardGRU512→256→binaryhead.21,293,770fresh parameters/86tensors; five focused CPU checks passed. Native teaching unchanged, explicit causal delay/fullgradients. **Candidate not trained; no new cloud job/budget.** The KDApod/guard/mirror/deadline below remain active.

**2026-10-02 — Fresh THREE-layer whole-sequence KDA training on RunPod, optimizer verified.** [Run evidence](../../SecondPass/SequenceKDA3/RUN_STATUS.md) and [depth comparison journal](../krauzlis-sequence-kda3-depth.md): A40pod`5hnvb87npqpqb4`,324488fresh trainable parameters,3pre-LNresidualKDA blocks/terminalCLS. Native tasks and Adam settings unchanged. SteadyGPUprofilepinsfull4216updates/134912episodes matching one-layer exposure. Checkpoint3/96episodes downloaded/hashverified/CPUreloaded; all55namedAdam states advanced. Guard/mirror active. Originalcap18:20:05UTC→02:20:05UTC /October2 7:20:05PM PDT preserved through initial unstarted-pod replacement; no renewal. Validation/finaltests pending. One-layer completion below remains the baseline.

**2026-10-02 — Single layer whole sequence KDA completed; task not acquired.** [Final results and artifacts](../../SecondPass/SequenceKDA/RUN_STATUS.md): all4,216updates/134,912episodes completed from fresh weights. Validation selected2,108; selected and terminal4,216 both give50% BA in B12/B20/B28 on200 fresh paired tests/condition, with all-positive decisions (57% raw accuracy,100% foil/catch false positives). Selected/terminal mean test AUC0.500102/0.502422. All45 manifest files retrieved and verified; both checkpoints CPU-reloaded with27 active Adam states. A40pod`8gamd8ems1pa0n` stopped and deleted after verified retrieval at17:56:01UTC. No renewed cap or further training. Earlier live entries are historical snapshots.

**2026-10-02 — Single layer whole sequence KDA TRAINING on RunPod, saved optimizer verified.** [Live-run specification/evidence](../../SecondPass/SequenceKDA/RUN_STATUS.md): A40pod`8gamd8ems1pa0n`, whole158340parameters fresh, exactlyoneKDA/terminalCLS, unchanged native KrauzlisB12/B20/B28. Profile pinned4216updates/134912episodes before production. Checkpoint3/96episodes downloaded/hashverified/CPUreloaded, all27namedAdam states advanced; startup snapshot11updates/352episodes. Guard/private status/mirror active. New8h/$5cap ends2026-10-03 00:15:49.999944UTC /October2 5:15:49PM PDT; no renewal. Held-out results pending. User correction to avoid exhaustive checks honored by removing repeated remote CPU suites. Preparation entries below are superseded.

**2026-10-02 — Current requested work: implement and train single layer whole sequence KDA on a pod.** [Specification](../../SecondPass/SequenceKDA/README.md) and [journal](../krauzlis-sequence-kda-design.md): native RGB frames as chronological patch tokens, spatial/time features, exactly one global KDA, terminal CLS and binary head. Entire model starts fresh; native B12/B20/B28 teaching unchanged. Constructed158340parameters, strictFP32 full-sequence parity passed, native CPUcheckpoint3 verifies96episodes/all27Adam states advanced. New8h/$5 bounded deployment is being prepared; no cloud optimizer evidence yet. Prior run entries remain historical and are not resumed.

**Probe adequacy completed — fresh600/150/300 grouped splits.** [Results](../krauzlis-probe-adequacy-diagnostic.md): matched raw-pixel ridge side BA0.500–0.512 despite endpoint optical-flow0.740; trained/random CNN probes remain near chance, removing projection does not rescue them. This probe family is not validated as an information-loss assay; no neural erasure or temporal-comparison lesion established. All72 fits/79 metric records replayed exactly, both frozen model identities preserved, original1800s cap retained. No training/task/cloud change.

**Upstream frozen selected2297 diagnostic completed on300 fresh native groups.** [Results](../krauzlis-upstream-motion-diagnostic.md): full-history pixel changed-side BA0.965, endpoint pixel0.760, direct CNN/KDA pre/post probes near0.50; no matched temporal-access rescue. Exact source/checkpoint parity and saved-prediction replay passed. No deployed training/task/cloud changes; original1200-second cap preserved. Mechanism remains unlocalized; no remedy selected.

**Frozen selected2297 factorized follow-up completed — exploratory reused-test.** [Results](../krauzlis-factorized-diagnostic.md). Original native n175, paired challenge175 groups. Disjoint component/calibrator fitting; matched final/external-phase access; fitted-model replay and grouped uncertainty. No model/stimulus/cloud changes.

**2026-10-01 — Local fresh Krauzlis run completed; cloud attempt02 launcher prepared, not yet launched by researcher.** [Evidence and protocol](../krauzlis-fresh-attempt02.md). Local saved report:4595updates/147040episodes; validation-selected2297 final BA0.500000/AUC0.534178, terminal4595 BA0.500000/AUC0.558973, all-positive decisions in all three conditions. This supersedes the local live snapshot below. Explicit NEW8h/$5 cloud retry remains wholly fresh; no checkpoint inheritance. Real CPU checkpoint3 now produces native verification and actual launcher readiness; source/guard/profile-handoff suites passed. Parent owns rental; no cloud calls/GPU work in this preparation. Prior logs/artifacts preserved.


**2026-10-01 00:54 UTC / September30 17:54 PDT — LOCAL Krauzlis-only WHOLEMODEL-FRESH production verified.** User explicitly selected whole-model from-scratch, not transferred weights. [Isolated adapter/protocol](../../SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisOnly/README.md) preserves current terminal spatial transformer and native B12/B20/B28 teaching; fp32/full BPTT, Adam1e-4, batch32/micro4, no clipping or curriculum. All used modules train; other task heads are inactive, not a frozen trunk. Runtime `/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run`. Observed37 completed updates; saved checkpoint3/96episodes independently SHA-256 verified and CPU-loaded:77 initial tensors equal direct construction, empty initial Adam/streams/counters,52 used learned tensors changed with advancing Adam, all three native condition streams advanced, MPS RNG persisted. Production is launchd-owned `org.vawm.krauzlis-wholemodel-fresh01`, supervisor91782/PPID1, exclusive MPS worker91913; harmless timeout probe passed, KeepAlive=false. Absolute eight-hour cap **2026-10-01T08:52:13.337213Z /01:52:13 PDT**, unchanged from first profile; includes evaluation/reporting. Profile pinned **4595 updates/147040 episodes**, reduced prospectively from10000: B12/B20 each1532updates/49024episodes; B281531/48992. Conservative optimizer estimate6.179h; first13 production updates0.678x matched profile. Validation2297/4595 selects mean three-cell AUC then BA; final selected/terminal test200/cell with target/foil/catch confusion and rates. **No production BA/AUC yet.** Through update37, first10/recent10 CE0.72633/0.69363 versus uniform0.69315; training loss is not acquisition evidence. Automatic final outputs: `report.json`, `REPORT.md`, `test_selected.json`, `test_terminal.json`. No cloud APIs/actions performed; older cloud live entries below are historical, not current lifecycle checks. Source/deployment/prior artifacts preserved.


**2026-09-30 05:48 UTC — Fresh terminal spatial-transformer variant TRAINING.** User explicitly selected terminal-only consolidation and a new 12h/$8 A40 cap. [Implementation and protocol](../../SecondPass/SpatialReadout/SpatialConsolidation/README.md). Entire CNN/three KDAs/ConvGRU/transformer/readout/heads fresh; no pretrained, predecessor or disposable-profile state. Pod `dwjm8cvx7qaqp5`, runtime `/Users/jonathanmorgan/VAWMRuntime/cloud_spatial_consolidation_01`. Observed 39 production updates/1248 episodes; step2 checkpoint downloaded, hash verified and independently loaded with advanced Adam. Measured allocation pinned BEFORE production:29874 updates/955968 episodes,2298 updates per task, unchanged13tasks/35conditions, batch32/micro4, fp32/full BPTT, Adam1e-4/no clipping. Validation14937/29874 then fresh selected/terminal tests; no production accuracy/AUC measured yet. First-cycle matched-profile ratio1.257; slowdown risk remains, hard cap takes precedence over completing exposure. A40$0.49/h plus storage; independent authenticated/publication-verified guard ends2026-09-30T17:36:34.532188Z (10:36AM PDT), scientific cutoff10min earlier. Manual checkpoint retrieval exercised; automatic mirror NOT installed, retrieval remains laptop-dependent. Existing ConvGRU run and cancelled older experiments untouched. This is not exposure-matched to the long ConvGRU continuation.

**2026-09-29 — Frozen checkpoint35039 neuroscience study: pinned grid completed, broader comparisons incomplete.** [Findings](../../SecondPass/SpatialReadout/NeuroscienceAnalysis/FINDINGS.md), [34-page atlas](../../SecondPass/SpatialReadout/NeuroscienceAnalysis/Neuroscience_atlas.pdf), [gallery](../../SecondPass/SpatialReadout/NeuroscienceAnalysis/index.html), [individual-trial/frame viewer](../../SecondPass/SpatialReadout/NeuroscienceAnalysis/viewer.html). 14,336 scored presentations (paired repeats, not independent scenes), 91 orientation and25 binding causal conditions at64 trials/cell; actual maps for10 acquired tasks. Native cued orientation/binding perfect atD0/4/12/24,128/cell; contour126/128. D12 orientation3/6/10 degrees yielded54.7/68.8/93.8% accuracy; these smaller magnitudes are OOD. Matched cue reassignment supports relevant-evidence use, but gate/coefficient maps do not demonstrate sustained target prioritization. Local finest-emission perturbations had small orientation effects (largest loss4/64, exploratory interval includes0) and no binding choice flips; do not claim primate-mechanism equivalence. Missing: independent foil factorial, magnitude/matched-cue allocation contrasts, binding random pulses; recognition excluded as partially acquired. All773 artifact hashes and14,336 saved decisions independently checked by parent; figures visually inspected. Model/checkpoint unchanged, no live analysis worker; completed within1438s of3600s cap. No cloud/training change or automatic follow-up. Earlier live progress below is historical, not refreshed by this analysis.

**2026-09-29 — Frozen analyses authorized while ConvGRU continuation remains active.** Saved live status at13:26UTC showed35012 cumulative updates,27273/104000 added, no recorded training failure. Completed validation28539 scored100%BA on seven sensory tasks, orientation_ring, orientation_cued and spatial_binding; recognition nonempty73.09%, cued duration26.17%, Krauzlis50%. These are validation results, not final tests. Periodic checkpoint35039 subsequently retrieved and hash-verified locally for analysis; it is not the validation-selected checkpoint. [Cued-motion pixel audit](../../SecondPass/SpatialReadout/CuedMotionAudit/PIXEL_FINDINGS.md) independently replayed128duration winners100%BA and200KrauzlisB20 test trials88.79%BA/.957AUC using image-only observers with disclosed fixed geometry/timing/history privileges; this supports usable rendered information, not a unique neural failure mechanism. Frozen-model diagnosis is separate. User now requests [neuroscience analysis](../../SecondPass/SpatialReadout/NeuroscienceAnalysis/BRIEF.md) of acquired tasks: psychometrics, real per-timestep spatial maps/allocation, paired local inhibition and calibrated microstimulation. Researcher preparing/executing one fixed35039 local analysis under a new finite3600s cap from first accelerator operation; no main-model updates, cloud changes or concurrent local accelerator worker. Prior pending-launch entries below are historical. Automatic launchd checkpoint mirror remains unactivated after the approval timeout; this35039 retrieval was a separately authorized one-shot analysis copy.

**2026-09-29 — User-authorized long continuation: original worker STOPPED/SAVED; new optimizer launch PENDING.** Fresh ConvGRU stopped cleanly at7739updates/247648episodes on request, not a model failure. Verified terminal SHA256 `7b87e8f21e4a972cbb78f8d657786e38c2aab29bf313329167575f9f564d5ff1`; all33 finalization artifacts retrieved locally. Original worker335/supervisor267 and mirrorPID24891 absent. Disclosed tenfold interpretation:104000 additional updates/3328000episodes, target cumulative111739/3575648. Same architecture/tasks/optimizer policy, preserving terminal full state rather than rewinding to validation5200. Same A40 pod5awq67fxgnsu1m restarted intentionally only to replace the old guard; new independent36h/$20 cap ends2026-09-30T18:14:46.431188Z (11:14AM PDT), provider readback/publication verified. Runtime `cloud_convgru_continuation_01`; researcher implementing minimal continuation harness. Validation5200 completed35cells: sensory learning strong, spatial/memory tasks near chance; no final test during the signal stop. Earlier fresh-run TRAINING status below is historical until new persisted progress is verified.

**2026-09-29 04:07 UTC — Fresh whole-model ConvGRU TRAINING, saved progress independently verified.** A40 pod `5awq67fxgnsu1m` at$0.49/hour plus storage; supervisor267/worker335. Observed89updates/2848episodes; checkpoint78 downloaded and SHA-256 verified locally. CPU reconstruction on the pod independently proved initial checkpoint tensors exactly equal a freshly seeded `SpatialReadout` constructor, empty initial Adam/streams/zero counters, and68 changed learned tensors by saved step65 across CNN/all3KDAs/ConvGRU/readout/heads. No trained state or profile state inherited. Native13tasks/35conditions unchanged. Profile pinned the full10400updates/332800episodes,800updates/task; validation5200/10400 with fresh selected/terminal finals. First production cycle1.077x profile, no material slowdown/failure. Hard deadline2026-09-29T12:02:21.369504Z; scientific cutoff11:52:21UTC, new8h/$5allocation, no renewal. Deadline/retrieval-only independent guard armed, provider status active, launchd mirrorPID24891/PPID1. Early losses near uniform; validation accuracy/AUC UNMEASURED. Runtime `/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_fresh_01`; [fresh-only harness](../../SecondPass/SpatialReadout/FreshRun/README.md). Earlier cancelled transformer pods remain stopped. Earlier preparation/no-new-run statements below are historical.

**2026-09-29 — Fresh whole-model ConvGRU run explicitly authorized; preparing, not yet training.** User requests ConvGRU from scratch on RunPod. Reuse only the original spatial-ConvGRU architecture code; every CNN/KDA/ConvGRU/readout/head weight and all optimizer/RNG/stream state initialize anew. Unchanged13tasks/35conditions, native supervision, no curriculum. Target10400updates/332800episodes subject to measured preproduction fit, new8h/$5 single-A40 allocation including finalization, no renewal. New harness `SecondPass/SpatialReadout/FreshRun`; runtime `cloud_convgru_fresh_01`. Verified deadline/retrieval-only guard and automatic mirror reused. No earlier pod resumed or trained checkpoint transferred; prior cancellations remain in force.

**2026-09-29 03:18 UTC — USER CANCELLED training; original KDA/global-GRU architecture restored as the active reference.** RunPod `7ij62e571pln8w` was stopped and provider-read back as EXITED. Mirror service removed and process absent; local checkpoint2899 SHA-256 verified. Disk retained for unmirrored evidence; storage charges remain. No restart or new training. The original `AccumulatorBaseline(stack=3, center=True, accumulator='kda')` has documented fresh initialization; later final-ConvGRU, comparator and CLS architecture branches inherited learned weights and are NOT clean from-scratch comparisons. The cancelled no-CLS retry itself was fresh. User's standing rule: NEVER inherit weights without explicit permission. No checkpoint was transferred for this rollback. [Active reference and evidence](../../SecondPass/ACTIVE_BASELINE.md). All running/authorization entries below are historical and superseded.

**2026-09-29 02:03 UTC — Fresh retry TRAINING with off-pod status and automatic checkpoint mirroring.** A40 pod `7ij62e571pln8w` at$0.49/hour, worker336/supervisor268;96updates/3072episodes, remote checkpoint91, automatic launchd mirror already downloaded/hash-verified checkpoint39. Persisted first13-update state verified empty initial Adam, advanced optimizer/native streams and105 changed parameter tensors; no inherited checkpoint. Unchanged13tasks/35conditions and exact fresh-only source/data bundle. New profile pinned3224updates/103168episodes before production (248updates/7936episodes per task); validation1612/3224, same selection/final protocol. First production cycle0.985x profile; no failure. Hard deadline2026-09-29T05:57:23.958244Z, new4h/$3maximum, no renewal. Guard has NO inactivity or completion-age stop: only verified manifest-bound full retrieval acknowledgement or hard deadline. Private never-deployed status template `l9zho77jzp` receives live progress/evaluation/error/stop snapshots and exact readback every60s; live off-pod update71 and publication_verified=true independently read. Launchd mirrorPID81181 (PPID1) copies metadata/checkpoints every120s while laptop online, then verifies every final artifact before acknowledging stop. Provider status and hard cap work independently of laptop connectivity; hard-cap shutdown can still leave unmirrored artifacts on retained disk. Early per-task CE remains near uniform prediction; no validation accuracy yet. Runtime `/Users/jonathanmorgan/VAWMRuntime/cloud_convdecoder_fresh_02`. Prior stopped attempt remains preserved/unresolved; preparation statements below are historical.

**2026-09-29 — User authorized a fresh retry after an opaque early cloud stop; preparation, NOT training yet.** Attempt01 pod `zab3qcxa59uxil` stopped at2026-09-28T22:32:55Z according to provider lifecycle (account-issued stop), before its hard deadline. Actual guard trigger and final optimizer/evaluation outcome remain unverified because CPU-only recovery lacked host memory and container logs timed out. Checkpoint442 is locally durable; do not infer it was the last remote update. Preserve its disk and artifacts. New `cloud_convdecoder_fresh_02` reuses the unchanged fresh-only source/data package, not learned state. Remove inactivity and completion-plus600s shutdown; retain only verified full-retrieval acknowledgement or a new finite4h/$3 hard cap. Private provider-side status record `l9zho77jzp` was create/update/readback verified without compute; it is a never-deployed template, avoiding edits that could restart a running pod. A bounded launchd artifact mirror is being installed before rental. No new model performance claim.

**2026-09-28 20:47 UTC — Fresh no-CLS model TRAINING on RunPod A40.** Pod `zab3qcxa59uxil`, worker278/supervisor247. Every learned tensor newly initialized; Adam initially empty, new RNG/native streams/scheduler, no checkpoint inputs or profile-state inheritance. Independent saved-state verification at26updates confirmed fresh initial optimizer, advancing streams/scheduler and105 changed parameter tensors; checkpoint52 downloaded and hash/size verified locally. New13-task/35-cell profile pinned4498updates/143936episodes (346updates/11072episodes per task), reduced BEFORE production from5200 to fit the new4h cap. Validation2249/4498 with prospective equal-task AUC then chance-normalized BA; fresh paired finals128/cell, Krauzlis200; empty specificity separate. All13tasks/35conditions and teaching unchanged. No new accuracy measurement yet. Maximum$3, A40$0.49/hour plusstorage; hard stop2026-09-29T00:41:16.233494Z, scientific deadline600s earlier. Authenticated pod-local guard verified. Runtime `cloud_convdecoder_fresh_01`; artifacts retrieved through checkpoint52. Previous pod/model untouched. Earlier migration and pending-launch statements are superseded.

**2026-09-28 — User correction: fresh-from-scratch no-CLS RunPod training authorized, launch preparation active.** User explicitly rejected weight transfer and requested a new pod training run. ALL learned components, Adam, RNG, scheduler and native task streams must initialize fresh; earlier migration plans below are superseded, not blockers. One no-CLS convolutional-decoder model, unchanged13tasks/35conditions. Target5200updates/166400episodes subject to measured fit before production, new finite4h/$3maximum including setup/evaluation/retrieval, modestA40$0.49/hour planned. Researcher prepares fresh-init worker/package; parent owns cloud launcher/independent stop. No new pod or persisted optimizer progress yet. Preserve every prior experiment. Runtime `/Users/jonathanmorgan/VAWMRuntime/cloud_convdecoder_fresh_01`.

**2026-09-28 — No-CLS convolutional readout implemented; NOT trained.** User requested replacing CLS with the earlier convolutional decoder on the selected spatial representations. New separate `SecondPass/SpatialRecurrentConvDecoder` removes CLS entirely:49 recurrent spatial tokens,49 queries/98 joint{Z,H} keys, unchanged-style two convolutional transformer blocks. Final64x7x7 H alone feeds ConvNormAct64→32→64, spatial mean/max,128→256 SiLU and native task heads. This adapts the older convolution-before-pooling pattern to one field; tasks/teaching unchanged. Saved CPU evidence verifies early-frame BPTT, decoder updates, reset/carry and all13heads on synthetic inputs only. Real checkpoint inspection hit an approval timeout and was not retried; trained-weight/Adam migration and training remain pending. Prior CLS model/results preserved. [Implementation/status](../../SecondPass/SpatialRecurrentConvDecoder/README.md).

**2026-09-28 — Recurrent transformer/CLS run COMPLETE; final artifacts recovered.** Finished17:21:43UTC with all2600 additional updates/83200episodes, selected1300 and terminal2600 each evaluated on35/35 fresh final conditions. Selected/terminal BA: motion46.88/25.78%, signed orientation79.69/100%, contrast100/100%, frequency92.97/100%, chromatic100/99.22%, contour50/57.03%, spectral100/97.66%, ring50/50.78%, cued orientation51.76/48.44%, cued duration23.63/25.78%, Krauzlis50/50.19%, binding50.39/52.34%, recognition51.91/50%. Empty recognition specificity100% separately. No acquisition of difficult spatial/memory tasks under this exposure; no matched control or proof of architecture incapacity. All40 published artifacts verified locally in `cloud_transformer_01/artifacts`, including both final checkpoints. Bounded CPU-only recovery succeeded after GPU restart lacked host capacity; no optimizer restart. Pod `txmfzvhbc9zm3b` verified EXITED after retrieval; retained storage still billable. No further run launched. [Full results and every primary condition](../../SecondPass/SpatialRecurrentTransformer/RESULTS.md). Earlier live-training entries are historical.

**2026-09-28 16:03 UTC — Recurrent spatial transformer/CLS TRAINING on A40.** Production started16:00:54UTC on pod `txmfzvhbc9zm3b`, worker312/supervisor283. Readback:75updates/2400episodes, saved65; checkpoint39 also downloaded/hash-verified locally. Independently reloaded production13 and verified all56 fresh parameter tensors (including recurrent CLS) changed, compatible Adam advanced and native streams progressed; exact migrated full-state checks pass. H has49spatial+1CLS tokens at64channels, two pre-norm convolutional attention/FFN blocks; H queries joint{Z,H}, H residual/output recurs, onlyfinalCLS→256→task head. Old ConvGRU/comparator/flatten-field readout removed. All learned encoder/transformer/head weights trainable; same13tasks/35conditions, oneAdam1e-4/fp32/fullBPTT. Full2600updates/83200episodes pinned from the new CUDA profile; estimated optimizer2.30h, conservative total3.53h, first production cycle0.960x profile. First validation1300, then2600; no new accuracy result yet. Original$12/21:46:03.902995UTC cap retained, independent pod-local guard authenticated/live. Local automatic retrieval service remains disabled; one-shot retrieval was exercised successfully. Runtime `/Users/jonathanmorgan/VAWMRuntime/cloud_transformer_01`, source [README](../../SecondPass/SpatialRecurrentTransformer/README.md). Earlier pending-launch statements below are historical.

**2026-09-28 — Recurrent spatial convolutional transformer with CLS authorized; implementation in progress, NOT launched.** User requests H-query cross-attention to joint {Z,H} keys/values after the KDA encoder, residual H and convolutional transformer output as the next recurrent H, plus one recurrent CLS token as the only decision representation. Replace ConvGRU/comparator/flatten-field readout; preserve compatible sensory/input/head weights and Adam from the now-retrieved completed comparison terminal (cumulative9360). All13tasks/35conditions and native teaching unchanged. Researcher implements one candidate; parent owns cloud. Target2600 new updates/83200episodes subject to one measured preproduction profile, within the unchanged $12 total ceiling and21:46:03.902995UTC enclosing deadline. No extra arm or budget renewal. Launch requires saved optimizer progress.

**Completed comparison artifacts recovered and verified.** Both selected1300 and terminal2600 final tests completed35/35cells; supervisor finished15:13:09UTC with no failure. All53 published artifact hashes/sizes verified locally in `cloud_comparison_03/artifacts`, including selected and terminal model checkpoints. Final selected BA: motion/orientation/contrast/frequency/chromatic100%; contour88.28%; natural spectral99.22%; ring48.44%; cued orientation50%; cued motion24.61%; Krauzlis50%; binding49.80%; recognition51.39%. Empty recognition specificity separate. No difficult-task rescue; no matched control. Old A40 pod briefly restarted solely for bounded retrieval, then verifiedEXITED. [Full final results](../../SecondPass/SpatialComparisonReadout/CloudRun/RESULTS.md). Earlier missing-final-artifact statements are superseded.

**2026-09-28 14:26 UTC — Cloud candidate TRAINING, with independently verified saved optimizer progress.** User said "can you try again". Secure A40 pod `txmfzvhbc9zm3b` ($0.49/GPU-hour plus storage) successfully deployed. Production began14:25:32 UTC after a PyTorch2.8 serialization-only repair: explicitly add legacy Adam `decoupled_weight_decay=False`; inherited moments, optimizer steps and all existing options remain unchanged. Failed pre-update receipts are archived rather than relabeled as training. Full model/optimizer/names/scheduler/native-stream/RNG migration checks pass. Independently reloaded production checkpoint26:832 fresh episodes, all four comparator tensors changed, Adam and streams advanced; live update27. Pinned full2600 additional updates/83200 episodes,200updates/6400episodes per task across unchanged13tasks/35conditions, no control or teaching change. First complete production cycle is0.905x matched profile duration; conservative estimated remaining train/evaluation/report time7014s. Original enclosing13:46:03–21:46:03 UTC cap and $12 ceiling remain unchanged, not reset on retries. Authenticated pod-local stop is armed, with stall and completion stops; it survives laptop disconnection, while artifact retrieval still requires connectivity. Local runtime/evidence: `/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/production_verified.json`. New validation and final results are pending; no model-performance improvement is claimed. Earlier zero-pod/failure snapshots below are historical.

**2026-09-28 14:04 UTC — Renewed cloud attempt blocked before training; ZERO pods remain.** After deleting both earlier pods on the user's instruction, the user explicitly requested restarting the work. The new attempt retained the $12 ceiling and fixed 13:46:03–21:46:03 UTC enclosing deadline. The prepared 67,946,799-byte deployment archive was uploaded and SHA256-verified on an A5000, then on an L4, but setup failed first on tar ownership restoration and then on the container's externally managed Python environment. Extraction now uses `--no-same-owner`; deployment now loads the image environment and permits the documented container package installation. Stopping these pods during repair released scarce capacity and both restart requests failed; both pods were deleted. Subsequent bounded replacement requests for listed Secure/Community GPUs were rejected for unavailable capacity. No accelerator profile or production optimizer update was verified. Authenticated provider listing at14:04:21 UTC returned `pods=[]`, `hasNextPage=false`; no GPU or attached pod storage remains. The earlier independent authenticated pod-stop self-test remains verified, but this is NOT a training result. Current-attempt billing records have not appeared yet; do not interpret empty scoped billing as zero spend. Local checkpoints/code/archive are preserved. Evidence: `/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/ATTEMPT_STATUS.json`; earlier stopped-pod statements below are historical and superseded.

**2026-09-28 — Cloud deployment attempt failed before verified training; pod is STOPPED.** Researcher lost provider/network access during deployment. The archive remained truncated/hash-mismatched in the last remote checks; no cloud production checkpoint or run-directory evidence was obtained. This is setup failure, not a scientific result. A fresh authenticated RunPod readback confirms pod `vj0rc2sb7da5m7` is `EXITED` with only start/terminate actions; SSH is refused. GPU compute is no longer running, but persistent storage remains billable. Stop time/cause and total charges are not established by this readback. Eight-hour deadline has passed; no restart/extension. Preserve local code, deployment bundle and terminal6760 parent; do not delete the stopped disk until recovery scope is resolved. Evidence: `/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_01/status_recovery.json`.

**2026-09-27 — Cloud candidate authorized; pod provisioned, training launch pending.** User requested putting the unfinished architecture on a suitable smaller GPU and explicitly added "and start training". This authorizes ONE learned spatial-comparison candidate from terminal6760; the cancelled local control remains cancelled. Cheaper3090/A5000 stock was unavailable; provisioned one24GB RTX4090, pod `vj0rc2sb7da5m7`, at$0.74/GPU-hour plus30GB container/20GB persistent storage. SSH verified real RTX4090/CUDA and PyTorch2.8.0. New immutable8h cap runs09:34:36.695–17:34:36.695 UTC, including setup/profile/evaluation/retrieval; target2600 additional updates/83,200episodes conditional on measured fit before production. All13tasks/35cells, native teaching and inherited optimizer/streams unchanged; CUDA platform migration disclosed. Run status `/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_01/pod.json`. A provisioned pod is NOT persisted optimizer progress; researcher is preparing the versioned CUDA wrapper. No causal architecture superiority can be inferred without the cancelled matched control.

**USER CANCELLED spatial-comparison training — 2026-09-25.** User instructed "kill the training pease". Stopped active control worker and supervisor, removed launchd job `org.vawm.spatial-comparison-v1-75feeabd250d`, and verified no experiment processes/job remain. Last saved control checkpoint is13 additional updates from terminal6760; candidate production had no checkpoint and its queued launch is cancelled. Preserve all files/checkpoints; no restart, second arm, final evaluation or budget renewal without new explicit authorization. Runtime cancellation record: `/Users/jonathanmorgan/VAWMRuntime/spatial_comparison_01/run/USER_CANCELLED.md`. This supersedes the launch authorization below; cancellation is not a measured model failure.

**Spatial comparison training authorized; implementation dispatched — 2026-09-25.** User approved one learned old-memory/current-visual comparison before compression versus ordinary continuation from the identical terminal6760, explicitly keeping architecture/research goals and avoiding excessive checks. All13 tasks/35cells and teaching unchanged; all weights trainable at inherited single LR. Candidate adds one12,416-parameter shared local residual comparator, no new memory/gates/oracle supervision. Target2600 additional updates/83,200 episodes per arm; one local sequential worker under a new total24h cap beginning first accelerator profile, equal exposure pinned from measured fit before production. Reuse the working Interactive launchd harness; one focused preflight, no repeated gates. Researcher owns launch; this records authorization/dispatch, not saved optimizer progress. [Brief](../../SecondPass/SpatialComparisonReadout/BRIEF.md).

**Matched temporal-access diagnostic COMPLETE — 2026-09-25.** On512 fresh test episodes/task, final ConvGRU-only auxiliary-structured decoding reaches63.09% orientation/85.74% binding versus deployed49.22%/50.00%. Matched time-separated access reaches61.33%/87.70%; both primary paired BA-difference intervals include zero. Final-only early features reach90.82%/100%, showing external stored activations are not required for the strong early D0 result. Final256 readout yields49.80%/62.50%. Within-layer capacity/supervision matched; across-layer capacity differs. All final-only arms are invariant to deleting/NaN-corrupting earlier feature timesteps. Main model unchanged;70.93s within1200s, worker exited. These are auxiliary-supervised accessibility results, not native-label acquisition or clean retention. [Report](../../SecondPass/SpatialReadout/TemporalAccessDiagnostic/REPORT.md). No further run launched.

**Frozen feature/comparator tests COMPLETE — 2026-09-25.** Native D0 cued orientation and binding each used1024/256/512 independent fit/validation/test episodes. Report-time cue location is100% decodable throughout; sign99.8–100%. Early sample orientations decode at1.33°/1.39°, but late probe precision is weaker (final256 readout30.06°/22.14°). Label-only diagnostic comparator49.22%/56.25% BA has no clear paired rescue over deployed48.63%/50.78%. A separately auxiliary-supervised structured comparator reaches91.60%/100% using predicted inputs only, but has explicit circular-comparison, known-timing and stored-sample-activation privileges: not a label-only or final-state-memory rescue. Main weights unchanged; bounded analysis/report completed360.82s, no further run. [Report](../../SecondPass/SpatialReadout/FeatureDiagnostic/REPORT.md). Earlier dispatch entry is historical.

**Frozen feature/comparator tests authorized and dispatched — 2026-09-25.** User approved locating cue/sign and local orientation accessibility and testing a small feature-only diagnostic comparator on native D0 cued orientation/binding. Terminal6760 remains frozen; independent train/validation/test episodes, no deployed-model training or task changes. One local extraction worker and CPU threads≤2 under a new finite1800-second extraction/fit/report allowance starting at first accelerator work. Launch/results require receipts; this entry records dispatch only. [Brief](../../SecondPass/SpatialReadout/FeatureDiagnostic/BRIEF.md).

**Frozen failure investigation complete — no new training.** Source/native audits and terminal6760 paired diagnostics localize failure to cue-conditioned evidence use, already atD0. Cue-location response biases are strong; sign/evidence flips weakly change decisions. All35 native cells pass bounded adapter/label checks; exercised BPTT paths reach early frames and all learned modules. Sensory-gradient dominance is measured but causal interference remains unproven. No memory/architecture remedy is established. [Consolidated findings](../../SecondPass/SpatialReadout/FailureAnalysis/FINDINGS.md) · [paired diagnostic](../../SecondPass/SpatialReadout/FailureAnalysis/DIAGNOSIS.md).

**2026-09-25 — Triple-length continuation COMPLETE.** All5070 added updates finished, cumulative ConvGRU6760 /216320 episodes; selected5486. Full35-cell selected and terminal final tests verified, supervisor exit0, no failure/cap trigger, worker exited06:37:41 UTC. Terminal motion100%, contour82.81%; selected motion63.28%, contour79.69%. Spatial selection/memory tasks remain near chance in both. Selection remains validation-only; terminal is not retroactively substituted. [Final tables and interpretation](../spatial-readout-convgru.md). Earlier running statements are historical.

**2026-09-24 15:38 UTC — Unchanged ConvGRU continuation RUNNING.** User authorized three times the preceding run as **5,070 additional updates**, target **6,760 cumulative ConvGRU updates /216,320 episodes**, adding390updates/12,480episodes per task. Exact terminal1690 model/name-mapped Adam/scheduler/native streams/RNG/history preserved. Independently loaded first saved update1691 and full-cycle checkpoint1703 /54,496 cumulative episodes; all four ConvGRU tensors changed and Adam advanced. Sole MPS worker88240, external launchd supervisor88230 (`gui/501/org.vawm.spatialreadout.continuation-v2-6fe2d49ea188`), Interactive policy, no automatic restart. New immutable cap: **2026-09-24 15:35:13.140134 UTC → 2026-09-25 15:35:13.140134 UTC**. First13-task cycle210.409s vs178.001s matched repaired-production estimate (1.182×, within1.25× budget margin). Planned optimizer13.967h; conservative total19.075h. Active artifacts: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2`; evidence `continuation_verified.json`. Four future validation looks; no new performance conclusion yet. [Continuation record](../spatial-readout-convgru.md). Earlier completion statements refer to the preserved predecessor.

**2026-09-24 — ConvGRU COMPLETE.** Finished12:01:55 UTC at1690/1690 added updates,54080 fresh episodes (130updates/4160episodes per task), selected1690. Complete35-cell final tests verified; selected and terminal identical and deduplicated. Supervisor exit0, no hard-cap trigger; worker exited. Strong basic orientation/contrast/chromatic/spectral performance, contour64.84% BA, but motion25% and spatial-memory tasks near chance. No demonstrated spatial-memory rescue or matched superiority. [Final results](../spatial-readout-convgru.md). All running/preparation entries below are historical.

**2026-09-24 07:17 UTC — ConvGRU TRAINING; scheduling slowdown repaired.** Exact-state handoff at branch124 changed only launchd Background to Interactive policy. Verified resumed model/Adam/sampler/streams/RNG equality and persisted optimizer progress; step145 /4640 new episodes, checkpoint143. One complete 13-task cycle now matches original profile speed (0.945× profiled time); 12 matched old/new conditions show3.71× speedup. Sole MPS worker14390, launchd supervisor14367. Active artifacts: `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_qos_repair`; original `run` preserved as clean signal stop. Target1690 and deadline13:33:53.442736 UTC unchanged. [Experiment record](../spatial-readout-convgru.md). Earlier status snapshots below are historical.

**2026-09-24 05:47 UTC — REVIEW READY, not production.** All35 MPS profiles completed in661.729s;78 final CPU tests passed. Pinned1690 additional updates /54080 episodes (130updates/task); projected optimizer work5.050h. New immutable deadline13:33:53.442736 UTC, latest conservative launch05:54:40.154780 UTC. Parent must review exact source/config and invoke the ready one-shot launchd command. Disposable migration and changed ConvGRU/Adam checkpoint independently verified; no worker remains active. [Ready marker](../../SecondPass/SpatialReadout/REVIEW_READY.json) · [full record](../spatial-readout-convgru.md). Earlier profiling/running text below is historical.

**2026-09-24 — Final-ConvGRU implementation/CPU checks passed; disposable profiling active, production NOT launched.** The new authorized branch keeps all three KDA encoder modules and replaces only final global recurrence/compression. Verified warm-start source is global-GRU3393 /108576 episodes; the old v4 job is stopped and its cap expired. New/suite/old-harness CPU tests:78 passed; source renderer checks35/35. Native launchd could not read Desktop under TCC, so a byte-verified local runtime copy was prepared under `/Users/jonathanmorgan/VAWMRuntime/final_convgru_01`; launchd ownership/deadline CPU probe passed there. Eight-hour cap began at epoch1790228033.4427361, deadline1790256833.4427361, never renewed. Parent reviews actual source/config after profiling, then launches the one-shot external owner. [Experiment and artifacts](../spatial-readout-convgru.md). Older running labels below are historical, not live state.

**V4 RECOVERED / RUNNING — verified 2026-09-23 08:14:42 UTC.** After normal v3 completion and the preserved pre-activation queue failure, explicit recovery passed all predecessor and unchanged feasibility gates. Independent CPU readback verified exact model/Adam/scheduler/stream/RNG migration from terminal 2860, then checkpoint 2861 with 42 changed model tensors and 42 advanced Adam states; persisted progress reached **2870 / 91,840 episodes**. Parent owns handed-off supervisor **`proc_b345f6fe2730` / PID15739**, sole MPS worker **15763**. The new cap began **08:13:26.181166 UTC** and expires **16:13:26.181166 UTC**, with no renewal. Target remains exactly **2145 added updates / 68,640 episodes**, endpoint **5005 / 160,160**. **66 CPU tests passed**; final evaluations remain pending. Monitor run `recovery_result.json`, `live_status.json` and `latest_checkpoint.json`; original `queue_result.json` deliberately preserves the blocked attempt. [Recovery details](../../SecondPass/JointTraining/RECOVERY_V4.md). All earlier statuses below are historical snapshots.

**2026-09-23 08:12 UTC — V3 COMPLETE; V4 blocked before activation:** v3 completed normally at2860 /91520 episodes, selected2314, complete35-cell selected and terminal finals. Both old processes exited. The v4 queue failed closed on a changed process identity at08:04:36 and had no budget or worker. Explicit recovery is CPU-verified (66tests), with full predecessor verification and unchanged28637.609s mean-estimator fit inside28800s. Launch/progress remains to be verified; earlier queued/running statements below are historical. [Recovery evidence](../../SecondPass/JointTraining/RECOVERY_V4.md).

**New authorization / QUEUED, 2026-09-23 07:46 UTC:** another **2,145 updates / 68,640 episodes**, reaching **5,005 / 160,160**, is durably queued after v3 finishes validation2860, both complete final evaluations and normal OS exit. The unchanged recipe adds 165 updates / 5,280 episodes per task. v3 is still the sole MPS worker (PID51433), last verified in `final_test_terminal` at step2860; it was not interrupted. The new eight-hour cap has **not started**. Parent owns queue **`proc_af5aedcba7d9` / PID11016**, transferred with completion delivery; queue expiry is **09:13:50.062308 UTC**. Receipt and exact-state/budget gates fail closed. No v4 optimizer progress is claimed. CPU tests: **49 passed**, compileall passed. See [v4 protocol](../../SecondPass/JointTraining/AMENDMENT_V4.md) and [queue handoff](../../SecondPass/JointTraining/runs/fresh_kda_joint_01_continuation_v4_8h/handoff.json). Earlier running/completed descriptions below are historical snapshots.


**Documentation update, 2026-09-23:** the [ten-page architecture/microstimulation report](../joint-kda-architecture-microstimulation.md) is complete. Its saved evidence cutoff is **06:48:15 UTC**: live step2647, latest completed validation2314, mean AUC0.69374. It explicitly separates that preliminary snapshot from final results and proposes—but does not run—causal interventions. Existing training scope/deadline are unchanged.

**New authorization / running, 2026-09-23:** the user explicitly requested “go ahead and set up a 8 hour training run for more trsining and more updates”. The versioned v3 continuation resumes **terminal 715**, not historical selected 117. It pins **2,145 additional updates / 68,640 additional episodes** (5,280/task), reaching **2,860 total updates / 91,520 episodes** (7,040/task). One local MPS worker; no architecture, loss, stimuli, sampling, optimizer or batch changes.

Verified live update733 and checkpoint728; hard deadline **2026-09-23 09:08:50.062308 UTC**. Parent owns guardian `proc_5cbdb07600dd` (sole MPS worker51433). Details and automatic final-report location: [joint-training record](../joint-suite-training.md). Earlier completion/no-further-authorization statements below are historical and superseded by this explicit new allowance.


**Completed 2026-09-22:** the fresh local joint KDA reached **715 updates /
22,880 episodes**, finishing normally at 10:04:27 UTC within the original
four-hour cap. Both checkpoints passed complete 35-cell final evaluation.
Terminal 715 learned contrast (100% BA), chromatic increments (99.22%) and
natural spectral detail (92.97%); motion and spatial/sequence tasks remain
near chance. The prespecified minimum-task-first validation rule selected
**117**, not terminal 715. Both are retained and reported separately in the
[completed results](../../SecondPass/JointTraining/RESULTS.md) and
[joint-training record](../joint-suite-training.md). No worker remains authorized
for additional training; no automatic extension or next experiment. The
launch/amendment snapshots below are historical.**

**Latest amendment, 2026-09-22: the user retained the ORIGINAL four-hour
limit. The evaluation-heavy 156-update allocation was a mistake and is
superseded by a progress-preserving continuation, pinned at 715 total updates /
22,880 episodes (1,760/task), not a new budget or model. The
[joint-training record](../joint-suite-training.md) and
[v2 protocol amendment](../../SecondPass/JointTraining/AMENDMENT_V2.md) document
the correction. Resumed training reached 122 updates / 3,904 episodes at
07:49:04 UTC; checkpoint 118 was SHA-verified and loaded, and migration proved
exact model/Adam/scheduler/stream/RNG preservation from step 117. Parent-owned
process `proc_e7bdb8db0d3f` runs one MPS worker. Deadline is unchanged:
10:47:56.585638 UTC (epoch 1790074076.5856378). All 35 cells remain in both
final checkpoint evaluations; scores are pending. More acquisition is not
a convergence or overnight-equivalence claim. This supersedes the
preparation-only status below.**

**Updated 2026-09-22: [unified task-suite assembly](../task-suite-assembly.md) is
complete for the 13 discussed tasks (35 primary cells). The user intends to
train across this suite later from fresh weights; this request authorizes no
training now. All native-renderer/replay CPU checks passed, and a fresh KDA
with all task heads passed forward-only checks without parameter updates.
The existing trained KDA remains an orientation-family model; these checks
are not new trained performance. Next: choose the later joint-training
allocation, reporting/selection implementation, exposure and finite budget.
See [suite documentation](../../SecondPass/TaskSuite/README.md).**

The dated entries below preserve earlier results and decisions; their launch
wording and next-step plans are historical, not active authorizations.

[Next-agent handoff](../../HANDOFF.md) ·
[paper-writing research handoff](../../PAPER_HANDOFF.md).

[Journal index](../README.md) · [Chronology](../CHRONOLOGY.md) · [Open questions](../OPEN_QUESTIONS.md)

**Updated 2026-09-17T12:10-07:00: experiment 27 is complete
([page](../experiments/27-accumulator-conv-stack.md)). On a RunPod 3090, with
everything trainable from scratch and the same recipe, the conv stack with a
gated spatial accumulator state after each scale (ConvGRU or KDA) passes the
from-scratch gate, learns the real orientation task at D0 by curriculum, and
is at ceiling (0.994-1.000) at D4, D12 and D24 on both seeds, where the plain
CNN+GRU control reaches 0.85-0.88 and the opponent traces with learned
retention are seed-dependent at D24 (0.55 / 1.00). This is the first
component shown to beat a plain baseline under the rules. The pod is stopped
and deleted (verified 404); results are local, terminal weights were not
retrieved. Nothing is running. Next: the same program on the other four
families with the ConvGRU model as the working baseline, then a
gating-versus-placement ablation.**

**Updated 2026-09-17T02:20-07:00: experiment 26 is complete for the
orientation family ([page](../experiments/26-plain-baseline-rung1.md)). The plain
CNN + GRU learns every two-way rung from scratch (single Gabor, ring-cued,
sign-glyph; all test BA 1.000 within 100k episodes) and not the three-way real
task (chance after 544k episodes from scratch), but learns the real task at D0
within 12.8k episodes from either two-way parent (test BA 0.998 and 1.000).
Continuing that model on mixed delays keeps D0 at 0.996 and leaves D4/D12/D24
at chance after 200k episodes: retention across blanks is a second, separate
break. Nothing is training. Next: a delay ladder from the D0 model, then the
same ladder for the other four families, then a second seed, before any
architectural component. Uncommitted work: `WorkingMemory/BatteryAudit/`,
`WorkingMemory/PlainBaseline/`, journal pages 25 and 26.**

**Updated 2026-09-16T23:45-07:00: rung 1 of the ladder is running locally
([experiment 26](../experiments/26-plain-baseline-rung1.md)). A plain per-frame
CNN + GRU with everything trainable collapses to a constant output within 150
updates on all five tasks under the recipe as specified (Adam 1e-3, batch 64):
the first Adam step makes the output input-independent, the same signature as
the v2 lineage's checkpoint. Ten recipes (frame stacking, LayerNorm, centred
input, zero head, lr 1e-3 to 3e-5) all leave the full orientation task at
chance in 100k episodes. A difficulty ladder inside the orientation family
locates the break: one uncued Gabor (rotation sign) and four Gabors with a ring
cue (location x rotation) are both learned to BA 1.00 within 40k episodes by
the centred stack-3 recipe at lr 1e-4; the real task adds the sign glyph
(location x sign x rotation) and is not learned. In progress: the sign-only
rung, and a 1M-episode run of the real task. Laptop GPU only; no cloud.**

**Updated 2026-09-16T21:45-07:00: the handoff's section-4 audit is done
([experiment 25](../experiments/25-battery-audit.md),
[WorkingMemory/BatteryAudit](../../WorkingMemory/BatteryAudit/README.md)). Hand-coded
observers reach BA 1.000 on orientation, binding and recognition, 0.954 on
motion duration and AUC 0.999 on Krauzlis at D0 from the rendered frames alone,
so the environments are not mis-specified. Streams are balanced, byte-exact on
resume and identical across processes. One-step gradient diagnostics on the v2
recipe show the clip at 1 scaling every update by 0.36 from step one, with
orientation and Krauzlis taking two thirds of the gradient and motion 5%, and
checkpoint 10240 producing input-invariant outputs with gradients below 0.003
on three tasks. Next: the plain per-frame CNN + GRU baseline, one task at D0,
on the laptop. Nothing is training locally or in the cloud.**

**Updated 2026-09-16T08:20-07:00: the v2 overnight pod `vqpgk21cpi53b6` was
user-stopped at step 10,621 / 424,840 episodes and deleted after complete
verified retrieval (49 checkpoints, all logs, validations 800–10000). Every
task stayed at chance; the user judged the arm a failure. No pod remains on
the account. Key reading: no from-scratch arm on this battery has learned even
binding or recognition, which the warm-started lineage solved, so the
scratch-optimisation regime, not the attention/readout changes, is the
binding constraint. See [experiment 24](../experiments/24-av-context-v2.md). The
local v2 run was also killed at the user's request at step ≈490, before any
validation. Nothing is training locally or in the cloud.**

**Earlier 2026-09-15T22:05-07:00: a local five-change AV-context v2 scratch
run ([experiment 24](../experiments/24-av-context-v2.md)) is training on the
laptop RTX 3070 in
`WorkingMemory/AttentionContextComparator/V2/runs/v2_local_20260915_220124`
(supervisor PID 34688). Changes: restored opponent sign/magnitude channels,
mixed avg+max pooling, a wide source-neutral attention head, non-zero
priority-selection init, and one learning rate for the scratch arm. The
identity gate (v2 minus C/D/E bit-identical to v1), the attention-locality
measurement (1.8e-17 mass beyond 3 cells on trained v1) and the linear probe
(all fields at chance) preceded launch. A pre-registered stopping rule at
validation 2,400 ends the arm if priority entropy and head-1 locality have
not moved. Profile 6.51 s/update; no validation yet.**

**Updated 2026-09-15T22:40-07:00: at the user's request both cloud pods were
stopped and deleted (`6mopzgoioemdp2` v1 AV-context at ≈6,400 updates,
`629g1utqkk8non` dual attention at ≈5,464 updates; account pod list empty).
Pre-deletion retrieval failed on both, so their checkpoints and full metric
logs are lost; the watcher mirrors keep validation results through 5600 and
4800 respectively (see [chronology](../CHRONOLOGY.md)). Neither arm produced a
held-out test. A single replacement pod is being provisioned for a long
overnight run of the v2 model; see [experiment 24](../experiments/24-av-context-v2.md).
The cloud statuses in the older blocks below are historical.**

**Earlier 2026-09-15T04:36Z: a third independent scratch architecture run is
live on pod `f7y0zw02f4fzum`. It uses width-64 pre-update attention, width-64
post-update attention over `C/H/R`, and a terminal `P_T`-only spatial-priority
decoder. Required remote construction/gradient/lineage checks and profiling
passed; production has finite metrics. Its own 45-second watcher and hard-stop
lifecycle do not touch control pod `ce00y2ooosl7wc` or comparator pod
`h1ygacz4zcsfj3`. No validation result exists yet.**

**Earlier 2026-09-15T03:53Z status: the inherited spatial-priority-readout attempt was
cancelled immediately after a lineage correction. Its local monitors were
stopped and Palladio RunPod `mbi39b005jp884` was stopped, deleted, and confirmed
absent. Last known display step was 8720 (320 updates; 12,800 fresh episodes);
no validation completed. The trace is not an architecture result. A separate
fully scratch replacement, pod `ce00y2ooosl7wc`, was provisioned at 03:47Z.
Remote scratch-construction checks and a three-update profile passed. At the
verified launch snapshot it had reached update 46 / 1,840 episodes; no
validation had completed. The older prospective-query run remains
user-stopped, retrieved and deleted.**

## Fully scratch replacement: early-training snapshot

[Pinned scratch model](../../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/model.py) ·
[pinned protocol](../../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/portable_bundle/WorkingMemory/SpatialPriorityReadout/protocol.py) ·
[lineage receipt](../../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/lineage_correction_receipt.json) ·
[launch receipt](../../WorkingMemory/SpatialPriorityReadout/launch_receipt.json) ·
[live status](../../WorkingMemory/SpatialPriorityReadout/runs/scratch_20260915_034720/live_status.json)

Version `spatial_priority_readout_scratch_v2` constructs every model tensor from
documented seeds and starts with an empty optimizer state. It loads zero parent
tensors and zero inherited Adam states. The unchanged target is 4,000 updates /
160,000 five-task episodes. The provider-ready artifact records a requested
RTX 3090 at listed $0.22/hour and deadline `2026-09-15T11:47:20Z`. The
three-update profile averaged3.4266seconds/update with1.014GB peak allocation,
projecting5.51hours/$1.21 including margin and evaluation/retrieval reserve.
At the launch receipt snapshot, worker PID332, lifecycle PID36128 and watcher
PID36812 were active. Latest batch: loss0.82499, aggregate accuracy0.550,
pre-clip gradient norm0.59707. Checkpoint0 was indexed and no
validation had completed. This is live operational evidence, not a result; the
scratch run is not initialization-matched to the prior trained models.

### Complementary local frozen-core result

The [analysis-only local diagnostic](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/report.md)
completed without touching the cloud run. From frozen motion-only step12200,
SHA256 `7cec4c48c3da81b4d4935c65098b58974a38a8ff3a825dfe12e4d8d16e9a0e26`,
a74,303-parameter pooled probe scored23.83% BA/0.4805 AUC and a74,373-parameter
spatial-priority probe24.32%/0.4832 on the same1,024 held-out delay
presentations (256 independent base movies). The paired BA gain was+0.49pp
(95% grouped bootstrap CI -1.76 to+2.64); no delay showed a reliable gain.
Both remain near25% chance.

The spatial map placed13.13% mass in an evaluation-only target region versus
5.33% uniform expectation, but this localization did not decode the motion
winner. This post-hoc negative decoding result neither substitutes for nor
predicts the end-to-end scratch experiment. Frozen checkpoint/state hashes
were unchanged; total local wall time was738.5seconds.

[Held-out predictions](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/predictions_test.jsonl) ·
[priority maps](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/priority_maps_test.npz) ·
[results](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/results.json) ·
[run manifest](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/run_manifest.json) ·
[completion receipt](../../WorkingMemory/SpatialPriorityReadout/LocalDiagnostic/runs/diagnostic_20260915_0350/completion_receipt.json)

## Cancelled spatial-priority-readout attempt

[Experiment 22](../experiments/22-spatial-priority-readout.md) ·
[implementation](../../WorkingMemory/SpatialPriorityReadout/README.md) ·
[historical run artifacts](../../WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605) ·
[superseding cleanup receipt](../../WorkingMemory/SpatialPriorityReadout/runs/priority_20260915_025605/lineage_correction_cleanup_receipt.json)

The launch inherited attention8400 model and Adam state, contrary to the
corrected requirement to initialize the complete trainable model and optimizer
from scratch. The cleanup receipt supersedes its “live” wording and snapshots.
No validation was completed before cancellation.

The sole forward change is the terminal decision readout. Final sensory
`H_T`, firing-rate `R_T` and comparator `C_T` fields remain aligned at 13×13,
are concatenated, and pass through 1×1 then 3×3 convolutions. Each task has a
spatial selection map and local class-evidence maps; a spatial softmax
weighted sum produces logits. Original attention queries and trained
source/locality biases are unchanged, and prospective gamma is absent.

That cancelled migration retained compatible tensors and 111 Adam states.
Those facts describe why it was invalidated; none of those inherited tensors
or states enter the replacement run.

**Historical immutable launch snapshot, not validation:** global step8512 /4,480 fresh
episodes, finite aggregate loss0.8246, accuracy0.45 and gradient norm0.4724.
Three profile updates averaged2.9833 seconds on the RTX4090, with1.032GB peak
allocated. Conservative training plus evaluation/retrieval projection is
about4.89hours/$1.66. The fixed target is step12400 /160,000fresh episodes;
the absolute deadline is `2026-09-15T10:56:06.615367Z`. No validation had
completed at this snapshot.

Detached lifecycle PID38436 and watcher PID30480 previously recorded rolling
aggregate metrics and GPU state. Both were stopped; the pod was deleted and
provider absence was confirmed by HTTP 404. No validation artifact was
produced.

## User-stopped prospective-query experiment

[Experiment 21](../experiments/21-prospective-query.md) · [Completion report](../../WorkingMemory/ProspectiveQuery/completion_report.md) · [Completion receipt](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/completion_receipt.json) · [Cleanup receipt](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/cleanup_receipt.json)

The sole architectural change is a learned scalar current-sensory residual in the pre-update query, initialized at zero:

`Q_t = W_Q(LN(R_{t-1}) + P + e_m + gamma * LN(H_t))`.

Focused tests show exact original behavior at `gamma=0`, a nonzero gamma gradient, query/routing sensitivity when enabled, strict original-attention8400 lineage, and unchanged source/locality terms and five-task protocol. The run stopped at logged step 9430 / 41,200 fresh episodes; durable checkpoint 9216 contains 32,640 episodes. The 214 later logged updates are not checkpointed.

The first setup/profile attempt consumed no training exposure. Its CRLF script and missing BSDS500 runtime payload were corrected without changing scientific settings, the source-hash ledger or the original deadline. The immutable source bundle is `a747fe…afa9`; the separately transferred exact dataset payload is `990e2b…883e`. Retrieval verified 29 artifacts. Pod deletion was confirmed by HTTP 404 and an empty pod list; `currentSpendPerHr` is zero.

Historical control comparisons are valid only at matched validation steps 9200/10000/10800/11600. The prior biased control was stopped before 12400/final held-out tests, so prospective terminal and final held-out scores will be unmatched exploratory results.

**Live validation 9200 (2026-09-15T01:54:44.3853582+00:00):**
[full matched table](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/validation_9200_report.md).
The 1,836 prospective and historical-control examples match exactly.
Prospective improves orientation normalized BA by 0.1719 and binding by 0.3203,
but recognition falls by 0.0799 and motion duration by 0.0260; Krauzlis remains
at chance-normalized zero. Mean task AUC rank rises from 0.6186 to 0.6422 and
the minimum normalized BA from -0.1016 to -0.0260. This is the first planned
validation snapshot, not checkpoint selection or final held-out evidence.

[Local diagnostic summary](../../WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/diagnostic_summary.json): changing only valid cue rings on 32 paired identical-evidence movies changed the frozen baseline prediction in 46.88% (Wilson 95% interval 30.87–63.55%) and changed attention routing. However, accuracy and correct-class probability remained near chance, so this is cue sensitivity rather than successful duration-rule use. A split local pre-pooling linear probe scored 25.98% overall versus 25% chance (23.44% target, 26.82% foil), providing no evidence that this particular final-frame local summary exposes longest-duration direction. It does not prove absence from other times or representations.

## Final shutdown and saved cloud results

[Cleanup receipt](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/cleanup_receipt.json) verifies stop `EXITED`, delete HTTP204 and an empty pod list. [Retrieval receipt](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/retrieval_receipt.json) verifies all73 saved artifacts. The local motion-only experiment is complete. No training, extraction or follow-up GPU experiment should resume without new authorization.

The cloud run is **user-stopped and partial**: logged global11923 /140,920new episodes; durable checkpoint11776 /135,040new episodes;147 logged updates were not checkpointed. Validation-selected so far is10000; last completed validation is11600. **No final held-out evaluation was performed.** These are distinct endpoints.

[Last completed validation11600](../../WorkingMemory/SpatialTaskBattery/BiasedTraining/runs/biased_20260913_194506/report.md): binding100% across delays; recognition N4 96.88%,N12 82.81%,N24 71.88–73.44%; signed orientation42.19–53.12%; motion duration23.44–25%; Krauzlis BA50%. These partial validation scores do not describe the durable11776 checkpoint and must not be presented as final held-out results. [Experiment19](../experiments/19-biased-spatial-battery.md) preserves the full table and history.

## Completed local branch: cued motion duration remains unlearned

[Experiment 20](../experiments/20-single-task-motion.md) · [Final report](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_report.md) · [Completion receipt](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/completion_receipt.json)

All 2,200 added updates / 17,600 motion episodes finished. Selected checkpoint11760 scores 24.80% / 25.78% / 25.00% / 25.00% at D0/D4/D12/D24; terminal12200 scores 25.20% / 25.00% / 25.78% / 25.00%, versus parent 25% at all delays. All paired gain intervals include zero. Terminal AUC is about 0.51–0.52 with severe restriction to a few output classes. This continuation did not acquire the task; failure at D0 cannot be explained solely by blank-period retention.

Losses/gradients stayed finite; last200-update CE was1.3877, close to log(4). The receipt verifies97 immutable artifacts and no remaining local model workers. Training/evaluation took56.4minutes, final analysis59.2minutes, within the120-minute cap. The branch retains current architecture, cues, optimizer and learning rates while training full motion CE. Its failure does not establish a capacity limit, bug or interference mechanism.

The earlier cloud five-task run was stopped at the user’s request and its pod
was subsequently deleted. Its completed motion validation through11600 is also
near chance; [exposure-aligned trajectories](../../WorkingMemory/SpatialTaskBattery/SingleTaskMotion/trajectory_comparison.md)
use different validation draws and must not be treated as paired comparisons.
No extra local GPU training is running or implied by this result; the later
prospective-query and spatial-priority pods have also been deleted.

## Historical context: bias removal and restored-bias launch (now stopped)

[Completed/cancelled experiment 18](../experiments/18-unbiased-attention-spatial-battery.md) · [Replacement experiment 19](../experiments/19-biased-spatial-battery.md) · [Architecture/training implementation](../../WorkingMemory/UnbiasedAttention/README.md) · [Five-task protocol](../../WorkingMemory/SpatialTaskBattery/PROTOCOL.md) · [Rendered task previews](../../WorkingMemory/SpatialTaskBattery/previews/index.html)

In the preceding bias-removal experiment, both arms independently started from attention 8400 and removed explicit source and distance penalties while preserving learned projections and compatible training state. **The completed old-task comparison is negative:** after equal 32,000-episode continuations, single-orientation D24 falls from 91.60% biased control to 57.62% with biases removed; binding D24 falls from 99.41% to 49.80%. Motion D0/D24 becomes 42.38%/42.38%, compared with 41.02%/41.99%; both change intervals include zero.

All five validation looks failed the preservation screen. The selected checkpoint is therefore the untouched **biased parent 8400**, not a trained bias-removed success. The terminal 12400 comparison reports the actual learned competitor. See [full 14-cell paired results](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/comparison.md) and [findings](../../WorkingMemory/UnbiasedAttention/Cloud/runs/cloud_20260913_184802/old_arm_report/findings.json). This is joint source/locality removal under one warm start and fixed budget; it cannot separate the two terms or establish a universal attention rule. Reused tests and cross-GPU training make it exploratory.

The user cancelled the five-task bias-removed arm at logged global 8550 (+150 updates / 6,000 episodes); the last durable checkpoint is 8448 (+48 updates / 1,920 episodes). [Cancellation receipt](../../WorkingMemory/UnbiasedAttention/Cloud/cancellation_receipt.json) verifies worker/supervisor stopped and no GPU compute processes remained. Logged uncheckpointed updates are not a saved endpoint, and this is not a completed acquisition result.

The subsequent original-bias five-task run started from intact attention8400, preserving learned bias terms and compatible Adam state, with fresh task heads. Its launch receipt is historical evidence, not current process status. It was later stopped before its planned160,000 episodes, retrieved and deleted as recorded above. Broader latent/probe analysis remains paused.

## Corrected attention viewer: every timestep (2026-09-14T01:05:05.388626+00:00)

[Open the individual-frame viewer](../../WorkingMemory/AttentionMaps/index.html) · [Journal correction history](../experiments/17-attention-maps.md) · [Per-frame receipt](../../WorkingMemory/AttentionMaps/perframe_receipt.json)

The first promoted figure showed one query and the overview averaged phases; those did not meet the user's request. The correction is now complete: all 13×13 receiving sites appear for **each of two heads and each of two source banks**, with cell values, a fixed 0–1 scale, actual scene/reference images, trial/task/delay selectors and every-timestep slider/playback. **No temporal or trial averaging** is used in the primary view. The optional source-key view averages queries within the selected frame only.

Capture covers 48 movies: eight paired base trials per task family × three families ×D0/D24, totaling 24 base trials and 912 frame observations. It took 6.76 seconds within the original AttentionMaps deadline; no extra budget, model update or intervention occurred. It uses frozen attention 8400, not the newly continued checkpoints. Broader latent/probe work remains paused.

[Cached locality evidence](../../WorkingMemory/AttentionMaps/locality_summary.json) explains why the original one-query map looked like one bright cell: representative orientation frames put about 94% joint attention at the query's own spatial coordinate, under a trained distance penalty about 4.03. This is local source routing, not evidence that the network failed to attend. The earlier phase averages below are historical aggregates, not instantaneous maps.

## Initial attention maps (historical summary)

[Open the interactive viewer](../../WorkingMemory/AttentionMaps/index.html) · [Journal entry](../experiments/17-attention-maps.md) · [Original report](../../WorkingMemory/AttentionMaps/report.md)

This is frozen **attention 8400**, the parent of the latest training comparison. Each of its two heads can read both current-sensory and old-memory sources; the viewer separates both banks within each head, shows condition means, and supports representative receiving-query inspection. A remembered-scene overlay is spatial context, not a reconstruction of memory.

Capture used 192 independent base episodes paired acrossD 0/D24, giving 384 presentations and 208 condition groups. It completed in 30.45 seconds with the checkpoint unchanged. Head 2 memory mass during blanks averages 77.75% for single orientation,84.71% for binding and 85.19% for motion; head 1 is around 12%. These totals show source allocation, not spatial focus or proof of retained task content. See [completion evidence](../../WorkingMemory/AttentionMaps/completion_receipt.json).

## Completed training-allocation comparison

[Stage1 journal](../experiments/16-training-exposure.md) · [Final report](../../WorkingMemory/TrainingExposure/report.md) · [Full analysis](../../WorkingMemory/TrainingExposure/analysis.json)

Both arms start the same attention 8400 checkpoint and complete 4,000 additional updates /32,000 episodes. Control retains 10% motion training; focused allocates 50%. Architecture, tasks, losses and inherited learning-rate policy are unchanged. Control selects terminal 12400; focused selects 11600 but also reports terminal 12400.

| Fresh held-out cell | Parent8400 | Control10% selected/terminal12400 | Focused50% selected11600 | Focused50% terminal12400 |
|---|---:|---:|---:|---:|
| Motion D0 |38.28%|41.02%|65.62%|49.80%|
| Motion D24 |34.57%|41.99%|57.62%|55.08%|
| Single orientation D12 |92.97%|96.88%|90.04%|90.23%|
| Single orientation D24 |80.47%|91.60%|83.98%|84.38%|

Ordinary continuation improves delayed orientation strongly while recovering some motion. Increased motion allocation recovers substantially more motion but gives weaker orientation outcomes. Focused single D12 falls 2.93 pp below parent (paired 95% CI−5.47 to−0.59); control has no observed point declines across 14 cells. Neither result establishes simultaneous all-cell preservation against the 2 pp margin. Preserve both checkpoints rather than naming an all-task winner.

Both have equal terminal exposure; their **selected** exposures differ:32,000 new episodes for control versus 25,600 for focused. The selected-model comparison includes the declared checkpoint selection; the terminal columns provide equal-endpoint exposure. Different local/cloud hardware remains a qualification.

The focused terminal model also develops severe up-choice bias: on the same 512 D0 test trials, down predictions fall 113→2 and up predictions 123→308 from selected 11600 to terminal 12400. AUC also decreases. [The dated instability note](../experiments/16-training-exposure.md) records the evidence; shared-task interference, representation/head mismatch and noisy constant-step continuation are still hypotheses rather than established mechanisms.

## Completed capabilities and their boundaries

| Capability | Evidence | Boundary |
|---|---|---|
| Two-step causal sensory discrimination | Opponent temporal model98.44–100% across7tasks | Does not establish long memory |
| Useful early motion in E/I state | Frozen-state linear readout gains9.47pp | Accessibility can exceed deployed decision quality |
| Motion judgment retained through24blanks | Earlier Retention14800 about79–80% across delays | This is a preserved earlier model/task evaluation, not the new attention model's score |
| Spatial feature-location binding | Spatial4400 bindingD24 93.36%; later models near99% | Full swaps can be detected using one remembered location; no two-item capacity claim |
| Improved native delayed orientation | Attention8400 79.30% versus58.40%continuation; latest control91.60%on its new test | Different experiments/draws must not be treated as one paired comparison |
| Dependence on blank-period memory routing | Frozen exclusion79.30%→50.00% | Does not separate refresh from rejecting blank drive |

The project has useful sensory, memory and attention components, not a complete Guided Search system or a biologically validated architecture. General search, independent multi-item capacity, broad transfer and seed replication remain unestablished.

## What is paused or unrun

The broader [LatentDynamics investigation](../../WorkingMemory/LatentDynamics/RESEARCH.md) was [paused before extraction](../../WorkingMemory/LatentDynamics/queue_receipt.json) at the user's request. Its unspent allowance was reused for the narrower attention maps. No UMAP/t-SNE, latent probes, new training, attention interventions or cloud restart were performed for that capture.

The proposed temporal-feature residual remains Stage2 design only. No optimizer sweep is included. The new authorization is specifically the two bias-removal arms above; it does not restart the paused latent/probe investigation.

## Preserved model ancestry

- Contour-focused PAV 2268 → causal opponent 4032 → broad sequence selected 6860.
- Sequence 6860 → LSTM 5000 and dense E/I 5000.
- E/I 5000 → existing-output refit 9840 → retention 14800.
- Retention 14800 → dense-comparator 4400 and spatial 4400, with a newly initialized spatial core.
- Spatial 4400 → ordinary continuation 8400, additive-feedback 8400 and pre-update-attention 8400.
- Attention 8400 → control 10% selected/terminal 12400 and focused 50% selected 11600/terminal 12400.

Step numbers are experiment-specific and do not imply equal cumulative lineage exposure. [Architecture math and tensor shapes](../ARCHITECTURE.md) describe the unchanged attention architecture.

## Previous experiment execution closure

Local TrainingExposure [exit receipt](../../WorkingMemory/TrainingExposure/runs/exposure_20260913_163542/exit.json): completed without error in 4,546.23 seconds. Cloud [retrieval](../../WorkingMemory/TrainingExposure/Cloud/retrieval_receipt.json) verified 141 files; [cleanup](../../WorkingMemory/TrainingExposure/Cloud/cleanup_receipt.json) deleted the pod at2026-09-14T00:21:33.450146Z, estimated cost$0.165 rather than an invoice. [AttentionMaps completion](../../WorkingMemory/AttentionMaps/completion_receipt.json) records 30.45 seconds and no model changes. No compute allowance is automatically renewed.
