## October 16: static-memory 2x continuation

The active local suite is `SecondPass/MemoryGuidedSearch/LocalRuntime/static_memory2x05`, resumed from update 173,247 toward 193,800 with 2:1 static-memory sampling. At the last verified publication status it reached 175,136; selected best 169,870 has monitoring mean AUC 0.9463159310. Doom is stopped and its cloud resources are deleted. See [publication scope](PUBLICATION_20261016.md) and the repository `AGENTS.md` for current operational instructions. Entries below are historical.

## 19-task search-suite continuation active - October 9, 18:10 PDT

User explicitly requested training from the last completed checkpoint. MemoryGuidedSearch/LocalRuntime/run01 continues VDA best16000, preserving all86 old Adam states and exact parent weights; three new four-location heads are fresh. Shared architecture/loss/Adam3e-4/clip100/batch32micro4/native task laws unchanged. Balanced19 tasks/437conditions, up to19000 additional updates under one eight-hour local allocation from18:08:57PDT, hard deadline October10 02:08:57PDT; no automatic restart/extension/cloud. Supervisor64808/worker52248 independently live-verified. Saved19 CPU-reloaded: all92parameter tensors advanced,92finite aligned Adamstates, all19tasks updated, original parent hash unchanged,29 imported source files copied and hash verified. Initial/best/latest preserve optimizer/scheduler/streams/allRNG state. Eleven contract tests passed. Monitoring uses a prespecified71-cell panel with32trials/cell atbaseline/19/every380/endpoint, preserving equal-task AUC selection; selected model full437-cell validation64 and fresh test256 are attempted within the same cap. Initial panel duration about60seconds; this is an update ceiling, not a promise of completion. Receipt MemoryGuidedSearch/LocalRuntime/run01/independent_production_verified.json. Separate RunPod Doom run is unchanged. Previous preparation-only search entries below are historical.

## Motion-memory search added - October 9

Prepared SecondPass/MemoryGuidedSearch v3 expands to19 tasks/437conditions with a third new four-location head. Motion memory: central random-dot sample, blank delay, then four simultaneous dot patches; choose the matching direction despite independent dot layouts. S3/6/12 x D0/4/12 x H3/6/12 x direction spacing20/45/90 gives81conditions. Coherent speed1.5pixels/frame,12dots,28pixel worlds/circular apertures; no direction/flow metadata enters model inputs or losses. Counterfactual queries preserve entire arrays and initial sample frame while changing only sample motion and correct answer; duration variants use exact prefixes. Motion stream42; prior18 tasks and their204xxx seeds preserved. Total382434parameters/92tensors, all86 parent Adamstates retained. Eleven CPU contract tests passed; current forward/example receipt is in preparation_verified.json. No local training launched or queued. Related neuroscience: Mendoza-Halliday et al.2014, doi10.1038/nn.3785; four-choice search and timings are adaptations. See MemoryGuidedSearch/README.md. Earlier preparation entries below are historical.

## Search suite expanded with pop-out and independent viewing durations - October 9

Prepared SecondPass/MemoryGuidedSearch v2 now has 18 tasks and 356 conditions. Memory-guided search uses independent S1/2/4/8 sample durations, D0/4/12 delays and H1/2/4/8 array durations across three families and two similarity levels (288 conditions). New popout_search uses color differences12/36/72 or orientation differences5/15/45 degrees crossed with H1/2/4/8 (24 conditions), no sample, four-location output. Matched timing comparisons preserve exact scene pixels and answers. Two new 64-to-4 heads add520 parameters; total382174 parameters/90 tensors; original86 Adam states and VDA best16000 are preserved. Fresh204xxx streams, balanced18-task schedule. Eight CPU tests and312 new-condition forward checks passed; exact-frame examples and preparation receipt retained. No local training started or queued. Old16 task laws and completed checkpoints remain unchanged. Earlier search-v1 preparation below is historical. See SecondPass/MemoryGuidedSearch/README.md.

## Memory guided search added to the local suite — October 9

Prepared SecondPass/MemoryGuidedSearch expands the completed16-task suite to17 tasks/80conditions. One new four-location belief head (260parameters) is shared across feature-conjunction objects, procedural silhouettes and spectral fractal textures. Each family has distinct/similar distractors, D0/4/12 blank delays and H2/6 search exposures:36newconditions. Exact novel raster matching; sample2frames, four candidate locations, target always present. Counterfactual helper changes only the sample while keeping search pixels identical. Fresh203xxxstreams include alloldtasks; parentVDA best16000 and86Adamstates preserved, newtotal381914parameters/88tensors. Six CPUcontract tests and all36newcondition forwards passed; examples export everyframe and hashes. No localtraining launched orqueued; architecture/loss ofoldtasks unchanged. SeparateDoom cloudlaunch remains documented in DeadlyCorridor/CloudRuntime. See MemoryGuidedSearch/README.md and preparation_verified.json. Previous experiment statuses below are historical.

## Deadly Corridor cloud training active — October 9, 17:22 PDT

User explicitly authorized starting Doom training on RunPod. Separate cloud run01 on RTX 4090 24GB pod jcfmgww5ubnvax (EU-RO-1, $0.89/hour), network volume hb9ruanesc (20GB, about $1.40/month). Parent is the prepared original suite best54808; VDA/suite originals remain untouched. Prepared discrete SAC with 54-action actor/twin32-quantile critics sharing the exact64D belief representation; encoder trains through critics, actionrepeat4, native skill5 rewards, Adam3e-4/clip100 unchanged. Independent CPU reload verified saved2354 decisions/37updates/35episodes, 62critic+4actor+1temperature finite aligned Adam states, 54 encoder tensors advanced, all34 source/checkpoint hashes verified. Worker945/supervisor944/guard779 live; GPU active. One eight-hour allocation from pod creation17:09:53PDT, hard deadline October10 01:09:53PDT; one-million decision ceiling, no automatic extension. Guard requests clean save then pod shutdown using provider-issued pod-scoped credentials; network volume persists. Four cloud contract tests and production-sized CUDA update passed. Initial runner stopped before learning due duplicate steps key in logging; fixed only log naming, archived failed attempt, deadline unchanged. PyTorch2.8/Python3.12 cloud runtime migration documented; WAD exact and config equivalent after CRLF normalization. No held-out performance claim yet. Evidence: SecondPass/DeadlyCorridor/CloudRuntime/run01/independent_production_verified.json; protocol CloudRuntime/README.md. Previous statements that Doom is unstarted are historical.

## VDA frozen analysis completed — October 9

VDA training completed 16,000 updates; selected best16000 and latest16000 independently verified with 86 finite Adam states. Frozen Analysis/run01 completed all 131,616 presentations (8,064 primary, 4,608 native, 18,144 psychometric, 100,800 exploratory interventions). The requested VDA-4 experiment used a 100% bottom-right cue, 18-degree top-left change, and bottom-right inhibition only at frames 5–6. Inhibition −4 increased top-left detection 48.18%→67.97% (+19.79 pp; paired 95% CI 15.62–24.22), while bottom-right detection fell 86.98%→6.51% and catch false alarms 3.65%→1.82%. Benefit survives Holm correction across eight planned tests. Separate mirrored-scene sweeps show a smaller VDA-9 gain and weak VDA-16 gain. Psychometrics, 35 exploratory arms per task, 90 attention groups/630 timesteps, nine PNG/PDF figures and three GIFs saved. All 1,080 sprites checked against raw pixels/maps; paired pixels, prechange attention, exact first-frame softmax effect and 5,760 global-control presentations verified. Worker66524/supervisor65640 exited0; GPU lock free; source/checkpoint hashes and tensors unchanged. Offline gallery: SecondPass/StabilizedAttentionVDA/Analysis/run01/gallery/index.html. Browser preview blocked by app security policy; static assets/links verified. This deliberately violates native 100%-cue support and tests computational priority-logit interventions, not biological equivalence or training-seed replication. No training or future compute queued; Doom remains unstarted. Evidence: completion_verified.json and viewer_verification.json. Earlier statuses below are historical.

## VDA explicitly resumed — October 9, 09:16 PDT

User requested continuing training. SecondPass/StabilizedAttentionVDA/LocalRuntime/continuation01 resumes run01/latest1280 with exact model/all86Adam states/scheduler/native streams/allRNG state. Original run01 best/latest1280 and frozen source remain unchanged; best1280 stays validation-eligible. Target remains16000 cumulative VDA updates,14720 remaining at restart; no architecture/loss/lr/clip/batch/data changes. Two CPU resume tests passed, next16 task/condition choices and native draws exact; disposable update identical after reload. CUDA RNG restoration verified. Worker44620/supervisor23228 independently live-verified, saved1296/live1310; all86 finite shape-aligned Adam states advanced correctly (shared+16, each head+1). Receipt continuation01/independent_production_verified.json. Only unused original allocation26607seconds is carried forward from activation09:15:14PDT, hard deadline16:38:41PDT October9; no automatic extension/cloud. Next full validation1600, every320/terminal, final256percondition. New stop request belongs in continuation01. Doom remains prepared and unstarted. Previous pause/live entries below are historical.

## Paused at user request — October 8, 21:59 PDT

VDA continuation run01 stopped cleanly after validation1280. Latest1280 and best1280 preserved with86 finite aligned Adam states each, optimizer/scheduler/native streams/all RNG state saved, zero unsaved updates. Worker44924 and supervisor58400 exited0, GPU lock independently free, original checkpoint/source hashes unchanged. No automatic restart or budget extension; resume only on user request. Doom remains prepared and unstarted. Evidence: SecondPass/StabilizedAttentionVDA/LocalRuntime/run01/pause_verified.json. Earlier running descriptions below are historical.

## Current — October8, 21:25 PDT

VDA4/9/16 continuation from successful suite54808 is active: worker44924/supervisor58400, saved16 independently verified with86 finite aligned Adam states and all80 old tensors advanced. Original artifacts unchanged. Balanced16tasks/44conditions, target16000additional updates under8h cap through05:23:14PDT October9. Deadly Corridor is prepared separately, not running or queued. [Protocol and evidence](vda-suite-and-deadly-corridor.md). The completed13-task suite and frozen psychometrics/intervention analysis remain preserved. Earlier entries below are historical.

Continuation independently verified: worker34492/supervisor17136/guard34524; saved13715/live13729. All80 parameter tensors advanced from latest13702; all80 finite aligned Adam states, each head1055updates after13newupdates. Exact restore and original/checkpoint/source hashes verified. Target27404 cumulative, first full validation13832. Deadline1:42:10AM PDT October8, no automatic extension. Evidence StabilizedAttentionSuite/LocalRuntime/continuation01/independent_production_verified.json.

## Explicit continuation authorization — October7

User approved doubling full-suite exposure. Resume StabilizedAttentionSuite run01/latest13702 with exact model/Adam/scheduler/native-stream/allRNG state; add13702 to cumulative27404 (2108updates/task,876928presentations). Architecture/loss/Adam3e-4/clipping100/batch32micro4 unchanged. Separate LocalRuntime/continuation01; original best13000/latest13702/results preserved. Original best stays validation-eligible. One new finite28800second LOCALcap at launch, no cloud/automatic extension. First full validation13832, global520cadence/terminal; Krauzlis monitoring global130cadence; final11192 fresh test namespace199104. CPU restoration and native-next-draw equality passed, both original checkpoints have80finite aligned Adamstates. Actual activation requires new runtime receipts. See SecondPass/StabilizedAttentionSuite/Continuation/README.md.

Production independently verified: StabilizedAttentionSuite worker41092/supervisor43836/guard13444; saved13/live40, all80 finite shape-aligned Adam states CPU-reloaded. Shared tensors have13updates and each task head1; all13tasks updated. Frozen source hashes and original Krauzlis best/latest hashes match. Profile pinned13702updates/438464fresh presentations/1054updates per task. First full validation130, then every520 and terminal; Krauzlis monitoring every130. Deadline4:19:41PM PDT October7, no automatic extension. No held-out learning result yet. Receipt SecondPass/StabilizedAttentionSuite/LocalRuntime/run01/independent_production_verified.json.

## Latest authorization — stabilized attention across the full suite, October7

User explicitly requests training the minimally changed stabilized model across the entire suite. New SecondPass/StabilizedAttentionSuite imports the original backbone unchanged and replaces only the final linear output with13 task-selected heads (381264parameters/80tensors). Fresh whole model; native13tasks/35conditions and photo splits, fresh199xxxstreams. Equal-task/within-task-condition balanced fresh32-trial batches replace single-task replay; Adam3e-4 FP32micro4, CE+.05pixelprediction and clipping100 unchanged. One new finite28800second LOCALcap starts35-condition profile; requested17160updates prospectively pinned to13-update cycles. Full validation130/every520/terminal, Krauzlis monitor every130; final11192. No cloud/automatic extension/extra arm. Exact backbone equivalence and13-task CPU optimizer checks passed. Preserve original best1200/latest1250 and all analysis. Actual activation requires runtime receipts; see LabJournal/stabilized-attention-suite.md.

## Frozen psychometrics and interventions complete — October7

StabilizedAttentionKrauzlis best1200 frozen analysis complete; no training. Fresh native3000: BA.857412 (bootstrap95% .844859–.869958), AUC.936479. Nine angle psychometrics0–60degrees and16,416 matched intervention presentations across96 scene clusters. Target inhibition dose-4/-12 reduces trained-angle hit86.46%→28.13%/2.08%. Distractor stimulation+12 reduces target hits92.71%→26.04%, increases distractor false alarms20.83%→87.50%; BA86.28%→31.82%. Target stimulation/distractor inhibition no binary-choice improvement in sampled26/28trials; cue-only effects smaller; off-site stimulation captures attention and yieldschanceBA. Priority-map logit interventions, not biological microstimulation equivalence; one checkpoint/fixed-left site; static/motion generality not established. Four figures, raw trials, curves, paired cluster intervals and report saved under SecondPass/StabilizedAttentionKrauzlis/Analysis/run01. Worker/supervisor exited0, GPUlockfree, best/latest hashes unchanged, no future compute queued. See REPORT.md and completion_verified.json.

## Latest authorization — frozen psychometrics and attention interventions, October7

User requests psychometric curves and attention stimulation/inhibition analysis. Training remains stopped. Analyze preserved StabilizedAttentionKrauzlis best1200 with all weights frozen and no optimizer. New Analysis/run01 has one finite2700second local analysis cap including native3000test and16,416 matched intervention presentations; no training/cloud/automatic extension. Original renderer unchanged; local adapter exposes angle/sign after consuming original RNG draws, native equivalence and zero-bias checks passed. Fresh198103 scene seeds,96 clusters balanced across3 baseline lengths/cue sides/signs; angles0/4/8/12/18/26/28/40/60, target/foil/catch separately. Fixed-left priority-logit biases +/-4,+/-12 during motion, cue-only+/-12, off-site+/-12, sham; identical pixels across interventions. Report observed psychometrics, catch/foil false alarms, catch-based dprime/criterion, and paired cluster bootstrap uncertainty. These are model priority-map interventions, not biological microstimulation or neural silencing. Native frozen3000test already completed BA.857412/AUC.936479; full intervention analysis pending. Sources/checkpoints preserved. See SecondPass/StabilizedAttentionKrauzlis/Analysis/run01.

## User-directed stop — October6

StabilizedAttentionKrauzlis stopped at logged update1250 at user request. Worker26664/supervisor26848 exited, guard normal_stop, GPU lock free. Best1200/latest1250 independently CPU verified with56 finite aligned Adam states each; 0 unsaved updates. Last validation1200: BA0.855093, AUC0.932545. No final3000test; incomplete user-stopped run, not a crash. Preserve checkpoints/source/logs. No automatic restart, additional training or analysis queued. Details StabilizedAttentionKrauzlis/LocalRuntime/run01/cancellation_receipt.json.

Production independently verified: StabilizedAttentionKrauzlis worker26664, saved1/live3, all56finite aligned Adam states CPU-reloaded; source hashes match and original checkpoints unchanged. Target1650updates/50000presentations/25000unique prospectively pinned by native profile. First validation33, then every50; no held-out learning result yet. Deadline12:26:09AM PDT October7, no renewal. Supervisor26848, independent guard43868.

## Latest authorization — stabilized attention, October6

User requests implementing suggested normalization/attention stabilization plus global gradient clipping100 and starting local training. New SecondPass/StabilizedAttentionKrauzlis: fresh379444parameters/56tensors; native task, single shared map and local comparisons unchanged. Separate per-patch gate/candidate LayerNorm in spatialGRU and CLSGRU; projected QK L2 with bounded scale1..16(initial8); alpha/(alpha+1/N) gates spatial update, replacing100*alpha input amplification. Adam3e-4 unchanged; global gradient clipping100 after batch accumulation, no gradient renormalization afterward. Model activation normalization is forward-pass. Fresh197xxxstreams, batch32micro4 FP32,1000movies/twoepochs. Requested2244updates prospectively pinned in66update pools; validation33/every50/terminal300trials, final3000. One new finite14400second LOCALcap starts native profile, includes evaluation; no automatic extension/cloud/additional arm. CPU changes verified; source frozen. Original best600/latest1700 and all diagnostics preserved. This combined intervention is not a one-factor causal ablation. No extra mnemonic processing. Launch and actual production require current runtime receipts. See candidate README and LabJournal/stabilized-attention-krauzlis.md.

## Stability investigation complete — October6

Original LocalComparisonKrauzlis run is interrupted at live1737; best600/latest1700 preserved, original worker/supervisor exited, cause unknown, no final3000test or completion receipt. Frozen600 passes600new diagnostic movies at75.7%BA/AUC.858. Collapsed1700 attention locks near fixation: target mass88.2%→0.69%. Swapping good attention into collapsed weights restores75.7%BA; opposite swap50%; fixation-only removal70.8%. Backward-only attention/memory edge tests reduce gradientnorm308→3.54or2.08 with identical logits. Three60-update matched continuation experiments: original lr3e-4 freshBA75.0%,lr3e-5 fresh78.2%,clip1at3e-4 fresh75.7%. Standard replay didnot collapse, so results indicate sensitivity, not a proven long-term fix. All diagnostics finished under finite3600s cap, originals hash-unchanged,43finiteAdamstates at660 per endpoint, GPUlockfree. No production restart/extension/future run authorized by this completion. Details LabJournal/local-comparison-krauzlis.md; Stability/run01/summary.json. Candidate continuation lower_lr/latest.pt is a recommendation only.

Production independently verified: LocalComparisonKrauzlis worker18300, saved1/live12,43finite aligned Adam states CPU-reloaded; frozen source hashes match. Fresh2244update target,68000presentations/34000unique. First validation66,then every100. Original deadline12:29:23AM PDT October6, no renewal. No held-out result yet.

## Latest authorization and launch — explicit local comparisons on Krauzlis

October5 user approves proceeding with targeted successor after frozen-model diagnostics. LocalComparisonKrauzlis fresh378675parameters/43tensors retains native task and one shared appearance/motion spatial-attention map. Adds explicit causal per-location reference8/recent4/gap4 population comparisons, selected by the same alpha and retained as735channels through a.8decay trace into the final classifier. No angle/event/cue-location metadata or privileged local probe labels enter training. SpatialGRU/CLSGRU retained; no inherited checkpoints. New196xxxstreams,FP32Adam3e-4,batch32micro4,1000movies/twoepochs. Requested6600updates prospectively pinned to whole66update pools; validate66/every100/terminal,300trials; final3000. Best/latest only. New finite14400second LOCALcap from2026-10-05T20:29:23.218975-07:00 through2026-10-06T00:29:23.218975-07:00, includes profile/all evaluations; no extension/cloud/other arm. CPU checks passed43gradients/states/tensor changes and causal-window/map tests. Source11files frozen; supervisor20944 launched. Tiny-fit8movies passed74updates; weights discarded. Actual production requires saved optimizer receipt. Old model remains user-cancelled and preserved. See LabJournal/local-comparison-krauzlis.md and SecondPass/LocalComparisonKrauzlis/LocalRuntime/run01.

## Diagnostics completed — October5

Original training remains stopped. Frozen-model probes found cue side100% decodable from spatial memory, local motion change81.4% BA before fusion/attention, final CLS change52.9% BA, and composed cue+motion probes82.9% BA on600 fresh movies. Probes use explicit event windows/local supervision: diagnostic, not an end-to-end solution. Clean simplified model100% BA; slowing clean movies to.375 preserves AUC.998 and yields97.7% BA after separate calibration. Strongest evidence points toward the recurrent comparison/selection path, not unreadable cue or absence of motion signal. All diagnostic processes exited, original checkpoints unchanged, GPU free; no successor queued. Detailed report: LabJournal/shared-attention-krauzlis.md; receipts Diagnostics/run01.

## User-directed stop — October5,7:55PM PDT

User explicitly cancelled SharedAttentionKrauzlis. Trainer30284 and CPU evaluator12188 stopped and identities verified exited. Last live update1339; preserved best1000/latest1300, all43 finite aligned Adam states independently CPU-verified. Latest completed monitoring validation1300: BA0.500000, AUC0.509180. No final test; incomplete user-cancelled run, not a spontaneous crash. No automatic restart, successor experiment, diagnostic training, queue or budget extension. Preserve source/checkpoints/logs. Runtime cancellation_receipt.json records ownership and stop evidence. Earlier active descriptions below are historical.


October5, after validation1000: user objects to the long gap before the next result. Added an independent CPU-only saved-checkpoint evaluator every100updates (roughly5-6minutes), using the identical300-trial validation set and originaldeadline. Below-normal-priority monitor12188, oneCPUthread; training process30284 remains unchanged. Results in LocalRuntime/run01/frequent_validation; these monitoring results do not change production best-checkpoint selection. No extra training, GPU worker, new budget, or automatic extension. Initial evaluation cross-checks CPU scores against the existing CUDA validation when available.

Production independently verified: worker30284, live step9, saved step1, all43 finite aligned Adam states CPU-reloaded; frozen source hashes match. Target2574updates/78000presentations/39000unique, deadline10:36:52PM PDT. No held-out result yet.

## Current — native Krauzlis shared-attention experiment, October5

CorrespondencePredictiveMemory is user-cancelled (live4324; retained best64/latest4250), with supervisor/guard exited. Its older active descriptions below are historical. SharedAttentionKrauzlis local supervisor22308 launched with one explicit shared spatial-attention head, native cue/distractor movies and spatialGRU/CLS memory. CPU checks passed; tiny eight-movie training check reached100% at80updates and its weights are discarded. Production requires the saved optimizer receipt; see SecondPass/SharedAttentionKrauzlis/LocalRuntime/run01. Finite4h cap starts6:36:52PM PDT, ends10:36:52PM PDT, no extension. Attention/stimulation analysis awaits held-out task learning.


Actual fresh production verified6:02PM PDT: worker37408, savedstep64, all43 finite aligned Adam states independently CPU-reloaded; nativeprofile pins25024 updates. Budget began6:01:32PM PDT and ends10:01:32PM PDT. Runtime root_startup_verification.json.

## October5,6:01PM PDT — explicit immediate handoff

User cancelled OutcomeContrastiveRViT; savedbest5000/latest24750 preserved and
CPU verified. Current successor CorrespondencePredictiveMemory supervisor14972
launched locally; native profile precedes production. See runtime receipts for
actual progress. Queue15136 cancelled in favor of the direct user-authorized launch.

## October5,5:55PM PDT — next model queued

OutcomeContrastiveRViT remains active; saved live22354/25024, latest validation
20000 BA50.26%/AUC0.50434, strong representation-collapse signal.
CorrespondencePredictiveMemory implemented/CPU checked and queued as the next
local model. Fixed pixel baseline97.92%/0.99590 on384 diagnostic trials is not a
trained model result. See [journal](correspondence-predictive-memory.md).
Runtime receipts take precedence over this timestamped snapshot.

# Current status — October 5, 2026

## Active training: outcome-contrastive RViT with spatial GRU

Architectureclarificationresolved byuser: patch-basedspatialGRUafterViT with
recurrentfeedback andcorrespondingCLSmemory. ImplementedfromexistingConvGRU
equations; all1,316,994parametersfresh. Sixrawframes,full-batchoutcomeSupCon+CE.
Supervisor13356launched5:06:05PM PDT; productionworker16140 is training.
The8-trialcheckreached100%trainingaccuracy/CE0.04289at35updates; thoseweights
were discarded. Nativeprofile pinsfull25024updates. Root independentlyCPUverified
best64/latest500, all36alignedfiniteAdamstates andfiniteweights; source/profile
match andactualworkeridentityverified. Deadline9:06:08PM PDT. Noheld-outresult
yet; tiny-setmemorizationisnotgeneralization. [Journal](outcome-contrastive-rvit.md).

## Latest completed: faster-dot transformer

**Completed4:36PM PDT October5:** all25,024updates /391,000unique movies /
782,000presentations. Finalheld-outBA49.772%, AUC0.498358, BCE0.693149 remains
chance acrossall sixcells. Selected10000/latest25024 CPU-verified24alignedfinite
Adamstates/finiteweights. Recovery lasted28min20sec; guard exitednormally and
alloldworkersgone. The separatelyauthorizedoutcomecontrastiveRViT is nowactive
above. [Result](weighted-sum-transformer-speed4.md).

## Original proposal October 5: outcome-contrastive RViT (historical)

User requests six raw frames, recurrent visual CLS and memoryCLS, terminalchoice
feedback and full-batch contrastive loss. Proposed outcome-derived SupCon+decision
objective is implemented and CPUchecked, including cross-chunk batch gradients.
Theinitialarchitecturequestionisnowresolved: useexistingConvGRUequations after
ViT andacorrespondinglinearCLS-GRU. Modelimplemented/trainingasrecordedabove.
[Design and sources](../SecondPass/OutcomeContrastiveRViT/README.md).

## Historical recovery launch: faster-dot transformer resumed October 5

October5: user explicitly requests continuation after computer shutdown. The
original workers are gone; checkpoint10000 (best/latest) is independently
verified with24alignedfinite Adam states, finiteweights, CUDA/global/shuffleRNG,
pool156/epoch0/cursor512. Lastlive10200 includes200unsaved updates to rollback.
Same cumulative25024target, same speed4/data/model/optimizer, no new arm.
Originalwallcapexpired; explicit recovery usesone finite3600-second localwindow
includingeval/reporting, preservingoldbudget/source/config. **Training locally:**
supervisor33412, worker15684, guard20572, launched4:07:42PM PDT; harddeadline
5:07:42PM PDT. Root independentlyCPU-verified saved10250 with24alignedfiniteAdam
states, finiteweights, exactRNGrestore and originalscientificconfig/budgetunchanged.
Boundedactualproduction CPU fixtures matched uninterruptedweights/Adam/sampler/RNG
bitwise at mid-epoch and endpool interruptions. Same cumulative25024target;
no final score yet. Recovery evidence in `Recovery/resume01`.
See [recovery journal](weighted-sum-transformer-speed4.md).

## Original launch: faster-dot transformer rerun (historical)

User requested increased dot speed and a rerun. **Training locally:** speeds
1.5/4/8 pixels per frame (4x original), same architecture, 26/28-degree changes,
labels/nuisances, initialization/shuffle seeds, optimizer, exposure and evaluation
schedule. Native profile pins all25,024 updates /391,000unique movies /
782,000presentations. Supervisor35380, worker41532; independently CPU-loaded
best64/latest250 with all24 aligned finite Adam states, finite weights and saved
speed_scale4. Actual worker identity and source/profile match verified.
No final score yet. New finite cap10:06:37PM October4 through2:06:37AM October5
PDT; no cloud or automaticextension. Previous completed-run statements below
describe their endpoints, not current process state. See [live journal](weighted-sum-transformer-speed4.md).

## Latest completed experiment: direct weighted-image transformer

**Completed October 4, 9:45 PM PDT:** all 25,024 updates / 391,000 unique movies / 782,000 presentations. Selected checkpoint 5,000 by validation BCE; latest 25,024 preserved. Final 3,072 test presentations / 1,536 paired contexts: **BA 49.25%, AUC 0.492292 — chance**. Zero-logit BA 49.77%; all six cells remain near chance.

The fresh 348,929-parameter model used the same chronological 1/2/3 weighted RGB image, 10x10 patches, one shared four-head transformer layer, a CLS token and a nonlinear classifier on ordered concatenated CLS outputs. Every learned parameter trained together on BCE; there was no VAE or direction/angle target. It did not solve the task at this exposure.

Both checkpoints independently CPU-verified: all 24 Adam states aligned to best5,000/latest25,024 and finite weights/optimizer values. Worker16652, supervisor32432 and guard39176 exited normally. All earlier artifacts preserved; the separately authorized faster-dot rerun is active above. See [journal](weighted-sum-transformer.md), [final report](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/FINAL_REPORT.md), [results](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/final_test.json) and [verification](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/root_final_verification.json).

Post-run read-only audit: severe CLS attention concentration was measured in both selected/latest transformer checkpoints, including one head attending almost entirely to CLS. Checked labels, renders, parameter updates and saved inference agree. Small-angle protocol choices and the missing tiny-dataset learning test remain material limitations; no cause or remedy has been established and no further training was launched. See [audit](../SecondPass/WeightedSumTransformer/LocalRuntime/run01/Audit/REPORT.md).

## Preceding experiment: weighted summed-image VAE and nonlinear decoder

**Completed October4,8:36PM PDT:** full25,024 VAE updates and10,000 classifier updates. Selected VAE25,024 andclassifier1,000. Final3,072 presentations/1,536 paired contexts: **change BA49.71%, AUC0.500107 — chance**.

The VAE itself did not preserve the weighted dot image: reconstructions are nearly uniform gray. Output variance is about0.125% of target variance; KL1.4867e-8 is consistent with posterior collapse. Image MSE is only marginally better than gray/train-mean baselines. This failure precedes the change decoder and does not isolate an intrinsic limit of the1/2/3 weighted representation.

All four best/latest checkpoints independently CPU-verified with20/6 aligned finite Adam states and finite weights. Frozen VAE immutability passed. Supervisor33384, worker40436 andguard42668 exited normally. No training is active or additional experiment queued for the night. All earlier artifacts remain preserved. See [journal](weighted-sum-vae.md), [final report](../SecondPass/WeightedSumVAE/LocalRuntime/run01/FINAL_REPORT.md), [results](../SecondPass/WeightedSumVAE/LocalRuntime/run01/final_test.json) and [reconstruction examples](../SecondPass/WeightedSumVAE/LocalRuntime/run01/weighted_reconstructions.png).

## Preceding experiment: three-context/three-prediction VAE

**Completed October4,7:22PM PDT:** all20,032updates /313,000unique twelve-frame movies /626,000presentations. Unsupervised validation selected terminal20,032. Final3,072change presentations in1,536paired nuisance contexts: **50.42% balanced accuracy, AUC0.504076**. Posterior/projection distance scores also remainednear chance (50.68%/50.59%BA). The tested recipe didnot solve simplified motion-change detection.

The model reconstructs three observed frames, recursively predicts three future frames withfull temporal gradients, anduses VICReg on two same-context augmented views. No direction/change labels enter optimization. Separate held-out support-balanced predictionerror improvesovercopy-last, but ordinaryMSE isworse andsavedforecasts arediffuse/blurred. Loss improvement alone isnot useful motionprediction evidence.

Both final checkpoints were independently CPU-loaded: finiteweights,all38Adam states alignedto20,032. Allworker/queue/guard processes exited. The original8:11:07PM PDT deadline wasrespected; no new training/cloud job waslaunched. The cancelledplain-VAE control andfailedfirstpredictive64-update attempt remainpreserved.

See [finalreport](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run02/FINAL_REPORT.md), [finalresults](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run02/final_test.json), [journal](predictive-consistency-vae.md) and [ten-page implementation specification](../output/pdf/predictive_consistency_vae.pdf). Live snapshots in the journal arehistorical; these finalheld-out results arecurrent authority.

## Previous pixel-derived VAE experiment

The pixel-loss arm completed all **25,024 updates**, selecting its terminal checkpoint. Final 3,072 presentations / 1,536 paired contexts: posterior distance BA 50.46%, AUC 0.50940; trained projection BA 51.27%, AUC 0.50594; raw-pixel scoring with validation calibration BA 60.84%, AUC 0.55002. This is a negative result for pixel-distance matching on the simplified distribution.

The user cancelled its plain-VAE control to proceed with the predictive model. Saved update 7,250 and all 20 Adam states are preserved; last live update was 7,268. No final control test was run, and the matched two-arm comparison is incomplete. See [journal](pixel-contrastive-vae.md), [joint final report](../SecondPass/PixelContrastiveVAE/LocalRuntime/run_20261004_2107/pixel_contrastive/FINAL_REPORT.md) and [cancellation receipt](../SecondPass/PixelContrastiveVAE/LocalRuntime/run_20261004_2107/vae_control/cancellation_receipt.json).

[Index](README.md) · [History](RESEARCH_HISTORY.md) · [Catalog](EXPERIMENT_CATALOG.md) · [Chronology](CHRONOLOGY.md)

## Preceding completed angular experiment

**AngularContrastiveMotion: completed, simplified comparison solved.** Fresh whole residual CNN, three ordered frames per clip, 128-dimensional normalized embedding, simulator-angle-supervised spring loss. Training completed at 12:06 PM PDT after 3 h 28 min 33 s: 25,024 updates, 782,000 pair presentations, 391,000 unique training pairs. Best checkpoint 19,000; latest 25,024. Both were independently loaded with all 64 Adam states at their saved steps.

The validation-selected distance threshold achieves 100% balanced accuracy and AUC 1.000 on 3,072 fresh comparison presentations: 512 per speed/angle cell, or 1,536 paired nuisance contexts total. Angular test loss is 0.00136432; distance/angle correlation is 0.987835. See the [full technical report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md), [results](../SecondPass/AngularContrastiveMotion/FINAL_REPORT.md) and [journal](angular-contrastive-motion.md).

This is one training seed on a new simplified persistent-dot distribution with explicit angular supervision. It does not establish native Krauzlis cue selection, distractor rejection, dot-lifetime robustness or recurrent memory.

## Preceding experiments

| Experiment | Final state | Main result |
|---|---|---|
| Frozen predictive encoder + FFN | Completed all 10,240 updates | 50% BA, AUC 0.49519, CE 0.693165; every test decision was no-change |
| Variational next-frame predictor | Completed 50,240 cumulative updates | Fresh test balanced MSE 0.00029275 vs copy-last 0.00385410 |
| Weighted-mean RViT, two epochs per pool | Completed 2,310 cloud updates; retrieved and pod deleted | Selected BA 50%, mean AUC 0.50397 |
| VAE-encoder RViT input×10 | Completed 330 local updates | BA 50%, mean AUC 0.49000 |
| Three-frame spatial VAE | Completed 9,900 updates | Successful reconstruction test; best 9,500/latest 9,900 retained |

## Runtime and next decision

The pixel-loss VAE arm completed; its plain control was cancelled. The revised predictive VAE completed as recorded above. The earlier angular-contrastive queue completed; it is not pending. The latest ephemeral cloud pods were retrieved and deleted. This experiment includes no cloud rental or automatic compute extension.

The current research question is whether image prediction and view consistency learn useful surprise/change information without explicit angular targets. Native-stimulus transfer remains untested. Preserve the successful encoder, its checkpoints and all negative evidence before choosing a transfer experiment.

## Historical status records

[Archived status snapshots through October 4](archive/CURRENT_STATUS_through_20261004.md) preserve previous launches, cancellations and recoveries. Their present-tense wording was valid at the recorded time; it is not current runtime authority. Individual run reports and receipts remain the source for each result. Earlier lineages are cataloged rather than silently treated as fresh training.
