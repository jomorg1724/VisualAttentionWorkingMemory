## 19-task search-suite continuation active - October 9, 18:10 PDT

User explicitly requested training from the last completed checkpoint. MemoryGuidedSearch/LocalRuntime/run01 continues VDA best16000, preserving all86 old Adam states and exact parent weights; three new four-location heads are fresh. Shared architecture/loss/Adam3e-4/clip100/batch32micro4/native task laws unchanged. Balanced19 tasks/437conditions, up to19000 additional updates under one eight-hour local allocation from18:08:57PDT, hard deadline October10 02:08:57PDT; no automatic restart/extension/cloud. Supervisor64808/worker52248 independently live-verified. Saved19 CPU-reloaded: all92parameter tensors advanced,92finite aligned Adamstates, all19tasks updated, original parent hash unchanged,29 imported source files copied and hash verified. Initial/best/latest preserve optimizer/scheduler/streams/allRNG state. Eleven contract tests passed. Monitoring uses a prespecified71-cell panel with32trials/cell atbaseline/19/every380/endpoint, preserving equal-task AUC selection; selected model full437-cell validation64 and fresh test256 are attempted within the same cap. Initial panel duration about60seconds; this is an update ceiling, not a promise of completion. Receipt MemoryGuidedSearch/LocalRuntime/run01/independent_production_verified.json. Separate RunPod Doom run is unchanged. Previous preparation-only search entries below are historical.

## Motion-memory search added - October 9

Prepared SecondPass/MemoryGuidedSearch v3 expands to19 tasks/437conditions with a third new four-location head. Motion memory: central random-dot sample, blank delay, then four simultaneous dot patches; choose the matching direction despite independent dot layouts. S3/6/12 x D0/4/12 x H3/6/12 x direction spacing20/45/90 gives81conditions. Coherent speed1.5pixels/frame,12dots,28pixel worlds/circular apertures; no direction/flow metadata enters model inputs or losses. Counterfactual queries preserve entire arrays and initial sample frame while changing only sample motion and correct answer; duration variants use exact prefixes. Motion stream42; prior18 tasks and their204xxx seeds preserved. Total382434parameters/92tensors, all86 parent Adamstates retained. Eleven CPU contract tests passed; current forward/example receipt is in preparation_verified.json. No local training launched or queued. Related neuroscience: Mendoza-Halliday et al.2014, doi10.1038/nn.3785; four-choice search and timings are adaptations. See MemoryGuidedSearch/README.md. Earlier preparation entries below are historical.

## Search suite expanded with pop-out and independent viewing durations - October 9

Prepared SecondPass/MemoryGuidedSearch v2 now has 18 tasks and 356 conditions. Memory-guided search uses independent S1/2/4/8 sample durations, D0/4/12 delays and H1/2/4/8 array durations across three families and two similarity levels (288 conditions). New popout_search uses color differences12/36/72 or orientation differences5/15/45 degrees crossed with H1/2/4/8 (24 conditions), no sample, four-location output. Matched timing comparisons preserve exact scene pixels and answers. Two new 64-to-4 heads add520 parameters; total382174 parameters/90 tensors; original86 Adam states and VDA best16000 are preserved. Fresh204xxx streams, balanced18-task schedule. Eight CPU tests and312 new-condition forward checks passed; exact-frame examples and preparation receipt retained. No local training started or queued. Old16 task laws and completed checkpoints remain unchanged. Earlier search-v1 preparation below is historical. See SecondPass/MemoryGuidedSearch/README.md.

Current local suite extension: [Memory guided search](MemoryGuidedSearch/README.md), prepared17tasks/80conditions; no localtraining launched. [Doom cloudrun](DeadlyCorridor/CloudRuntime/README.md) is separate. Prior status entries below are historical.

Current: [VDA frozen analysis complete](StabilizedAttentionVDA/Analysis/run01/gallery/index.html). Training finished at 16,000 updates. All 131,616 analysis presentations verified; the requested VDA-4 reciprocal effect is present. Offline gallery, psychometrics, attention maps and interventions are ready. Browser preview was policy-blocked; static exports verified. No training queued; Doom remains unstarted. Earlier status entries below are historical.

Current: VDA continuation is running again from1280 in LocalRuntime/continuation01; saved1296 independently verified October9. Same16000-update target and unused prior local allocation. Doom remains prepared only. Previous pause entries below are historical.

Current: VDA continuation is paused at1280 at user request; checkpoints and full training state preserved. Doom remains prepared, not running. No automatic restart. See the VDA run01 pause_verified.json receipt. Earlier active descriptions below are historical.

Current: [StabilizedAttentionVDA](StabilizedAttentionVDA/README.md), continuation from successful54808 across16tasks/44conditions, production verified at16updates. [Deadly Corridor](DeadlyCorridor/README.md) is prepared separately, not running or queued. The completed13-task suite and frozen analysis remain intact. [Experiment record](../LabJournal/vda-suite-and-deadly-corridor.md).

## Frozen psychometrics and interventions complete — October7

StabilizedAttentionKrauzlis best1200 frozen analysis complete; no training. Fresh native3000: BA.857412 (bootstrap95% .844859–.869958), AUC.936479. Nine angle psychometrics0–60degrees and16,416 matched intervention presentations across96 scene clusters. Target inhibition dose-4/-12 reduces trained-angle hit86.46%→28.13%/2.08%. Distractor stimulation+12 reduces target hits92.71%→26.04%, increases distractor false alarms20.83%→87.50%; BA86.28%→31.82%. Target stimulation/distractor inhibition no binary-choice improvement in sampled26/28trials; cue-only effects smaller; off-site stimulation captures attention and yieldschanceBA. Priority-map logit interventions, not biological microstimulation equivalence; one checkpoint/fixed-left site; static/motion generality not established. Four figures, raw trials, curves, paired cluster intervals and report saved under SecondPass/StabilizedAttentionKrauzlis/Analysis/run01. Worker/supervisor exited0, GPUlockfree, best/latest hashes unchanged, no future compute queued. See REPORT.md and completion_verified.json.

## User-directed stop — October6

StabilizedAttentionKrauzlis stopped at logged update1250 at user request. Worker26664/supervisor26848 exited, guard normal_stop, GPU lock free. Best1200/latest1250 independently CPU verified with56 finite aligned Adam states each; 0 unsaved updates. Last validation1200: BA0.855093, AUC0.932545. No final3000test; incomplete user-stopped run, not a crash. Preserve checkpoints/source/logs. No automatic restart, additional training or analysis queued. Details StabilizedAttentionKrauzlis/LocalRuntime/run01/cancellation_receipt.json.

Production independently verified: StabilizedAttentionKrauzlis worker26664, saved1/live3, all56finite aligned Adam states CPU-reloaded; source hashes match and original checkpoints unchanged. Target1650updates/50000presentations/25000unique prospectively pinned by native profile. First validation33, then every50; no held-out learning result yet. Deadline12:26:09AM PDT October7, no renewal. Supervisor26848, independent guard43868.

## Latest authorization — stabilized attention, October6

User requests implementing suggested normalization/attention stabilization plus global gradient clipping100 and starting local training. New SecondPass/StabilizedAttentionKrauzlis: fresh379444parameters/56tensors; native task, single shared map and local comparisons unchanged. Separate per-patch gate/candidate LayerNorm in spatialGRU and CLSGRU; projected QK L2 with bounded scale1..16(initial8); alpha/(alpha+1/N) gates spatial update, replacing100*alpha input amplification. Adam3e-4 unchanged; global gradient clipping100 after batch accumulation, no gradient renormalization afterward. Model activation normalization is forward-pass. Fresh197xxxstreams, batch32micro4 FP32,1000movies/twoepochs. Requested2244updates prospectively pinned in66update pools; validation33/every50/terminal300trials, final3000. One new finite14400second LOCALcap starts native profile, includes evaluation; no automatic extension/cloud/additional arm. CPU changes verified; source frozen. Original best600/latest1700 and all diagnostics preserved. This combined intervention is not a one-factor causal ablation. No extra mnemonic processing. Launch and actual production require current runtime receipts. See candidate README and LabJournal/stabilized-attention-krauzlis.md.

## Stability investigation complete — October6

Original LocalComparisonKrauzlis run is interrupted at live1737; best600/latest1700 preserved, original worker/supervisor exited, cause unknown, no final3000test or completion receipt. Frozen600 passes600new diagnostic movies at75.7%BA/AUC.858. Collapsed1700 attention locks near fixation: target mass88.2%→0.69%. Swapping good attention into collapsed weights restores75.7%BA; opposite swap50%; fixation-only removal70.8%. Backward-only attention/memory edge tests reduce gradientnorm308→3.54or2.08 with identical logits. Three60-update matched continuation experiments: original lr3e-4 freshBA75.0%,lr3e-5 fresh78.2%,clip1at3e-4 fresh75.7%. Standard replay didnot collapse, so results indicate sensitivity, not a proven long-term fix. All diagnostics finished under finite3600s cap, originals hash-unchanged,43finiteAdamstates at660 per endpoint, GPUlockfree. No production restart/extension/future run authorized by this completion. Details LabJournal/local-comparison-krauzlis.md; Stability/run01/summary.json. Candidate continuation lower_lr/latest.pt is a recommendation only.

**Current successor:** [Local-comparison Krauzlis](LocalComparisonKrauzlis/README.md), explicit per-location temporal evidence routed through one shared attention map. Previous shared-attention model is stopped; earlier active descriptions below are historical.

## User-directed stop — October5,7:55PM PDT

User explicitly cancelled SharedAttentionKrauzlis. Trainer30284 and CPU evaluator12188 stopped and identities verified exited. Last live update1339; preserved best1000/latest1300, all43 finite aligned Adam states independently CPU-verified. Latest completed monitoring validation1300: BA0.500000, AUC0.509180. No final test; incomplete user-cancelled run, not a spontaneous crash. No automatic restart, successor experiment, diagnostic training, queue or budget extension. Preserve source/checkpoints/logs. Runtime cancellation_receipt.json records ownership and stop evidence. Earlier active descriptions below are historical.

**Current experiment:** [Shared-attention native Krauzlis](SharedAttentionKrauzlis/README.md), one shared appearance/motion attention map, spatialGRU/CLS memory. Launched locally October5; runtime receipts establish production status. Previous simplified-dot runs are cancelled and preserved.

[Correspondence Predictive Memory](CorrespondencePredictiveMemory/README.md) — next local model, queued October5; correspondence, transport prediction and spatial memory.

# Second pass

**Historical, now cancelled:** [Outcome-contrastive RViT with spatial GRU](OutcomeContrastiveRViT/README.md), sixrawframes, recurrentvisualCLS+memoryCLS, full-batchcontrast fromterminalchoicefeedback. Localworker16140, full25024target, saved500CPUverified36alignedfiniteAdamstates. Eight-trialmemorizationpassed35updates; freshproductionweights. Noheld-outresultyet. [Journal](../LabJournal/outcome-contrastive-rvit.md).

**Latest completed result:** faster-dot transformer (speeds1.5/4/8) completedall25,024updates October5,4:36PM PDT. FinalBA49.772%, AUC0.498358 remainschance; increasing speed didnotresolvelearning. Best10000/latest25024 independentlyCPUverified, oldworkers/guardexited. [Journal](../LabJournal/weighted-sum-transformer-speed4.md).

**Latest completed run:** [Direct weighted-image transformer](WeightedSumTransformer/README.md), one shared four-head layer and CLS pair classifier trained end to end. Completed25,024updates October4,9:45PM PDT; final change BA49.25%, AUC0.492292 remained at chance. Previous weighted VAE also completed at chance with near-gray reconstruction.

**October 4, 2026: angular contrastive motion learning completed successfully on the simplified task. The [pixel-derived contrastive VAE](PixelContrastiveVAE/README.md) completed near chance; the user cancelled its plain-VAE control at saved update 7,250. The [predictive consistency VAE](PredictiveConsistencyVAE/README.md) now uses three observed frames to predict three future frames recursively, with same-clip view consistency. Its corrected fresh run completed20,032updates; finalchangeBA50.42%, AUC0.504076 remainednear chance. Allfinalartifacts arepreserved.**

The [technical report](AngularContrastiveMotion/TECHNICAL_REPORT.md) documents the fresh CNN, direction-supervised spring loss and 100% held-out before/after change decisions. The original native cue/distractor task remains untested for this model. Use [current status](../LabJournal/CURRENT_STATUS.md), the [experiment catalog](../LabJournal/EXPERIMENT_CATALOG.md) and the [research history](../LabJournal/RESEARCH_HISTORY.md) for the distinction between final results and older launch snapshots.

## Latest motion-learning sequence

| Implementation | Question | Observed result |
|---|---|---|
| [WeightedMeanConvGRU](WeightedMeanConvGRU/README.md) | Does a decayed raw-frame mean plus CNN/GRU solve the cue-free single-stimulus task? | Cancelled at570updates for the VAE; no final test |
| [ThreeFrameConvVAE](ThreeFrameConvVAE/FINAL_REPORT.md) | Can a spatial variational representation reconstruct three motion frames? | 9,900 updates; reconstruction improved; no response decision |
| [VAERViT](VAERViT/README.md) | Does the explicitly transferred VAE encoder help recurrent change decisions? | Cancelled at 227 updates for the input×10 experiment |
| [VAERViTInput10](VAERViTInput10/README.md) | Does scaling visual tokens by 10 improve learning? | 330 updates; final BA 50%, mean AUC about 0.490 |
| [WeightedMeanRViT](WeightedMeanRViT/README.md) | Can a fresh CNN over the weighted mean feed RViT? | 10-epoch cloud run replaced at 1,304 updates; preserved cancellation evidence |
| [WeightedMeanRViTEpoch2](WeightedMeanRViTEpoch2/FINAL_REPORT.md) | Does refreshing after two epochs improve generalization? | 2,310 updates; selected final BA 50%, mean AUC 0.504 |
| [VariationalMotionPredictor](VariationalMotionPredictor/FINAL_REPORT.md) | Does prediction of frame 4 from frames 1–3 learn useful motion structure? | 50,240 updates; fresh prediction MSE 92.4% below copy-last |
| [PredictiveMotionChange](PredictiveMotionChange/FINAL_REPORT.md) | Can a frozen predictive representation plus a concatenation FFN detect a 26°/28° change? | 10,240 FFN updates; final BA 50%, AUC 0.495 |
| [AngularContrastiveMotion](AngularContrastiveMotion/TECHNICAL_REPORT.md) | Can a fresh CNN learn direction similarity using proportional angular supervision? | 25,024 updates; 100% BA/AUC 1.000 in all six simplified comparison cells |

These are different objectives and stimulus distributions. They are not an exposure-matched ablation proving one causal mechanism. Prediction loss, angular loss and final response accuracy are separate measures.

## Earlier second-pass implementations

- [JointTraining](JointTraining/README.md), [TaskSuite](TaskSuite/README.md) and the [task atlas](TaskSuite/Demo/README.md): unified 13-task/35-condition battery and learned KDA joint learner.
- [SpatialReadout](SpatialReadout/FreshRun/README.md), [spatial comparison](SpatialComparisonReadout/README.md), [recurrent transformer](SpatialRecurrentTransformer/README.md), [convolutional recurrent decoder](SpatialRecurrentConvDecoder/README.md): spatial readout/core changes with explicit fresh-versus-transferred lineages.
- [SequenceKDA](SequenceKDA/README.md), [SequenceKDA3](SequenceKDA3/README.md), [SequenceKDA16](SequenceKDA16/README.md): whole-sequence single-layer, deeper and multi-head alternatives on native Krauzlis movies.
- [DelayedFrameGRU](DelayedFrameGRU/README.md), [TwoFrameRViT](TwoFrameRViT/README.md), [TwoFrameRViTReplay](TwoFrameRViTReplay/FINAL_REPORT.md), [RandomFrameRViT](RandomFrameRViT/FINAL_REPORT.md): separate temporal encoders, visual/memory attention, replay schedules and selected-frame gradient policy.
- [StructuredMotionRViT](StructuredMotionRViT/FINAL_REPORT.md), [TrainingPathAudit](TrainingPathAudit/REPORT.md), [SingleStimulusRViT](SingleStimulusRViT/README.md): motion-energy inductive bias, objective/gradient audit and cue/distractor removal.
- [KDA paper](KDA_paper/KDA_paper.md), [bibliography](papers/BIBLIOGRAPHY.md), [pod runbook](PODS.md), [analysis protocol](../ANALYSIS_SOP.md): source rationale and historical operational records.

The [original active-baseline record](ACTIVE_BASELINE.md) is a historical architectural rollback. It is not a current training instruction or evidence that the latest contrastive model was initialized from that baseline. Superseded live-status entries are archived in the journal.
