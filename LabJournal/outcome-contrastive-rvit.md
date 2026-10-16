# Six-frame outcome-contrastive RViT with spatial GRU

**Actualproductionverified:** worker16140is trainingtowardthefull25,024updates.
RootCPU-loadedbest64/latest500; all36Adamstatesaligned/finite, allweightsfinite,
source/profileunchangedandactualprocessidentityverified. BudgetstartsOctober5,
5:06:08.373PM PDT andends9:06:08.373PM. Profile0.07408sec/update,
1.77120sec/1000-moviegeneration,0.005399sec/evaluationtrial, reserve638.87sec.
Thebounded8-trialcheckreached100%trainingaccuracyat35updates (CE0.04289);
itsweightswerediscarded. Thisestablisheslimitedmemorizationcapacityonly.
Initialvalidation64isapipelinehealthsnapshot, notafinalperformanceclaim.

October5,2026. The user proposed terminalchoicecorrectness feedback and a
full-batch contrastive loss on a persistent CLS memory token, then clarified
that RViT must include a patch-based spatialGRU/LSTM after the visualtransformer.
The earlier architecture-reference hold is resolved.

## Actual architecture

Six raw100x100RGBframes, preserving the continuous3before/3after dotmovie.
Perframe10x10patches ->100tokens, width128; prependlearnedvisualCLS/positions.
One4-headblock uses visualselfattention plusprevious-memorycrossattention;
the resultingspatialrepresentations drivea3x3ConvGRU, andthevisualCLSdrivesa
correspondinglinearGRU. Persistentmemoryis128x10x10 plus128DCLS. Bothstates
feedbackatthenextframe. OnlyfinalmemoryCLSdrivesbeliefs andcontrastiveloss.
FullBPTTthroughall6frames, all1,316,994parameters/36tensorsfreshandtrainable.

Spatialgateequations are adaptedfrom
`SecondPass/SpatialReadout/model.py::FinalConvGRU`, also presentin
`PreAttentiveVision/TemporalIntegration/accumulators.py::SpatialConvGRU`.
TheCPUcheckloadsreferenceweightsatwidth64 andverifiesexactmatchingoutputs.
CLSuseslinearmapswiththesamewrite/reset/candidateequations andisnotafakegridcell.
Thisisthenewuser-requestedcomposition;olderTwoFrameRViTdidnotcontainthisGRU.

## Objective and experiment

Terminalargmaxchoiceisassigned+1forcorrect,-1forincorrect. Inbinaryclassification,
choicepluscorrectnessrecovers thecorrectanswer. Same-answertrialsaresupervised
contrastivepositives; oppositesare negatives, withselfexcluded. TP/FNareoneanswer
group;TN/FPtheother. Incorrectanchorsgetweight2, correctanchorsweight1.
OptimizechoiceCE +0.1weightedSupCon, temperature0.1, directlyonnormalized
memoryCLS. Theseweightchoices/jointtrainingadaptSupCon;thisisnotCPCorRL.
Noangles,velocities,phasesorlabelsenterforwardinputs.

Speeds1.5/4/8, changes26/28degrees, fresh193xxxstreams. Adam3e-4, FP32,
TF32off, noclip/weightdecay, CPUthreads2, onelocalGPU. Fullbatch32includes
16/class; each1000-freshmoviepoolhas2shuffledepochsandbalancedtail8. One
contrastiveobjectivecoversthewholebatchwithallcandidategradientsretained.
Target25,024updates /391,000unique movies /782,000presentations, prospectively
pinnedfromnativeprofileincomplete64-updatepoolsunderanewfinite4hourcap.
Val384at64/5000/10000/15000/20000/terminal selectsminimumdecisionCE. Final3072
presentations/1536pairedcontexts reportallsixcells andTP/TN/FP/FN,BA,AUC,CE.
Actualargmaxisusedwithoutthresholdcalibration. Nootherproductionarm/cloud.

Bounded8-triallearningcheck(max256updates), profiling, production/evaluation
andreporting allsharethesamecap. Itsweightsarediscarded;productionstartsfresh.
Small-setmemorizationdoesnotdemonstrategeneralization. Onlybest/latest
checkpointswithAdam/RNG/samplerstatepersist; independentdeadlineguardowns
onlythisworker. Earlierexperiments/checkpointsremainpreserved.

## Execution evidence

Objectivechecks, causalstreaming/fulltemporalgradients andactualCPUproduction
save/validation/finalizationfixturepassed. Supervisor13356launchedOctober5,
5:06:05PM PDT; nativeprofileworker25152beganwiththeboundedlearningcheck.
Thisrecordsactivation, notyetafullproductionclaimorheld-outresult. Actualstatus
andcheckpointsarein `SecondPass/OutcomeContrastiveRViT/LocalRuntime/run01`.

[Implementation, mathematics and primary sources](../SecondPass/OutcomeContrastiveRViT/README.md)
