# Local variational motion prediction pilot

User question: whether next-frame prediction teaches useful motion features before response decoding. Fresh syntheticconstant-velocitydot data, fourRGB100x100frames; posterior seesframes0,1,2 only and decoderpredictsframe3 fromone512-dimensional Gaussianvector. Directionuniformoverallangles acrosssamples, ALLdots shareonevelocity within sample. Persistent16–48dots, speed.375/1/2pixels perframe, graybackground andGaussianσ.7/contrast.42. Periodicwrapping preserves identities withoutunpredictablerespawns or directionalbiasfromboundaryrejection. This is a newsyntheticbenchmark, not a Krauzlisreplication. No cues/fixation/distractor/classificationlabels.

Predictivebottleneck support-balancedfutureMSE+meanGaussianKL, beta warmup0→1e-4 over250updates. Supportunioncontains past+targetcontrastpixels; targetweightsare used onlyinside loss, neverencoding. Report ordinaryfullMSE/background/motion-support/change-regionMSEseparately and compare copyframe3 andgray. Deterministicdecode(mu) evaluation; stochasticz duringtraining. Beatingcopylast onfreshdraws isprimarysignal, especially at1/2pixel speeds; successfulpredictionwouldnotaloneestablishchange-detectiontransfer or provea particularlatentrepresentation.

Initiallocalpilot: finite20min cap startsfirstacceleratorprofile, includesdisposable3updateprofile/training/evaluation. Target1024updates pinnedbeforefreshproduction tocomplete64updatepools, each1000freshtrials×2epochs (32updatesperepoch; final8trialbatch). Batch32/micro4, FP32/fullgradients/Adam1e-4/no clipping/CPU2, all15,463,363paramsfreshtrainable; noVAE9900orpredecessorweights. Onlybest/latest replacedatomically; dataregeneratedfromseed/countersratherthancheckpointingpixels. Preserveallpreviousmodels. ParentlaunchesaftershortCPUchecks; production requires savedAdam evidence. No cloud model or rental.

## Initial local pilot result

Completed256updates /8000presentations /4000unique samples in fourtwo-epoch pools. The20-minute limit was a ceiling; measured conservativeallocation completedearly atOctober3,7:37:52PM PDT. Selectedbest256/latest256 retainedandCPUverified, all138Adam states; allfinal128per-speedcells complete. Foreground-union-balanced futureMSE(model/copylast/gray):

|Speed(px/frame)|Model|Copylast|Gray|
|---|---:|---:|---:|
|0.375|0.006248|0.001002|0.007367|
|1.0|0.004695|0.004149|0.005312|
|2.0|0.003484|0.006367|0.003726|

MeanmodelpredictionMSE0.004809 vs copylast0.003839 andgray0.005469. Theearlydecoder produceslargelyflat/blurred images; beatingcopylastat2px/speed doesnotestablishmotionuse, sincecopylast canbe worsethanblankpredictionforlargedisplacements. Requirecomparisonwith BOTHbaselines andlaterhistory-use/direction-access testsbeforeclaimingmotionrepresentation. Thisshortpilot testsimplementationandinitialoptimization, notadequatetraininghorizon. No autonomouscontinuationornewcloudrun.


## Substantial local continuation — actually running

User explicitly requested real training after the short pilot. Resumed the complete model, Adam, RNG and data-pool state from update 256, with no architecture/objective changes. Target 49,984 additional updates (50,240 cumulative; approximately 1.57 million cumulative presentations) or the new eight-hour wall limit, whichever comes first. Batch 32 / microbatch 4, Adam 1e-4, MPS FP32, two CPU threads. Continue two epochs per 1,000 fresh four-frame samples, then refresh. First validation at 356, subsequently every 1,000 updates; selection by validation predictive MSE. Final test uses fresh index offset 3,000,000. Best/latest are overwritten in the existing run directory, with no extra checkpoint copies. Earlier pilot metadata preserved under LocalRuntime/continuation01/pilot_metadata. Independent deadline guard and exclusive local accelerator lock armed; hard stop 2026-10-04T11:45:34.384345+00:00. Persisted optimizer progress independently verified at update 257, all 138 Adam states advanced. See LocalRuntime/continuation01/production_verified.json. No cloud compute.


Live validation snapshot, October 3 approximately 10:56 PM PDT: training active at 13,026 updates /407,064 presentations /204,000 generated unique samples. At validation13,000 balanced prediction MSE0.00076596 versus copy-last0.00384705 (~80.1% lower), beating copy-last at all three speeds. Selected best12,000 MSE0.00075362. Latest batch total loss0.00113526, reconstruction0.00078389, KL3.51368 with beta1e-4. These are live validation results, not final test results or evidence of downstream Krauzlis classification. Training continues under unchanged eight-hour cap; only best/latest retained.


## Completed substantial local predictive training — October 4

Completed50,240 cumulative updates (49,984 additional), 1,570,000 presentations and785,000 unique four-frame samples. Finished2026-10-04T11:29:05.750221+00:00, approximately7.73 hours after continuation start; normal owner/guard exit, no active training. Best50,000/latest50,240 retained, independently reloaded with all138 Adam states at their respective steps; no additional checkpoint copies. Final held-out test used768 fresh sequences (256/speed, test index offset3,000,000) with complete coverage.

| Speed(px/frame) | Predictor balanced MSE | Copy-last balanced MSE |
|---|---:|---:|
| 0.375 | 0.00036810 | 0.00100461 |
| 1.0 | 0.00028531 | 0.00415830 |
| 2.0 | 0.00022482 | 0.00639938 |

Mean balanced MSE0.00029275 versus copy-last0.00385410 and gray0.00548914: 92.4% lower than copy-last. This is next-frame prediction on the new constant-velocity benchmark, not downstream change-detection or Krauzlis performance. No claim yet that the latent supports direction/change decoding. Only best/latest checkpoints remain; no autonomous follow-up or cloud launch. Evidence: [final report](../SecondPass/VariationalMotionPredictor/LocalRuntime/attempt01/run/continuation_report.json), completion_verified.json and prediction panels.
