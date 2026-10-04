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
