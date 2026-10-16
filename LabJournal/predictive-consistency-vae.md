# Predictive consistency VAE

October 4, 2026. Current protocol: **three observed context frames → three recursively predicted future frames**, following the user's explicit correction. Fresh weights, one local model, no plain-VAE control or cloud. The ten-page mathematical document has been updated and all ten rendered pages reviewed.


## Final result — completed October 4, 7:22 PM PDT

The corrected local run completed all20,032updates,313,000unique twelve-frame movies and626,000presentations, with selectedcheckpoint20,032. Final3,072change presentations/1,536paired contexts: primary forecast-surpriseBA50.42%, AUC0.504076; posteriorBA50.68%, AUC0.508740; projectionBA50.59%, AUC0.508057. Allsix primarycells remainnear chance (49.80–50.98%BA). Copy-last balancederror scoringBA60.55%, AUC0.535031. These are finalheld-out results, superseding the live snapshots below.

Independent root CPUverification confirmed best/latest finite weights andall38Adam states at20,032. Queue/worker/guard exited; no automatic follow-up. Completion was within the original8:11:07PM PDT deadline. Separate objective384-movie test: balancedpredictionerror0.00635051 beatscopy-last0.00879748, but ordinaryMSE0.00158658 is worse thancopy-last0.00113222. Visually inspected forecastexamples show diffuse blurredpatterns rather than faithful individual-dot forecasts. Loss reduction didnot produce useful operational change detection. This doesnot ruleout image-based predictive learning generally or establish what information diagnosticreadouts could recover.

See [final report](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run02/FINAL_REPORT.md), [full final results](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run02/final_test.json), [forecast examples](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run02/forecast_examples.png) and [root verification](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run02/root_final_verification.json).

## Question and implementation

Can reconstruction, temporal prediction and appearance-preserving view consistency learn change information from raw images without direction, angular-distance or binary-change learning targets? The small convolutional VAE has 2,124,961 parameters in 38 tensors. Its nine-channel encoder reconstructs the three observed frames through a 128-dimensional Gaussian latent. A separate nine-channel context encoder and 128-dimensional predictor use the shared decoder to generate one next image. A rolling three-frame context appends that prediction and repeats for three future frames. All three future losses backpropagate through earlier generated frames; there is no teacher forcing or future-target conditioning.

The objective combines support-balanced observed reconstruction and equal-weight three-horizon prediction, coordinate-mean KL warmed to beta 1e-4 over 250 updates, and VICReg view invariance/variance/covariance weights 1/1/0.04. The two views are coherent translations/contrast changes with independent pixel noise of the SAME ordered observed clip. Fresh namespaces are 189101/189102/189103. Training pairs contain two consecutive six-frame stable-motion segments of one twelve-frame movie; the second segment occasionally changes direction. Each segment supplies three context frames and three targets.

Adam uses one learning rate 1e-4, epsilon 1e-8, FP32 and no weight decay or clipping. Effective batches contain 32 paired movies and 64 observed clips per view. Four-pair microbatches use an exact two-pass cache of whole-batch VICReg output gradients. All learned tensors receive optimizer states; the final projector bias can legitimately have zero update because VICReg is invariant to a common translation.

## Evaluation and verification

Checkpoint selection minimizes the independent matched three-horizon unsupervised objective: 384 twelve-frame movies with an internal index offset of 60 million. The separate original three-before/three-after change assay remains unchanged. Forecasts begin with all three before frames and use only their own outputs thereafter; errors against the last two after frames supply the primary surprise score. Paired changed/unchanged examples have identical first target frames, which are an exact chance negative control. Validation alone fits score sign/threshold. Final change evaluation contains 3,072 presentations / 1,536 paired nuisance contexts, separately from the held-out objective sample. Every speed/angle cell, raw MSE, copy-last baseline and secondary latent-distance score remains visible.

CPU checks verify causality, later-horizon gradients into the first prediction, coherent augmentation and global cached-gradient agreement within 1.34e-8. The ACTUAL production runner also passed a disposable CPU engineering fixture through 64 updates, validation, best/latest checkpoints with all 38 Adam states, selected-weight final evaluation, image/report exports and completion receipts. This is a software test, not an experiment outcome or GPU profile.

## Actual corrected launch

Queue26328 activated at2026-10-04T23:27:18UTC. The profile measured0.45264seconds/update and3.37593seconds/1,000-movie pool; with the inherited deadline,1.25margin and745.50seconds evaluation/reporting reserve it pinned **20,032updates /313,000unique movies /626,000presentations** before fresh production. Profile weights were discarded. Worker6192 and independent guard40384 are active. Actual GPU validation64 passed; best/latest were independently loaded on CPU with all38 Adam states at64 and finite weights, then live training advanced past114. See `LocalRuntime/run02/root_checkpoint_verification.json`, `profile.json`, `production_config.json` and `queue04/root_launch_receipt.json`. These are startup/validation observations, not final held-out outcomes.


## Live validation snapshot — October 4, 5:08 PM PDT

The worker remained active at5,030/20,032updates. Validation5000 completed and selected checkpoint5000 by unsupervised loss0.691846 (initial64:1.008166). Primary forecast-surprise change BA51.82%, AUC0.508925; posterior/projection BA51.82%, AUC0.510539/0.508341. These are validation scores calibrated on this same set, not final held-out outcomes. Support-balanced prediction decreased0.009049→0.007696, but ordinary prediction MSE increased0.001177→0.002587, so aggregate loss improvement should not be equated with useful motion prediction. The view term still dominates0.676398; prediction/reconstruction are0.007696/0.007724. Preserve the planned horizon and unchanged objective; next validation10,000. Deadline8:11:07PM PDT remains unchanged.


## Live validation snapshot — October 4, 6:16 PM PDT

Training active at12,930/20,032updates (64.5%); latestsaved12,750/all38Adam states. Validation10,000 selected checkpoint10,000: totalobjective0.362789, primaryforecast-surpriseBA52.60%, AUC0.509006, posteriorBA51.82%, AUC0.511013. Predictionbalancederror0.00769694 andMSE0.00257793 are essentially unchanged from5,000 despite loweroverallloss. These are validation observations, not final test outcomes. Nextvalidation15,000; continue unchanged within original8:11:07PM PDT deadline.


## Live validation snapshot — October 4, 6:50 PM PDT

Workerhealthy at16,770/20,032updates (83.7%); latestsaved16,750/all38Adam states. Validation15,000 selectedcheckpoint15,000: objective0.266602, primarychangeBA52.34%, AUC0.512451. Balancedpredictionerror0.00679618 improved11.7% from10,000 (0.00769694), andordinaryMSE0.00193830 improved24.8%; operationalchangeclassification remains near chance. This is live validation, not finalheld-out evidence. Continue pinnedexposure unchanged; expectedtrainingfinisharound7:20–7:25PM, originaldeadline8:11:07PM PDT.

## Preserved failed attempt and finite deadline

The user cancelled the previous plain-VAE control at saved update 7,250 and authorized immediate predictive training. The first predictive attempt (`LocalRuntime/run01`) trained 64 updates then failed on an undefined variable in validation checkpoint saving. Latest weights and all 38 Adam states are preserved; the failure is acknowledged rather than recorded as a healthy run. The correction uses `save(best=better)` and the complete production-path fixture above.

The revised model starts fresh in `LocalRuntime/run02`; no failed weights or Adam states transfer. Its profile inherits ONLY the original budget origin 1791155467.341355 and deadline 1791169867.341355: October 4, **4:11:07–8:11:07 PM PDT**. This correction does not renew the four-hour allowance. Requested exposure is 25,024 updates; native profiling may prospectively reduce it in complete 64-update pools before fresh production. Runtime receipts establish the actual pinned exposure and progress.

Predictive coding and temporal-association theory motivate the learning question; VAE and VICReg supply the implemented mathematical operations. This is not a cortical circuit or attention model. The separate context branch can learn useful prediction even if the full-clip posterior retains static scene information. Prediction and latent-distance outcomes therefore answer separate questions. Comparison with the preceding recipe changes several components and is not a single-factor ablation.

See [implementation protocol](../SecondPass/PredictiveConsistencyVAE/README.md), [ten-page mathematical document](../output/pdf/predictive_consistency_vae.pdf), [current status](CURRENT_STATUS.md) and [original supersession receipt](../SecondPass/PredictiveConsistencyVAE/LocalRuntime/run01/superseded_receipt.json).

<details><summary>Historical initial queue and two-context/one-target implementation</summary>

# Predictive consistency VAE

October 4, 2026. The user requested a similar small convolutional architecture with the proposed predictive and view-consistency objective, queued as the next model, and a ten-page mathematical document. One new model is authorized. All existing progress remains preserved; no cloud or inherited weights.

## Question and implementation

Can temporal prediction and appearance-preserving consistency produce useful change information from raw images, without an explicit motion direction, angular distance or change label? The model has 2,124,313 parameters in 38 tensors: three-frame 128-dimensional Gaussian VAE; separate causal six-channel CNN with the same widths; 128-dimensional predictor using the shared three-frame decoder's final output; and unnormalized 64-dimensional view projector. Every active learned parameter trains with one Adam learning rate, 1e-4, epsilon 1e-8, FP32, no clipping or weight decay.

The loss adds support-balanced reconstruction, causal third-frame prediction, KL with beta warming to 1e-4 over 250 updates, and VICReg-style invariance/variance/covariance weights 1/1/0.04. Covariance penalties are averaged across views. Each view applies a coherent toroidal translation, contrast scaling and independent pixel noise to one ordered clip. It never attracts the two separate clips indiscriminately. Global view statistics use 64 clips per full update (16 for partial tails), preserved with exact first-derivative gradient caching and microbatch recomputation. Dataset namespaces 188101/188102/188103 are fresh.

The primary evaluation forecasts the second clip from the first clip's final two frames, then uses predicted frames recursively. Actual second-clip frames are targets only. Late support-balanced surprise is primary; the identical first boundary frame is a negative control. Within-clip predictions, copy-last baselines, latent distances and every speed/angle cell remain visible. Labels only calibrate validation score sign and threshold; checkpoint selection uses unsupervised validation loss.

## Execution plan and checks

The queue waits for BOTH current PixelContrastiveVAE arms to publish successful 25,024-update completion and final-test artifacts, then for release of the shared OS GPU lock. Its requested horizon is 25,024 fresh updates, 391,000 unique paired movies and 782,000 pair presentations. A new four-hour local cap begins at its first GPU profile; prospective profiling can pin a lower complete-pool exposure before production. The queue performs no CUDA work while pending, never extends an old budget, and starts only one new model. Final evaluation is 3,072 presentations in 1,536 paired nuisance contexts, with 384 validation presentations.

CPU model checks passed on Torch 1.13.1: all 38 tensors have finite nonzero gradients and Adam progress; target changes leave the causal prediction unchanged; augmentation shares spatial transformations across time; recursive predictions have the correct dependency. Two independent checks found cached-view gradients agree with direct whole-batch gradients within 1.34e-8. Runner fixtures check validation calibration, the identical first forecast, final-image export, successful-predecessor requirements and worker exit. Actual-source read-only review found and corrected checkpoint replacement retries, validation CUDA RNG restoration, optional queue telemetry retry and held-handle process identity checks.

At 2026-10-04T21:41:21Z, **queue worker 34400 was armed in `LocalRuntime/queue02/`**. Its pending receipt verifies matching source hashes, the current model's held GPU lock, and absence of any new profile, budget or production config. Requested production output is `LocalRuntime/run01/`. The first unactivated queue 31760 was explicitly replaced: a real Windows occupied-byte-lock check raised PermissionError during the initial file read, before the lock call. The probe is now covered by occupied-lock error handling; CPU tests verify occupied current/trial locks wait and released locks can be acquired. Queue01 never activated or spent a GPU budget. GPU training has not started for the new model.

## Theory and interpretation

Predictive coding motivates residual learning; temporal association motivates appearance invariance; VICReg supplies the operational variance/covariance safeguards; VAE reparameterization supplies the reconstruction latent. The implementation is not InfoNCE/CPC, a cortical circuit, or an attention mechanism. A separate context branch might carry useful motion while the full-clip mean retains static scene information; primary predictive and secondary latent scores therefore answer different questions. Autoregressive rollout uses generated context frames although training uses actual context frames; that distribution shift can accumulate prediction errors. The known background/support threshold also limits natural-scene transfer claims.

The preceding pixel-distance worker crashed at update 2,873 because a live-status file was temporarily locked. It resumed as PID 20352 from a checkpoint with all 22 compatible Adam states and exact shuffle/data/RNG progress, under the original deadline. Critical files require successful bounded-retry writes; optional telemetry can warn and skip. The new queue/runner incorporates this reliability lesson and validation RNG isolation.

See [implementation protocol](../SecondPass/PredictiveConsistencyVAE/README.md), [ten-page document](../output/pdf/predictive_consistency_vae.pdf), and [current status](CURRENT_STATUS.md). These are implementation and queued-execution claims, not held-out model outcomes.

</details>
