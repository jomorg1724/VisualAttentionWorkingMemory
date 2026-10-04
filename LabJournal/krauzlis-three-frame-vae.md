# Three-frame VAE before response learning

User requested replacing weighted CNN-GRU with reconstruction pretraining,
then attaching a response architecture later. The old model was stopped and
saved at570updates/17,288presentations, all70Adamstates CPUverified; no finaltest.

The fresh VAE uses three ordered RGB100x100 frames, a residual convolutional
encoder, diagonal-Gaussian latent256x13x13, and latent-only residual decoder
reconstructing allthree. All9,507,913parameters/126tensors train. StandardGaussian
posterior/reparameterization follows [Kingma and Welling](https://arxiv.org/abs/1312.6114).
The contrast-balanced reconstruction +1e-4meanKL is a disclosed objective adaptation,
not the standard unweighted pixel ELBO. No classification or temporal loss.

Same no-cue/single-stimulus renderer. Uniform ordered triplet starts7..T-4,
fully inside moving-dot frames7..T-2; no crops, labels or privileged motion inputs.
1,000freshmovies×10shuffled epochs, newwindow per presentation. Count movies and
triplet presentations separately. NativeMPS profile3warmup+3steady/8movies percell
pins9,900updates/300,000triplets/30,000uniquemovies in30whole330updatepools.
FP32/Adam1e-4/no clipping/batch32micro4/twoCPUthreads. Val64movies/cell×3windows
at100/250/every500/terminal, reconselection; pairedfreshfinals128/cell×3windows.
Gray/copy-middle baselines and reconstruction PNGs retained. Group by movie.

Actual production as of 2026-10-03 15:57 UTC: checkpoint3/96 parentCPUverified, all126Adamstates
and tensors advanced, fresh constructor/emptyinitialAdam/MPS RNG verified.
Source-only28file runtime, no profile/oldmodel inheritance. Independentguard21104,
supervisor21106, worker21185. New8hLOCALcap15:49:10UTC to23:49:10UTC
(October3,8:49:10AM to4:49:10PM PDT), no extension. Cloud remains closed.

Early validation100 mean balancedrecon0.009875, gray0.029108,
copy-middle0.003412. It beatsgray on the balanced objective but
still trails copy-middle; image/temporal reconstruction not yet faithful.
FullMSE0.004273, KL0.323425, temporalDiff0.001664.
These are early validation snapshots. Reconstruction success would motivate feature
analysis; it would not establish correct downstream responses. No downstream model
was implemented or trained. [Run](../SecondPass/ThreeFrameConvVAE/RUN_STATUS.md).

## Live validation snapshot — 2026-10-03 16:05 UTC

At2,121updates/64,288triplets, last100mean total loss0.002985.
Validation2000 on64movies/cell×3windows: balancedrecon0.003108,
copy-middle0.003412, gray0.029108. Allthreeconditions now beat copy-middle
on the balanced objective. Full-image MSE0.00027665 remains worse than
copy-middle0.00005612; temporalDiff0.00015595 versus0.00008418.
Reconstruction images retain the rough stimulus position, but individual dots form soft blobs and show artifacts;
this is earlyvalidation, not finaltest or proof of useful motion features.
Run continues under unchanged allocation/cap, no architecture/objective change.

## Live validation snapshot — 2026-10-03 16:31 UTC

At5,686updates/172,320triplet presentations, last100 mean total loss0.00100025,
reconstruction0.00097246. Validation5500: balancedrecon0.00101607 versuscopy-middle
0.00341202; temporalDiffMSE0.000078845 versus0.000084177, nowlower inallthreecells.
That is a6.3percent mean movement-error improvement on these fixedvalidationmovies,
not a significance result or evidence of correct motion-direction/change decisions.
UnweightedfullMSE0.00009455 stillworse than copy-middle0.00005612.
Run/allocation/objective/cap unchanged; no new experiment launched.

## Disk failure, cleanup and authorized recovery — October 3

Checkpoint5700 failed at9:31AM PDT with ENOSPC; the latest complete saved state was5600 /169,736triplet presentations /17,000unique movies. The next100updates /3,032presentations were physically executed but unsaved and will be replayed from restored state. No completed fresh final tests. System logs confirmed real disk exhaustion. The project/runtime accumulated1,522checkpoint/model files totaling54.03GiB. User explicitly authorized removing old checkpoints:1,518files /57.9GB deleted; only full VAE5600 retained, independently CPU verified with126Adam states. Code/logs/reports survive; historical checkpoint links are unavailable. [Cleanup receipt](../SecondPass/ThreeFrameConvVAE/LocalRuntime/checkpoint_cleanup.json).

User now authorizes continuing the SAME run from5600 with exactly TWO retained checkpoint files, best and latest, replaced atomically. No architecture, objective, batch, optimizer, RNG, stream or allocation change. Original9900target and23:49:10UTC harddeadline remain. Best5500 file was removed, so restored5600 will be re-evaluated to initialize available-best selection; older validation scores remain historical. Recovery code uses full saved model/Adam/RNG/stream/scheduler state, no new step-zero file and no initial-checkpoint dependency. Resume launch/progress evidence will be recorded after actual advancement.

## Actual recovery launch and persisted progress — October 3, 18:59 UTC

**October 3, 11:59 AM PDT — VAE RESUMED and optimizer progress verified.**
Restored complete update5600 model/Adam/RNG/streams/scheduler. Latest independently CPU verified saved update5700 /172,768triplet presentations, all126Adam states at the saved step; live update5784. Only `resume01/latest.pt` and `resume01/best.pt` exist. Original5600 filename promoted/removed after verification; preserved checkpoint content/state. Startupval5600 balancedrecon0.001003128 initializes available best; historical5500 weights unavailable. Same9900target and original4:49:10PM PDT hardstop, no renewal; cloudclosed. Resume worker15761/guard15751. [Recovery evidence](../SecondPass/ThreeFrameConvVAE/LocalRuntime/resume_verified.json).

Recovery uses the original immutable dependency sources plus a separately hashed adapter snapshot; no original source was overwritten. First5601/5602/5603advancement saved to overwritten latest with small JSON proof receipts. Parent independently reloaded saved5700 and verified its digest, all126Adam steps, complete finite model/state, unchanged budget and exactlytwo checkpoint filenames. Latest stable saved5700 replays the100updates that were unsaved before the disk failure; count those extra attempts separately. This is continuation evidence, not completed finaltest or proof that reconstructed motion solves the response task.

**2026-10-03 19:34 UTC — VAE COMPLETED9900updates; only best9500/latest9900 retained.**
300,000triplet presentations /30,000unique movies, same original allocation/cap. Fresh paired final tests complete (384independentmovies /1152triplets per model). Last100total loss0.00063257/recon0.00060685. Selected held-out balancedrecon0.00064548 versuscopy-middle0.00342465; temporal-differenceMSE0.00005794 versus0.00008455 (31.5%lower). Full-imageMSE0.00005657 approximatelycopy-middle0.00005637. No response model/accuracy tested. ParentCPUverified both full checkpoints and126Adam steps; guard exited normally. [Final report](../SecondPass/ThreeFrameConvVAE/FINAL_REPORT.md) | [Completion proof](../SecondPass/ThreeFrameConvVAE/LocalRuntime/completion_verified.json). No continuation or new run launched.

