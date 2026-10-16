# Weighted image VAE and subsequent change decoder

Each ordered three-frame clip becomes the single RGB image
`S = X[t-2] + 2 X[t-1] + 3 X[t]`. The VAE consumes and reconstructs `S/6`;
multiplying its reconstruction by six returns the requested sum without clipping.
This is a reversible scale normalization, not a different weighting.

The fresh 128-dimensional convolutional VAE trains with ordinary full-image MSE
and mean-coordinate Gaussian KL (beta warms to 1e-4 over 250 updates). It receives
no direction, change labels, prediction targets, contrastive targets or pretrained
weights. Its validation selection minimizes deterministic reconstruction MSE.
Reconstruction reports include raw-sum MSE, foreground-support error, gray and
train-only image-mean baselines and saved actual/reconstructed weighted images;
background accuracy alone is not evidence of useful dot reconstruction.

After the entire VAE allocation finishes, its selected checkpoint is frozen in
evaluation mode. Fresh independent balanced classifier training pairs supply
ordered `[mu_before, mu_after]` vectors. Coordinate centering/scaling fits only
the training cache and is recorded for deployment. A new 256→256→128→1 GELU
network learns change/no change with binary cross entropy. VAE weights receive
no classifier gradients and are checked unchanged after final evaluation.

Default requested exposure is 25,024 VAE updates (1,000 fresh continuous
six-frame movies per pool, two shuffled epochs, effective 32 pairs/microbatch8)
then 10,000 classifier updates at batch128 over 64,000 independently generated
cached training pairs. The classifier replays those fixed training latents;
presentation count and unique movies are reported separately. Classifier
selection minimizes validation BCE; a positive-sign scalar threshold is fitted
on validation only. Final evaluation has 3,072 presentations / 1,536 paired
nuisance contexts across three speeds × two angles. No threshold is fitted on test.

All local work is sequential under one GPU lock. A new finite four-hour ledger
starts at the first GPU profile and covers both stages, cache extraction,
validation, evaluation and reporting. Actual VAE exposure is pinned in complete
64-update pools before production, reserving measured classifier/extraction
costs and at least 30 minutes. Profile weights are discarded. No cloud work,
additional arm or automatic deadline renewal is included. Each stage retains
only its best and latest checkpoint with compatible optimizer/RNG state.

`python -m SecondPass.WeightedSumVAE.launch --run ABSOLUTE_RUN_DIRECTORY`
owns the single profile/production worker and independent deadline guard.
`python -m SecondPass.WeightedSumVAE.runner_check` runs a disposable, CPU-only
engineering fixture through 64 real VAE updates, validation/checkpoint saving,
frozen-latent extraction, 64 real classifier updates, selected-model loading and
paired final evaluation. It does not activate the scientific experiment.

These are simplified persistent coherent dots in two adjacent three-frame clips.
They are not a native Krauzlis replication or a biological-validation result.


Actual local launch: October4,7:51PM PDT, `LocalRuntime/run01/`; worker40436,
supervisor33384, guard42668. Profile pinned25,024 VAE and10,000 decoder updates.
Actual GPU validation64 passed; root CPU-verified20 aligned finite VAE Adam states.
Both stages share the11:51:50PM PDT deadline. Training is live, with no final
accuracy claim. The classifier starts automatically after selected VAE extraction.


Final result: completedOctober4,8:36PM PDT in45minutes, bothfulltargets.
Selected VAE25,024/classifier1,000. FinalBA49.71%, AUC0.500107 on3,072
presentations/1,536paired contexts. Weighted reconstruction collapsed to nearly
gray; inspect `LocalRuntime/run01/weighted_reconstructions.png` and VAE baselines.
All checkpoints verified/preserved; workers exited and no extra experiment queued.
[Final report](LocalRuntime/run01/FINAL_REPORT.md) supersedes launch/live statements.
