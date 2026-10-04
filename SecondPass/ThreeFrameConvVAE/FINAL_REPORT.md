# Three-frame convolutional VAE: final results

Completed9900updates /300,000triplet presentations /30,000unique movies on October3 at19:34:23UTC (12:34PM PDT). Best validation selects9500; latest is9900. Original eight-hour cap preserved; no extension or cloud compute. Both full model/Adam/RNG/stream/scheduler checkpoints independently CPU loaded and hashed,126Adam states at the corresponding update. Only `best.pt` and `latest.pt` remain, about218MiB total. Independent guard exited normally.

Three ordered native100×100RGB frames feed a9,507,913parameter convolutional VAE with diagonal-Gaussian256×13×13latent and latent-only decoder. Native no-cue/single-stimulus B12/B20/B28 movies, original positions/dynamics and full spatial frames; no response labels or additional motion loss. Same1000movie pools×10shuffled epochs, batch32/micro4, Adam1e-4, FP32/no clipping/full gradients.

Last100updates: total loss 0.00063257; balanced reconstruction 0.00060685; mean KL 0.257178; unweightedpixelMSE 0.00005120.

Fresh paired selected/latest tests:128independentmovies per condition, threewindows each,384movies /1152triplets per model. Windows are grouped by movie; paired evaluations are not additional independent samples. Selection uses validation only.

| Metric, mean across B12/B20/B28 | Best9500 | Latest9900 | Copy-middle baseline |
| --- | ---: | ---: | ---: |
| Balanced reconstruction | 0.00064548 | 0.00066858 | 0.00342465 |
| Unweighted pixel MSE | 0.00005657 | 0.00005946 | 0.00005637 |
| Temporal-difference MSE | 0.00005794 | 0.00005714 | 0.00008455 |

| Condition | Best reconstruction | Latest reconstruction | Best temporal-difference MSE |
| --- | ---: | ---: | ---: |
| B12 | 0.00065931 | 0.00067925 | 0.00005841 |
| B20 | 0.00063988 | 0.00066530 | 0.00005806 |
| B28 | 0.00063725 | 0.00066118 | 0.00005736 |

The selected model reduces the balanced reconstruction objective by81.2% and temporal-difference MSE by31.5% versus copying the middle frame. Full-image MSE is approximately equal to that baseline (0.36% higher). These measure reconstruction, not motion direction/change decisions or usefulness of the latent for downstream response training. No downstream response model was trained.

Checkpoint5700originally failed with ENOSPC; restored5600 exactly after user-authorized removal of prior checkpoints.100unsaved updates /3032triplet presentations were repeated and are extra physical attempts, not additional unique training exposure (303,032physical presentations including discarded attempts). Historical best5500 weights were deleted; resumed5600 initialized available-best selection, and9500subsequently won on validation. Historical code/logs/reports remain; old checkpoint links are unavailable.

[Parent completion verification](LocalRuntime/completion_verified.json) | [Cleanup manifest](LocalRuntime/checkpoint_cleanup.json) | [Architecture](README.md).
Runtime: `/Users/jonathanmorgan/VAWMRuntime/three_frame_conv_vae_local01/resume01`.
