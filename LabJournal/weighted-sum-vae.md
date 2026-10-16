# Weighted summed-image VAE and nonlinear change decoder

October 4, 2026. The user explicitly requested this as the final experiment for the night: return to a decayed sum with weights 3 for the newest frame, 2 for the preceding frame, and 1 for the oldest. Train a VAE to reconstruct that summed image, then concatenate the latents of two consecutive three-frame clips and train a nonlinear change/no-change decoder.

## Final result — completed October 4, 8:36 PM PDT

Both stages completed in45minutes:25,024 VAE updates, then10,000 classifier updates. Selected checkpoints:VAE25,024, classifier1,000; terminal10,000 classifier also retained. Final3,072 presentations/1,536 paired contexts: **change BA49.71%, AUC0.500107, BCE0.694719**. No model solved change detection in this run.

The reconstruction stage itself failed: target weighted dot trails are visible, while VAE outputs are almost uniform gray. Held-out reconstruction MSE0.000583310 vs gray0.000599152/train-mean0.000585566; output variance7.238e-7 vs target5.8124e-4, KL1.4867e-8. These observations are consistent with an effectively collapsed VAE. Sparse-image MSE rewarded background reconstruction rather than preserving the requested pattern. This is a negative finding for the implemented VAE-plus-decoder pipeline; it does not isolate a fundamental limit of the weighted input representation.

All four checkpoints independently CPU-verified with aligned20/6 Adam states and finite weights/optimizer tensors. Frozen VAE immutability passed. All worker/supervisor/guard processes exited normally, within the original new four-hour cap. No additional experiment for the night. See [final report](../SecondPass/WeightedSumVAE/LocalRuntime/run01/FINAL_REPORT.md), [results](../SecondPass/WeightedSumVAE/LocalRuntime/run01/final_test.json), [reconstructions](../SecondPass/WeightedSumVAE/LocalRuntime/run01/weighted_reconstructions.png) and [verification](../SecondPass/WeightedSumVAE/LocalRuntime/run01/root_final_verification.json). Launch/live statements below are historical.

## Implemented question

For a chronological clip `(I0, I1, I2)`, the requested image is `S = I0 + 2 I1 + 3 I2`. The VAE input and reconstruction target are `S/6`, a reversible normalization that keeps RGB intensities in the decoder's range. Reported raw reconstructions are multiplied by six; the requested sum is never clipped. One shared image encoder processes both clips independently.

The fresh convolutional VAE has a 128-dimensional diagonal Gaussian latent and a single RGB reconstruction. Its loss is ordinary image MSE plus a coordinate-mean Gaussian KL penalty, warmed to beta 1e-4 over 250 updates. There is no prediction, contrastive, direction or change-label loss in this stage. Raw MSE is 36 times normalized MSE. Sparse-image foreground error and visual reconstructions are retained as diagnostics because ordinary image MSE can be dominated by background.

After completing VAE training and selecting its checkpoint on reconstruction validation, freeze that newly trained VAE in evaluation mode. Concatenate deterministic posterior means `[mu_before, mu_after]` and train a fresh `256 -> 256 GELU -> 128 GELU -> 1` decoder with binary cross-entropy. Change labels enter only this supervised decoder stage. A train-only normalization of latent features may condition optimization; validation/test never fit that normalization.

Both clips continue the same six-frame dot movie. Original simplified speeds 0.375/1/2 and changes 26/28 degrees remain. Labels/directions are renderer metadata during VAE training, not encoder inputs. Fresh split namespaces 190101/190102/190103 and separate mode domains distinguish VAE training, supervised training and evaluation. Classifier training uses balanced independent fresh movies; final evaluation pairs unchanged/changed movies by nuisance context and reports all six cells.

## Execution contract

Implementation is in `SecondPass/WeightedSumVAE/`. This is one fresh two-stage local experiment, not a continuation of the failed predictive VAE. Both stages and evaluation share a new finite four-hour cap beginning at the first GPU profile. Requested targets are 25,024 VAE updates and 10,000 decoder updates; profiling pins feasible exposure before production. Only one local GPU worker runs, with an independent owned-process deadline guard. No cloud, extra arm, automatic budget extension or additional overnight experiment is included.

Researcher agents implement and independently review actual code. CPU checks must include the actual stage transition, validation/checkpoint saving and final evaluation path. A launch requires persisted optimizer progress before being reported as training. All earlier models and results remain preserved.

## Actual execution

**Running locally:** supervisor33384 launched at7:51:46PM PDT, production worker40436 and independent guard42668. New finite budget started 2026-10-04T19:51:50.023232-07:00 and ends 2026-10-04T23:51:50.023232-07:00; both stages share that one deadline. Native profiling pinned the full25,024 VAE updates (391,000 unique pairs,782,000 pair presentations) and10,000 classifier updates (1,280,000 presentations of64,000 independent cached training pairs). Profile weights were discarded. Measured VAE update0.05969seconds plus2.39005seconds/1,000-movie pool, extraction0.002760seconds/pair andclassifierupdate0.003570seconds; reserve2,117.10seconds includes at least30minutes for the second stage.

The complete CPU engineering fixture passed64 VAE and64 classifier updates through both validation/checkpoint paths and finalization. Independent actual-source review passed. Actual GPU validation64 and save succeeded; the root independently loaded VAE best64/latest250 with all20 aligned Adam states and finite model/optimizer tensors. Production continued past1053. Startup loss is not a reconstruction-success claim; final images and gray/mean-image/support diagnostics remain required. Classification has not started and no final held-out score exists yet.

Artifacts: `SecondPass/WeightedSumVAE/LocalRuntime/run01/`. Frozen source snapshot, `profile.json`, `production_config.json`, `root_startup_verification.json`, worker identities and guard receipts establish the actual launch. Source hashes remain unchanged. The nonlinear decoder starts automatically after VAE selection and latent extraction; no additional experiment is queued.

[Current status](CURRENT_STATUS.md) · [Previous predictive result](predictive-consistency-vae.md)
