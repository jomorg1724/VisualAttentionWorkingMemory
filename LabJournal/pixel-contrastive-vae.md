# Three-frame VAE with a pixel-derived pair objective

October 4, 2026. User-authorized full local training. The pixel-loss arm completed and was tested. The user subsequently cancelled the plain-VAE control to proceed with predictive learning. Its latest checkpoint is update 7,250 (last live update 7,268), with all 20 compatible Adam states preserved; no final control test was run. The matched comparison is therefore incomplete. Earlier launch statements below are historical.

The user proposes replacing explicit direction/direction-difference targets with similarities derived from decayed raw frames, and supplementing a three-frame VAE with that pair objective. The two clips continue one movie; the second sometimes changes direction. A fresh small VAE and a matched plain-VAE control will train sequentially with no inherited weights or cloud use.

The pixel target is cosine dissimilarity of centered exponential traces with weights 1/7, 2/7, 4/7. The pair objective matches deterministic embedding dissimilarity to this detached target. This is pixel-derived metric regression. Neither simulator angles nor change labels enter the model's training loss.

A pre-training CPU assay found that the raw target measures positional overlap/travel as well as motion: slow turns sometimes make clips more similar, while faster speeds nearly saturate the score. The proposed loss is retained for a direct empirical test, with raw-pixel scoring and a plain VAE as controls. No direction-based substitute or additional variance-floor objective is included.

The finite total local budget is eight hours from the first GPU profile. Profiling measured about0.140s/control update and0.142s/joint update, plus2.60s to render a1,000-pair pool. It pinned the full25,024 updates per arm before production:391,000 unique pairs and782,000 presentations per model. Shared budget started2026-10-04T21:07:01.519Z and ends2026-10-05T05:07:01.519Z; it does not renew. Production worker24780 and independent owned-process guard29628 were launched at21:09:26UTC. The joint arm trains first, control follows. All22 joint Adam states and learned tensor changes were verified at the first saved update; the parent independently loaded saved weights/Adam on CPU. Existing trained progress remains preserved.

One independent actual source review found and corrected a final-evaluation checkpoint error before production: loading selected weights while keeping terminal Adam must never cause the exception handler to overwrite latest. The worker now suppresses that save after the selected weights load. A post-profile bookkeeping amendment added explicit Adam coverage assertions and moved final draws to untouched test indices36,000,000. `production_profile.json` records old/new source hashes and unchanged measured computation, original budget, target and deadline; `profile.json` is preserved. No second budget or GPU profile was started.

Runtime artifacts live in `SecondPass/PixelContrastiveVAE/LocalRuntime/run_20261004_2107/`; source copies, launch receipt, profile, guard identity, production proof and CPU verification are retained. Completion, validation selection and final per-cell outcomes remain pending.

See [protocol](../SecondPass/PixelContrastiveVAE/README.md). This experiment concerns simplified coherent motion, not cue selection or native Krauzlis behavior.

## Windows file-lock failure and exact recovery

The original worker stopped at update2,873 when Windows denied atomic replacement of `live_status.json`; weights/gradients were finite. The failure handler preserved latest weights, all22 Adam states at2,873, RNG states and the exact epoch/order/cursor (pool44, epoch1, cursor800). A versioned resume restored the remaining200 presentations/7updates of that epoch without resampling. Resume worker20352, guard15636 and awake helper22396 launched at21:30:34UTC. Test count3,072, update target25,024 perarm and original05:07:01UTC deadline remain unchanged; no new profile or budget. At21:41:22UTC production had advanced to5,708 updates, with no error in resume.stderr.log.

File replacements now retry transient Windows sharing denials. Optional live status can warn/skip after bounded retries; critical checkpoint/receipt writes must still succeed. CPU fault injection verifies that distinction, checkpoint preservation and bitwise reproducible restored CPU updates. Original launch/profile/failure records remain available alongside `resume_worker_receipt.json`, `resume_cpu_check.json` and versioned source copies. This recovery is not a new scientific arm.

## Pixel-loss arm completed; control remains active

At23:05UTC/4:05PM PDT, verified pixel_contrastive completion at25,024updates, selected25,024. Final3,072presentations/1,536pairedcontexts: posterior BA50.46%, AUC0.509398; projection BA51.27%, AUC0.505942; raw pixel score withvalidationcalibration BA60.84%, AUC0.550024. The learned cosine scores remain near chance on this distribution. This does not rule out predictive or other self-supervised objectives. Plain-VAE control observed5,824/25,024updates, latest5,000validation near chance; finaltestpending. PredictiveConsistencyVAE queue34400 remainsalive/pending, no newbudget/profile. [Final arm report](../SecondPass/PixelContrastiveVAE/LocalRuntime/run_20261004_2107/pixel_contrastive/FINAL_REPORT.md).
