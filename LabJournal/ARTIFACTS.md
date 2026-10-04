# Artifacts, local checkpoints and reproduction

[Index](README.md) · [Latest technical report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md)

## Versioned research record

Git includes the source implementations, experiment designs, final and intermediate result JSON, structured training progress, run budgets/guards/retrieval receipts, authored analyses, stimulus demos, figures and documents. The experiment catalog links source evidence rather than inferring completion from an old authorization note. Source-only runtime snapshots are retained when they record deployed code.

Generated browser/dependency binaries, compiler caches, disposable test directories, Finder metadata and stdout logs that duplicate structured progress are ignored. They remain on disk. Research outputs, authored plans, and stderr failure records are preserved. Raw artifacts and portable demo/report archives are included where already present; some are large. No checkpoint was deleted for this documentation/publication task.

`.gitattributes` keeps file bytes unchanged during Git checkout, important for historical source/checkpoint receipts. Existing `*.pt` and `*.npz` exclusions continue to apply. A cloned repository contains reports/identities, not the local learned weights or cached feature arrays. Archives do not secretly reintroduce model weights.

## Latest retained weights

| Model | Best | Latest | Where the local files live |
|---|---:|---:|---|
| Angular contrastive CNN | 19,000 | 25,024 | `SecondPass/AngularContrastiveMotion/LocalRuntime/run/{best,latest}.pt` |
| Predictive encoder + FFN | 5,632 | 10,240 | `SecondPass/PredictiveMotionChange/LocalRuntime/run/{best,latest}.pt` |
| Variational next-frame predictor | 50,000 | 50,240 | `SecondPass/VariationalMotionPredictor/LocalRuntime/attempt01/run/{best,latest}.pt` |
| Three-frame spatial VAE | 9,500 | 9,900 | `/Users/jonathanmorgan/VAWMRuntime/three_frame_conv_vae_local01/resume01/{best,latest}.pt` |
| Weighted RViT, two epochs | 2,000 | 2,310 | `SecondPass/WeightedMeanRViTEpoch2/CloudRuntime/artifacts/{best,latest}.pt` |
| VAE–RViT input×10 | 330 | 330 | `/Users/jonathanmorgan/VAWMRuntime/vae_rvit_input10_local01/run/{best,latest}.pt` |

The latest contrastive [checkpoint manifest](../SecondPass/AngularContrastiveMotion/LocalRuntime/checkpoint_manifest.json) records exact byte sizes/SHA256s. [Completion verification](../SecondPass/AngularContrastiveMotion/LocalRuntime/completion_verified.json) binds saved steps, all64 Adam states and final metrics. FFN/predictor receipts separately document their states. Some older checkpoints were removed by the user's October3 cleanup; historical scores/reports do not imply those weights are still recoverable. No deleted lineage is reconstructed here.

## Reference environment

The latest model ran on an Apple M4 Max using MPS, FP32 and two CPU threads, with Python3.11.15, PyTorch2.8.0, NumPy2.4.6 and Pillow12.3.0. [Documented environment](../SecondPass/AngularContrastiveMotion/LocalRuntime/environment_documented.json) records the actual local platform. Matplotlib3.11.2 was installed after training only to plot saved results; it was not a training dependency.

No inference is necessary to read/rebuild the plots:

```sh
python -m SecondPass.AngularContrastiveMotion.render_learning_curves
python -m unittest SecondPass.test_publication_guards -v
```

The plotting command reads committed progress/validation JSON and produces PNG/SVG, with no model or accelerator execution. Install the root requirements in a separate compatible Python environment before using those commands. The reference versions above describe the measured run; cross-platform bitwise equivalence is not claimed.

## Training and checkpoint limits

The latest [launcher](../SecondPass/AngularContrastiveMotion/launch.py) and [worker](../SecondPass/AngularContrastiveMotion/train.py) preserve the original macOS runtime paths. Fresh-machine execution requires adapting the interpreter and local GPU-lock path, and selecting an unused run directory. Importing source does not start training. Invoking the launcher starts a paid-in-time local experiment with an8h ceiling and independent guard, so it should only be used for an explicitly requested new run. The completed repository's default run directory is already populated and is intentionally rejected.

Only model, loss and data code were reused in the successful from-scratch run. The weights and Adam are fresh. The model accepts `[B,3,3,100,100]` FP32 pixels and returns `[B,128]`; directions supervise the loss only. To evaluate existing weights, load the local `best.pt` model state and use its saved validation threshold. Test evaluation now rejects a missing threshold, preventing accidental fitting on test labels.

The FFN worker now rejects existing attempts before any metadata rewrite. That safeguard and the angular test-threshold guard were added after training; they do not change completed results. Generic mid-pool FFN resume is not implemented. The predictive continuation preserves full RNG/Adam/pool state but its source hash map was empty; its measured outputs remain documented without claiming immutable source binding. Training receipts retain original machine-local paths and source references as historical evidence.

No reproduction or new training is executed by this documentation update. All current local workers have completed; the latest cloud pods were retrieved and deleted.
