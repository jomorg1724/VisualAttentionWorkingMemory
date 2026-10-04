# Variational next-frame predictor — final local result

The model predicts frame four from three ordered frames through a single 512-dimensional Gaussian vector. A fresh residual CNN and latent-only decoder trained on persistent full-field dots with a constant direction per sample and speeds 0.375, 1 and 2 px/frame. The prediction objective balances motion-support/background MSE and adds a small warmed-up KL term.

## Exposure and checkpoint lineage

A short pilot completed 256 updates. The user then explicitly requested substantial training, which continued the complete model, Adam, RNG and data-pool state for 49,984 additional updates. Final totals: **50,240 updates, 1,570,000 presentations and 785,000 unique four-frame samples**. The continuation took about 7 h 44 min and completed at 4:29 AM PDT on October 4, within its new eight-hour cap.

The selected checkpoint is **50,000**; latest is **50,240**. Both were independently loaded with all 138 Adam states at their saved steps. Only best/latest remain. This continuation is separate from the original small pilot result; see the [continuation completion receipt](LocalRuntime/continuation01/completion_verified.json).

## Fresh final prediction test

There were 768 independently generated four-frame sequences, 256 per speed, at test index offset 3,000,000. The validation-selected model was evaluated using deterministic `decode(mu)`; this is not the expectation of nonlinear decoded posterior samples.

| Speed (px/frame) | Predictor balanced MSE | Copy-last balanced MSE |
|---|---:|---:|
| 0.375 | 0.00036810 | 0.00100461 |
| 1.0 | 0.00028531 | 0.00415830 |
| 2.0 | 0.00022482 | 0.00639938 |

Mean balanced MSE was **0.00029275**, versus **0.00385410** for copying the last frame and **0.00548914** for gray prediction: **92.4% lower than copy-last**. See the [full run report](LocalRuntime/attempt01/run/continuation_report.json), [prediction panel](LocalRuntime/attempt01/run/continuation_test_best_speed2.png) and [journal](../../LabJournal/variational-motion-prediction.md).

This establishes next-frame prediction on the new constant-velocity benchmark. It is not a native Krauzlis response result. The subsequent [frozen-encoder FFN comparison](../PredictiveMotionChange/FINAL_REPORT.md) remained at chance, so good predictive MSE did not by itself provide a successful simple change readout.

No model is training as part of this completed run. The continuation implementation preserves complete training state but currently does not bind source hashes (`source_hashes={}`); checkpoint identity, code and saved reports are retained, while stronger source binding was not part of this run.
