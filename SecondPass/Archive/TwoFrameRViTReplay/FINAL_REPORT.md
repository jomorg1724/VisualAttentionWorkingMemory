# Older cloud RViT — final report

Reported 2026-10-03T03:12:48.794988+00:00.

Completed 2,310 updates / 70,000 presentations; validation selected update 1,250. Training loss averaged **0.49881** over the final 100 updates.

Fresh final tests use 200 trials per condition. These are independent of training and validation.

| Checkpoint | Condition | Balanced accuracy | AUC | Target hit | Foil false alarm | Catch false alarm |
|---|---|---:|---:|---:|---:|---:|
| selected | B12 | 50.0% | 0.4662 | 100.0% | 100.0% | 100.0% |
| selected | B20 | 50.0% | 0.5090 | 100.0% | 100.0% | 100.0% |
| selected | B28 | 50.0% | 0.4265 | 100.0% | 100.0% | 100.0% |
| terminal | B12 | 47.1% | 0.4824 | 57.0% | 62.1% | 64.3% |
| terminal | B20 | 50.6% | 0.5105 | 57.0% | 55.2% | 57.1% |
| terminal | B28 | 45.8% | 0.4837 | 50.9% | 62.1% | 53.6% |

Training within each repeated pool:

| Pool | Unique movies | First epoch mean loss | Last epoch mean loss |
|---|---:|---:|---:|
| 1 | 1,000 | 0.69087 | 0.66776 |
| 2 | 1,000 | 0.68429 | 0.20551 |
| 3 | 1,000 | 0.75993 | 0.68244 |
| 4 | 1,000 | 0.68383 | 0.66352 |
| 5 | 1,000 | 0.69230 | 0.66112 |
| 6 | 1,000 | 0.69001 | 0.42636 |
| 7 | 1,000 | 0.74355 | 0.36000 |

The within-pool losses measure fitting reused training movies; fresh test scores measure generalization.

The selected model does not meet the recorded acquisition criterion of at least 70% balanced accuracy in every condition.

A longer continuation is a possible next experiment. This report does not extend the existing budget or launch another run.

[Original final evidence](CloudRuntime/artifacts/REPORT.md) · [Verified retrieval](CloudRuntime/retrieval_verified.json)
