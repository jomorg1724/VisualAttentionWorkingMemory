# Random-frame RViT: completed at chance

All 3,960 updates / 120,000 presentations / 12,000 unique movies completed. Final 100-update mean training loss: 0.684553. Validation selected checkpoint 3500.

Fresh final tests used 200 trials per condition for each checkpoint. Both selected and terminal balanced accuracy are 50% in all three conditions. Selected mean AUC is 0.488279; terminal mean AUC is 0.479090.

| Checkpoint | Condition | Balanced accuracy | AUC |
|---|---|---:|---:|
| selected | B12 | 50.00% | 0.476974 |
| selected | B20 | 50.00% | 0.502448 |
| selected | B28 | 50.00% | 0.485414 |
| terminal | B12 | 50.00% | 0.488780 |
| terminal | B20 | 50.00% | 0.454100 |
| terminal | B28 | 50.00% | 0.494390 |

The sampled-frame gradient scheme and direct memory carry did not produce held-out task learning in this run. This does not establish that either idea is ineffective in general. No further training or cap renewal was launched.

Final artifacts were retrieved and verified, both final checkpoints CPU reloaded, and exact pod `5us0rp5jwmg5bu` deleted. [Retrieval](CloudRuntime/retrieval_verified.json) · [Cleanup](CloudRuntime/cleanup_verified.json) · [Full report](CloudRuntime/artifacts/report.json).
