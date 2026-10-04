# Current status — October 4, 2026

[Index](README.md) · [History](RESEARCH_HISTORY.md) · [Catalog](EXPERIMENT_CATALOG.md) · [Chronology](CHRONOLOGY.md)

## Latest completed experiment

**AngularContrastiveMotion: completed, simplified comparison solved.** Fresh whole residual CNN, three ordered frames per clip, 128-dimensional normalized embedding, simulator-angle-supervised spring loss. Training completed at 12:06 PM PDT after 3 h 28 min 33 s: 25,024 updates, 782,000 pair presentations, 391,000 unique training pairs. Best checkpoint 19,000; latest 25,024. Both were independently loaded with all 64 Adam states at their saved steps.

The validation-selected distance threshold achieves 100% balanced accuracy and AUC 1.000 on 3,072 fresh comparison presentations: 512 per speed/angle cell, or 1,536 paired nuisance contexts total. Angular test loss is 0.00136432; distance/angle correlation is 0.987835. See the [full technical report](../SecondPass/AngularContrastiveMotion/TECHNICAL_REPORT.md), [results](../SecondPass/AngularContrastiveMotion/FINAL_REPORT.md) and [journal](angular-contrastive-motion.md).

This is one training seed on a new simplified persistent-dot distribution with explicit angular supervision. It does not establish native Krauzlis cue selection, distractor rejection, dot-lifetime robustness or recurrent memory.

## Preceding experiments

| Experiment | Final state | Main result |
|---|---|---|
| Frozen predictive encoder + FFN | Completed all 10,240 updates | 50% BA, AUC 0.49519, CE 0.693165; every test decision was no-change |
| Variational next-frame predictor | Completed 50,240 cumulative updates | Fresh test balanced MSE 0.00029275 vs copy-last 0.00385410 |
| Weighted-mean RViT, two epochs per pool | Completed 2,310 cloud updates; retrieved and pod deleted | Selected BA 50%, mean AUC 0.50397 |
| VAE-encoder RViT input×10 | Completed 330 local updates | BA 50%, mean AUC 0.49000 |
| Three-frame spatial VAE | Completed 9,900 updates | Successful reconstruction test; best 9,500/latest 9,900 retained |

## Runtime and next decision

No current local model is training. The latest ephemeral cloud pods were retrieved and deleted. The briefly held contrastive queue was activated and completed; it is not pending. No new inference, training, cloud rental or compute extension is implied by this status page.

The next research decision is how the learned angular representation transfers to native stimuli and an actual selected-stimulus change decision. That work has not been run. Preserve the successful encoder, its checkpoints and all negative evidence before choosing a transfer experiment.

## Historical status records

[Archived status snapshots through October 4](archive/CURRENT_STATUS_through_20261004.md) preserve previous launches, cancellations and recoveries. Their present-tense wording was valid at the recorded time; it is not current runtime authority. Individual run reports and receipts remain the source for each result. Earlier lineages are cataloged rather than silently treated as fresh training.
