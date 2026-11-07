# Frozen predictive encoder + FFN — final result

Two deterministic 512-dimensional representations from the predictor's selected update 50,000 are concatenated into a fresh FFN (LayerNorm, 256/GELU, 64/GELU, two logits). The encoder stayed frozen. Only 281,026 classifier parameters in eight tensors trained, using balanced two-class cross-entropy and Adam 1e-3.

## Completed allocation

All **10,240 planned updates** finished in **23 min 06 s** at 8:05 AM PDT on October 4: **640,000 presentations of 320,000 unique training trials**. Training used batch 64, two epochs per fresh 1,000-trial pool, then pool replacement. The validation-selected checkpoint is **5,632**; latest is **10,240**. Both were independently loaded with all eight Adam states at their respective steps.

## Final held-out comparison

The continuous six-frame stimulus supplies three before frames and three after frames. Each clip has constant direction internally. Half the trials maintain direction; half turn by ±26° or ±28°, at 0.375, 1 or 2 px/frame. Dots keep moving in no-change trials. Adjacent no-change/change examples share the same nuisance variables, complete before clip and first after frame.

The test has **3,072 presentations from 1,536 matched nuisance contexts**, with 512 presentations / 256 contexts in each speed/angle cell. Splits are separate. The selected classifier achieved:

| Measure | Final result |
|---|---:|
| Mean six-cell balanced accuracy | 50.00% |
| Mean six-cell AUC | 0.49519 |
| Cross-entropy | 0.693165 |
| Change sensitivity | 0% in every cell |
| No-change specificity | 100% in every cell |

Every test decision was no-change. The last 100 training batches averaged CE 0.693178. This particular frozen-encoder/concatenation readout did not learn change detection; it does not establish that all motion information is absent from the representation. No end-to-end fine-tuning was performed.

Evidence: [final test](LocalRuntime/run/final_test.json), [run report](LocalRuntime/run/report.json), [checkpoint verification](LocalRuntime/completion_verified.json), [journal](../../LabJournal/predictive-motion-change.md). The worker and deadline guard exited normally. Best/latest and predictive source checkpoints remain local; no model is training.

Classifier checkpoints retain model/Adam/RNG and pool counters but not generic in-pool order/cursor/epoch. The retained best/latest happen to be complete-pool boundaries. No generic mid-pool resume guarantee is claimed. Publication-time guards now reject an existing run directory; this entry-point change was made after training and does not change the completed losses or model.
