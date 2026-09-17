# Complementary local frozen-core readout diagnostic

**Status: complete; analysis-only/post-hoc.** No deployed model weight was updated and the active cloud run was not queried or modified.

## Design and exposure

The frozen source was motion-only step 12200, SHA256 `7cec4c48c3da81b4d4935c65098b58974a38a8ff3a825dfe12e4d8d16e9a0e26`. It had received 30,400 cumulative motion episodes. Encoder, opponent traces, original pre-update attention, spatial E/I memory and comparator stayed in eval mode. Both fresh probes consumed the identical cached float32 `H_T/R_T/C_T` fields.

Independent base episodes per delay were 512/128/256 for train/validation/test, using seeds 731091/731092/731093. The four delay presentations of each base movie were grouped. Probe A has 74,303 parameters; probe B has 74,373 parameters (70, 0.094%, more).

Both probes selected epoch 20 at validation-only look 3: 640 optimizer updates and 40,960 repeated training presentations each. Both were run through the fixed epoch-40 validation schedule (1,280 updates / 81,920 presentations each) before restoring the selected states. These presentations reuse 512 independent training base episodes across epochs and four delays; they are not fresh episodes.

## Once-only held-out test

| Probe | Accuracy | Balanced accuracy (95% CI) | Macro OVR AUC (95% CI) |
|---|---:|---:|---:|
| Pooled | 23.83% | 23.83% [21.48%, 26.27%] | 0.4805 [0.4493, 0.5122] |
| Spatial priority | 24.32% | 24.32% [22.07%, 26.66%] | 0.4832 [0.4543, 0.5112] |

Paired spatial-minus-pooled BA is +0.49%, 95% grouped bootstrap CI [-1.76%, +2.64%].

| Delay | Pooled BA | Spatial BA | Spatial−pooled BA (95% CI) | Pooled / spatial AUC |
|---|---:|---:|---:|---:|
| D0 | 22.27% | 22.66% | +0.39% [-4.30%, +4.69%] | 0.4889 / 0.4885 |
| D4 | 25.00% | 24.22% | -0.78% [-5.08%, +3.91%] | 0.4707 / 0.4682 |
| D12 | 23.05% | 25.39% | +2.34% [-0.78%, +5.47%] | 0.4774 / 0.4749 |
| D24 | 25.00% | 25.00% | +0.00% [+0.00%, +0.00%] | 0.4756 / 0.4916 |

## Priority-map alignment

Mean target-region mass was 13.13% (95% CI 11.88%–14.56%); peak hit rate was 20.80% (95% CI 17.58%–24.12%). Uniform-map expectation is 5.33%. Target coordinates were used only for this post-test metric, not training. Maps are exposed in
[`priority_maps_test.npz`](runs/diagnostic_20260915_0350/priority_maps_test.npz)
and [`predictions_test.jsonl`](runs/diagnostic_20260915_0350/predictions_test.jsonl).

| Delay | Target-region mass | Peak-hit rate |
|---|---:|---:|
| D0 | 15.54% | 17.58% |
| D4 | 18.02% | 32.03% |
| D12 | 13.21% | 30.86% |
| D24 | 5.75% | 2.73% |

## Interpretation and limits

This is one frozen checkpoint, one stimulus generator and one probe initialization. A post-hoc probe can demonstrate decodable information for its function class, not that the deployed output uses it, that the map is causal attention, or that end-to-end cloud training will obtain the same result. The pooled comparator is capacity-matched within 0.094%, not architecturally identical in inductive bias. The 3×3 cue region is an approximate feature-grid projection rather than pixel attribution. Across-delay presentations repeat base evidence and are not counted as independent episodes.

Production computation took 414.6 seconds; total bounded wall time from the first smoke extraction through finalization was 738.5 seconds. See
[`results.json`](runs/diagnostic_20260915_0350/results.json),
[`run_manifest.json`](runs/diagnostic_20260915_0350/run_manifest.json), feature
receipts and selected probe checkpoints for reproducible details.
