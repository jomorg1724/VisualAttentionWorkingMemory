# Spatial versus dense E/I comparison

The spatial competitor learned the new location-binding task substantially better: at24 inserted blank frames, balanced accuracy was93.36% versus49.61% for dense, a paired gain of43.75percentage points (95%CI39.84–47.86). At unseen location centers it scored94.53% versus49.61%. This supports keeping the spatial model as a binding candidate, but it is not a general replacement: motion decisions deteriorated, and both models remained near chance on the native single-item task atD24.

Single-item D12 improved to79.69% for spatial, versus46.48% for the newly trained dense competitor and63.28% for the untouched parent on the exact same examples. AtD24, spatial scored49.80%, dense48.24%, and parent51.37%. The binding and native single-item tasks differ in rendering, change sizes and query content, so their contrast does not establish a two-item capacity advantage.

Both models completed 35,200 new training episodes under the fixed shared schedule. Checkpoints were selected using validation only. The table reports paired held-out evidence, with confidence intervals clustered by the binding four-case blocks.

| Condition | Dense BA% | Spatial BA% | Spatial−dense pp [95%CI] |
|---|---:|---:|---:|
| single_D0 | 96.29 | 97.85 | +1.56 [-0.20, +3.32] |
| single_D4 | 92.77 | 97.27 | +4.49 [+1.95, +7.03] |
| single_D12 | 46.48 | 79.69 | +33.20 [+27.73, +38.29] |
| single_D24 | 48.24 | 49.80 | +1.56 [-1.18, +4.49] |
| binding_D0 | 85.16 | 96.29 | +11.13 [+7.81, +14.26] |
| binding_D4 | 82.23 | 95.31 | +13.09 [+9.57, +16.99] |
| binding_D12 | 52.73 | 94.92 | +42.19 [+37.89, +46.48] |
| binding_D24 | 49.61 | 93.36 | +43.75 [+39.84, +47.86] |
| binding_D0_locations | 84.38 | 96.48 | +12.11 [+9.18, +14.84] |
| binding_D4_locations | 81.84 | 96.88 | +15.04 [+11.91, +18.16] |
| binding_D12_locations | 50.98 | 95.51 | +44.53 [+41.02, +47.85] |
| binding_D24_locations | 49.61 | 94.53 | +44.92 [+41.02, +48.63] |
| motion_D0 | 70.12 | 46.68 | -23.44 [-27.54, -18.95] |
| motion_D24 | 67.58 | 25.00 | -42.58 [-46.29, -38.67] |

[Comparison curves](comparison_curves.png) · [Full BA/AUC/confusions/spacing and paired intervals](analysis.json) · [Validation trajectories and run records](results.json)

## Preserved parent and motion trade-off

The unchanged Retention14800 parent was evaluated after model selection on the same512 base examples per existing-task cell. This was frozen inference only, with no new fitting or binding zero-shot test. Parent-to-competitor differences combine added training and the model changes; they do not isolate a comparator effect.

| Existing task | Parent BA% | Dense BA% | Spatial BA% |
|---|---:|---:|---:|
| Single D0 |95.51|96.29|97.85|
| Single D4 |92.58|92.77|97.27|
| Single D12 |63.28|46.48|79.69|
| Single D24 |51.37|48.24|49.80|
| Motion D0 |79.10|70.12|46.68|
| Motion D24 |77.54|67.58|25.00|

Spatial's single-D12 gain over the parent was16.41pp (paired95%CI11.91–21.09). Its motion-D24 loss was52.54pp (48.82–56.45); dense also lost9.96pp (6.25–13.87). Motion received10% of training updates and was excluded from checkpoint selection, so this screen prioritized the orientation tasks rather than preserving all prior skills. Spatial motion-D24 AUC remained0.762 despite25% argmax accuracy: poor decisions do not establish that all motion information vanished. Full parent scores and paired intervals are in [parent_reference.json](parent_reference.json).

The practical result is a promising spatial binding specialist with a clear retention/decision trade-off. Preserve both the new candidate and the old parent; these data do not justify discarding the trained motion system. No further training was launched.

[Paired parent comparison figure](parent_tradeoff.png)

## Exposure and resources

- dense_comparator: selected update 4400 (35,200 new episodes); terminal 4400 updates, 556,160 logical frames. Measured training collection/BPTT 3524.4s; clipping on 97.2% of updates; median early-rate gradient 0.043248. Selected per-cell exposure is retained in analysis.json.
- spatial_ei: selected update 4400 (35,200 new episodes); terminal 4400 updates, 556,160 logical frames. Measured training collection/BPTT 3915.9s; clipping on 83.8% of updates; median early-rate gradient 0.015563. Selected per-cell exposure is retained in analysis.json.

Both selected checkpoints are the terminal update4400. Each primary task/delay cell received3,960 episodes; each motion cell received1,760. Dense validation mean primary AUC rose0.6380→0.6747→0.7155→0.7223; spatial was0.8412→0.9179→0.9160→0.9286. High clipping frequency and nonzero early-state/encoder gradients are optimization observations, not proof that the training horizon was sufficient. The two models have797,474 and528,538 parameters respectively; spatial's42.25-times larger recurrent state is distinct from parameter count. Measured peak allocated GPU memory was567.6MB dense and606.7MB spatial, excluding desktop/context use.

The main experiment and saved-score report took8,321.1s, and the optional parent reference took99.1s. The final completion receipt includes all elapsed reporting time against the original14,400s deadline, selected checkpoint/source hashes, and process-exit evidence. All computation used sequential local GPU workers, fp32 and two CPU threads; no cloud was used.

## Interpretation limits

Dense inherits its trained memory core whereas the spatial core is newly initialized. Both inherit learned sensory weights and receive equal new exposure, but comparator geometry, state size and parameter sharing also differ. This compares practical trained systems; it does not isolate locality or establish a biological mechanism. Spatial states contain 42.25 times as many new memory scalars; the common sensory traces remain additional persistent history. All learned components were allowed to adapt.

The binding inventory and probe marginals are balanced; label cannot be solved from orientation inventory or probe alone. However, because both items always exchange positions on swap trials, retaining and checking just one location can solve the task. These results test feature-location association and do not prove both items were stored or measure two-item capacity. Binding also uses localized patches and 15/30/60-degree separations, whereas native single-item recall includes 7.5-degree differences, a full-field grating and different query content; cross-task scores are not a controlled load curve.

New phases/noise remove literal raster matching. The held-out center grid tests spatial interpolation, not extrapolation. Near/far results are descriptive strata with their actual denominators. Four-case balancing causes within-block dependence, accounted for in the primary bootstrap. Location grids are summarized separately, never pooled as independent repeats. The finite acquisition horizon does not establish ultimate capacity or failure to learn. No unrequested follow-up or cloud run was launched.

Final closure: 8591.8s (143.2min) elapsed within the14,400s allowance. All owned GPU workers exited; the original source inventory still matches its pinned hashes. The independent reviewer accepted the final interpretation. Both selected checkpoint identities and the original primary completion receipt are preserved.
