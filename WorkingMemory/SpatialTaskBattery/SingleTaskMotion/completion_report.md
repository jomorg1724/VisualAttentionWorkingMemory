# Completed motion-only experiment

**The model did not learn this task under the tested continuation.** Another 17,600 motion-only training episodes left held-out direction accuracy near the 25% chance level, including trials with no inserted blank delay.

| Blanks | Parent BA | Selected BA | Terminal BA | Terminal AUC | Terminal prediction counts (classes0/1/2/3) |
|---|---:|---:|---:|---:|---|
| D0 | 25.00% | 24.80% | 25.20% | 0.521 | [87, 365, 0, 60] |
| D4 | 25.00% | 25.78% | 25.00% | 0.521 | [1, 2, 0, 509] |
| D12 | 25.00% | 25.00% | 25.78% | 0.514 | [0, 43, 21, 448] |
| D24 | 25.00% | 25.00% | 25.00% | 0.518 | [0, 256, 256, 0] |

The selected checkpoint 11760 used 14,080 added motion episodes; the terminal checkpoint 12200 used all 17,600. Both are shown because validation selection did not produce a meaningful held-out improvement. The parent had already received 12,800 motion episodes in the new battery.

| Blanks | Selected minus parent BA | Terminal minus parent BA | Terminal minus selected BA |
|---|---:|---:|---:|
| D0 | -0.20pp [-3.33, +2.73] | +0.20pp [-3.71, +3.71] | +0.39pp [-2.54, +3.32] |
| D4 | +0.78pp [-1.95, +3.32] | +0.00pp [-2.15, +2.15] | -0.78pp [-3.12, +1.56] |
| D12 | +0.00pp [-2.93, +2.93] | +0.78pp [-2.73, +4.30] | +0.78pp [-1.37, +2.93] |
| D24 | +0.00pp [-3.12, +2.93] | +0.00pp [-4.10, +4.30] | +0.00pp [-3.12, +3.12] |

Intervals use 2,000 class-stratified paired bootstrap replicates. Each delay reuses the same 512 held-out base episodes; all paired labels and generation metadata match across models. All gain intervals include zero. Confusions show severe restriction to a small subset of outputs; selected D12/D24 predicts a single class throughout. Near-chance AUC also shows little useful ranking of correct directions in these final outputs.

Training losses and gradients were finite throughout; first/last 200-update mean CE was 1.3916/1.3877, compared with log(4)=1.3863. The run completed in 56.4minutes, within its 120-minute cap. No local model worker remains; no additional GPU test was launched.

The cloud five-task validation trajectory also remains around chance through checkpoint 11600. Its validation samples differ from the local set, so the trajectory comparison is descriptive. Full motion CE versus averaged five-task gradients changes optimization; failure of this continuation neither proves a task implementation bug nor establishes an architecture capacity limit. It does show that this amount of additional task-only training did not solve acquisition. Failure already at D0 cannot be explained solely by losing information during blank intervals.

The cloud run and its deadline remain unchanged. See trajectory_comparison.md for exposure-normalized cloud/local validation points and completion_findings.json for full confusions, probability-ranking scores and paired comparisons.