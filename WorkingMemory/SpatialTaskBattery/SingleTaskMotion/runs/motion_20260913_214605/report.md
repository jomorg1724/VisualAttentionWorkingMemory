# Motion-only continuation

Added motion episodes: 17600. Parent step10000; selected step11760; terminal step12200.

| Delay | Parent BA | Selected BA | Terminal BA | Selected gain (paired95% interval) |
|---|---:|---:|---:|---:|
| D0 | 25.00% | 24.80% | 25.20% | -0.20pp [-3.32, +2.73] |
| D4 | 25.00% | 25.78% | 25.00% | +0.78pp [-1.95, +3.32] |
| D12 | 25.00% | 25.00% | 25.78% | +0.00pp [-2.93, +3.12] |
| D24 | 25.00% | 25.00% | 25.00% | +0.00pp [-3.12, +2.74] |

Each delay has512 scored presentations of the same512 base episodes; repeats across delays are paired, not additional independent episodes. The2,000-replicate bootstrap is class-stratified and conditional on these trained models. All confusion matrices and probabilities remain in the summaries/predictions.

The cloud run contributes eight motion episodes per mixed update, matching the local motion sample count per update. Local training uses a full motion loss; the cloud averages five task losses and combines their gradients. Changes therefore do not isolate gradient interference. Cloud training and its deadline were left unchanged.