# Completed old-task comparison

**Removing the explicit source and distance biases reduced delayed orientation and binding accuracy, without a clear motion improvement at equal training exposure.** Both arms continued the same attention8400 parent for 4,000 updates / 32,000 episodes. These are terminal 12400 results.

| Task / inserted blanks | Biased control | Biases removed | Change (paired 95% interval) |
|---|---:|---:|---:|
| motion_D0 | 41.02% | 42.38% | +1.37pp [-0.98, +3.52] |
| motion_D24 | 41.99% | 42.38% | +0.39pp [-2.54, +3.52] |
| single_D0 | 99.80% | 98.24% | -1.56pp [-2.93, -0.39] |
| single_D4 | 99.80% | 99.22% | -0.59pp [-1.56, +0.20] |
| single_D12 | 96.88% | 82.81% | -14.06pp [-17.58, -10.74] |
| single_D24 | 91.60% | 57.62% | -33.98pp [-38.67, -29.10] |
| binding_D0 | 99.41% | 97.85% | -1.56pp [-2.93, -0.39] |
| binding_D4 | 99.41% | 93.36% | -6.05pp [-8.20, -3.71] |
| binding_D12 | 99.61% | 58.98% | -40.62pp [-43.95, -37.50] |
| binding_D24 | 99.41% | 49.80% | -49.61pp [-52.73, -46.29] |
| binding_D0_locations | 99.61% | 98.24% | -1.37pp [-2.73, -0.20] |
| binding_D4_locations | 99.61% | 94.34% | -5.27pp [-7.23, -3.32] |
| binding_D12_locations | 99.61% | 59.38% | -40.23pp [-43.55, -37.30] |
| binding_D24_locations | 99.61% | 53.91% | -45.70pp [-48.83, -42.38] |

All five unbiased validation checkpoints failed the predeclared preservation screen. The automatic selected checkpoint is therefore the untouched, still-biased 8400 parent. It must not be described as a successful trained unbiased model; the table above reports the actual unbiased terminal model.

Binding at 24 blanks fell to chance in trained locations and nearly chance at held-out locations. Immediate binding remained 97.85%, while delayed binding deteriorated progressively. Orientation shows the same delay dependence. Motion changed by only +1.37 points without blanks and +0.39 points after 24 blanks; both paired intervals include zero.

This tests removal of the source and locality terms together from a model already trained with them. It does not separate their contributions, prove that unforced attention cannot learn, or establish a general biological requirement. The comparison uses one continuation per arm on different GPUs with matching software; previously used test seeds make this exploratory. No model was calibrated or retrained for this report.

Uncertainty: 2,000 paired bootstrap replicates; complete four-case counterbalance blocks for binding, class-stratified episodes otherwise. All 14 test cells are shown; no aggregate score hides the regressions.

The new five-task arm is a separate acquisition experiment and continues unchanged on the same pod toward 160,000 episodes. No result for that arm is claimed here. The original deadline remains 2026-09-14 09:36:25UTC.

Detailed selected/terminal metrics and class confusions: [report.md](report.md). Paired numerical results: [paired_terminal_comparison.json](paired_terminal_comparison.json).