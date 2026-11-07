# Local structured-motion RViT: completed

Completed660updates/20,000presentations/2,000unique movies. Validation selected250. Fresh final tests200/cell: selected mean balanced accuracy50.37%, terminal50.67%.

| Checkpoint | Cell | Balanced accuracy | AUC |
|---|---|---:|---:|
| selected | B12 | 49.64% | 0.535700 |
| selected | B20 | 50.38% | 0.522032 |
| selected | B28 | 51.08% | 0.492146 |
| terminal | B12 | 54.17% | 0.564667 |
| terminal | B20 | 51.69% | 0.604141 |
| terminal | B28 | 46.14% | 0.424929 |

This small one-seed run did not show consistent task generalization across all three native conditions. Progress and final artifacts are preserved in the original local runtime. No continuation was launched. The new weighted CNN–GRU starts fresh on a different, single-stimulus diagnostic task.
