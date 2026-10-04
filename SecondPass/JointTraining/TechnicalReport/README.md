# Spatial KDA architecture and microstimulation — technical report

Ten-page technical research note requested by the user, 23 September 2026.

- **Read:** [architecture_microstimulation.pdf](architecture_microstimulation.pdf).
- **Edit:** [architecture_microstimulation.md](architecture_microstimulation.md); generated LaTeX is also supplied.
- **Measurements:** [metrics_snapshot.json](metrics_snapshot.json), frozen at **2026-09-23 06:48:15 UTC**. Six complete saved evaluations, each covering all 13 tasks / 35 primary conditions. The continuation had reached step 2647, but its latest measured validation was step 2314. Later results are intentionally not folded into this preliminary report.
- **Architecture/data audit:** [architecture_metrics_notes.md](architecture_metrics_notes.md).
- **Paper audit:** [paper_comparison_notes.md](paper_comparison_notes.md), with page-indexed text, inspected Figure 5 and verified quotations in `paper_evidence/`.
- **Checks:** `report_numeric_verification.json` and `build_verification.json`.

## Content

1. Executive assessment and architecture schematic.
2. Layer shapes, shared and task-specific computation, exact parameter/state counts.
3. KDA update equations, conditional retention bound, readout bottleneck and raw-frame history.
4. Native task/report semantics and training exposure.
5. All-task preliminary metrics, explicit chance levels and degenerate response policies.
6. Saved validation curves, checkpoint selection and inference limits.
7. Morgan, Albanna & Herman Figure 5: documented intervention and limits of equivalence.
8. Proposed emission and associative-state pulses, independent calibration, dose and controls.
9. Paired site/epoch/dose design; psychometrics, sensitivity, criterion and uncertainty.
10. Expected outcomes, biological distinction, interpretation and reproducibility sources.

## Rebuild without training or inference

Requirements: Python with `matplotlib` and `PyMuPDF` (`fitz`), Pandoc, Tectonic, and Times New Roman / Arial / Menlo fonts used on this Mac. Fonts are embedded in the delivered PDF; exact typesetting on a different platform requires installing equivalent fonts or changing the YAML/header.

From this directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /tmp/vawm-task-suite-venv/bin/python verify_report_numbers.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /tmp/vawm-task-suite-venv/bin/python build_report.py
```

Replace the interpreter path with an appropriately provisioned local Python if using the source archive elsewhere. Figure generation reads the frozen JSON, not live training progress. No checkpoint or accelerator is required for rebuilding. `audit_saved_metrics.py --verify` additionally checks recorded original source/checkpoint hashes inside the original repository; it is not a standalone archive-only verification.

## Verification and scope

- All 39 displayed task-table values checked programmatically against the snapshot, using round-half-up for displayed percentages.
- Six saved 35-cell evaluations checked for exact cell inventory, confusion/stratum denominators and aggregate consistency; parameter and checkpoint-shape audit recorded separately.
- Citation ledger verified with literal source evidence for both external references.
- Ten PDF pages verified, with section/page alignment, text presence, bounds and rendered visual inspection.
- The report makes **no measured microstimulation claim**. Proposed interventions were not implemented or run, model weights and active training artifacts were not changed, and no additional GPU or cloud work was launched.
- Completed v2 tests and repeated v3 validation are explicitly separated. The continuation is exploratory; new episode draws do not create new held-out photograph identities.
