# Keeping the lab journal current

[Index](README.md) · [Experiment template](EXPERIMENT_TEMPLATE.md)

The journal is a readable record, not a new experiment harness or approval gate. The researcher who implements or analyzes an experiment updates its page alongside the existing report/receipt. Routine updates should take minutes and use saved artifacts; they should not launch inference or delay healthy training.

## At design and launch

Create the next numbered experiment page using the template. Record the question, source rationale, exact intervention/control, parent checkpoint, changed versus retained weights/optimizer/stream semantics, tasks, planned exposure, selection and finite resource allocation. Mark design-only or running. Link the page from the index, chronology and current status. State any change requested by the user that supersedes an earlier plan.

## During a run

Append a dated snapshot when there is a meaningful validation look, failure/recovery or changed interpretation. Keep update counts, episodes and checkpoint identity together. Label validation as provisional and distinguish completed training from completed final evaluation. A live metrics.csv row is evidence of progress, not of test completion. No need to copy every minibatch into prose: link logs and curves.

## At completion

Use the final report, predictions/analysis and completion receipt. Record actual exposure, selected and terminal checkpoints, task-specific BA/AUC/confusion, important paired uncertainty, regressions and resulting decision. Include parent fallback and terminal regressions if relevant. State dataset/split identity and pairing unit; never manufacture pairing across different test seeds. Record actual compute, recovery history and cloud retrieval/cleanup status from receipts. Update CURRENT_STATUS.md and CHRONOLOGY.md, then mark the experiment completed.

## When evidence changes an interpretation

Keep the earlier hypothesis and date the correction. For example, “possible early-motion loss” became “useful early information accessible; refit output first.” Avoid rewriting history as though the correct mechanism was known initially. Do not overwrite immutable run snapshots or recovered negative results. A lightweight dated correction block on the affected page is sufficient.

## Minimal evidence convention

Each measured statement should link the report or numeric artifact that supports it. The initial journal's evidence_manifest.json records source hashes at compilation, not a promise that mutable aggregate files will never change. For later updates, append a short source/timestamp note or extend the manifest. Do not deserialize large checkpoints just for documentation; use their recorded identity when available.

The initial build_initial_journal.py is a one-time documentation assembly script using standard-library file reads only. It copies selected result tables from original reports and authors the surrounding research narrative. **Do not rerun it over later edits.** Update Markdown directly after the baseline, or deliberately revise the builder and preserve subsequent changes.

## Current Stage1 completion handoff

Update experiments/16-training-exposure.md and CURRENT_STATUS.md after both arms have completed evaluation and combined analysis. Report all 14 cells, including held-out centers, and parent/selected/terminal outcomes. Confirm cloud cleanup separately from local completion. The existing completion monitor should ask the responsible researcher to make this update from saved results. It does not authorize a new run, residual training or compute extension.
