# Learned spatial comparison before compression

The authorized contract is [BRIEF.md](BRIEF.md). This experiment adds only a shared local residual comparison of previous ConvGRU memory and current spatial evidence. The recurrent carry stays unchanged. It tests end-to-end acquisition of cue-conditioned comparison, not a handcoded solver, a whole Guided Search system, or biological attention.

## Active execution

Canonical run: `/Users/jonathanmorgan/VAWMRuntime/spatial_comparison_01/run`

- External owner: `gui/501/org.vawm.spatial-comparison-v1-75feeabd250d`, launchd Interactive, KeepAlive=false.
- One shared deadline: **2026-09-26T14:34:42.498420+00:00**, from the first profile at 2026-09-25T14:34:42.498420+00:00. It never renews.
- Control first, candidate second, one MPS worker and shared optimizer lock, at most two CPU threads.
- Pinned **2,600 additional updates / 83,200 episodes per arm**, 200 updates / 6,400 episodes per task. ConvGRU-lineage endpoint 9,360; original shared-trunk global endpoint 12,753.
- Shared fresh baseline, midpoint 1,300 and terminal 2,600 validation, target8 mean AUC then suite equal-task AUC, earlier ties. Baseline remains eligible. Selected and terminal final tests use paired fresh draws across all 35 cells, 128 per cell / 200 per Krauzlis cell.
- Planning used every previous 5,070 production update and every unique completed prior validation/final cell plus one disposable inherited 13-task cycle per arm. Conservative projected remaining time was 21.47 hours, including margins/evaluations/reporting; 1.5x training sensitivity is 25.25 hours and therefore a schedule risk, not a promise of completion.

The source runtime is a separately copied, verified tree outside Desktop, reusing the already-proven local execution path. Old source and all prior models remain unchanged. Profile model/Adam/stream/RNG state is discarded. Both production arms independently restore terminal6760 and its named Adam, queues, streams and RNG. The candidate alone adds 12,416 fresh parameters and empty new-parameter Adam state.

## Files and evidence

- `experiment.py`: minimal model/migration adapters, inherited training/evaluation loop integration, sequential absolute-cap supervisor and saved-state verifier.
- `preflight.py`: focused CPU exact-logit, recurrent-carry, inherited Adam, RNG and next-draw checks. Comparator gradients and saved weight/Adam changes are checked in the actual-launcher disposable profile and automatically at production updates 2 and 13.
- `report.py`: paired final BA/AUC differences and label-stratified bootstrap uncertainty; every task and cell remains visible. Full empty-recognition specificity and Krauzlis event strata remain in each final result.
- Runtime `allocation.json`, `activation.json`, `profile_*/profile.json`, `control/progress.jsonl`, `candidate/progress.jsonl`, per-arm `latest_checkpoint.json` and `persisted_progress_verification.json` are authoritative. Process liveness or disposable profiles are not production progress.
- Automatic final outputs: runtime `REPORT.md`, `comparison.json`, and each arm's `report.json` / `test_selected.json` / `test_terminal.json`. Failure/cap yields `incomplete_report.json` and an explicit incomplete report, not a successful comparison.

Inspect saved production progress without accelerator use:

```sh
cd /Users/jonathanmorgan/VAWMRuntime/spatial_comparison_01/repo
/tmp/vawm-task-suite-venv/bin/python -m SecondPass.SpatialComparisonReadout.experiment verify /Users/jonathanmorgan/VAWMRuntime/spatial_comparison_01/run/control
```

Do not relaunch, kickstart, renew the cap, or start another worker. The supervisor automatically owns the arm transition. An additive read-only checkpoint-verification hook was completed before production pinning; its old/new source digests and unchanged deadline are recorded in `preproduction_checkpoint_receipt_hook.json`.
