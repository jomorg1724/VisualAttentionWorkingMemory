# Frozen neuroscience atlas

Open [index.html](index.html) for scientific figures and [viewer.html](viewer.html) for actual per-trial per-timestep maps. [Neuroscience_atlas.pdf](Neuroscience_atlas.pdf) has one figure per page. [FINDINGS.md](FINDINGS.md) states measured outcomes and missing comparisons.

## Data and reproduction
- `data/trials.jsonl`: complete logits, labels, predictions, native metadata and pair IDs. `trial_scores.csv` is normalized export.
- `data/paired_causal_effects.csv`, `condition_summary.csv`, allocation and psychometric CSVs: exact numerical plot sources.
- `maps/*.npz`: real float32 images/gates/signed coefficients, masks, axes and reconstruction errors; matching JSON metadata.
- `data/calibration.npz`, `calibration.json`: frozen independent feature localizer and held-out validation.
- `intervention_grid.json`, `achieved_interventions.jsonl`: sites/epochs/doses and achieved RMS.
- `budget.json`, `plan.json`, `execution.json`, `completion.json`: fixed cap, profiled/prepinned grid, actual exposure and completion.

CPU tests: `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python -m pytest SecondPass/SpatialReadout/NeuroscienceAnalysis/test_analysis.py -q`. Plot tests use system `python3 -m pytest .../test_render.py -q`. CPU preparation: same torch interpreter `-m SecondPass.SpatialReadout.NeuroscienceAnalysis.run --prepare`. The completed `--execute` refuses budget renewal; rerunning requires new explicit authorization, not deleting the budget. Regenerating only figures uses existing system `python3 SecondPass/SpatialReadout/NeuroscienceAnalysis/render.py`; do not use this to evade the original cap.

No files outside this analysis subtree were modified. Reference guide is independently authored. Scientific caveats: not softmax attention, not physiological inhibition/current, not classical cue-validity or reaction-time analysis. See captions and findings.

## Final verification
`verification.json` checks every scored row, full paired-condition coverage, map axes, mask equality, PDF pages and the actual offline viewer. `model_replay_verification.json` checks CPU replay and untouched finest stored state with downstream propagation. `pulse_verification.json` checks achieved calibrated RMS and one-update timing. `exposure_audit.json` counts all accelerator presentations, including parity calls. `checks/viewer_screenshot.png` is an actual standalone-browser capture. No accelerator worker remains.

Reproduce reporting from saved data with `python3 .../render.py`, then `python3 .../verify.py`; optional CPU-only substrate replay uses the torch interpreter with `-m SecondPass.SpatialReadout.NeuroscienceAnalysis.verify_model`. `python3 .../finalize.py` writes the final audit notes/manifest. Original deadline guards intentionally prevent unapproved execution after expiry. Browser runtime shutdown warnings do not invalidate the verified DOM/screenshot.
