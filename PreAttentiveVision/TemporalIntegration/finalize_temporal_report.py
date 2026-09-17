"""Render a concise Markdown account from saved temporal results; no model imports."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def percent(value):
    return f'{100 * value:.2f}%'


def interval(cell, key):
    values = cell.get(key)
    return 'not estimated' if values is None else f'[{100 * values[0]:.2f}, {100 * values[1]:.2f}]'


def render():
    result = json.loads((HERE / 'results_temporal.json').read_text())
    root = Path(result['run_root'])
    if result['status'] != 'completed':
        raise RuntimeError('Final report requires completed run evidence')
    exit_record = json.loads((root / 'exit.json').read_text())
    if exit_record['status'] != 'completed':
        raise RuntimeError('Supervisor completion receipt required')
    cfg = result['config']
    tasks = list(cfg['task_classes'])
    runs = result['runs']
    reference = result['pair_reference']
    passed = [r['model'] for r in runs if r['test']['engineering_criterion_met']]
    lines = ['# Causal temporal integration: measured comparison', '',
        'All three accumulators completed the same fresh training exposure with the selected spatial encoder frozen. '
        + ('The per-task 95% point-estimate screen was met by ' + ', '.join(passed) + '.' if passed else
           'None met the 95% balanced-accuracy point-estimate screen on every task.'), '',
        'This tests learned two-step sensory integration with these fitted systems. It does not establish long-duration memory, '
        'a population architecture ranking, or a biological mechanism. A spending cap is not a sufficient acquisition horizon.', '',
        'The opponent model scored100% on motion and spatial frequency, and98.44–99.78% on the other five tasks. '
        'KDA scored31.70% on motion while scoring96.88–99.33% on the other tasks; ConvGRU scored75.67% on motion and94.64–99.11% elsewhere. '
        'These are outcomes after the stated exposure, not claims that the other architectures cannot learn motion.', '',
        '## Held-out task performance', '',
        'Balanced accuracy, fixed argmax decisions. Each column uses exactly the same 448 fresh generated pairs per task. '
        'The preserved reference retains the original direct pair decoder; the three new models receive only the current field and their causal state.', '',
        '| Task | Preserved pair reference | ' + ' | '.join(r['model'] for r in runs) + ' |',
        '|---|' + '---:|' * (len(runs) + 1)]
    for task in tasks:
        cells = [percent(reference['tasks'][task]['overall']['balanced_accuracy'])]
        cells += [percent(r['test']['tasks'][task]['overall']['balanced_accuracy']) for r in runs]
        lines.append('| ' + task + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Task AUC and uncertainty are retained in results_temporal.json and the interactive HTML report. '
        'There is no pooled raw accuracy across binary and four-class tasks.', '',
        '## Paired changes from the preserved reference', '',
        'Differences are percentage points in balanced accuracy, with paired 95% percentile bootstrap intervals. '
        'Intervals resample procedural pairs or natural-image source photos, preserving each model/reference pairing. '
        'The two-percentage-point retention tolerance is descriptive; this is not a formal noninferiority analysis.', '',
        'For the opponent model, contour BA increased3.35percentage points versus the preserved reference (paired95% interval+1.58 to+5.05). '
        'Natural-image spectral BA decreased1.12points (source-photo-cluster interval−2.40 to−0.22). '
        'All point differences meet the descriptive two-point retention tolerance, but the natural-image interval extends beyond it. '
        'The contour increase is a system-level comparison that also includes new modules/readout and more training; it does not isolate a benefit of the fixed energy channels.', '',
        '| Accumulator | Task | Delta BA, pp | 95% interval, pp | Delta AUC |',
        '|---|---|---:|---:|---:|']
    for comparison in result['paired_comparisons']:
        for task in tasks:
            cell = comparison['tasks'][task]
            lines.append(f"| {comparison['model']} | {task} | {100*cell['delta_ba']:+.2f} | {interval(cell, 'delta_ba_ci95')} | {cell['delta_auc']:+.4f} |")
    lines += ['', '## Acquisition and checkpoint selection', '',
        'Checkpoints were selected on validation only: highest minimum task BA, then mean task OVR-AUC, with exact ties resolved in favor of the earlier checkpoint. '
        'The fixed training exposure was completed regardless of early scores. Final test results did not choose the checkpoint.', '',
        '| Accumulator | Selected update | Terminal update | Validation minimum BA at successive looks | Validation mean AUC at successive looks |',
        '|---|---:|---:|---|---|']
    for r in runs:
        lines.append(f"| {r['model']} | {r['best_step']} | {r['step']} | " +
            ', '.join(f"{v['step']}: {percent(v['min_task_ba'])}" for v in r['val_curve']) + ' | ' +
            ', '.join(f"{v['step']}: {v['macro_ovr_auc']:.4f}" for v in r['val_curve']) + ' |')
    lines += ['', '## Temporal diagnostics', '',
        'Each diagnostic repeats the same test pairs. Reset-before-second deletes prior state and may create an out-of-distribution state. '
        'A decrease supports dependence on prior state without identifying a neural mechanism. '
        'Frame swap reverses direction labels and flips binary interval/signed-orientation labels. '
        'Some task marginals may retain single-frame information; no fixed chance threshold was imposed.', '',
        'For the opponent model, motion BA fell from100% to25.89% after resetting state before frame2, and was99.78% after reversing the frames and transforming labels. '
        'This supports use of temporal history and the tested order sensitivity; the reset removes the whole state and does not establish that the energy branch is necessary.', '',
        '| Accumulator | Task | Normal BA | Reset-before-second BA | Reversed-order, transformed-label BA |',
        '|---|---|---:|---:|---:|']
    for r in runs:
        for task in tasks:
            normal = r['test']['tasks'][task]['overall']['balanced_accuracy']
            reset = r['test']['diagnostics']['reset_before_second']['tasks'][task]['overall']['balanced_accuracy']
            swap = r['test']['diagnostics']['frame_swap']['tasks'][task]['overall']['balanced_accuracy']
            lines.append(f"| {r['model']} | {task} | {percent(normal)} | {percent(reset)} | {percent(swap)} |")
    lines += ['', '## Exposure and measured cost', '',
        f"Each model trained for {cfg['updates']:,} updates at batch32 ({cfg['training_pairs_per_arm']:,} fresh generated pairs), "
        'with identical per-task stream prefixes and the interleaved contour-focused twelve-update schedule. '
        'These are new projection/core/readout updates; the inherited encoder had zero parameter updates. '
        'The encoder retains prior training exposure, and all three depend on the same trained parent.', '',
        '| Task | Terminal pairs per model | ' + ' | '.join(r['model'] + ' selected pairs' for r in runs) + ' |',
        '|---|' + '---:|' * (len(runs) + 1)]
    for task in tasks:
        lines.append('| ' + task + ' | ' + f"{cfg['training_pairs_per_task'][task]:,}" + ' | ' +
            ' | '.join(f"{r['selected_per_task_updates'][task]*32:,}" for r in runs) + ' |')
    lines += ['', '| Accumulator | Trainable parameters | Core parameters | State MiB/example | State MiB/batch32 | Profile peak allocated MiB | Profile mean training ms/update | Production worker wall seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        p = result['profiles'][r['model']]
        lines.append(f"| {r['model']} | {p['trainable_params']:,} | {p['core_params']:,} | {p['state_bytes_per_example']/2**20:.3f} | "
            f"{p['state_bytes_per_batch']/2**20:.2f} | {p['peak_vram_bytes']/2**20:.1f} | {1000*p['train_step_mean']:.1f} | {r['training_seconds']:.2f} |")
    lines += ['', f"Supervisor receipt: completed in {exit_record['wall_seconds']:.2f} seconds of the new 7,200-second allowance, including profiles, validation, final inference and paired analysis. "
        'The previous unused allowance was not consumed or renewed. One GPU worker ran at a time, fp32, no AMP/TF32, two CPU PyTorch threads. '
        'Persistent state bytes exclude activations and gradients; peak allocated VRAM excludes desktop/driver allocations. '
        'Profile48-update fits were separate, preserved diagnostics and were not used as production initialization. Their shared initial stream prefixes are repeated presentations, not extra unique production data. '
        'Production worker wall time includes process startup and checkpointing; profile update time measures a different scope.', '',
        '## Reproduction and limits', '',
        f"Run directory: `{root}`. The immutable source snapshot, fixed_config.json, dataset_manifest.json, budget.json, worker logs, metrics.csv, "
        'checkpoint indices, full-state checkpoints, raw predictions and exit.json are retained there. '
        'results_temporal.json is the readable aggregate; paired_analysis.json contains the saved-score comparison.', '',
        f"Frozen parent: `{cfg['parent_checkpoint']}`; SHA256 `{cfg['parent_sha256']}`. "
        'Its recorded training step is2268; the reference evaluator does not train it.', '',
        'Training base seed310001, validation910001, test1010001, with task seeds base+100003*(registry index+1). '
        'Model seed30301, common projection/readout initialization30312, temporal-core initialization30313. '
        'Fresh Adam lr.001, weight decay.0001, gradient norm clip5; full two-step BPTT through the new modules. '
        'Source manifests and versioned full state define strict resumption; do not resume a prior optimizer for new temporal weights.', '',
        'Normal task intervals use1,000 bootstrap replicates. Natural-image sources are clustered because fresh generated crops/pairs reuse the held-out BSDS photo pool; '
        'procedural pairs are the unit for other tasks. Repeated diagnostics/checkpoints are not independent data or training seeds. '
        'Point-perfect empirical bootstrap intervals can collapse at100% and do not imply population certainty. '
        'Per-difficulty summaries and confusion matrices remain in the aggregate. No held-out tuning or additional architecture sweep was performed.', '',
        'A two-step recurrent state can retain enough information to implement an online pair comparator. '
        'These results therefore concern replacing direct two-encoding access under the tested objectives; persistent working memory, longer accumulation, '
        'speed/coherence tuning and MT pattern-motion correspondence require separately authorized tasks.', '']
    text = '\n'.join(lines)
    (root / 'report.md').write_text(text, encoding='utf-8')
    (HERE / 'report.md').write_text(text, encoding='utf-8')
    print(root / 'report.md')


if __name__ == '__main__':
    render()
