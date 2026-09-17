"""Render a final WM Markdown account from saved results only; no ML imports."""
import json,csv
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent


def render():
    r=json.loads((HERE/'results.json').read_text());root=Path(r['run_root'])
    receipt=json.loads((root/'exit.json').read_text())
    if r['status']!='completed' or receipt['status']!='completed':raise RuntimeError('Completed evidence required')
    cfg=r['config'];test=r['test'];tasks=list(cfg['task_classes']);cells=test['cells'];batch=cfg['recipe']['batch_size']
    def cell(f,c):return cells[f+'/'+c]['overall']
    def pct(x):return f'{100*x:.2f}%'
    def ci(v,key='balanced_accuracy_ci95'):
        a=v.get(key)
        if a is None:return 'not estimated'
        return f'[{a[0]:.4f},{a[1]:.4f}]' if key.startswith('macro_ovr_auc') else f'[{100*a[0]:.2f},{100*a[1]:.2f}]'
    protocol_auc={p:sum(c['overall']['macro_ovr_auc'] for c in cells.values() if c['protocol']==p)/sum(c['protocol']==p for c in cells.values()) for p in ['anchor','integration','recall']}
    lines=['# Learned opponent model: sequence-memory battery','',
        f"The warm-started opponent model completed {cfg['total_episodes']:,} fresh training episodes. "
        f"The selected checkpoint is update {r['selected_step']:,} ({r['selected_step']*batch:,} episodes); "
        f"training continued to the fixed endpoint of {cfg['total_steps']:,} updates. "
        'Task-specific held-out results below separate sensory/cue acquisition, accumulation, decision retention and retrospective probes.','',
        f"Held-out mean OVR-AUC was {protocol_auc['anchor']:.4f} for sensory anchors, {protocol_auc['integration']:.4f} for integration/decision-retention cells, "
        f"and {protocol_auc['recall']:.4f} for retrospective recall cells (chance ranking: 0.5). "
        'These descriptive protocol means average the declared cells equally; the minimal-delay results below determine how much a harder-condition deficit can be attributed to retention rather than incomplete rule acquisition.','',
        'The new sequence rules were not reliably acquired across the minimal-delay bridge cells. '
        'Consequently, the longer-delay/load scores do not provide a clean estimate of memory capacity in this run. '
        'Near-chance decisions must also be separated from ranking: the motion minimal retrospective probe has '
        f"BA {pct(cell('motion_direction','bridge_recall')['balanced_accuracy'])} but OVR-AUC {cell('motion_direction','bridge_recall')['macro_ovr_auc']:.4f}; "
        'biased class decisions therefore coexist with some useful score ordering. No calibration was fitted on these held-out results.','',
        'The trace update remains fixed at 0.25/0.75. All 408,728 learned encoder, projection, temporal-output and readout parameters were allowed to adapt. '
        'No separate working-memory module, stored prior prediction or hidden sample cache was added.','',
        '## Sensory and cue bridge at the selected checkpoint','',
        'Held-out balanced accuracy. Anchors retain the original two-frame tasks. Integration and retrospective columns use the new cue/rule formats; previous two-frame scores are not a paired baseline for those tasks.','',
        '| Family | Sensory anchor | Minimal integration, visible cue | Minimal integration, flashed cue | Minimal retrospective probe |',
        '|---|---:|---:|---:|---:|']
    for f in tasks:lines.append('| '+f+' | '+' | '.join(pct(cell(f,c)['balanced_accuracy']) for c in ['anchor','bridge_integrate_visible','bridge_integrate_transient','bridge_recall'])+' |')
    tables=[('Integration length and cue use',['integrate_L8','integrate_L16','integrate_L32','integrate_transient']),
            ('Decision retention and interference',['integrate_L8','decision_D2','decision_D4','decision_D8','decision_D16','decision_distractor']),
            ('Retrospective probe delay',['recall_D0','recall_D2','recall_N1','recall_D8','recall_D16']),
            ('Retrospective load, binding conditions and timing controls',['recall_N1','recall_N2','recall_N4','recall_matched_one','recall_distractor','recall_precue','recall_postcue'])]
    for title,conditions in tables:
        lines+=['','## '+title,'','| Family | '+' | '.join(conditions)+' |','|---|'+'---:|'*len(conditions)]
        for f in tasks:lines.append('| '+f+' | '+' | '.join(pct(cell(f,c)['balanced_accuracy']) for c in conditions)+' |')
    lines+=['','The load comparison changes target age unless controlled. Read recall_N4 alongside the matched-duration one-item condition and the recorded target-age/serial-position strata. '
            'Precues, retrocues and postcues differ in what can be selected when; a score difference alone does not isolate an internal selection mechanism. '
            'Decision retention can store an already-computed category. The retrospective binary probe cannot be answered before the new probe appears.','',
            '## Cell uncertainty and score ranking','',
            'Each cell has 128 fresh held-out episodes. Intervals are 500-replicate percentile bootstraps, conditional on this fitted model. '
            'Natural cells cluster by the queried target photo; reuse of uncued photos is not separately clustered. '
            'The source pools are the existing held-out BSDS photos, not newly independent source images. '
            'A perfect empirical interval can collapse at 100% and is not a guarantee of population certainty.','',
            '| Family / condition | BA | BA 95% interval | OVR-AUC | AUC 95% interval | Source groups |','|---|---:|---:|---:|---:|---:|']
    for key,c in cells.items():
        v=c['overall'];lines.append(f"| {key} | {pct(v['balanced_accuracy'])} | {ci(v)} | {v['macro_ovr_auc']:.4f} | {ci(v,'macro_ovr_auc_ci95')} | {v['unique_base_groups']} |")
    lines+=['','Confusion matrices, binary hit/miss/false-alarm/correct-rejection counts, sensitivity/criterion, class recall intervals and condition-specific strata are in results.json. '
            'Binary and four-class chance accuracy differ; no pooled raw accuracy is a capacity claim. High ranking with biased argmax decisions is not evidence of absent information.','',
            '## Acquisition trajectory and selection','',
            'The bridge was trained first and its compatible weights/Adam state continued into the joint mixture. There was no accuracy gate or restart. '
            'Bridge validation is reported separately and never competes directly with joint checkpoint selection. '
            'Joint selection uses mean task OVR-AUC: equal family/protocol groups, with conditions averaged inside each group. Worst-cell BA is descriptive.','',
            '| Stage | Update | Episodes | Validation mean AUC | Validation worst-cell BA |','|---|---:|---:|---:|---:|']
    for v in r['validation']:lines.append(f"| {v['stage']} | {v['step']} | {v['step']*batch:,} | {v['selection_mean_auc']:.4f} | {pct(v['worst_cell_ba'])} |")
    lines+=['','Weak minimal-delay sensory/cue/probe performance makes stronger memory-capacity interpretations inconclusive. '
            'A finite exploratory exposure is not a sufficient acquisition horizon. Conversely, good trained-condition performance does not establish arbitrary-length retention or general working-memory competence.','',
            '## Exposure and execution','',
            f"Batch {batch}; {cfg['bridge_episodes']:,} bridge episodes and {cfg['joint_episodes']:,} joint episodes. "
            'The joint mixture is 10% sensory anchors, 45% integration/decision-retention, 45% retrospective probes, with families balanced within groups. '
            'Every episode contributes one terminal cross-entropy loss; blank/cue/distractor frames still update the state. There is no padding or temporal gradient truncation.','']
    lines+=['Executed scalar/recall renderers use nine discrete levels. Orientation spacing is 7.5°; contrast spans 0.08–0.36 in log modulation amplitude; '
            'spatial frequency spans 4–10 cycles/image in log spacing; the chromatic-axis coordinate spans −0.12…+0.12; natural beta spans +0.6…−0.6. '
            'New contour paths span bend −9…9 pixels with 2° jitter; the old sensory anchors retain their original difficulty sampling. '
            'Thus mismatch values in the raw strata are level differences, not a continuous human precision measurement.','']
    selected=Counter();terminal=Counter();selected_frames=0
    with (root/'opponent/metrics.csv').open() as f:
        for row in csv.DictReader(f):
            key=row['stage']+'/'+row['family']+'/'+row['condition'];terminal[key]+=batch
            if int(row['step'])<=r['selected_step']:selected[key]+=batch;selected_frames=int(row['visual_updates'])
    lines+=[f"Terminal encoder/frame presentations: {r['training']['counts']['visual_updates']:,}; selected-checkpoint presentations: {selected_frames:,}. "
            'These counts include informative frames and repeated/blank/cue observations, which are not independent episodes.','',
            '| Stage / family / condition | Selected-checkpoint episodes | Terminal episodes |','|---|---:|---:|']
    for key in terminal:lines.append(f'| {key} | {selected[key]} | {terminal[key]} |')
    lines+=['','Detailed event/cue/probe/blank/distractor counts: `training.metadata_counts` in the aggregate. '
            'These are overlapping event tags (for example, a probe can also be the report frame), not an additive partition of total images. '
            'Logical observed-frame updates exclude exact backward activation recomputation; recomputation costs time but supplies no new observation. '
            'Profiles were separate preserved fits, never production initialization; their training/evaluation presentations are not added to production exposure.','',
            f"The supervisor completed in {receipt['wall_seconds']:.2f} seconds of the new 14,400-second allowance. "
            'One GPU worker ran at a time, fp32, no AMP/TF32, two PyTorch CPU threads. '
            'Exact non-reentrant encoder/projection activation checkpointing preserved RNG and full temporal gradients. '
            'The focused CPU check matched old two-frame logits, showed nonzero earliest-frame/encoder gradients and zero checkpointed-versus-uncheckpointed parameter-gradient difference.','',
            '## Reproduction and interpretation limits','',
            f"Run: `{root}`. Source/data manifests, fixed_config.json, budget.json, supervisor/worker logs, metrics.csv, "
            'immutable full-state checkpoints and byte indices, raw held-out predictions and exit.json are preserved. '
            'Initial trained parent, exact seeds, all condition dictionaries and fixed exposure are in fixed_config.json. '
            'The new optimizer and sampler are versioned; no strict continuation of the old temporal optimizer is claimed.','',
            'The held-out set contains the declared trained-condition slices. Longer 64-transition, delay 32 and load 6 extrapolations were not included in this fixed allocation. '
            'The generator has a verified matched-motion construction; a separate model counterfactual sweep was not silently added. '
            'Standard motion-duration results alone therefore cannot rule out every correlated summary heuristic.','',
            'For identical future image suffixes, earlier-history trace differences decay by 0.25^D/0.75^D. '
            'Trainable per-frame encoding can use a simultaneously visible cue but has no history feedback or learned retention coefficient. '
            'This structure predicts a useful stress axis; small residuals/readout amplification and finite precision prevent a universal accuracy theorem. '
            'Any proposed memory addition must address an observed, acquired-task deficit and compete with changes to the existing computation. '
            'This experiment does not force the conclusion that a separately named working-memory module is necessary.','']
    text='\n'.join(lines);(root/'report.md').write_text(text,encoding='utf-8');(HERE/'report.md').write_text(text,encoding='utf-8');print(root/'report.md')

if __name__=='__main__':render()
