"""Saved-score paired analysis and final reports; never invokes a model/GPU."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,time,html,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.evaluate_multitask import classification


def read(path):return json.loads(Path(path).read_text())


def paired(first,second,kind,seed=1393001):
    a=[json.loads(x) for x in Path(first).read_text().splitlines()]
    b=[json.loads(x) for x in Path(second).read_text().splitlines()]
    assert len(a)==len(b)
    assert all((x['task'],x['condition'],x['label'],x['trial_id'],x['metadata'])==(y['task'],y['condition'],y['label'],y['trial_id'],y['metadata']) for x,y in zip(a,b))
    result=[];rng=np.random.default_rng(seed)
    for key in sorted({(r['task'],r['condition']) for r in a}):
        ix=[j for j,r in enumerate(a) if (r['task'],r['condition'])==key]
        y=np.array([a[j]['label'] for j in ix]);p=np.array([a[j]['probabilities'] for j in ix]);q=np.array([b[j]['probabilities'] for j in ix])
        cp,cq=classification(y,p),classification(y,q);boot=[]
        strata=[np.flatnonzero(y==k) for k in range(p.shape[1])]
        for _ in range(1000):
            sample=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata])
            u,v=classification(y[sample],p[sample]),classification(y[sample],q[sample])
            boot.append([v['balanced_accuracy']-u['balanced_accuracy'],v['macro_ovr_auc']-u['macro_ovr_auc']])
        intervals=np.quantile(np.array(boot),[.025,.975],axis=0)
        pa,pb=p.argmax(1)==y,q.argmax(1)==y
        result.append(dict(comparison=kind,family=key[0],condition=key[1],n=len(y),
            delta_ba=cq['balanced_accuracy']-cp['balanced_accuracy'],delta_ba_ci95=intervals[:,0].tolist(),
            delta_auc=cq['macro_ovr_auc']-cp['macro_ovr_auc'],delta_auc_ci95=intervals[:,1].tolist(),
            both_correct=int((pa&pb).sum()),first_only_correct=int((pa&~pb).sum()),second_only_correct=int((~pa&pb).sum()),both_wrong=int((~pa&~pb).sum())))
    return result


def training_summary(root,arm,selected_step,batch):
    rows=list(csv.DictReader((root/arm/'metrics.csv').open()));terminal=Counter();selected=Counter();states=[]
    for r in rows:
        terminal[r['condition']]+=batch
        if int(r['step'])<=selected_step:selected[r['condition']]+=batch
        d=json.loads(r['diagnostics'])
        if d:states.append(dict(step=int(r['step']),condition=r['condition'],**d))
    return dict(terminal_cell_episodes=dict(terminal),selected_cell_episodes=dict(selected),
        clipping_fraction=float(np.mean([int(r['clipped']) for r in rows])),
        preclip_norm_quantiles=np.quantile([float(r['grad_norm']) for r in rows],[.5,.9,.99,1]).tolist(),
        state_diagnostics=states,logical_visual_updates=int(rows[-1]['visual_updates']))


def run():
    started=time.time();r=read(HERE/'results.json');root=Path(r['run_root']);exit_receipt=read(root/'exit.json')
    assert r['status']==exit_receipt['status']=='completed'
    cfg=r['config'];arms=cfg['arms'];runs=r['runs'];comparisons=[]
    comparisons+=paired(runs['lstm']['test']['predictions'],runs['ei_adaptive']['test']['predictions'],'EI minus LSTM')
    for arm in arms:comparisons+=paired(runs[arm]['reset_test']['predictions'],runs[arm]['test']['predictions'],arm+': normal minus reset')
    summaries={a:training_summary(Path(runs[a].get('local_artifact_root',root)),a,runs[a]['selected_step'],cfg['recipe']['batch_size']) for a in arms}
    analysis=dict(comparisons=comparisons,training=summaries,bootstrap='1000 paired class-stratified episode resamples per cell; conditional on fitted models and observed class counts; no correction for multiple exploratory cells',
        matched_raw_metadata_verified=True,analysis_seconds=time.time()-started)
    (HERE/'analysis.json').write_text(json.dumps(analysis,indent=2));(root/'analysis.json').write_text(json.dumps(analysis,indent=2))
    def pct(x):return f'{100*x:.2f}%'
    def metric(arm,key,view='test'):return runs[arm][view]['cells'][key]['overall']
    lines=['# Two recurrent memories after the opponent visual system','',
        f"Both models completed {cfg['episodes_per_arm']:,} fresh episodes ({cfg['total_steps']:,} updates each). "
        f"The shared WM parent was update 6,860. The experiment completed after {exit_receipt['wall_seconds']/60:.2f} elapsed minutes. "
        'The user subsequently moved the E/I arm to the Palladio RunPod account while local LSTM training continued. Each location retained its declared absolute deadline.','',
        '| Held-out cell | LSTM BA | LSTM AUC | E/I BA | E/I AUC |','|---|---:|---:|---:|---:|']
    for key in runs['lstm']['test']['cells']:
        u,v=metric('lstm',key),metric('ei_adaptive',key)
        lines.append(f"| {key} | {pct(u['balanced_accuracy'])} | {u['macro_ovr_auc']:.4f} | {pct(v['balanced_accuracy'])} | {v['macro_ovr_auc']:.4f} |")
    lines+=['','Both models acquired the focused new tasks. LSTM scored higher on the longer motion-duration cell; the small E/I advantage on delayed orientation recall has a paired interval that includes zero. '
        'These results concern the tested tasks and fitted models, rather than a general memory-capacity or biological ranking.','',
        'Motion chance BA is 25%; orientation chance BA is 50%. AUC chance is 0.5 for both. '
        'Fixed argmax decisions and score ranking are both reported; test scores never select or calibrate the model. '
        'This is one initialization per arm, not replication across training seeds. '
        'The six cells contain 3,072 unique held-out episodes shared across both models and both state conditions: 12,288 scored presentations, not 12,288 independent episodes. '
        'Likewise, each model trains on the same 40,000-episode procedural stream; 80,000 model-episode presentations are not 80,000 independent source episodes.','',
        '## Paired effects and the recurrent-history intervention','',
        'Positive EI-minus-LSTM deltas favor EI. Positive normal-minus-reset deltas mean that preserving the new recurrent history helped the trained model. '
        'Resetting the added state before every frame leaves opponent traces and the current-frame memory computation intact. '
        'It is not a separately trained memory-free or parameter-matched control.','',
        '| Comparison / cell | BA difference (percentage points) |95% interval| AUC difference |95% interval|','|---|---:|---:|---:|---:|']
    for c in comparisons:
        lo,hi=c['delta_ba_ci95'];al,ah=c['delta_auc_ci95']
        lines.append(f"| {c['comparison']} / {c['condition']} | {100*c['delta_ba']:+.2f} | [{100*lo:+.2f},{100*hi:+.2f}] | {c['delta_auc']:+.4f} | [{al:+.4f},{ah:+.4f}] |")
    lines+=['','Resetting the new recurrent history reduces both motion-integration decisions to chance and substantially lowers their AUC. '
        'For delayed orientation recall, it reduces BA by 7.81 points in LSTM and 10.35 in E/I, while OVR-AUC changes by less than 0.002 in either model. '
        'The orientation effect therefore does not establish additional score-ranking information; decision bias and use of retained sensory traces remain relevant alternatives.','',
        'Intervals use 1,000 paired class-stratified episode resamples. Episode labels, identities and complete metadata match across each comparison. '
        'These intervals condition on the trained models; they neither measure seed variability nor correct for multiple exploratory cells. '
        'Empirical perfect scores can produce collapsed percentile intervals, which are not guarantees of population certainty.','',
        '## Selection, exposure and acquisition','',
        'The arithmetic mean of all six cell OVR-AUCs selects a checkpoint at four planned validation looks, with exact ties favoring the earlier checkpoint. '
        'Training still reaches the same final endpoint for both models.','',
        '| Arm | Selected update | Selected episodes | Terminal episodes | Clipped updates |','|---|---:|---:|---:|---:|']
    for a in arms:lines.append(f"| {a} | {runs[a]['selected_step']:,} | {runs[a]['selected_step']*cfg['recipe']['batch_size']:,} | {cfg['episodes_per_arm']:,} | {pct(summaries[a]['clipping_fraction'])} |")
    lines+=['','| Arm / cell | Selected-checkpoint episodes | Terminal episodes |','|---|---:|---:|']
    for a in arms:
        for cell,n in summaries[a]['terminal_cell_episodes'].items():lines.append(f"| {a} / {cell} | {summaries[a]['selected_cell_episodes'].get(cell,0):,} | {n:,} |")
    lines+=['','| Arm | Validation update | Mean six-cell AUC |','|---|---:|---:|']
    for a in arms:
        for v in runs[a]['validation']:lines.append(f"| {a} | {v['step']:,} | {v['selection_mean_auc']:.4f} |")
    lines+=['','Clipping was frequent in both runs and is a material optimization qualification, despite finite computation and observed acquisition. '
        'No clipping-based restart or hidden tuning was performed. The following state maxima are from the planned diagnostic batches, not a scan of every activation.','',
        '| Arm | Median pre-clip norm | Maximum pre-clip norm | Largest logged state magnitude |','|---|---:|---:|---:|']
    for a in arms:
        s=summaries[a];peak=max(v for d in s['state_diagnostics'] for v in d['state_peak'].values())
        lines.append(f"| {a} | {s['preclip_norm_quantiles'][0]:.3f} | {s['preclip_norm_quantiles'][-1]:.3f} | {peak:.3f} |")
    lines+=['','## Numerical opportunity and limits','',
        'LSTM has 1,011,872 learned parameters; E/I has 714,912. Each adds 512 recurrent state values. '
        'All 408,728 transferred parameters remain trainable; unused task heads receive no task loss. '
        'Shared interface and residual tensors begin identically, and fresh task-local streams match across arms. '
        'The two cores have different initialization and parameterization, so equal exposure does not equate optimization difficulty.','',
        'LSTM forget biases initialize nominal retention time constants 4–128; its independent gate normalization precedes biases and never normalizes the stored cell. '
        'E/I starts with balanced incoming excitation/inhibition, rate time constants 2–8 and weak 0.05 adaptation. '
        'Dale signs remain constrained by positive magnitudes times presynaptic column signs. Rate/adaptation dynamics use old states synchronously; neither state nor recurrent current is normalized. '
        'Bounded leak timescales do not guarantee stability of the full recurrent loop.','',
        'Adam uses shared epsilon 1e-10, new-parameter LR 3e-4 and transferred LR 3e-5; clipping threshold 1. '
        'Raw sign-constrained parameters, biases, normalization and timescale/adaptation parameters have no generic decay. '
        'Profiles and planned training diagnostics report effective recurrent-weight changes, raw-gradient scales, clipping, state RMS/max, gate saturation or E/I activity, and early/late representation gradients. '
        'A nonzero gradient is not proof of a useful learned recurrent computation; neither a small gradient nor clipping alone proves lost information.','',
        'Minimal and longer cells must be read together. Weak minimal-rule acquisition prevents a clean memory-capacity interpretation. '
        'Improvement over the earlier broad battery could reflect focused exposure or the new shared interface as well as the recurrent core. '
        'A reset effect establishes sensitivity to this trained recurrent history under the tested intervention; it does not establish biological correspondence, universal memory capacity, or the necessity of a separately named WM module. '
        'No attention, extra models, independent sweep or automatic budget extension was added.','',
        '## Platform migration','',
        'LSTM trained locally on Windows/RTX3070 Laptop; E/I trained remotely on Linux/RTX3090 at the user’s explicit request. '
        'Torch 1.13.1+cu117, NumPy 1.23.1, SciPy 1.8.1 and Pillow 9.1.1 matched. Python patch versions differed (3.10.6/3.10.12). '
        'The remote initializer contained exact unfitted local E/I tensors, and its common input/output tensors were checked against local LSTM update 0. '
        'It was not an after-profile model. Full task metadata matching is checked for paired analysis; cross-platform rendering/training is not claimed bitwise identical. '
        'Platform variation is an additional confound for differences between models. The original local supervisor was replaced without interrupting its active LSTM worker, and local E/I production was excluded. '
        'Production wall times on different GPUs are not an architecture-throughput comparison. '
        'Cloud artifacts were retrieved separately; pod cleanup and actual billing duration require the saved cloud cleanup receipt.','',
        '## Reproduction','',
        f"Run: `{root}`. Read README.md for exact cells, frame counts, equations and initialization. "
        'The run stores pinned source/config/parent identities, full optimizer/sampler/RNG checkpoints, metric/state logs and raw paired predictions. '
        'results.json contains every cell confusion matrix, class recall, binary hit/miss/false-alarm/correct-rejection counts, sensitivity/criterion, uncertainty and strata. '
        'analysis.json contains paired intervals and selected/terminal exposure. Profile fits remain separate from production.','']
    text='\n'.join(lines)
    for path in (HERE/'report.md',root/'report.md'):path.write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='completed',analysis_seconds=time.time()-started,report=str(HERE/'report.md'))))

if __name__=='__main__':run()
