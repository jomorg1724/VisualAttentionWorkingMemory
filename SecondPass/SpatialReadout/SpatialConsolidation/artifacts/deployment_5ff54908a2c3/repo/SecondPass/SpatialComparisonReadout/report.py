"""Paired final reporting, without any post-test model selection."""
import json
from pathlib import Path
import numpy as np
from SecondPass.JointTraining.core import atomic_json, summarize
from SecondPass.SpatialReadout.continuation_v2 import verify_coverage
from WorkingMemory.BatteryAudit.observers import auc_binary

TARGETS=('orientation_cued','spatial_binding')


def paired_cell(control,candidate,rng,repeats=2000):
    a=control['paired_observations'];b=candidate['paired_observations']
    if a['labels']!=b['labels'] or a['metadata_sha256']!=b['metadata_sha256']:raise ValueError('Unpaired final draws')
    y=np.asarray(a['labels']);pa=np.asarray(a['probabilities']);pb=np.asarray(b['probabilities'])
    groups=[np.flatnonzero(y==k) for k in range(pa.shape[1])]
    delta=(pb.argmax(1)==y).astype(float)-(pa.argmax(1)==y).astype(float)
    empty=control['balanced_accuracy'] is None
    draws=[]
    for ix in groups:
        if len(ix):draws.append(delta[rng.choice(ix,size=(repeats,len(ix)),replace=True)].mean(1))
    boot=np.mean(draws,axis=0)
    point=float(np.mean([delta[ix].mean() for ix in groups if len(ix)]))
    result=dict(task=control['task'],cell=control['cell'],n=len(y),
        control_ba=control['balanced_accuracy'],candidate_ba=candidate['balanced_accuracy'],
        ba_difference=None if empty else point,ba_difference_ci95=None if empty else np.quantile(boot,[.025,.975]).tolist(),
        specificity_difference=point if empty else None,
        control_auc=control['auc'],candidate_auc=candidate['auc'],
        auc_difference=None if control['auc'] is None else candidate['auc']-control['auc'],
        control_events=control.get('events'),candidate_events=candidate.get('events'))
    if control['task'] in TARGETS:
        # Paired, label-stratified bootstrap; draws stay paired across arms.
        auc_boot=[]
        for _ in range(repeats):
            ix=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
            auc_boot.append(float(auc_binary(y[ix],pb[ix,1])-auc_binary(y[ix],pa[ix,1])))
        result['auc_difference_ci95']=np.quantile(auc_boot,[.025,.975]).tolist()
        return result,boot,np.asarray(auc_boot)
    return result,None,None


def report(directory):
    directory=Path(directory);rng=np.random.default_rng(95492763)
    reports={a:json.loads((directory/a/'report.json').read_text()) for a in ('control','candidate')}
    allocation=json.loads((directory/'allocation.json').read_text())
    if any(r['terminal_step']!=allocation['max_steps'] or not r['final_coverage_complete'] or r['failure'] is not None for r in reports.values()):
        raise ValueError('Incomplete or unequal arms; no completed comparison')
    out=dict(protocol='learned_spatial_comparison_v1',allocation=allocation,arms={a:{k:r[k] for k in ('terminal_step','selected_step','selection_history','optimizer_seconds','terminal_checkpoint')} for a,r in reports.items()},roles={})
    lines=['# Learned spatial comparison versus unchanged continuation','',
        'One terminal6760 parent, unchanged all-task teaching, one learned residual comparator. This is a pilot package comparison, not biological attention or a comparison-versus-capacity isolation.','']
    for role in ('selected','terminal'):
        results={}
        for arm in reports:
            r=json.loads((directory/arm/f'test_{role}.json').read_text());r=r.get('results',r)
            verify_coverage(r,128,200);results[arm]=r
        lookup={(r['task'],r['cell']):r for r in results['candidate']['cells']}
        rows=[];ba_boot=[];auc_boot=[]
        for a in results['control']['cells']:
            row,bb,ab=paired_cell(a,lookup[a['task'],a['cell']],rng)
            rows.append(row)
            if bb is not None:ba_boot.append(bb);auc_boot.append(ab)
        assert len(rows)==35 and len(ba_boot)==8
        target=[r for r in rows if r['task'] in TARGETS]
        gain=float(np.mean([r['ba_difference'] for r in target]));ci=np.quantile(np.mean(ba_boot,axis=0),[.025,.975]).tolist()
        signal=gain>=.05 and ci[0]>0
        tasks={t:dict(control=results['control']['summary']['tasks'][t],candidate=results['candidate']['summary']['tasks'][t]) for t in results['control']['summary']['tasks']}
        for t,row in tasks.items():
            row['ba_difference']=row['candidate']['balanced_accuracy']-row['control']['balanced_accuracy']
            row['auc_difference']=row['candidate']['auc']-row['control']['auc']
        summary=dict(target_mean_ba_difference=gain,target_mean_ba_difference_ci95=ci,
            target_mean_auc_difference=float(np.mean([r['auc_difference'] for r in target])),
            target_mean_auc_difference_ci95=np.quantile(np.mean(auc_boot,axis=0),[.025,.975]).tolist(),
            practical_signal=signal,cells=rows,tasks=tasks)
        out['roles'][role]=summary
        lines.extend([f'## {role.title()}',f'Target8 mean BA gain: {gain*100:.2f} pp; paired 95% interval [{ci[0]*100:.2f}, {ci[1]*100:.2f}] pp. Prespecified practical signal: {signal}.','',
            '| Task | Cell | n | Control BA | Candidate BA | BA difference | Control AUC | Candidate AUC |','|---|---|---:|---:|---:|---:|---:|---:|'])
        for r in rows:lines.append('| '+ ' | '.join(str(r[k]) for k in ('task','cell','n','control_ba','candidate_ba','ba_difference','control_auc','candidate_auc'))+' |')
        lines+=['','### All-task differences','| Task | BA difference | AUC difference |','|---|---:|---:|']
        for t,r in tasks.items():lines.append(f"| {t} | {r['ba_difference']} | {r['auc_difference']} |")
    lines+=['','Intervals are paired episode-bootstrap intervals, stratified by label within cell, conditional on this one trained pair. They do not include training-seed uncertainty or imply fresh photo identities. Delay cells are independent. No post-test selection. Empty recognition specificity and Krauzlis target/foil/catch remain in comparison.json and every arm test file.']
    atomic_json(directory/'comparison.json',out)
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return out


if __name__=='__main__':
    import sys
    report(sys.argv[1])
