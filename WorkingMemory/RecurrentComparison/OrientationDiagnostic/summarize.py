"""Saved-prediction uncertainty and scientific tables only; no model inference."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,time,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_text())
def write(name,data):(P/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def errors(p,theta):return np.rad2deg(np.abs((np.arctan2(p[:,1],p[:,0])/2-theta+np.pi/2)%np.pi-np.pi/2))
def main():
    s=read('summary.json');data=dict(np.load(P/'test_features.npz'));p=dict(np.load(P/'heldout_predictions.npz'));rng=np.random.default_rng(27975001)
    boots=np.stack([np.concatenate([rng.choice(np.flatnonzero(data['y']==c),sum(data['y']==c),replace=True) for c in (0,1)]) for _ in range(1000)])
    transfers={};angular_strata={}
    for name,v in p.items():
        if name not in s['recovery']:continue
        err=errors(v,data['probe_angle'] if name.startswith('probe_angle') else data['sample_angle'])
        angular_strata[name]={str(x):dict(n=int((data['mismatch']==x).sum()),mae_degrees=float(err[data['mismatch']==x].mean()),within_half_change=None if x==0 else float((err[data['mismatch']==x]<x/2).mean())) for x in np.unique(data['mismatch'])}
    for stage in ('endblank','preprobe'):
        for d in (0,4,12,24):
            early=errors(p[f'transfer_sample_r_to_{stage}_D{d}'],data['sample_angle']);late=errors(p[f'{stage}_r_D{d}'],data['sample_angle']);delta=late-early
            transfers[f'{stage}_D{d}']=dict(late_fit_minus_transferred_mae_degrees=float(delta.mean()),ci95=np.quantile(delta[boots].mean(1),[.025,.975]).tolist())
    change_intervals={}
    for name,r in s['comparison'].items():
        change_intervals[name]={}
        for value,row in r['actual_change_degrees'].items():
            n=row['n'];fraction=row['accuracy'];z=1.95996398454;center=(fraction+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n)
            change_intervals[name][value]=dict(**row,wilson_ci95=[center-half,center+half],meaning='specificity' if float(value)==0 else 'changed-trial sensitivity')
    write('paired_analysis.json',dict(transfer=transfers,angular_errors_by_actual_change=angular_strata,comparison_change_strata=change_intervals,bootstrap='1000 class-stratified paired independent-base resamples; delay variants/models never inflate n'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    delays=[0,4,12,24]
    for prefix,label,color in [('preprobe_r_D','Firing rates r','#21618c'),('preprobe_ra_D','Rates + adaptation (diagnostic)','#8e44ad'),('preprobe_sensory_D','Sensory path with traces','#148f77'),('transfer_sample_r_to_preprobe_D','Sample-trained r decoder','#d68910')]:
        rows=[s['recovery'][prefix+str(d)] for d in delays];mean=np.array([x['mae_degrees'] for x in rows]);ci=np.array([x['mae_ci95'] for x in rows]);axes[0].errorbar(delays,mean,yerr=np.maximum(0,np.stack((mean-ci[:,0],ci[:,1]-mean))),marker='o',label=label,color=color,capsize=3)
    axes[0].axhline(45,ls=':',color='.6',label='Independent uniform angle:45° MAE');axes[0].axhline(7.5,ls='--',color='.6');axes[0].set(xlabel='Inserted blank frames',ylabel='Axial angle MAE (degrees)',title='Sample recovery immediately before probe',xticks=delays);axes[0].legend(fontsize=7)
    for prefix,label,color,ds in [('existing_D','Deployed model','#21618c',delays),('comparator_D','r + independent probe MLP64','#c0392b',[0,24]),('circular_D','Decoded-angle distance','#148f77',[0,24])]:
        rows=[s['comparison'][prefix+str(d)] for d in ds];mean=np.array([x['ba']*100 for x in rows]);ci=np.array([x['ba_ci95'] for x in rows])*100;axes[1].errorbar(ds,mean,yerr=np.maximum(0,np.stack((mean-ci[:,0],ci[:,1]-mean))),marker='o',label=label,color=color,capsize=3)
    axes[1].axhline(50,ls=':',color='.6');axes[1].set(xlabel='Inserted blank frames',ylabel='Balanced accuracy (%)',title='Held-out operational comparison',ylim=(35,101),xticks=delays);axes[1].legend(fontsize=8)
    fig.savefig(P/'orientation_diagnostic.png',dpi=180);fig.savefig(P/'orientation_diagnostic.svg');plt.close(fig)
    def f(x):return f'{x:.2f}'
    lines=['# Frozen orientation diagnostic','', 'Selected Retention E/I checkpoint14800;512 independent held-out episodes, each shown at four delays. No parent-model updates.','', '## Pre-probe orientation recovery','', '| State/stage | D0 MAE° | D4 | D12 | D24 |','|---|---:|---:|---:|---:|']
    for prefix,label in [('endblank_r_D','r before query'),('preprobe_r_D','r after query / before probe'),('preprobe_ra_D','[r,a] before probe'),('preprobe_sensory_D','sensory path before probe'),('transfer_sample_r_to_preprobe_D','sample-trained r decoder transferred')]:lines.append('| '+label+' | '+' | '.join(f(s['recovery'][prefix+str(d)]['mae_degrees']) for d in delays)+' |')
    lines+=['',f"Sample-stage r ridge MAE: {f(s['recovery']['sample_r']['mae_degrees'])}°. Isolated-probe ridge MAE: {f(s['recovery']['probe_angle']['mae_degrees'])}°. Smallest actual change is7.5°; a3.75° half-change benchmark is descriptive, not a sufficient comparator guarantee.",'','## Operational label comparison','','| Readout | BA% [paired95%CI] | AUC | Change vs deployed model, pp [95%CI] |','|---|---:|---:|---:|']
    for name,r in s['comparison'].items():lines.append(f"| {name} | {f(r['ba']*100)} [{f(r['ba_ci95'][0]*100)}, {f(r['ba_ci95'][1]*100)}] | {r['auc']:.3f} | {f(r['delta_vs_existing']*100)} [{f(r['delta_ci95'][0]*100)}, {f(r['delta_ci95'][1]*100)}] |")
    lines+=['','## Fixed nonlinear angle fits','','| Probe | Angular MAE° | 95%CI | Within3.75° |','|---|---:|---:|---:|']
    for name,r in s['recovery'].items():
        if name.endswith('_mlp'):lines.append(f"| {name} | {f(r['mae_degrees'])} | [{f(r['mae_ci95'][0])}, {f(r['mae_ci95'][1])}] | {f(r['fraction_within3_75']*100)}% |")
    lines+=['','## Actual-change detail atD24','','| Actual change° | Trials | Deployed correct% | MLP comparator correct% | Circular comparator correct% |','|---|---:|---:|---:|---:|']
    for value,row in s['comparison']['existing_D24']['actual_change_degrees'].items():lines.append('| '+value+' | '+str(row['n'])+' | '+' | '.join(f(s['comparison'][name]['actual_change_degrees'][value]['accuracy']*100) for name in ('existing_D24','comparator_D24','circular_D24'))+' |')
    lines+=['','[Figure](orientation_diagnostic.png) · [Complete numeric results](summary.json) · [Paired transfer and change-size detail](paired_analysis.json) · [Frozen fit/threshold choices](fit_selection.json)','','## Limits and accounting','','Targets are actual rendered axial angles, not the latent level. All angular predictions use only pre-probe states; probe-angle decoding uses an independently initialized sensory path. The ordinary sensory feature includes opponent traces. Thus a comparator restricted to r plus an independently encoded probe omits that history route, and a D0 comparator failure can reflect that omission as well as limited probe fitting. Adaptation access is diagnostic-only. Weak probe performance never establishes erased information. An early decoder transfer penalty is evidence about decoder stability, not by itself a demonstration of a changed code.','','Operational classifiers receive no angles or labels as inputs. Circular comparisons use inferred angles and a validation-only threshold. Postprobe label fits, if present, are operational rescue tests and make no sample-reconstruction claim. AUC uses continuous score differences. Change-size rows in summary.json contain changed-trial sensitivity (and zero-change specificity), not balanced accuracy within an all-positive stratum. Intervals condition on this one frozen trained model and use independent base episodes; repeated delays are paired presentations.','','Train2048/validation512/test512 independent groups produce12288 model-delay presentations. All choices were frozen before held-out collection. Neural extraction and MLP fits use fp32; NumPy/SciPy ridge uses float64. No original optimizer was constructed and no parent weights were updated. Exact checkpoint and runtime source identities are in config.json; all candidate traces and selected fitted weights are retained.']
    (P/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary_ready=True,wall_seconds=s['wall_seconds'])),flush=True)
if __name__=='__main__':main()
