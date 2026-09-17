"""Paired saved-score analysis only, with binding four-case block resampling."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,csv,time,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def metrics(y,p):
    k=p.shape[1];guess=p.argmax(1);conf=np.zeros((k,k),int);np.add.at(conf,(y,guess),1);recall=np.diag(conf)/np.maximum(1,conf.sum(1));aucs=[]
    for label in range(k):
        positive=y==label;n=positive.sum()
        if n and n<len(y):
            r=rankdata(p[:,label]);aucs.append((r[positive].sum()-n*(n+1)/2)/(n*(len(y)-n)))
    return dict(ba=float(recall.mean()),auc=float(np.mean(aucs)),confusion=conf.tolist(),n=len(y))
def main():
    agg=read(HERE/'results.json');assert agg['status']=='completed';run=Path(agg['run_root']);budget=read(run/'budget.json');deadline=budget['deadline_unix'];arms=('dense_comparator','spatial_ei');records={};grouped={}
    for arm in arms:
        for evaluation in agg['runs'][arm]['validation']+[agg['runs'][arm]['test']]:evaluation['interpretation']='Checkpoint selection is equal mean AUC over eight primary single/binding-by-delay validation cells. Motion anchors and held-out locations are descriptive. No reset intervention is performed. Paired delays repeat the same base evidence.'
    agg['reporting_note']='Legacy generic evaluator six-cell/reset boilerplate corrected in this final aggregate; numerical metrics and worker artifacts unchanged.'
    write(HERE/'results.json',agg);write(run/'aggregate.json',agg)
    for arm in arms:
        records[arm]=[json.loads(x) for x in Path(agg['runs'][arm]['test']['predictions']).read_text().splitlines()];grouped[arm]={}
        for row in records[arm]:grouped[arm].setdefault(row['condition'],[]).append(row)
        for rows in grouped[arm].values():rows.sort(key=lambda r:int(r['paired_base_id'].split('/')[-1]))
    rng=np.random.default_rng(36973001);results={};deltas={};strata={};indices={}
    for cell,first in grouped[arms[0]].items():
        second=grouped[arms[1]][cell];assert len(first)==len(second) and all(a['paired_base_id']==b['paired_base_id'] and a['label']==b['label'] and a['metadata']==b['metadata'] for a,b in zip(first,second))
        y=np.array([r['label'] for r in first]);pred={arm:np.array([r['probabilities'] for r in grouped[arm][cell]]) for arm in arms};kind=('binding_locations' if cell.endswith('_locations') else 'binding') if cell.startswith('binding') else ('single' if cell.startswith('single') else 'motion')
        if kind not in indices:
            if kind.startswith('binding'):
                assert len(y)%4==0
                # Shuffled complete four-case blocks carry fixed marginal labels.
                indices[kind]=np.array([np.concatenate([np.arange(4*b,4*b+4) for b in rng.integers(len(y)//4,size=len(y)//4)]) for _ in range(1000)])
            else:indices[kind]=np.array([np.concatenate([rng.choice(np.flatnonzero(y==v),sum(y==v),replace=True) for v in np.unique(y)]) for _ in range(1000)])
        boots=indices[kind];draws={};results[cell]={}
        for arm in arms:
            results[cell][arm]=metrics(y,pred[arm]);draws[arm]=np.array([[metrics(y[ix],pred[arm][ix])[key] for key in ('ba','auc')] for ix in boots]);results[cell][arm].update(ba_ci95=np.quantile(draws[arm][:,0],[.025,.975]).tolist(),auc_ci95=np.quantile(draws[arm][:,1],[.025,.975]).tolist())
        delta=draws['spatial_ei']-draws['dense_comparator'];deltas[cell]=dict(ba=results[cell]['spatial_ei']['ba']-results[cell]['dense_comparator']['ba'],ba_ci95=np.quantile(delta[:,0],[.025,.975]).tolist(),auc=results[cell]['spatial_ei']['auc']-results[cell]['dense_comparator']['auc'],auc_ci95=np.quantile(delta[:,1],[.025,.975]).tolist())
        if cell.startswith('binding'):
            strata[cell]={}
            for spacing in ('near','far'):
                ix=np.array([r['metadata']['spacing']==spacing for r in first]);strata[cell][spacing]={arm:metrics(y[ix],pred[arm][ix]) for arm in arms}
        if time.time()>deadline-30:raise TimeoutError('Analysis reached original cap')
    training={}
    for arm in arms:
        rows=list(csv.DictReader((run/arm/'metrics.csv').open()));grads=[json.loads(r['diagnostics']) for r in rows if json.loads(r['diagnostics'])]
        selected=agg['runs'][arm]['selected_step'];selected_rows=[r for r in rows if int(r['step'])<=selected];exposure={}
        for row in selected_rows:exposure[row['cell']]=exposure.get(row['cell'],0)+8
        training[arm]=dict(selected_step=selected,selected_episodes=8*selected,selected_cell_exposure=exposure,terminal_updates=len(rows),terminal_episodes=8*len(rows),frames=int(rows[-1]['frames_total']),train_seconds=sum(float(r['seconds']) for r in rows),clipping_fraction=float(np.mean([float(r['clipped']) for r in rows])),median_preclip_norm=float(np.median([float(r['gradient_norm']) for r in rows])),diagnostic_records=len(grads),early_rate_grad_median=float(np.median([r['early_r_gradient_norm'] for r in grads])),encoder_grad_median=float(np.median([r['encoder_gradient_norm'] for r in grads])),final_state_rms=grads[-1]['late_r_rms'],raw_recurrent_update_median=float(np.median([r['effective_recurrent_relative_update'] for r in grads])))
    result=dict(status='completed',cells=results,spatial_minus_dense=deltas,spacing=strata,training=training,uncertainty='1000 paired draws; binding resamples128 four-case blocks, single/motion class-stratified512 base episodes. Shared resample draws retain all paired delay/model presentations. Intervals describe sampled episodes for one trained model per architecture, not seed uncertainty.',independent_test_groups=dict(single=512,motion=512,binding_id=128,binding_locations=128),binding_groups_are_four_case_blocks=True)
    write(HERE/'analysis.json',result);write(run/'analysis.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,prefix,title in zip(axes,('single','binding','binding_locations'),('Single-item orientation','Binding: trained location grid','Binding: unseen location centers')):
        for arm,color,label in [('dense_comparator','#21618c','Dense + comparator'),('spatial_ei','#a93226','Spatial E/I + comparator')]:
            names=[f'binding_D{d}_locations' if prefix=='binding_locations' else f'{prefix}_D{d}' for d in (0,4,12,24)];r=[results[n][arm] for n in names];mean=np.array([v['ba'] for v in r])*100;ci=np.array([v['ba_ci95'] for v in r])*100;ax.errorbar([0,4,12,24],mean,yerr=np.maximum(0,np.stack((mean-ci[:,0],ci[:,1]-mean))),marker='o',color=color,label=label,capsize=3)
        ax.axhline(50,ls=':',color='.6');ax.set(title=title,xlabel='Inserted blank frames',ylabel='Balanced accuracy (%)',xticks=[0,4,12,24],ylim=(35,101));ax.legend(fontsize=8)
    fig.savefig(HERE/'comparison_curves.png',dpi=180);fig.savefig(HERE/'comparison_curves.svg');plt.close(fig)
    lines=['# Spatial versus dense E/I comparison','',f"Both models completed {agg['config']['episodes_per_arm']:,} new training episodes under the fixed shared schedule. Checkpoints were selected using validation only. The table reports paired held-out evidence, with confidence intervals clustered by the binding four-case blocks.",'','| Condition | Dense BA% | Spatial BA% | Spatial−dense pp [95%CI] |','|---|---:|---:|---:|']
    for cell,rs in results.items():
        delta=deltas[cell];lines.append(f"| {cell} | {rs['dense_comparator']['ba']*100:.2f} | {rs['spatial_ei']['ba']*100:.2f} | {delta['ba']*100:+.2f} [{delta['ba_ci95'][0]*100:+.2f}, {delta['ba_ci95'][1]*100:+.2f}] |")
    lines+=['','[Comparison curves](comparison_curves.png) · [Full BA/AUC/confusions/spacing and paired intervals](analysis.json) · [Validation trajectories and run records](results.json)','','## Exposure and resources','']
    for arm,v in training.items():lines.append(f"- {arm}: selected update {v['selected_step']} ({v['selected_episodes']:,} new episodes); terminal {v['terminal_updates']} updates, {v['frames']:,} logical frames. Measured training collection/BPTT {v['train_seconds']:.1f}s; clipping on {100*v['clipping_fraction']:.1f}% of updates; median early-rate gradient {v['early_rate_grad_median']:.5g}. Selected per-cell exposure is retained in analysis.json.")
    lines+=['','## Interpretation limits','','Dense inherits its trained memory core whereas the spatial core is newly initialized. Both inherit learned sensory weights and receive equal new exposure, but comparator geometry, state size and parameter sharing also differ. This compares practical trained systems; it does not isolate locality or establish a biological mechanism. Spatial states contain 42.25 times as many new memory scalars; the common sensory traces remain additional persistent history. All learned components were allowed to adapt.','','The binding inventory and probe marginals are balanced; label cannot be solved from orientation inventory or probe alone. However, because both items always exchange positions on swap trials, retaining and checking just one location can solve the task. These results test feature-location association and do not prove both items were stored or measure two-item capacity. Binding also uses localized patches and 15/30/60-degree separations, whereas native single-item recall includes 7.5-degree differences, a full-field grating and different query content; cross-task scores are not a controlled load curve.','','New phases/noise remove literal raster matching. The held-out center grid tests spatial interpolation, not extrapolation. Near/far results are descriptive strata with their actual denominators. Four-case balancing causes within-block dependence, accounted for in the primary bootstrap. Location grids are summarized separately, never pooled as independent repeats. The finite acquisition horizon does not establish ultimate capacity or failure to learn. No unrequested follow-up or cloud run was launched.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='analysis_completed',run=str(run),seconds_from_budget_start=time.time()-budget['started_unix'])),flush=True)
if __name__=='__main__':main()
