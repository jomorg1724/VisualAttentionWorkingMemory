"""Saved paired predictions only; architecture-only comparison and acute interruption."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import time,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.analysis.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for attempt in range(100):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)
def metrics(y,p):
    k=p.shape[1];guess=p.argmax(1);conf=np.zeros((k,k),int);np.add.at(conf,(y,guess),1);aucs=[]
    for label in range(k):
        positive=y==label;n=positive.sum()
        if n and n<len(y):aucs.append((rankdata(p[:,label])[positive].sum()-n*(n+1)/2)/(n*(len(y)-n)))
    return dict(ba=float((np.diag(conf)/np.maximum(1,conf.sum(1))).mean()),auc=float(np.mean(aucs)),confusion=conf.tolist(),n=len(y))
def loadrows(evaluation):
    grouped={}
    for line in Path(evaluation['predictions']).read_text().splitlines():
        r=json.loads(line);grouped.setdefault(r['condition'],[]).append(r)
    for rows in grouped.values():rows.sort(key=lambda x:int(x['paired_base_id'].split('/')[-1]))
    return grouped
def main():
    agg=read(P/'results.json');assert agg['status']=='completed';run=Path(agg['run_root']);budget=read(run/'budget.json')
    arms=('continuation','controller_feedback');data={arm:loadrows(agg['runs'][arm]['test']) for arm in arms};data['parent']=loadrows(agg['parent_reference']);off=loadrows(agg['feedback_interruption'])
    # Numerical selection was always eight primary cells; remove inherited legacy prose.
    for e in [agg['parent_reference'],agg['feedback_interruption']]+[r for arm in arms for r in agg['runs'][arm]['validation']+[agg['runs'][arm]['test']]]:
        e['interpretation']='Existing eight primary-cell mean validation AUC selects checkpoints; motion and new centers are descriptive. Repeated delays/models share the same base evidence.'
    for p in (P/'results.json',run/'aggregate.json'):write(p,agg)
    rng=np.random.default_rng(48973001);indices={};cells={};deltas={};interruption={};spacing={}
    for cell,base in data['continuation'].items():
        y=np.array([r['label'] for r in base]);kind=('binding_locations' if cell.endswith('locations') else 'binding') if cell.startswith('binding') else ('single' if cell.startswith('single') else 'motion')
        if kind not in indices:
            if kind.startswith('binding'):indices[kind]=np.array([np.concatenate([np.arange(4*b,4*b+4) for b in rng.integers(len(y)//4,size=len(y)//4)]) for _ in range(1000)])
            else:indices[kind]=np.array([np.concatenate([rng.choice(np.flatnonzero(y==v),sum(y==v),replace=True) for v in np.unique(y)]) for _ in range(1000)])
        boot=indices[kind];draws={};cells[cell]={};prob={}
        variants=list(data)+( ['feedback_off_blanks'] if cell in off else [])
        for arm in variants:
            rows=off[cell] if arm=='feedback_off_blanks' else data[arm][cell]
            assert len(rows)==len(base) and all(x['paired_base_id']==z['paired_base_id'] and x['label']==z['label'] and x['metadata']==z['metadata'] for x,z in zip(base,rows))
            p=np.array([r['probabilities'] for r in rows]);prob[arm]=p;cells[cell][arm]=metrics(y,p)
            draws[arm]=np.array([[metrics(y[ix],p[ix])[key] for key in ('ba','auc')] for ix in boot]);cells[cell][arm].update(ba_ci95=np.quantile(draws[arm][:,0],[.025,.975]).tolist(),auc_ci95=np.quantile(draws[arm][:,1],[.025,.975]).tolist())
        def difference(a,b):
            values=draws[a]-draws[b]
            return dict(ba=cells[cell][a]['ba']-cells[cell][b]['ba'],auc=cells[cell][a]['auc']-cells[cell][b]['auc'],ba_ci95=np.quantile(values[:,0],[.025,.975]).tolist(),auc_ci95=np.quantile(values[:,1],[.025,.975]).tolist())
        deltas[cell]=dict(feedback_minus_control=difference('controller_feedback','continuation'),control_minus_parent=difference('continuation','parent'),feedback_minus_parent=difference('controller_feedback','parent'))
        if cell in off:interruption[cell]=difference('feedback_off_blanks','controller_feedback')
        if cell.startswith('binding'):
            spacing[cell]={s:{arm:metrics(y[ix],prob[arm][ix]) for arm in data} for s in ('near','far') for ix in [np.array([r['metadata']['spacing']==s for r in base])]}
        if time.time()>budget['deadline_unix']-30:raise TimeoutError('Original allowance reached')
    training={}
    for arm in arms:
        rows=list(csv.DictReader((run/arm/'metrics.csv').open()));selected=agg['runs'][arm]['selected_step'];selected_rows=[r for r in rows if int(r['step'])<=selected];exposure={}
        for r in selected_rows:exposure[r['cell']]=exposure.get(r['cell'],0)+8
        diag=[json.loads(r['diagnostics']) for r in rows if json.loads(r['diagnostics'])]
        training[arm]=dict(selected_global_step=selected,selected_new_episodes=len(selected_rows)*8,selected_cell_exposure=exposure,terminal_new_updates=len(rows),terminal_new_episodes=len(rows)*8,new_frames=sum(int(r['frames'])*8 for r in rows),train_seconds=sum(float(r['seconds']) for r in rows),clipping_fraction=float(np.mean([float(r['clipped']) for r in rows])),median_preclip_norm=float(np.median([float(r['gradient_norm']) for r in rows])),diagnostic_records=diag,validation_curve=[dict(step=v['step'],primary_mean_auc=v['selection_mean_auc']) for v in agg['runs'][arm]['validation']])
    result=dict(status='completed',cells=cells,paired_differences=deltas,feedback_interruption_off_minus_normal=interruption,spacing=spacing,training=training,uncertainty='1000 paired resamples. Binding uses128 independent four-case blocks per location-grid split; single/motion use512 class-stratified base episodes. Paired model/delay rows never inflate sample size. Conditional on these trained models, not across training seeds.')
    for p in (P/'analysis.json',run/'analysis.json'):write(p,result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,prefix,ds,title in zip(axes.flat,('single','binding','locations','motion'),([0,4,12,24],[0,4,12,24],[0,4,12,24],[0,24]),('Single-item orientation','Binding: trained centers','Binding: unseen centers','Motion-duration decision')):
        for arm,label,color in [('parent','Unchanged spatial parent','#777777'),('continuation','Ordinary continuation','#21618c'),('controller_feedback','Controller feedback','#a93226')]:
            names=[f'binding_D{d}_locations' if prefix=='locations' else f'{prefix}_D{d}' for d in ds];rs=[cells[n][arm] for n in names];mean=np.array([r['ba'] for r in rs])*100;ci=np.array([r['ba_ci95'] for r in rs])*100
            ax.errorbar(ds,mean,yerr=np.maximum(0,np.stack((mean-ci[:,0],ci[:,1]-mean))),marker='o',capsize=3,color=color,label=label)
        ax.axhline(25 if prefix=='motion' else 50,ls=':',color='.6');ax.set(title=title,xlabel='Inserted blank frames',ylabel='Balanced accuracy (%)',xticks=ds,ylim=(18,102));ax.legend(fontsize=8)
    fig.savefig(P/'retention_curves.png',dpi=180);fig.savefig(P/'retention_curves.svg');plt.close(fig)
    lines=['# Additive controller feedback: unchanged teaching','',f"Both arms completed {agg['config']['episodes_per_arm']:,} additional episodes from the same spatial4400 parent. All images, cues, objectives, task heads, comparator and90/10 allocation were unchanged. Checkpoints were selected using the predeclared eight-primary-cell validation AUC.",'','| Condition | Parent BA% | Continued BA% | Feedback BA% | Feedback−continued pp [95%CI] |','|---|---:|---:|---:|---:|']
    for name,r in cells.items():
        d=deltas[name]['feedback_minus_control'];lines.append(f"|{name}|{r['parent']['ba']*100:.2f}|{r['continuation']['ba']*100:.2f}|{r['controller_feedback']['ba']*100:.2f}|{d['ba']*100:+.2f} [{d['ba_ci95'][0]*100:+.2f},{d['ba_ci95'][1]*100:+.2f}]|")
    lines+=['','[Retention curves](retention_curves.png) · [Full BA/AUC/confusions and paired intervals](analysis.json) · [Source, checkpoints and validation curves](results.json)','','## Acute feedback interruption','','Only the24 inserted blank frames were interrupted, on exactly paired examples and unchanged selected weights. This is an out-of-distribution intervention, not the ordinary forward path.','']
    for cell,d in interruption.items():lines.append(f"- {cell}: feedback-off minus normal BA {d['ba']*100:+.2f}pp [95%CI {d['ba_ci95'][0]*100:+.2f},{d['ba_ci95'][1]*100:+.2f}].")
    lines+=['','## Exposure and limits','']
    for arm,r in training.items():lines.append(f"- {arm}: selected global update{r['selected_global_step']} ({r['selected_new_episodes']:,} added episodes), terminal{r['terminal_new_episodes']:,} added episodes and{r['new_frames']:,} logical frames; train{r['train_seconds']:.1f}s, clipping on{100*r['clipping_fraction']:.1f}% of updates. Per-cell selected exposure and gradient/coupling trajectories are saved in analysis.json.")
    lines+=['','The feedback arm adds45,537 parameters and64 controller rate/adaptation state scalars. The controller receives the full sensory/memory summaries and may itself retain content, although it has no direct classifier output. This compares the added controller-feedback package, not attention isolated from extra capacity. Both arms preserve the same learned parent and optimizer histories; equal added exposure does not establish a sufficient acquisition horizon. No new cue, teaching target, angle supervision or sampling intervention was introduced.','','Binding full swaps can be detected by remembering one location; they do not establish two-item capacity. Binding and native single-item renderings/change sizes differ. Held-out centers measure spatial interpolation. Four-case block uncertainty and per-model seed limitations remain. Motion keeps the prior10% allocation and is excluded from checkpoint selection. Acute feedback interruption can indicate functional dependence but is not a complete causal decomposition of training benefits. No additional run or cloud compute was launched.']
    for p in (P/'report.md',run/'report.md'):p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='analysis_completed',elapsed_seconds=time.time()-budget['started_unix'])),flush=True)
if __name__=='__main__':main()
