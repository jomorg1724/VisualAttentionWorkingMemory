"""Saved-artifact-only motion audit. No Torch import, inference, or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,csv,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];started=time.time();sources={}
def read(path):
    path=Path(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
def rows(path):
    path=Path(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();out={}
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['condition'].startswith('motion'):out.setdefault(r['condition'],[]).append(r)
    for v in out.values():v.sort(key=lambda r:int(r['paired_base_id'].split('/')[-1]))
    return out
def auc(y,s):
    n=int(sum(y));m=len(y)-n
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None
def metrics(rs):
    y=np.array([r['label'] for r in rs]);p=np.array([r['probabilities'] for r in rs]);q=p.argmax(1);n=len(y);cm=np.zeros((4,4),int);np.add.at(cm,(y,q),1)
    ac=[auc(y==k,p[:,k]) for k in range(4)];conf=p.max(1);correct=q==y;ece=0.
    for lo in np.arange(0,1,.1):
        ix=(conf>=lo)&(conf<(lo+.1) if lo<.9 else conf<=1)
        if ix.any():ece+=ix.mean()*abs(correct[ix].mean()-conf[ix].mean())
    return dict(n=n,accuracy=float(correct.mean()),balanced_accuracy=float(np.mean([correct[y==k].mean() for k in range(4) if (y==k).any()])),confusion=cm.tolist(),predicted_counts=np.bincount(q,minlength=4).tolist(),true_counts=np.bincount(y,minlength=4).tolist(),class_auc=ac,macro_auc=float(np.mean([a for a in ac if a is not None])),mean_probabilities=p.mean(0).tolist(),mean_confidence=float(conf.mean()),ece10=float(ece),nll=float(-np.log(np.clip(p[np.arange(n),y],1e-15,1)).mean()),brier_sum=float(((p-np.eye(4)[y])**2).sum(1).mean()))
def strata(rs):
    groups={}
    for r in rs:
        m=r['metadata'];d=m['directions'];sw=sum(a!=b for a,b in zip(d,d[1:]));keys=[f"count_margin={m['count_margin']}",f"winner_is_final={r['label']==m['final_direction']}",f"switches={sw}"]
        for key in keys:groups.setdefault(key,[]).append(r)
    return {k:metrics(v) for k,v in groups.items()}
attention=read(ROOT/'WorkingMemory/PreUpdateAttention/results.json');local=read(ROOT/'WorkingMemory/SelectiveMaintenance/results.json');remote=Path(read(ROOT/'WorkingMemory/PreUpdateAttention/retrieval_receipt.json')['results'])
datasets={'attention':rows(remote/'test/predictions.jsonl'),'parent':rows(local['parent_reference']['predictions'])}
for arm in ('continuation','controller_feedback'):datasets[arm]=rows(local['runs'][arm]['test']['predictions'])
results={};comparisons={};rng=np.random.default_rng(57973001)
for cell,base in datasets['attention'].items():
    results[cell]={};comparisons[cell]={};y=np.array([r['label'] for r in base]);boot=np.array([np.concatenate([rng.choice(np.flatnonzero(y==k),sum(y==k),replace=True) for k in range(4)]) for _ in range(1000)])
    for arm,rs in datasets.items():
        group=rs[cell];assert all((a['paired_base_id'],a['label'],a['metadata'])==(b['paired_base_id'],b['label'],b['metadata']) for a,b in zip(base,group)) and len(base)==len(group)
        results[cell][arm]=dict(overall=metrics(group),strata=strata(group))
    a=np.array([np.argmax(r['probabilities'])==r['label'] for r in base])
    for arm in ('continuation','controller_feedback','parent'):
        b=np.array([np.argmax(r['probabilities'])==r['label'] for r in datasets[arm][cell]]);delta=a.astype(float)-b
        comparisons[cell][arm]=dict(attention_minus_arm=float(delta.mean()),ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),both_correct=int(sum(a&b)),attention_only=int(sum(a&~b)),arm_only=int(sum(~a&b)),both_wrong=int(sum(~a&~b)))
trajectories={};training={}
for arm,agg,key,runroot in [('attention',attention,'preupdate_attention',remote),('continuation',local,'continuation',Path(local['run_root'])),('controller_feedback',local,'controller_feedback',Path(local['run_root']))]:
    run=agg['runs'][key];trajectories[arm]=[]
    for v in run['validation']:
        raw=v['predictions'];path=runroot/raw.split('remote_results/')[-1] if raw.startswith('/workspace/') else Path(raw)
        trajectories[arm].append(dict(step=v['step'],selection_auc=v['selection_mean_auc'],motion={c:metrics(r) for c,r in rows(path).items()}))
    path=runroot/key/'metrics.csv';sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();log=list(csv.DictReader(path.open()));bycell=Counter(r['cell'] for r in log);blocks=[]
    for end in (5400,6400,7400,8400):
        batch=[r for r in log if end-1000<int(r['step'])<=end];norm=np.array([float(r['gradient_norm']) for r in batch]);motion=[r for r in batch if r['cell'].startswith('motion')]
        blocks.append(dict(end_step=end,clipping_fraction=float(np.mean([int(r['clipped']) for r in batch])),gradient_norm_median=float(np.median(norm)),gradient_norm_max=float(norm.max()),finite=bool(np.isfinite(norm).all()),motion_loss=float(np.mean([float(r['loss']) for r in motion])),motion_accuracy=float(np.mean([float(r['accuracy']) for r in motion]))))
    cfg=agg['config']['recipe'];assert Counter(cfg['cycle'])==Counter({**{k:9 for k in cfg['cells'] if not k.startswith('motion')},'motion_D0':4,'motion_D24':4})
    training[arm]=dict(updates=len(log),episodes=len(log)*cfg['batch_size'],updates_per_cell=dict(bycell),episodes_per_cell={k:v*cfg['batch_size'] for k,v in bycell.items()},motion_fraction=sum(v for k,v in bycell.items() if k.startswith('motion'))/len(log),blocks=blocks,diagnostics=[dict(step=int(r['step']),cell=r['cell'],values=json.loads(r['diagnostics'])) for r in log if r['diagnostics']!='{}'])
spatial=read(ROOT/'WorkingMemory/SpatialComparison/results.json');oldspatial=rows(spatial['runs']['spatial_ei']['test']['predictions']);oldparent=rows(ROOT/'WorkingMemory/SpatialComparison/parent_predictions.jsonl');prior={}
for cell,rs in oldspatial.items():
    assert all((a['paired_base_id'],a['label'],a['metadata'])==(b['paired_base_id'],b['label'],b['metadata']) for a,b in zip(rs,oldparent[cell]))
    prior[cell]=dict(spatial4400=metrics(rs),retention14800=metrics(oldparent[cell]))
oldcomparison=read(ROOT/'WorkingMemory/SpatialComparison/parent_reference.json');interruption=read(ROOT/'WorkingMemory/SelectiveMaintenance/analysis.json')
for path in ['WorkingMemory/PreUpdateAttention/train.py','WorkingMemory/SelectiveMaintenance/train.py','WorkingMemory/SpatialComparison/stimuli.py']:
    p=ROOT/path;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
calibration={}
for arm,agg,key,rr in [('attention',attention,'preupdate_attention',remote),('continuation',local,'continuation',Path(local['run_root']))]:
    selected=next(v for v in agg['runs'][key]['validation'] if v['step']==8400);raw=selected['predictions'];path=rr/raw.split('remote_results/')[-1] if raw.startswith('/workspace/') else Path(raw);val=rows(path)['motion_D0'];y=np.array([r['label'] for r in val]);lp=np.log(np.clip(np.array([r['probabilities'] for r in val]),1e-15,1))
    def objective(b):
        offsets=np.r_[b,-sum(b)];z=lp+offsets;pr=softmax(z,axis=1);err=pr-np.eye(4)[y];g=err.mean(0)+.01*offsets
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])+.005*(offsets**2).sum()),g[:3]-g[3]
    fit=minimize(objective,np.zeros(3),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12,'gtol':1e-8});offsets=np.r_[fit.x,-sum(fit.x)];out={}
    for cell,rs in datasets[arm].items():
        pr=softmax(np.log(np.clip(np.array([r['probabilities'] for r in rs]),1e-15,1))+offsets,axis=1);changed=[dict(r,probabilities=p.tolist()) for r,p in zip(rs,pr)];yy=np.array([r['label'] for r in rs]);old=np.array([np.argmax(r['probabilities'])==r['label'] for r in rs]);new=pr.argmax(1)==yy;d=new.astype(float)-old
        b=np.array([np.concatenate([rng.choice(np.flatnonzero(yy==k),sum(yy==k),replace=True) for k in range(4)]) for _ in range(1000)])
        out[cell]=dict(metrics=metrics(changed),delta_accuracy=float(d.mean()),delta_ci95=np.quantile(d[b].mean(1),[.025,.975]).tolist())
    calibration[arm]=dict(training_n=len(val),fitted_on='selected8400 existing D0 validation only',lambda_fixed=.01,objective='mean cross entropy + .01/2 sum of all4 zero-sum offsets squared',offsets=offsets.tolist(),optimizer_success=bool(fit.success),iterations=int(fit.nit),message=str(fit.message),test=out)
result=dict(status='completed',class_order=['right','up','left','down'],motion=results,attention_paired_differences=comparisons,validation=trajectories,training=training,calibration=calibration,prior_independent_matched_comparison=prior,prior_paired_summary=oldcomparison,additive_saved_analysis={k:v for k,v in interruption.items() if 'interruption' in k or 'intervention' in k},source_hashes=sources,seconds=time.time()-started,compute='CPU only, threads1, no torch/model import or inference; two fixed3-parameter offset fits only',limits='Same512 base episodes repeated across delays/models are not independent replicates. Strata are descriptive and conditional on trained seeds. Prior spatial-versus-retention matched comparison uses a different test seed from current four-model comparison; do not pair across them. AUC ranks probability scores within each class, not across classes. No causal attribution of training allocation or clipping.')
(HERE/'findings.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{a:v['overall'] for a,v in arms.items()} for k,arms in results.items()},indent=2));print('SECONDS',result['seconds'])
