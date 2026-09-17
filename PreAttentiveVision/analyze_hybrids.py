"""Paired CPU analysis of the predeclared hybrid comparison; no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import sys
import time
import json
import hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def auc(y,s):
    a=s[y==1];b=np.sort(s[y==0])
    return float((np.searchsorted(b,a,'left')+np.searchsorted(b,a,'right')).mean()/(2*len(b)))


def score(y,p):
    c=p.shape[1];pred=p.argmax(1)
    return np.array([np.mean([(pred[y==k]==k).mean() for k in range(c)]),
                     np.mean([auc((y==k).astype(int),p[:,k]) for k in range(c)])])


def main(root):
    root=Path(root);began=time.monotonic();aggregate=json.loads((root/'aggregate.json').read_text())
    if aggregate['status']!='completed':raise ValueError('Wait for all final evaluations before paired analysis')
    names={r['model']:r['test']['predictions'] for r in aggregate['runs']}
    if aggregate.get('parent_reference',{}).get('status')=='completed':names['parent_step756']=aggregate['parent_reference']['predictions']
    rows={};hashes={}
    for name,path in names.items():
        raw=Path(path).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest();rows[name]=[json.loads(line) for line in raw.splitlines()]
    pairs=[('convnext_gabor_residual','convnext_grn'),('convnext_se_residual','convnext_grn')]
    if 'parent_step756' in rows:pairs.append(('convnext_grn','parent_step756'))
    comparisons=[]
    for candidate,reference in pairs:
        tasks={}
        for task in aggregate['config']['task_classes']:
            a=[r for r in rows[candidate] if r['task']==task];b=[r for r in rows[reference] if r['task']==task]
            assert [(r['trial_id'],r['label'],r['base_id']) for r in a]==[(r['trial_id'],r['label'],r['base_id']) for r in b]
            y=np.array([r['label'] for r in a]);p=np.array([r['probabilities'] for r in a]);q=np.array([r['probabilities'] for r in b])
            ps=score(y,p);qs=score(y,q);pc=p.argmax(1)==y;qc=q.argmax(1)==y
            groups={}
            for i,r in enumerate(a):groups.setdefault(r['base_id'] or r['trial_id'],[]).append(i)
            group_rows=list(groups.values());rng=np.random.default_rng(202609131);samples=[]
            for _ in range(1000):
                ix=np.concatenate([group_rows[j] for j in rng.integers(0,len(group_rows),len(group_rows))])
                samples.append(score(y[ix],p[ix])-score(y[ix],q[ix]))
            ci=np.quantile(samples,[.025,.975],axis=0)
            tasks[task]=dict(n=len(y),source_groups=len(groups),candidate_ba=float(ps[0]),reference_ba=float(qs[0]),
                candidate_auc=float(ps[1]),reference_auc=float(qs[1]),delta_ba=float(ps[0]-qs[0]),delta_auc=float(ps[1]-qs[1]),
                delta_ba_ci95=ci[:,0].tolist(),delta_auc_ci95=ci[:,1].tolist(),
                counts=dict(both_correct=int((pc&qc).sum()),candidate_only_correct=int((pc&~qc).sum()),
                            reference_only_correct=int((~pc&qc).sum()),both_wrong=int((~pc&~qc).sum())))
        criterion=bool(tasks['contour']['delta_ba']>=.03 and tasks['motion_direction']['delta_ba']>=-.02)
        comparisons.append(dict(model=candidate,reference=reference,tasks=tasks,
            engineering_criterion_met=criterion if reference=='convnext_grn' else None))
    result=dict(status='completed',run_root=str(root.resolve()),comparisons=comparisons,
        criterion='Contour BA gain>=3 percentage points and motion BA loss<=2 points versus continued control; engineering screen on point estimates, not formal proof',
        uncertainty='1000 paired source-image-cluster bootstrap resamples for natural task; paired generated-pair bootstrap otherwise; conditional on one parent/seed and selected checkpoints',
        interpretation='Fresh generator evaluation seeds compared with earlier development run; natural test photographs reused as source population, not new independent source images. No test-driven weight or threshold tuning.',
        prediction_sha256=hashes,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_seconds=time.monotonic()-began)
    for path in (root/'paired_analysis.json',HERE/'results_hybrids_analysis.json'):path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
