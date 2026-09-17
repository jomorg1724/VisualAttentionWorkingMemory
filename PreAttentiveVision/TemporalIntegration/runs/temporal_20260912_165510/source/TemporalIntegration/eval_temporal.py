"""Final temporal diagnostics and paired saved-score comparisons."""
import numpy as np
import time
from PreAttentiveVision.evaluate_multitask import evaluate_tasks

MOTION_INVERSE=(2,3,0,1)
BINARY_TASKS=('orientation','contrast','spatial_frequency','chromatic_increment','contour','natural_spectrum')


def swapped_labels(labels,task):
    if task=='motion_direction':
        # Right/up/left/down become left/down/right/up after time reversal.
        return labels.new_tensor(MOTION_INVERSE)[labels]
    if task in BINARY_TASKS:return 1-labels
    raise ValueError('Undeclared frame-swap label mapping')


def summarize_modes(rows,task_classes,resamples=0):
    normal=[r for r in rows if r['mode']=='normal']
    result=evaluate_tasks(normal,task_classes,resamples=resamples)
    result['min_task_ba']=min(v['overall']['balanced_accuracy'] for v in result['tasks'].values())
    result['engineering_criterion_met']=all(v['overall']['balanced_accuracy']>=.95 for v in result['tasks'].values())
    result['diagnostics']={}
    for mode in ('reset_before_second','frame_swap'):
        subset=[r for r in rows if r['mode']==mode]
        if subset:result['diagnostics'][mode]=evaluate_tasks(subset,task_classes,resamples=0)
    result['diagnostic_interpretation']='Reset-before-second removes prior state and can be out of the trained state distribution. Swapped pairs use inverse direction or flipped binary labels. These are repeated presentations of the same pairs, not additional independent episodes.'
    return result


def ranking(y,s):
    positive=s[y==1];negative=np.sort(s[y==0])
    return float((np.searchsorted(negative,positive,'left')+np.searchsorted(negative,positive,'right')).mean()/(2*len(negative)))


def scores(y,p):
    predicted=p.argmax(1);classes=p.shape[1]
    return np.array([np.mean([(predicted[y==k]==k).mean() for k in range(classes)]),
                     np.mean([ranking((y==k).astype(int),p[:,k]) for k in range(classes)])])


def paired_comparison(candidate_rows,reference_rows,resamples=1000,deadline=None):
    tasks={}
    for task in sorted({r['task'] for r in candidate_rows if r['mode']=='normal'}):
        a=[r for r in candidate_rows if r['task']==task and r['mode']=='normal']
        b=[r for r in reference_rows if r['task']==task and r['mode']=='normal']
        assert [(r['trial_id'],r['label'],r['base_id']) for r in a]==[(r['trial_id'],r['label'],r['base_id']) for r in b]
        y=np.array([r['label'] for r in a]);p=np.array([r['probabilities'] for r in a]);q=np.array([r['probabilities'] for r in b])
        ps,qs=scores(y,p),scores(y,q);groups={}
        for i,r in enumerate(a):groups.setdefault(r['base_id'] or r['trial_id'],[]).append(i)
        group_rows=list(groups.values());rng=np.random.default_rng(30389);boot=[]
        for _ in range(resamples):
            if deadline is not None and time.time()>=deadline-1:raise TimeoutError('Paired analysis reached experiment deadline')
            ix=np.concatenate([group_rows[j] for j in rng.integers(0,len(groups),len(groups))])
            boot.append(scores(y[ix],p[ix])-scores(y[ix],q[ix]))
        intervals=np.quantile(boot,[.025,.975],axis=0)
        tasks[task]=dict(n=len(y),source_groups=len(groups),candidate_ba=float(ps[0]),reference_ba=float(qs[0]),
            candidate_auc=float(ps[1]),reference_auc=float(qs[1]),delta_ba=float(ps[0]-qs[0]),delta_auc=float(ps[1]-qs[1]),
            delta_ba_ci95=intervals[:,0].tolist(),delta_auc_ci95=intervals[:,1].tolist())
    return tasks
