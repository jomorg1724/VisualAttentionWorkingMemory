"""Task-specific classification summaries for ordered two-frame tasks."""
import numpy as np
from PreAttentiveVision.evaluate import auc, point, wilson


def classification(y, probabilities):
    y=np.asarray(y,dtype=int);p=np.asarray(probabilities);nclasses=p.shape[1]
    pred=p.argmax(1)
    confusion=np.zeros((nclasses,nclasses),dtype=int)
    np.add.at(confusion,(y,pred),1)
    counts=confusion.sum(1)
    recalls=[float(confusion[k,k]/counts[k]) if counts[k] else None for k in range(nclasses)]
    aucs=[auc((y==k).astype(int),p[:,k]) for k in range(nclasses)]
    result=dict(n=len(y),classes=nclasses,confusion_true_rows_pred_columns=confusion.tolist(),
        class_counts=counts.tolist(),recall=recalls,
        recall_ci95=[wilson(int(confusion[k,k]),int(counts[k])) for k in range(nclasses)],
        accuracy=float((pred==y).mean()),balanced_accuracy=float(np.mean(recalls)) if None not in recalls else None,
        chance_accuracy=1/nclasses,macro_ovr_auc=float(np.mean(aucs)) if None not in aucs else None,
        per_class_ovr_auc=aucs,decision_rule='Fixed argmax; no held-out calibration or threshold tuning')
    if nclasses==2:result['binary']=point(y,p[:,1],.5)
    return result


def summarize(rows,resamples=0,seed=7331):
    y=np.array([r['label'] for r in rows]);p=np.array([r['probabilities'] for r in rows])
    result=classification(y,p);groups={}
    for i,r in enumerate(rows):groups.setdefault(r.get('base_id') or r['trial_id'],[]).append(i)
    result.update(unique_base_groups=len(groups),uncertainty_unit='Source-image clusters when base_id exists; generated pairs otherwise; conditional on this trained model')
    if resamples:
        group_rows=list(groups.values());rng=np.random.default_rng(seed);boot=[]
        for _ in range(resamples):
            ix=np.concatenate([group_rows[j] for j in rng.integers(0,len(groups),len(groups))])
            cell=classification(y[ix],p[ix])
            if cell['balanced_accuracy'] is not None and cell['macro_ovr_auc'] is not None:
                boot.append([cell['balanced_accuracy'],cell['macro_ovr_auc']])
        if boot:
            for j,key in enumerate(('balanced_accuracy','macro_ovr_auc')):
                result[key+'_ci95']=np.quantile(np.array(boot)[:,j],[.025,.975]).tolist()
        result['bootstrap_valid_replicates']=len(boot)
    return result


def evaluate_tasks(rows,task_classes,resamples=0):
    tasks={}
    for task in task_classes:
        subset=[r for r in rows if r['task']==task]
        tasks[task]={'overall':summarize(subset,resamples),
            'difficulty':{d:summarize([r for r in subset if r.get('difficulty','unspecified')==d],0)
                          for d in sorted(set(r.get('difficulty','unspecified') for r in subset))}}
    return dict(tasks=tasks,
        macro_balanced_accuracy=float(np.mean([v['overall']['balanced_accuracy'] for v in tasks.values()])),
        mean_normalized_balanced_accuracy=float(np.mean([(v['overall']['balanced_accuracy']-v['overall']['chance_accuracy'])/(1-v['overall']['chance_accuracy']) for v in tasks.values()])),
        macro_ovr_auc=float(np.mean([v['overall']['macro_ovr_auc'] for v in tasks.values()])),
        interpretation='Different class counts have different chance accuracy. Read task scores and confusion matrices separately; one training seed is not seed replication.')
