"""CPU-only posthoc combination diagnostic from saved probability vectors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json
import time
import hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/'combination_protocol.json').read_text())
ROOT=HERE/PROTOCOL['source_run']


def write(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def auc(y,s):
    a=s[y==1];b=np.sort(s[y==0])
    return float((np.searchsorted(b,a,'left')+np.searchsorted(b,a,'right')).mean()/(2*len(b)))


def stats(y,p):
    pred=p.argmax(1);classes=p.shape[1]
    return dict(balanced_accuracy=float(np.mean([(pred[y==c]==c).mean() for c in range(classes)])),
        macro_ovr_auc=float(np.mean([auc((y==c).astype(int),p[:,c]) for c in range(classes)])))


def load(split,identities):
    result={}
    for model in PROTOCOL['models']:
        path=ROOT/f'{model}_seed20271'/('eval_val_000756' if split=='val' else 'eval_test')/'predictions.jsonl'
        data=path.read_bytes();identities[str(path.relative_to(HERE))]=hashlib.sha256(data).hexdigest()
        rows=[json.loads(line) for line in data.splitlines()]
        result[model]={task:[r for r in rows if r['task']==task] for task in PROTOCOL['tasks']}
    for task in PROTOCOL['tasks']:
        reference=[(r['trial_id'],r['label'],r['base_id']) for r in result[PROTOCOL['models'][0]][task]]
        for model in PROTOCOL['models']:
            assert reference==[(r['trial_id'],r['label'],r['base_id']) for r in result[model][task]]
    return result


def arrays(rows):return np.array([r['label'] for r in rows]),np.array([r['probabilities'] for r in rows])


def main():
    started=time.monotonic();identities={};validation=load('val',identities);choices=[]
    for first,second in PROTOCOL['ensembles']:
        tasks={}
        for task in PROTOCOL['tasks']:
            y,p=arrays(validation[first][task]);_,q=arrays(validation[second][task])
            tasks[task]={'first':stats(y,p),'second':stats(y,q),'ensemble':stats(y,.5*p+.5*q)}
        choices.append(dict(first=first,second=second,tasks=tasks,
            selection_score=float(np.mean([v['ensemble']['macro_ovr_auc'] for v in tasks.values()]))))
    chosen=max(choices,key=lambda x:x['selection_score'])
    selection=dict(protocol=PROTOCOL,choices=choices,selected_pair=[chosen['first'],chosen['second']],
        selection_saved_before_test_load=True)
    write('combination_validation_selection.json',selection)
    test=load('test',identities);output=[]
    for first,second in PROTOCOL['ensembles']:
        for task in PROTOCOL['tasks']:
            y,p=arrays(test[first][task]);_,q=arrays(test[second][task]);e=.5*p+.5*q
            pc=p.argmax(1)==y;qc=q.argmax(1)==y;ec=e.argmax(1)==y
            metrics={'first':stats(y,p),'second':stats(y,q),'ensemble':stats(y,e)}
            oracle=pc|qc
            counts=dict(both_correct=int((pc&qc).sum()),first_only_correct=int((pc&~qc).sum()),
                second_only_correct=int((~pc&qc).sum()),both_wrong=int((~pc&~qc).sum()),
                ensemble_fixes_first_error=int((~pc&ec).sum()),ensemble_breaks_first_correct=int((pc&~ec).sum()),
                ensemble_fixes_second_error=int((~qc&ec).sum()),ensemble_breaks_second_correct=int((qc&~ec).sum()),
                first_correct=int(pc.sum()),second_correct=int(qc.sum()),ensemble_correct=int(ec.sum()),
                oracle_correct=int(oracle.sum()))
            diff={}
            for key in ('balanced_accuracy','macro_ovr_auc'):
                diff[key+'_vs_first']=metrics['ensemble'][key]-metrics['first'][key]
                diff[key+'_vs_second']=metrics['ensemble'][key]-metrics['second'][key]
                diff[key+'_vs_best_test_constituent']=metrics['ensemble'][key]-max(metrics['first'][key],metrics['second'][key])
            rng=np.random.default_rng(202609129);boots=[]
            for _ in range(2000):
                ix=rng.integers(0,len(y),len(y));z=y[ix]
                cells=[stats(z,s[ix]) for s in (p,q,e)]
                boots.append([cells[2][key]-cells[j][key] for key in ('balanced_accuracy','macro_ovr_auc') for j in (0,1)])
            boots=np.array(boots);intervals={}
            for j,key in enumerate(('balanced_accuracy_vs_first','balanced_accuracy_vs_second','macro_ovr_auc_vs_first','macro_ovr_auc_vs_second')):
                intervals[key]=np.quantile(boots[:,j],[.025,.975]).tolist()
            output.append(dict(task=task,first=first,second=second,n=len(y),metrics=metrics,counts=counts,
                paired_differences=diff,paired_difference_ci95=intervals,
                oracle_accuracy=float(oracle.mean()),oracle_interpretation='True-label-informed any-constituent-correct upper bound for selecting these two predictions; unavailable to a deployable selector'))
    a=json.loads((ROOT/'aggregate.json').read_text());trajectories={}
    for m in a['runs']:
        if m['model'] in PROTOCOL['models']:
            trajectories[m['model']]=[dict(step=v['step'],tasks={t:v['tasks'][t]['overall'] for t in PROTOCOL['tasks']}) for v in m['val_curve']]
    result=dict(protocol=PROTOCOL,validation_selection=selection,test_analysis=output,
        validation_trajectories=trajectories,source_prediction_sha256=identities,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_seconds=time.monotonic()-started,original_model_updates=0,model_inferences=0)
    write('combination_analysis.json',result)
    print(json.dumps({'selected':selection['selected_pair'],'validation':choices,'test':output,'cpu_seconds':result['cpu_seconds']},indent=2))


if __name__=='__main__':main()
