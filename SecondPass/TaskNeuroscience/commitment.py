"""Check how threshold-defined early commitments relate to terminal choices."""
import json
import numpy as np
from .design import OUT,TASKS,THRESHOLDS,neural_cells,onset,write


def analyze(task):
    root=OUT/task;rows=[];settling=[]
    for spec in TASKS[task]['conditions']:
        cell=spec['id'];a=np.load(root/'native'/f'{cell}.npz');meta=json.loads((root/'native'/f'{cell}.json').read_text())
        latency=[];good=[]
        for p,y,m in zip(a['temporal'],a['y'],meta):
            begin=onset(task,m,len(p));choice=p[begin:].argmax(1);different=np.flatnonzero(choice!=choice[-1]);latency.append(int(different[-1])+2 if len(different) else 1);good.append(bool(choice[-1]==y))
        settling.append(dict(cell=cell,n=len(latency),latencies=latency,terminal_correct=good,median_frames=float(np.median(latency)),mean_frames=float(np.mean(latency))))
        for threshold in THRESHOLDS:
            r=dict(cell=cell,threshold=threshold,n=len(a['y']),crossings=0,changed_choice=0,rescued_error=0,spoiled_correct=0,early_correct=0,terminal_correct_among_crossers=0)
            for p,y,m in zip(a['temporal'],a['y'],meta):
                begin=onset(task,m,len(p));ix=np.flatnonzero(p[begin:].max(1)>=threshold)
                if not len(ix):continue
                early=int(p[begin+ix[0]].argmax());last=int(p[-1].argmax());r['crossings']+=1;r['changed_choice']+=int(early!=last);r['rescued_error']+=int(early!=y and last==y);r['spoiled_correct']+=int(early==y and last!=y);r['early_correct']+=int(early==y);r['terminal_correct_among_crossers']+=int(last==y)
            rows.append(r)
    pooled=[]
    for threshold in THRESHOLDS:
        rr=[r for r in rows if r['threshold']==threshold];pooled.append(dict(threshold=threshold,**{k:sum(r[k] for r in rr) for k in rr[0] if k not in ('cell','threshold')}))
    result=dict(cells=rows,pooled=pooled,settling=settling,interpretation='First post-evidence confidence crossing is compared with the unchanged model terminal choice on the same movie. Rescue means an incorrect early commitment becomes correct; spoiled means a correct early commitment becomes incorrect. Pooled counts are descriptive presentations and preserve all native conditions; the representative panel uses one independent 128-trial condition. Settling latency is one plus the last post-evidence index at which a later choice revision still occurs: the first frame whose argmax matches the terminal argmax through the end. It is retrospective and uses future observations, so it describes readout dynamics rather than a causal stopping policy.')
    write(root/'commitment.json',result)
    import matplotlib.pyplot as plt
    from .figures import save
    cell=neural_cells(task)[0];rr=[r for r in rows if r['cell']==cell];fig,ax=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    for key,label in [('early_correct','At first crossing'),('terminal_correct_among_crossers','At trial end, same crossers')]:
        ax[0].plot(THRESHOLDS,[r[key]/r['crossings'] if r['crossings'] else np.nan for r in rr],marker='o',label=label)
    ax[0].set(xlabel='Confidence threshold',ylabel='Correct fraction',ylim=(0,1.04),title='Early commitment versus final choice');ax[0].legend(fontsize=9)
    for key,label in [('changed_choice','Any changed choice'),('rescued_error','Early error corrected'),('spoiled_correct','Early correct answer lost')]:
        ax[1].plot(THRESHOLDS,[r[key]/r['crossings'] if r['crossings'] else np.nan for r in rr],marker='o',label=label)
    ax[1].set(xlabel='Confidence threshold',ylabel='Fraction among crossing trials',ylim=(0,1.04),title='Revisions after commitment');ax[1].legend(fontsize=9)
    save(fig,root/'figures/commitment')
    r=next(r for r in settling if r['cell']==cell);latency=np.array(r['latencies']);good=np.array(r['terminal_correct']);xx=np.arange(1,int(latency.max())+1);fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True)
    for mask,label in ((good,'Terminally correct'),(~good,'Terminally incorrect')):
        if mask.any():ax.step(xx,[float((latency[mask]<=t).mean()) for t in xx],where='post',label=f'{label} (n={mask.sum()})')
    ax.set(xlabel='Frames since evidence onset',ylabel='Fraction settled to final choice',ylim=(0,1.04),title='Retrospective choice settling');ax.legend(fontsize=10);save(fig,root/'figures/choice_settling');return result
