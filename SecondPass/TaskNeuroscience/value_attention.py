"""Native VDA probability/value stratification without additional inference."""
import json
import numpy as np
from .design import OUT,TASKS,sites,write
from .statistics import interval


def analyze(task):
    if not task.startswith('vda'):return None
    from .figures import save
    import matplotlib.pyplot as plt
    root=OUT/task;trials=[]
    for condition in TASKS[task]['conditions']:
        cell=condition['id'];a=np.load(root/'native'/f'{cell}.npz');meta=json.loads((root/'native'/f'{cell}.json').read_text())
        for i,m in enumerate(meta):
            mask=sites(task,m)[0];mass=(a['attention'][i].reshape(7,100).astype(float)*mask).sum(1)
            trials.append(dict(cell=cell,validity=m['displayed_validity'],value=m['value'],cue_color=m['cue_color'],event='catch' if a['y'][i]==0 else 'cued' if m['change_index']==m['cue_index'] else 'uncued',correct=bool(a['p'][i].argmax()==a['y'][i]),report=bool(a['p'][i].argmax()==1),cue_mass=mass.tolist(),uniform_mass=float(mask.mean())))
    rows=[]
    for validity in sorted(set(t['validity'] for t in trials)):
        for event in ('cued','uncued','catch'):
            selected=[t for t in trials if t['validity']==validity and t['event']==event]
            if not selected:continue
            k=sum(t['report'] for t in selected);rows.append(dict(validity=validity,event=event,n=len(selected),report=k/len(selected),ci95=interval(k,len(selected))))
    reward=[];priority=[];rng=np.random.default_rng(2903)
    for value in sorted(set(t['value'] for t in trials)):
        selected=[t for t in trials if t['value']==value];k=sum(t['correct'] for t in selected)
        reward.append(dict(value=value,cue_color=selected[0]['cue_color'],n=len(selected),accuracy=k/len(selected),ci95=interval(k,len(selected))))
        for validity in sorted(set(t['validity'] for t in trials)):
            selected=[t for t in trials if t['value']==value and t['validity']==validity]
            if not selected:continue
            x=np.array([t['cue_mass'][4]-t['uniform_mass'] for t in selected]);boot=x[rng.integers(len(x),size=(3000,len(x)))].mean(1)
            priority.append(dict(value=value,validity=validity,n=len(x),prechange_excess_cue_mass=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    result=dict(native_trials=len(trials),event_by_validity=rows,by_value=reward,prechange_priority=priority,reward_fraction=float(sum(t['correct']*t['value'] for t in trials)/sum(t['value'] for t in trials)),trial_clock='Prechange priority uses frame 4, the final sample frame before either change frame. Uniform mass is the fraction of 100 sites inside the cue mask.',sampling='Descriptive native stratification pooled over three independent angle-condition streams; Wilson report intervals and trial-bootstrap mean-attention intervals. Colors carry value and cue geometry carries probability; this is an observational stratification.')
    write(root/'value_attention.json',result)
    fig,ax=plt.subplots(1,3,figsize=(10,4),constrained_layout=True)
    for event in ('cued','uncued','catch'):
        rr=[r for r in rows if r['event']==event];x=np.array([r['validity'] for r in rr]);y=np.array([r['report'] for r in rr]);bounds=np.array([r['ci95'] for r in rr])
        ax[0].errorbar(x,y,yerr=[y-bounds[:,0],bounds[:,1]-y],marker='o',capsize=2,label=event)
    ax[0].set(xlabel='Displayed cue probability',ylabel='Change report fraction',ylim=(0,1.04),title='Hits and false reports');ax[0].legend(fontsize=9)
    for value in sorted(set(t['value'] for t in trials)):
        rr=[r for r in priority if r['value']==value];x=np.array([r['validity'] for r in rr]);y=np.array([r['prechange_excess_cue_mass'] for r in rr]);bounds=np.array([r['ci95'] for r in rr])
        ax[1].errorbar(x,y,yerr=[y-bounds[:,0],bounds[:,1]-y],marker='o',capsize=2,label=f'value {value:g}')
    ax[1].axhline(0,ls='--',color='#8899a4');ax[1].set(xlabel='Displayed cue probability',ylabel='Cue mass above uniform',title='Priority before the change');ax[1].legend(fontsize=9)
    yy=np.array([r['accuracy'] for r in reward]);bounds=np.array([r['ci95'] for r in reward]);ax[2].bar(range(len(reward)),yy,color=['#286690','#8373ad','#a84458']);ax[2].errorbar(range(len(reward)),yy,yerr=[yy-bounds[:,0],bounds[:,1]-yy],fmt='none',color='#173447',capsize=3);ax[2].set_xticks(range(len(reward)),[f"{r['value']:g}\nn={r['n']}" for r in reward]);ax[2].set(xlabel='Trial value',ylabel='Accuracy',ylim=(0,1.04),title='Performance by value')
    save(fig,root/'figures/value_attention');return result
