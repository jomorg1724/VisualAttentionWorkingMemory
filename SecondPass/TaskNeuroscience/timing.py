"""Matched-scene effects of native timing manipulations."""
import json
import numpy as np
from .design import OUT,TASKS,write
from .statistics import holm

def analyze(task):
    if task not in ('memory_guided_search','motion_memory_search','popout_search'):return []
    import matplotlib.pyplot as plt
    from .figures import save
    factors=('exposure',) if task=='popout_search' else ('sample_frames','delay','exposure')
    group_fields={'memory_guided_search':('family','similarity'),'motion_memory_search':('spacing',),'popout_search':('feature','difference')}[task]
    cells=TASKS[task]['conditions'];groups=sorted(set(tuple(c['kwargs'][k] for k in group_fields) for c in cells));rows=[]
    for group in groups:
        selected=[c for c in cells if tuple(c['kwargs'][k] for k in group_fields)==group]
        for factor in factors:
            levels=sorted(set(c['kwargs'][factor] for c in selected));low,high=levels[0],levels[-1];differences=[]
            for c in selected:
                if c['kwargs'][factor]!=low:continue
                wanted=dict(c['kwargs'],**{factor:high});other=next(z for z in selected if z['kwargs']==wanted)
                p=OUT/task/'native'/f"{c['id']}.npz";q=OUT/task/'native'/f"{other['id']}.npz";a=np.load(p);b=np.load(q)
                am=json.loads(p.with_suffix('.json').read_text());bm=json.loads(q.with_suffix('.json').read_text());assert np.array_equal(a['y'],b['y'])
                assert all(x['trial_id']==y['trial_id'] and x['target_index']==y['target_index'] for x,y in zip(am,bm))
                if task=='motion_memory_search':
                    assert all(x['seed']==y['seed'] and x['directions']==y['directions'] and x['array_first_frame_sha256']==y['array_first_frame_sha256'] for x,y in zip(am,bm))
                    if factor!='exposure':assert all(x['array_sha256']==y['array_sha256'] for x,y in zip(am,bm))
                elif 'array_sha256' in am[0]:assert all(x['array_sha256']==y['array_sha256'] for x,y in zip(am,bm))
                if factor=='exposure':
                    assert np.allclose(a['p'],b['temporal'][:,a['temporal'].shape[1]-1],atol=3e-5), 'Longer exposure must preserve the earlier causal output'
                differences.append((b['p'].argmax(1)==b['y']).astype(float)-(a['p'].argmax(1)==a['y']))
            d=np.mean(differences,axis=0);rng=np.random.default_rng(1612);boot=d[rng.integers(len(d),size=(3000,len(d)))].mean(1);bounds=np.quantile(boot,[.025,.975]).tolist()
            if np.all(d==0):bound=1-.025**(1/len(d));bounds=[-bound,bound]
            null=(rng.choice([-1,1],size=(9999,len(d)))*d).mean(1);pval=(1+(np.abs(null)>=abs(d.mean())-1e-12).sum())/10000
            rows.append(dict(group=' / '.join(map(str,group)),factor=factor,low=low,high=high,paired_condition_pairs=len(differences),scenes=len(d),delta_accuracy=float(d.mean()),ci95=bounds,sign_flip_p=float(pval),exposure_prefix_output_verified=factor=='exposure'))
    for r,p in zip(rows,holm([r['sign_flip_p'] for r in rows])):r['holm_p']=p
    fig,axes=plt.subplots(1,len(factors),figsize=(max(6,3.8*len(factors)),max(4,len(groups)*.55)),squeeze=False,constrained_layout=True)
    for ax,factor in zip(axes[0],factors):
        r=[z for z in rows if z['factor']==factor];v=100*np.array([z['delta_accuracy'] for z in r]);b=100*np.array([z['ci95'] for z in r]);ax.barh(range(len(r)),v,color='#164d70');ax.errorbar(v,range(len(r)),xerr=[v-b[:,0],b[:,1]-v],fmt='none',color='#333333',capsize=2);ax.set_yticks(range(len(r)),[z['group'] for z in r],fontsize=9);ax.axvline(0,ls='--',color='#999999');ax.set(title=factor.replace('_',' '),xlabel='Long minus short accuracy (percentage points)')
    save(fig,OUT/task/'figures/timing_effects');write(OUT/task/'timing_effects.json',rows);return rows
