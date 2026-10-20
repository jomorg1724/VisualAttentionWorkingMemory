"""Diagnostic literature-linked reanalysis of existing trials; no model inference."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .design import OUT

def read(p): return json.loads(p.read_text(encoding='utf8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;e=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-e),float(c+e)]
def csv_rows(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(task,name,fig):
    p=OUT/task/'figures'/name
    fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def main():
    provenance={};rows=[];task='motion_duration_cued'
    for p in sorted((OUT/task/'native').glob('*.json')):
        m=read(p);a=np.load(p.with_suffix('.npz'));y=a['y'];pred=a['p'].argmax(-1)
        seq=np.array([x['directions_by_patch'][x['target_location']] for x in m])
        assert np.array_equal(y,np.array([np.bincount(s,minlength=4).argmax() for s in seq]))
        lastthree=[]
        for s in seq:
            c=np.bincount(s[-3:],minlength=4);lastthree.append(next(int(v) for v in s[-3:][::-1] if c[v]==c.max()))
        heuristics={'first_transition':seq[:,0],'last_transition':seq[:,-1],'last_three_transitions':np.array(lastthree)}
        for name,h in heuristics.items():
            conflict=h!=y;n=int(conflict.sum());k=int(((pred==y)&conflict).sum())
            rows.append(dict(cell=p.stem,delay=m[0]['condition']['delay'],strategy=name,n=len(y),
                model_accuracy=float((pred==y).mean()),heuristic_accuracy=float((h==y).mean()),
                conflict_trials=n,model_correct_on_conflict=k,model_conflict_accuracy=k/n if n else None,
                model_conflict_ci95=wilson(k,n)))
        for q in (p,p.with_suffix('.npz')):provenance[str(q.relative_to(OUT))]=digest(q)
    result=dict(method='Descriptive diagnostic subsets, separately within each native delay condition. Last-three plurality ties choose the most recent tied direction. No fitting, model changes or new inference.',rows=rows)
    write(OUT/task/'literature_analysis.json',result);csv_rows(OUT/task/'source_tables/literature_temporal_shortcuts.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.1),layout='constrained')
    for i,name in enumerate(heuristics):
        rr=sorted([r for r in rows if r['strategy']==name],key=lambda r:r['delay']);x=[r['delay'] for r in rr]
        axes[0].plot(x,[r['heuristic_accuracy'] for r in rr],'o-',label=name.replace('_',' '))
        lo=np.array([r['model_conflict_ci95'][0] for r in rr]);hi=np.array([r['model_conflict_ci95'][1] for r in rr]);v=np.array([r['model_conflict_accuracy'] for r in rr])
        axes[1].errorbar(np.array(x)+(i-1)*.16,v,yerr=[v-lo,hi-v],fmt='o-',capsize=3,label=name.replace('_',' '))
    axes[0].plot(x,[r['model_accuracy'] for r in rr],'ks-',label='trained model',lw=2)
    for ax in axes:
        ax.set(xlabel='Blank delay (frames)',ylabel='Accuracy',ylim=(0,1.05),xticks=[0,4,12,24]);ax.axhline(.25,color='gray',ls=':',lw=1);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Whole native condition');axes[1].set_title('Model when the shortcut is wrong')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    save(task,'literature_temporal_shortcuts',fig)

    rows=[];task='image_recognition'
    for p in sorted((OUT/task/'native').glob('*.json')):
        m=read(p);a=np.load(p.with_suffix('.npz'));y=a['y'];pred=a['p'].argmax(-1)
        assert all(int(mm['label'])==int(yy) for mm,yy in zip(m,y))
        n1=int((y==1).sum());n0=int((y==0).sum());h=int(((y==1)&(pred==1)).sum());f=int(((y==0)&(pred==1)).sum())
        zh=ndtri((h+.5)/(n1+1)) if n1 else None;zf=ndtri((f+.5)/(n0+1)) if n0 else None
        rows.append(dict(cell=p.stem,load=m[0]['study_load'],probe_hold=m[0]['probe_hold'],repeat_trials=n1,novel_trials=n0,
            hits=h,false_alarms=f,hit_rate=h/n1 if n1 else None,false_alarm_rate=f/n0 if n0 else None,
            dprime=float(zh-zf) if n1 and n0 else None,criterion=float(-.5*(zh+zf)) if n1 and n0 else None))
        for q in (p,p.with_suffix('.npz')):provenance[str(q.relative_to(OUT))]=digest(q)
    write(OUT/task/'literature_analysis.json',dict(method='Condition-specific descriptive hit/false-alarm rates. d-prime and criterion use (count+0.5)/(opportunities+1). No sensitivity estimate for an empty study list. Trial-level discrimination uses equal-variance Gaussian SDT as a descriptive coordinate system.',rows=rows))
    csv_rows(OUT/task/'source_tables/literature_recognition_sdt.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.9),layout='constrained')
    for j,hold in enumerate(sorted(set(r['probe_hold'] for r in rows))):
        rr=sorted([r for r in rows if r['probe_hold']==hold and r['load']>0],key=lambda r:r['load']);x=[r['load'] for r in rr];color=f'C{j}'
        axes[0].plot(x,[r['hit_rate'] for r in rr],'o-',color=color,label=f'Hits; hold {hold}')
        axes[0].plot(x,[r['false_alarm_rate'] for r in rr],'s--',color=color,label=f'False alarms; hold {hold}')
        axes[1].plot(x,[r['dprime'] for r in rr],'o-',color=color,label=f'hold {hold}')
    axes[0].set(ylabel='Report probability',ylim=(0,1.04),title='Repeat acceptance and false recognition')
    axes[1].set(ylabel="Sensitivity d-prime",title='Sensitivity by study load')
    for ax in axes:ax.set_xlabel('Study images');ax.set_xticks([4,12,24]);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    save(task,'literature_recognition_sdt',fig)
    write(OUT/'literature_reanalysis_provenance.json',dict(source_files=provenance,new_model_forwards=0,
        tasks=['motion_duration_cued','image_recognition'],checkpoint_update=204289))
    print('Literature reanalysis complete; zero new model forwards.',flush=True)

if __name__=='__main__':main()
