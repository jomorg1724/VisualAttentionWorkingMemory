"""Fresh-pair scoring, calibration and source-image-aware uncertainty for PAV."""
import math
from statistics import NormalDist
import numpy as np

FAMILIES=('gabors','dots','shapes','natural')


def auc(y,s):
    positive=s[y==1];negative=np.sort(s[y==0])
    if not len(positive) or not len(negative): return None
    return float((np.searchsorted(negative,positive,'left')+np.searchsorted(negative,positive,'right')).mean()/(2*len(negative)))


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def point(y,s,threshold):
    y=np.asarray(y);s=np.asarray(s);p=s>=threshold
    npos=int((y==1).sum());nneg=int((y==0).sum())
    h=int(((y==1)&p).sum());fa=int(((y==0)&p).sum())
    hr=h/npos if npos else None;fr=fa/nneg if nneg else None
    d=c=None
    if npos and nneg:
        zh=NormalDist().inv_cdf((h+.5)/(npos+1));zf=NormalDist().inv_cdf((fa+.5)/(nneg+1))
        d=zh-zf;c=-(zh+zf)/2
    return dict(n=len(y),signal_n=npos,noise_n=nneg,hits=h,misses=npos-h,false_alarms=fa,
        correct_rejections=nneg-fa,hit_rate=hr,false_alarm_rate=fr,
        hit_rate_ci95=wilson(h,npos),false_alarm_rate_ci95=wilson(fa,nneg),
        balanced_accuracy=(hr+1-fr)/2 if npos and nneg else None,
        auroc=auc(y,s),dprime=d,criterion=c)


def calibrate(rows):
    y=np.asarray([r['label'] for r in rows]);s=np.asarray([r['prob_change'] for r in rows])
    family=np.asarray([r['family'] for r in rows])
    candidates=np.r_[np.unique(s),np.nextafter(s.max(),np.inf)]
    best=None
    for t in candidates:
        macro=float(np.mean([point(y[family==f],s[family==f],t)['balanced_accuracy'] for f in FAMILIES]))
        if best is None or (macro,float(t))>(best['macro_balanced_accuracy'],best['threshold']):
            best=dict(threshold=float(t),macro_balanced_accuracy=macro)
    best['rule']='Maximize equal-family validation BA; exact ties choose largest threshold. One common threshold across all tasks.'
    return best


def summarize_rows(rows,threshold=.5,resamples=0,seed=7331):
    y=np.asarray([r['label'] for r in rows]);s=np.asarray([r['prob_change'] for r in rows])
    result=point(y,s,threshold)
    groups={}
    for i,r in enumerate(rows):
        key=r.get('base_id') or r['trial_id']
        groups.setdefault(key,[]).append(i)
    result['unique_base_groups']=len(groups)
    result['uncertainty_unit']='CIFAR source-image cluster for natural pairs; generated pair otherwise'
    boot=[]
    if resamples and result['auroc'] is not None:
        group_rows=list(groups.values());rng=np.random.default_rng(seed)
        for _ in range(resamples):
            sampled=rng.integers(0,len(group_rows),size=len(group_rows))
            ix=np.concatenate([group_rows[j] for j in sampled])
            cell=point(y[ix],s[ix],threshold)
            if cell['auroc'] is not None:
                boot.append([cell['auroc'],cell['balanced_accuracy'],cell['dprime'],cell['criterion']])
        for k,name in enumerate(['auroc','balanced_accuracy','dprime','criterion']):
            result[name+'_ci95']=np.quantile(np.asarray(boot)[:,k],[.025,.975]).tolist()
    return result,np.asarray(boot)


def evaluate_rows(rows,threshold=.5,resamples=0,seed=7331):
    families={};macro_boot=[]
    for fidx,family in enumerate(FAMILIES):
        subset=[r for r in rows if r['family']==family]
        overall,boot=summarize_rows(subset,threshold,resamples,seed+100*fidx)
        difficulties={}
        for didx,d in enumerate(['easy','medium','hard']):
            selected=[r for r in subset if r['difficulty']==d]
            difficulties[d]=summarize_rows(selected,threshold,resamples,seed+100*fidx+didx+1)[0]
        families[family]=dict(overall=overall,difficulty=difficulties,
            unique_base_images=len({r['base_id'] for r in subset if r.get('base_id')}))
        if len(boot):macro_boot.append(boot)
    macro={name:float(np.mean([v['overall'][name] for v in families.values()])) for name in ['auroc','balanced_accuracy']}
    if macro_boot and len({len(a) for a in macro_boot})==1:
        aggregate=np.mean(macro_boot,axis=0)
        for k,name in enumerate(['auroc','balanced_accuracy']):macro[name+'_ci95']=np.quantile(aggregate[:,k],[.025,.975]).tolist()
    return dict(threshold=threshold,macro=macro,families=families,
        interpretation='Intervals condition on trained checkpoint and validation-selected threshold. Paired test seeds shared across model/seed runs; repeated source images clustered, not independent natural images.')
