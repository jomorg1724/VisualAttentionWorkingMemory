"""Exploratory Weibull fits, with matched-nuisance bootstrap uncertainty."""
import json
import numpy as np
from scipy.optimize import minimize
from .design import OUT,TASKS,write

def curve(x,v,gamma,decreasing=False):
    scale,shape,lapse=np.exp(v[0]),np.exp(v[1]),v[2]
    exponent=np.minimum((np.maximum(x,0)/scale)**shape,700)
    return gamma+(1-gamma-lapse)*(np.exp(-exponent) if decreasing else -np.expm1(-exponent))

def fit_one(x,good,gamma,start=None,decreasing=False):
    n=good.shape[1];k=good.sum(1);positive=x[x>0]
    bounds=[(np.log(positive.min()/100),np.log(positive.max()*100)),(np.log(.1),np.log(20)),(0,.2)]
    def loss(v):
        p=np.clip(curve(x,v,gamma,decreasing),1e-9,1-1e-9)
        return -np.sum(k*np.log(p)+(n-k)*np.log1p(-p))
    starts=[start] if start is not None else [np.array([np.log(np.median(positive)),np.log(s),.01]) for s in (.5,2,5)]
    fits=[minimize(loss,v,bounds=bounds,method='L-BFGS-B') for v in starts]
    return min(fits,key=lambda r:r.fun)

def analyze(task,reps=300):
    folder=OUT/task;groups={}
    for path in sorted((folder/'psychometric').glob('*.npz')):
        strength,event=path.stem.split('_',1)
        if event!='native':continue # Event-conditioned report curves remain nonparametric.
        a=np.load(path);groups.setdefault(event,[]).append((float(strength),a['p'].argmax(1)==a['y']))
    result=[]
    for event,rows in groups.items():
        rows.sort(key=lambda r:r[0]);x=np.array([r[0] for r in rows]);good=np.array([r[1] for r in rows]);gamma=1/TASKS[task]['classes']
        if len(x[x>0])<4:continue
        decreasing=task=='contour'
        f=fit_one(x,good,gamma,decreasing=decreasing);v=f.x;scale,shape=np.exp(v[:2]);lapse=v[2]
        # Threshold is halfway from fixed guessing floor to fitted asymptote.
        threshold=scale*np.log(2)**(1/shape);slope=(-1 if decreasing else 1)*(1-gamma-lapse)*shape/(2*scale)*np.log(2)**((shape-1)/shape)
        rng=np.random.default_rng(1914);boot=[]
        for _ in range(reps):
            ix=rng.integers(good.shape[1],size=good.shape[1]);q=fit_one(x,good[:,ix],gamma,start=v,decreasing=decreasing)
            if q.success:boot.append(float(np.exp(q.x[0])*np.log(2)**(1/np.exp(q.x[1]))))
        pred=curve(x,v,gamma,decreasing);observed=good.mean(1)
        boundary=shape<.101 or shape>19.9 or lapse>.199
        result.append(dict(event=event,gamma=gamma,scale=float(scale),shape=float(shape),lapse=float(lapse),decreasing=decreasing,
            threshold=float(threshold),threshold_ci95=np.quantile(boot,[.025,.975]).tolist() if boot else None,
            threshold_within_tested_range=bool(x.min()<=threshold<=x.max()),slope_at_threshold=float(slope),
            optimization_success=bool(f.success),boundary_parameter=bool(boundary),bootstrap_successes=len(boot),bootstrap_repetitions=reps,
            max_absolute_residual=float(abs(pred-observed).max()),x=x.tolist(),predicted=pred.tolist(),observed=observed.tolist(),
            interpretation='Exploratory descriptive fit; midpoint is between chance and fitted asymptote. Bootstrap resamples matched nuisance indices jointly across signal levels. Event-conditioned report curves are retained empirically.'))
    write(folder/'psychometric_fit.json',result)
    if result:
        import matplotlib.pyplot as plt
        from .figures import save,AXES
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for r in result:
            xx=np.linspace(0,max(r['x']),500);v=np.array([np.log(r['scale']),np.log(r['shape']),r['lapse']])
            ax.plot(xx,curve(xx,v,r['gamma'],r['decreasing']),label='Fitted Weibull');ax.scatter(r['x'],r['observed'],label='Observed',zorder=3)
            ax.axvline(r['threshold'],ls='--',color='#777777',label='Midpoint threshold')
        ax.set(xlabel=AXES.get(task,'Signal strength'),ylabel='Correct fraction',ylim=(0,1.04),title='Descriptive psychometric fit');ax.legend();save(fig,folder/'figures/psychometric_fit')
    return result
