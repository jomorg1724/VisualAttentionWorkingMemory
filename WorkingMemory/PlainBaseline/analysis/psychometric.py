"""Psychometric sweeps for a trained orientation_cued model (any arm): magnitude, delay, cue reliability, distraction.

Usage: python -m WorkingMemory.PlainBaseline.analysis.psychometric --checkpoint <terminal.pt> --out <dir> [--n 512] [--sweeps magnitude,delay,cue,distraction]
Writes <out>/psychometric.json and <out>/psychometric.md. Fits a cumulative Gaussian with lapse to the magnitude sweep
(guess rate 0.5, maximum likelihood over a grid, bootstrap CIs), an exponential to the delay sweep, and reports every
point with a 2,000-replicate trial bootstrap. All sweeps use PSYCH_SEED, distinct from train/val/test.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
np.seterr(over='ignore',invalid='ignore')
from scipy.stats import norm
from scipy.optimize import minimize
from WorkingMemory.PlainBaseline.baseline import PlainBaseline,ALL_TASK_CLASSES
from WorkingMemory.PlainBaseline.analysis.psych_stream import PsychOrientationStream,PSYCH_SEED
from WorkingMemory.BatteryAudit.observers import balanced_accuracy,auc_binary

def load_model(path,device):
    saved=torch.load(path,map_location=device);a=saved['args']
    if a.get('encoder')=='accum':
        from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline
        m=AccumulatorBaseline(ALL_TASK_CLASSES,stack=a['stack'],feature_norm=a['feature_norm'],center=a['center'],accumulator=a['accumulator'])
    else:m=PlainBaseline(ALL_TASK_CLASSES,stack=a['stack'],feature_norm=a['feature_norm'],center=a['center'])
    m.load_state_dict(saved['model']);m.to(device).eval();return m,a

def run_point(model,stream,delay,n,device,chunk=32):
    ys=[];ps=[];pr=[]
    with torch.no_grad():
        for off in range(0,n,chunk):
            x,y,_=stream.batch(min(chunk,n-off),'orientation_cued',dict(delay=delay));lg=model(x.to(device),'orientation_cued').float().cpu()
            ys+=y.tolist();ps+=lg.argmax(1).tolist();pr+=lg.softmax(1)[:,1].tolist()
    ys=np.array(ys);ps=np.array(ps);pr=np.array(pr);correct=(ys==ps).astype(float)
    rng=np.random.default_rng(0);boot=[balanced_accuracy(ys[i],ps[i],2) for i in (rng.integers(0,len(ys),len(ys)) for _ in range(2000))]
    return dict(n=int(len(ys)),ba=balanced_accuracy(ys,ps,2),ba_ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],auc=auc_binary(ys,pr),accuracy=float(correct.mean()),correct=correct.tolist())

def fit_cumulative_gaussian(xs,ks,ns,guess=.5):
    """Maximum-likelihood fit of psi(x)=guess+(1-guess-lapse)*Phi((x-mu)/s); returns mu (threshold at the midpoint), s, lapse."""
    xs=np.array(xs,float);ks=np.array(ks,float);ns=np.array(ns,float)
    def nll(p):
        mu,logs,lap=p;lap=1/(1+np.exp(-lap))*.1;psi=guess+(1-guess-lap)*norm.cdf((xs-mu)/np.exp(logs));psi=np.clip(psi,1e-6,1-1e-6)
        return -(ks*np.log(psi)+(ns-ks)*np.log(1-psi)).sum()
    best=None
    for mu0 in np.linspace(xs.min(),xs.max(),6):
        for logs0 in (0.,1.,2.):
            r=minimize(nll,[mu0,logs0,-3.],method='Nelder-Mead',options=dict(maxiter=4000))
            if best is None or r.fun<best.fun:best=r
    mu,logs,lap=best.x;lap=1/(1+np.exp(-lap))*.1;s=float(np.exp(logs))
    if not (xs.min()-2*s<=mu<=xs.max()+2*s):mu=float('nan')  # curve is flat within the tested range: no threshold
    return dict(threshold_75=float(mu),slope=float(1/s),spread=s,lapse=float(lap),nll=float(best.fun))

def fit_exponential(ds,ys):
    ds=np.array(ds,float);ys=np.array(ys,float)
    def sse(p):
        a,b,logtau=p;return ((a+(b-a)*np.exp(-ds/np.exp(logtau))-ys)**2).sum()
    best=min((minimize(sse,[ys.min(),ys.max(),lt],method='Nelder-Mead') for lt in (0.,2.,4.)),key=lambda r:r.fun)
    a,b,logtau=best.x;return dict(asymptote=float(a),initial=float(b),tau_frames=float(np.exp(logtau)),sse=float(best.fun))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);ap.add_argument('--n',type=int,default=512)
    ap.add_argument('--sweeps',default='magnitude,delay,cue,distraction');ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model,args=load_model(a.checkpoint,a.device);res=dict(checkpoint=a.checkpoint,model_args=args,n_per_point=a.n,seed=PSYCH_SEED,sweeps={})
    sweeps=a.sweeps.split(',');t0=time.time()
    if 'magnitude' in sweeps:
        mags=[2.5,5,7.5,10,15,22.5,30,45];res['sweeps']['magnitude']={}
        for delay in (0,4,12,24):
            pts=[]
            for mg in mags:
                r=run_point(model,PsychOrientationStream(magnitudes=(mg,)),delay,a.n,a.device);r['magnitude']=mg;pts.append({k:v for k,v in r.items() if k!='correct'});r['_ks']=sum(r['correct'])
                pts[-1]['_ks']=r['_ks']
            fit=fit_cumulative_gaussian(mags,[p.pop('_ks') for p in pts],[a.n]*len(mags))
            res['sweeps']['magnitude'][f'D{delay}']=dict(points=pts,fit=fit);print(f"magnitude D{delay}: "+' '.join(f"{p['magnitude']}:{p['ba']:.3f}" for p in pts)+f" | thr75 {fit['threshold_75']:.1f} deg lapse {fit['lapse']:.3f}",flush=True)
    if 'delay' in sweeps:
        delays=[0,1,2,4,8,12,16,24,32,48];pts=[]
        for d in delays:
            r=run_point(model,PsychOrientationStream(magnitudes=(15,)),d,a.n,a.device);r.pop('correct');r['delay']=d;pts.append(r)
        fit=fit_exponential(delays,[p['ba'] for p in pts]);res['sweeps']['delay']=dict(points=pts,fit=fit)
        print('delay (15 deg): '+' '.join(f"{p['delay']}:{p['ba']:.3f}" for p in pts)+f" | tau {fit['tau_frames']:.1f} frames asymptote {fit['asymptote']:.3f}",flush=True)
    if 'cue' in sweeps:
        pts=[]
        for scale in (1.0,.5,.25,.1):
            for jit in (0,2,4):
                r=run_point(model,PsychOrientationStream(cue_scale=scale,glyph_jitter_px=jit),4,a.n,a.device);r.pop('correct');r.update(cue_scale=scale,jitter=jit);pts.append(r)
        res['sweeps']['cue']=dict(points=pts,delay=4);print('cue (D4): '+' '.join(f"s{p['cue_scale']}j{p['jitter']}:{p['ba']:.3f}" for p in pts),flush=True)
    if 'distraction' in sweeps:
        pts=[]
        for nd in (0,1,2,3):
            for same in (False,True):
                r=run_point(model,PsychOrientationStream(n_distractors=nd,distractor_same_mag=same),12,a.n,a.device);r.pop('correct');r.update(n_distractors=nd,same_magnitude=same);pts.append(r)
        res['sweeps']['distraction']=dict(points=pts,delay=12);print('distraction (D12): '+' '.join(f"n{p['n_distractors']}{'s' if p['same_magnitude'] else ''}:{p['ba']:.3f}" for p in pts),flush=True)
    res['seconds']=time.time()-t0;(out/'psychometric.json').write_text(json.dumps(res,indent=1),encoding='utf-8')
    L=[f"# Psychometric sweeps\n\nCheckpoint `{a.checkpoint}`; {a.n} trials per point at seed {PSYCH_SEED}; BA with 95% trial-bootstrap intervals.\n"]
    for name,sw in res['sweeps'].items():
        L.append(f'## {name}\n')
        if name=='magnitude':
            for d,v in sw.items():L.append(f"**{d}**: "+', '.join(f"{p['magnitude']}deg {p['ba']:.3f} [{p['ba_ci'][0]:.3f},{p['ba_ci'][1]:.3f}]" for p in v['points'])+f"; fit threshold(75%) {v['fit']['threshold_75']:.2f} deg, slope {v['fit']['slope']:.3f}/deg, lapse {v['fit']['lapse']:.3f}\n")
        else:
            L.append(', '.join(f"{ {k:vv for k,vv in p.items() if k in ('delay','cue_scale','jitter','n_distractors','same_magnitude')} } {p['ba']:.3f} [{p['ba_ci'][0]:.3f},{p['ba_ci'][1]:.3f}]" for p in sw['points'])+('\n'+json.dumps(sw['fit']) if 'fit' in sw else '')+'\n')
    (out/'psychometric.md').write_text('\n'.join(L),encoding='utf-8');print('wrote',out)

if __name__=='__main__':main()
