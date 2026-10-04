"""Input-only multichannel small-displacement SSD matching and fresh analysis fit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def velocity(frames):
    """T,C,H,W -> dx,dy and normalized matching residual. No labels/metadata."""
    z=np.asarray(frames,dtype=np.float64)
    mid=(z[1:]+z[:-1])/2
    gx=(mid[...,1:-1,2:]-mid[...,1:-1,:-2])/2
    gy=(mid[...,2:,1:-1]-mid[...,:-2,1:-1])/2
    dt=(z[1:]-z[:-1])[...,1:-1,1:-1]
    a=np.stack([gx.ravel(),gy.ravel()],1);b=-dt.ravel()
    energy=(a*a).sum();reg=max(energy*1e-6,1e-12)
    w=np.ones(len(b));v=np.zeros(2)
    for _ in range(3):
        v=np.linalg.solve(a.T@(w[:,None]*a)+reg*np.eye(2),a.T@(w*b))
        residual=b-a@v
        # Soft robust downweighting of replacements; no pixel-mass/speed metadata.
        scale=max(float(np.sqrt(np.mean(residual**2))),1e-10)
        w=1/np.maximum(1,abs(residual)/(2*scale))
    err=float(np.mean(residual**2)/(np.mean(b*b)+1e-12))
    return v,np.array([err,float(energy/max(a.shape[0],1))])

def patch_descriptor(phases, stride=1):
    assert len(phases)==2
    a,qa=velocity(phases[0]);b,qb=velocity(phases[1])
    a=a*stride;b=b*stride
    sa=np.linalg.norm(a);sb=np.linalg.norm(b)
    ua=a/(sa+1e-10);ub=b/(sb+1e-10)
    cos=np.clip(ua@ub,-1,1)
    return np.array([1-cos,abs(ua[0]*ub[1]-ua[1]*ub[0]),np.linalg.norm(ua-ub),abs(sa-sb)/(sa+sb+1e-10),sa+sb,qa[0],qb[0],abs(qa[0]-qb[0])])

def describe(patches):
    """N,phase,patch,time,channel,H,W -> N,patch,8."""
    return np.stack([[patch_descriptor(row[:,k]) for k in range(2)] for row in patches])

def fit(x,y):
    """Fresh convex patch-shared logistic, balanced classes, fixed L2=1."""
    x=x.reshape(-1,8).astype(float);y=y.reshape(-1).astype(float)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-10]=1
    z=(x-mean)/scale;z=np.c_[z,np.ones(len(z))]
    weights=np.where(y==1,len(y)/(2*max(y.sum(),1)),len(y)/(2*max((1-y).sum(),1)))
    def objective(w):
        s=z@w;p=expit(s);reg=w.copy();reg[-1]=0
        return float(np.sum(weights*(np.logaddexp(0,s)-y*s))+.5*(reg@reg)),z.T@(weights*(p-y))+reg
    result=minimize(objective,np.zeros(9),jac=True,method='L-BFGS-B',options=dict(maxiter=200,ftol=1e-12,gtol=1e-7))
    assert result.success,result.message
    return dict(mean=mean,scale=scale,weight=result.x,iterations=np.array(result.nit))

def predict(f,x):
    return expit(((x-f['mean'])/f['scale'])@f['weight'][:8]+f['weight'][8])
