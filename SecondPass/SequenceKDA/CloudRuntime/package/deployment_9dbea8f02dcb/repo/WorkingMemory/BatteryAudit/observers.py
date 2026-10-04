"""Hand-coded ideal observers for the five-task spatial battery.

Each observer reads only the rendered frames (plus, where stated, the trial
metadata for scoring) and returns a decision or a scalar score. They measure
how much label information survives rasterisation at 100x100, independently
of any learned model. No observer is tuned on the trials it is scored on
except where a threshold is explicitly reported as oracle-selected.
"""
from __future__ import annotations
import numpy as np
from WorkingMemory.SpatialTaskBattery.stimuli import CENTERS
from PreAttentiveVision.neuroscience_stimuli import DIRECTION_VECTORS

# ----------------------------------------------------------------------------
# Cue decoding from pixels
# ----------------------------------------------------------------------------
_YY,_XX=np.mgrid[:100,:100]
_RING_MASKS={k:(np.abs(np.sqrt((_XX-x)**2+(_YY-y)**2)-14)<.8) for k,(x,y) in enumerate(CENTERS)}

def decode_ring(frame):
    """Which of the four locations carries the thin 0.95 ring? Returns (index, score margin)."""
    g=frame[0]
    scores=np.array([g[m].mean() for m in _RING_MASKS.values()])
    order=np.argsort(scores)[::-1]
    return int(order[0]),float(scores[order[0]]-scores[order[1]])

def decode_sign_glyph(frame):
    """Locate the 10x10 +/- glyph 23 px above one of the centres; return (target, sign, contrast)."""
    g=frame[0];best=None
    for k,(x,y) in enumerate(CENTERS):
        x0,y0=int(x)-5,int(y)-23;patch=g[y0:y0+10,x0:x0+10]
        # The glyph is a dark horizontal bar (rows 2:4, cols 2:8) and, for +, a dark vertical bar (rows 0:6, cols 4:6).
        dark=(patch<.3).sum()
        if best is None or dark>best[2]:best=(k,patch,dark)
    k,patch,dark=best
    vertical=(patch[0:2,4:6]<.3).mean()+(patch[4:6,4:6]<.3).mean()
    sign=1 if vertical>.5 else -1
    return k,sign,int(dark)

# ----------------------------------------------------------------------------
# Orientation estimation with a quadrature Gabor bank
# ----------------------------------------------------------------------------
class GaborBank:
    def __init__(self,n_orient=36,wavelength=6.,sigma=4.5,half=12):
        self.thetas=np.arange(n_orient)*np.pi/n_orient
        yy,xx=np.mgrid[-half:half+1,-half:half+1].astype(np.float64)
        env=np.exp(-(xx**2+yy**2)/(2*sigma**2))
        self.even=np.stack([env*np.cos(2*np.pi*(xx*np.cos(t)+yy*np.sin(t))/wavelength) for t in self.thetas])
        self.odd=np.stack([env*np.sin(2*np.pi*(xx*np.cos(t)+yy*np.sin(t))/wavelength) for t in self.thetas])
        self.even-=self.even.mean(axis=(1,2),keepdims=True)
        self.half=half
    def energy(self,gray,x,y):
        h=self.half;patch=gray[int(y)-h:int(y)+h+1,int(x)-h:int(x)+h+1]-.5
        e=(self.even*patch).sum(axis=(1,2));o=(self.odd*patch).sum(axis=(1,2))
        return e*e+o*o
    def angle(self,gray,x,y):
        """Orientation in [0,pi) by argmax with parabolic refinement over the circular bank."""
        en=self.energy(gray,x,y);i=int(np.argmax(en));n=len(en)
        a,b,c=en[(i-1)%n],en[i],en[(i+1)%n];den=a-2*b+c
        frac=0. if den==0 else .5*(a-c)/den
        return float(((i+frac)*np.pi/n)%np.pi),float(en.max()/(en.mean()+1e-12))

def wrap_half_pi(d):
    """Wrap an orientation difference into (-pi/2, pi/2]."""
    return (d+np.pi/2)%np.pi-np.pi/2

# ----------------------------------------------------------------------------
# Motion: block matching of consecutive frames inside an aperture window
# ----------------------------------------------------------------------------
def block_match_scores(prev,cur,cx,cy,half=14,shifts=(1,2)):
    """Score each cardinal direction by coincidence of previous dots shifted by s pixels with current dots.

    Returns array (4,) of max-over-shift correlation, plus the zero-shift score for reference.
    """
    p=prev[0]-.5;c=cur[0]-.5;cy,cx=int(cy),int(cx)
    win=c[cy-half:cy+half+1,cx-half:cx+half+1]
    def corr(dx,dy):
        src=p[cy-half-dy:cy+half+1-dy,cx-half-dx:cx+half+1-dx]
        return float((src*win).sum())
    zero=corr(0,0);out=np.zeros(4)
    for k,(dx,dy) in enumerate(DIRECTION_VECTORS):
        out[k]=max(corr(int(round(s*dx)),int(round(s*dy))) for s in shifts)
    return out,zero

def motion_observer(frames,cx,cy,moving_frames,reference_frame):
    """Per-transition direction scores; returns (hard vote label, soft sum label, per-frame argmax list, score matrix)."""
    seq=[reference_frame]+list(moving_frames);scores=[]
    for a,b in zip(seq[:-1],seq[1:]):
        s,_=block_match_scores(frames[a],frames[b],cx,cy);scores.append(s)
    scores=np.stack(scores)
    per_frame=scores.argmax(axis=1)
    counts=np.bincount(per_frame,minlength=4)
    hard=int(counts.argmax());soft=int(scores.sum(axis=0).argmax())
    return hard,soft,per_frame.tolist(),scores

# ----------------------------------------------------------------------------
# Krauzlis: Lucas-Kanade global translation per transition, direction change pre/post event
# ----------------------------------------------------------------------------
def lucas_kanade(prev,cur,cx,cy,radius=10):
    p=prev[0].astype(np.float64);c=cur[0].astype(np.float64);m=.5*(p+c)
    cy,cx=int(round(cy)),int(round(cx));r=radius
    ys,xs=slice(cy-r,cy+r+1),slice(cx-r,cx+r+1)
    Ix=.5*(m[ys,cx-r+1:cx+r+2]-m[ys,cx-r-1:cx+r]);Iy=.5*(m[cy-r+1:cy+r+2,xs]-m[cy-r-1:cy+r,xs]);It=c[ys,xs]-p[ys,xs]
    A=np.array([[(Ix*Ix).sum(),(Ix*Iy).sum()],[(Ix*Iy).sum(),(Iy*Iy).sum()]]);b=-np.array([(Ix*It).sum(),(Iy*It).sum()])
    if np.linalg.cond(A)>1e8:return np.zeros(2)
    return np.linalg.solve(A,b)

def krauzlis_observer(frames,centers,baseline,reference_frame=7):
    """Return per-patch (pre_angle, post_angle, delta_deg, pre_speed, post_speed)."""
    out=[]
    for (cx,cy) in centers:
        flows=[]
        for t in range(baseline+8):
            flows.append(lucas_kanade(frames[reference_frame+t],frames[reference_frame+t+1],cx,cy))
        flows=np.array(flows);pre=flows[:baseline].mean(axis=0);post=flows[baseline:].mean(axis=0)
        a0=np.arctan2(pre[1],pre[0]);a1=np.arctan2(post[1],post[0]);d=np.rad2deg((a1-a0+np.pi)%(2*np.pi)-np.pi)
        out.append(dict(pre_angle=float(np.rad2deg(a0)),post_angle=float(np.rad2deg(a1)),delta_deg=float(d),pre_speed=float(np.linalg.norm(pre)),post_speed=float(np.linalg.norm(post)),flows=flows))
    return out

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def balanced_accuracy(y,yhat,n_classes):
    y=np.asarray(y);yhat=np.asarray(yhat);accs=[]
    for k in range(n_classes):
        m=y==k
        if m.any():accs.append((yhat[m]==k).mean())
    return float(np.mean(accs))

def auc_binary(y,score):
    """Rank AUC of score for label 1 versus 0 (ties count half)."""
    y=np.asarray(y);s=np.asarray(score,dtype=np.float64);pos=s[y==1];neg=s[y==0]
    if len(pos)==0 or len(neg)==0:return float('nan')
    gt=(pos[:,None]>neg[None,:]).sum();eq=(pos[:,None]==neg[None,:]).sum()
    return float((gt+.5*eq)/(len(pos)*len(neg)))

def best_threshold_ba(y,score):
    y=np.asarray(y);s=np.asarray(score);best=(0.,None)
    for th in np.unique(s):
        ba=balanced_accuracy(y,(s>=th).astype(int),2)
        if ba>best[0]:best=(ba,float(th))
    return best
