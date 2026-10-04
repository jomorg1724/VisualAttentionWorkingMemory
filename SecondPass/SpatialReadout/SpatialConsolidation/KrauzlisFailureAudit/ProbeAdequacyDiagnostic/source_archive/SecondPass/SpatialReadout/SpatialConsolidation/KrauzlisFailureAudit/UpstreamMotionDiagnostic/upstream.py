"""Analysis-only, frozen selected2297. Imported legacy modules never run main()."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import sys, json, time, signal, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/Users/jonathanmorgan/Desktop/VisualAttentionWorkingMemory')
OUT=Path(__file__).parent
OLD=OUT.parent/'FrozenDiagnostic'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
import diagnostic as d
from SecondPass.SpatialReadout.CuedMotionAudit import pixel_observer as px
import torch

def access(data,site,kind,projection):
    """Only layer activations; neither truth, cue nor native logits accepted."""
    phases=('final','final') if kind=='final' else ('baseline','post')
    z=np.concatenate([np.roll(data[site+'__'+p],j*(projection.shape[0]//2),axis=1)@projection for j,p in enumerate(phases)],axis=1)
    i,j=np.triu_indices(z.shape[1])
    return np.concatenate([z,z[:,i]*z[:,j]],axis=1)

def circular_change(pre,post):
    return np.rad2deg(np.angle(np.exp(1j*(post-pre))))

def flow_angles(sums,counts,b):
    # Full baseline/post, or last TWO transitions of each phase (stack3).
    out=[]
    for windows in ((slice(0,b),slice(b,b+8)),(slice(b-2,b),slice(b+6,b+8))):
        phases=[]
        for sl in windows:
            v=sums[sl].sum(0)
            phases.append(np.arctan2(v[:,1],v[:,0]))
        out.append(np.stack(phases))
    return np.stack(out)

def pixel_summary(frames):
    # B determined from visible sequence length, not supplied trial metadata.
    b=len(frames)-17;gray=frames[:,0];sums=np.zeros((b+8,2,2));counts=np.zeros((b+8,2),int)
    for k,(x,y) in enumerate(px.K_CENTERS):
        patch=gray[7:b+16,y-11:y+12,x-11:x+12]-.5
        for t,(a,z) in enumerate(zip(patch[:-1],patch[1:])):
            flow=px.patch_flow(a,z);sums[t,k]=flow.sum(0);counts[t,k]=len(flow)
    cue,_=px.decode_cue(gray[0],px.K_CENTERS,9.925,.7)
    return dict(angles=flow_angles(sums,counts,b),cue=cue,counts=counts,flow_sums=sums)
