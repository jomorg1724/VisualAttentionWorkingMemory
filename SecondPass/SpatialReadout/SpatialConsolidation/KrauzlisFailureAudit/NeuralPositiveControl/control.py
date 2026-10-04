"""Fresh conventional full-history control; no inherited worker or model code."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from SecondPass.SpatialReadout.SpatialConsolidation.Krauzlis90Fresh.stimuli import SpatialBatteryStream
TASK='krauzlis_cued_motion'

class NativeStream:
    def __init__(self,seed,split): self.native=SpatialBatteryStream(seed,split)
    def batch(self,n,baseline):
        return self.native.batch(n,TASK,{'baseline_transitions':baseline})

class Attention(nn.Module):
    def __init__(self,d=48):
        super().__init__(); self.norm=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.out=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x):
        n,l,d=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(n,l,3,4,d//4).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(n,l,d)
        x=x+self.out(a)
        return x+self.ff(x)

class Model(nn.Module):
    """All frames, full native field. Length-homogeneous batches need no padding.

    Time-neighbor RGB concatenation plus stride-one 3x3 convolution is exactly
    a learned 3x3x3 space-time convolution (8 outputs), implemented as Conv2d
    for a reliable MPS path. No optical-flow computation or event/cue indices.
    Learned 5x5 and 2x2 patch aggregation gives 100 spatial tokens per frame.
    Two axial temporal/spatial attention blocks preserve every time token.
    A learned per-site temporal attention pool and dense spatial readout classify.
    """
    def __init__(self):
        super().__init__()
        self.motion=nn.Conv2d(9,8,3,padding=1)
        self.patch=nn.Sequential(nn.GELU(),nn.Conv2d(8,24,5,stride=5),nn.GELU(),nn.Conv2d(24,48,2,stride=2))
        self.space_pos=nn.Parameter(torch.randn(1,1,100,48)*.02)
        self.time_pos=nn.Parameter(torch.randn(1,45,1,48)*.02)
        self.temporal=nn.ModuleList([Attention(),Attention()])
        self.spatial=nn.ModuleList([Attention(),Attention()])
        self.pool=nn.Linear(48,1)
        self.readout=nn.Sequential(nn.LayerNorm(4800),nn.Linear(4800,128),nn.GELU(),nn.Linear(128,2))
    def forward(self,images):
        b,t,c,h,w=images.shape
        if (c,h,w)!=(3,100,100) or t>45: raise ValueError(images.shape)
        x=images-.5
        zero=torch.zeros_like(x[:,:1])
        triples=torch.cat([torch.cat([zero,x[:,:-1]],1),x,torch.cat([x[:,1:],zero],1)],2)
        z=self.patch(self.motion(triples.reshape(b*t,9,h,w))).flatten(2).transpose(1,2).reshape(b,t,100,48)
        z=z+self.space_pos+self.time_pos[:,:t]
        for temporal,spatial in zip(self.temporal,self.spatial):
            z=temporal(z.permute(0,2,1,3).reshape(b*100,t,48)).reshape(b,100,t,48).permute(0,2,1,3)
            z=spatial(z.reshape(b*t,100,48)).reshape(b,t,100,48)
        weights=self.pool(z).softmax(dim=1)
        return self.readout((weights*z).sum(dim=1).flatten(1))
