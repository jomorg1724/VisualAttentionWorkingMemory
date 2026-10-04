"""Accumulator-in-the-conv-stack encoder: the plain baseline's conv blocks with a recurrent state after each of the
three coarser scales (25x25, 13x13, 7x7). Everything is trainable; there is no frozen encoder and no fixed retention.

At scale s with block output H_t^s (C_s channels):
    U_t^s = P_s H_t^s                      (1x1 conv to 32 channels)
    O_t^s, S_t^s = A_s(U_t^s, S_{t-1}^s)   (accumulator update, 32-channel output field, state carried across frames)
    next block input = concat(H_t^s, O_t^s)  (C_s + 32 channels)
The deepest [H^4, O^4] (160x7x7) is flattened to the same 256-d feature and GRU/heads as the plain baseline, so the
only difference from `PlainBaseline` is the spatial state inside the conv stack.

Accumulator arms (`--accumulator`):
    convgru  : SpatialConvGRU from PreAttentiveVision/TemporalIntegration/accumulators.py (learned write/reset gates).
    opponent : fast/slow leaky traces with LEARNED per-channel retention (initialised at the lineage's 0.25/0.75) and the
               lineage's fixed quadrature energy channels; the retention is the only change from OpponentAccumulator.
    kda      : SpatialKDA associative update from the same file.
    none     : the plain baseline (states are never created), for an exact control on the same code path.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from PreAttentiveVision.TemporalIntegration.accumulators import SpatialConvGRU, SpatialKDA, OpponentAccumulator

class OpponentTracesLearned(OpponentAccumulator):
    """Opponent accumulator whose fast and slow retentions are learned per channel (sigmoid-parametrised)."""
    def __init__(self):
        super().__init__()
        self.raw_fast=nn.Parameter(torch.full((32,),float(torch.logit(torch.tensor(.25)))))
        self.raw_slow=nn.Parameter(torch.full((32,),float(torch.logit(torch.tensor(.75)))))
    def forward(self,current,state=None):
        rf=self.raw_fast.sigmoid()[None,:,None,None];rs=self.raw_slow.sigmoid()[None,:,None,None]
        if state is None:fast=slow=current
        else:fast=rf*state[0]+(1-rf)*current;slow=rs*state[1]+(1-rs)*current
        energies,_=self.energy_channels(fast,slow)
        return self.output(torch.cat((.5*(fast+slow),fast-slow,energies),1)),(fast,slow)

def make_accumulator(kind):
    return {'convgru':SpatialConvGRU,'opponent':OpponentTracesLearned,'kda':SpatialKDA}[kind]()

class AccumulatorBaseline(nn.Module):
    def __init__(self,task_classes,hidden=256,stack=1,feature_norm='none',center=False,zero_head=False,accumulator='convgru'):
        super().__init__();self.stack=int(stack);self.center=bool(center);self.kind=accumulator
        def block(i,o,k,s):return nn.Sequential(nn.Conv2d(i,o,k,s,k//2),nn.GroupNorm(8,o),nn.ReLU(inplace=True))
        widths=(32,64,96,128);extra=0 if accumulator=='none' else 32
        self.blocks=nn.ModuleList([block(3*self.stack,32,5,2),block(32,64,3,2),block(64+extra,96,3,2),block(96+extra,128,3,2)])
        if accumulator!='none':
            self.proj=nn.ModuleList([nn.Conv2d(c,32,1) for c in widths[1:]])
            self.acc=nn.ModuleList([make_accumulator(accumulator) for _ in widths[1:]])
        self.feat=nn.Linear((128+extra)*7*7,hidden);self.norm=nn.LayerNorm(hidden) if feature_norm=='layernorm' else nn.Identity()
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.heads=nn.ModuleDict({t:nn.Linear(hidden,int(n)) for t,n in task_classes.items()})
        if zero_head:
            for h in self.heads.values():nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
    def frames(self,images):
        if self.center:images=images-0.5
        if self.stack==1:return images
        B,T=images.shape[:2];pad=images.new_zeros(B,self.stack-1,*images.shape[2:]);x=torch.cat([pad,images],1)
        return torch.cat([x[:,k:k+T] for k in range(self.stack)],2)
    def encode_frame(self,x,states):
        """One frame through the stack; states is a list of three accumulator states (or None)."""
        h=self.blocks[0](x);new=[]
        for s in range(3):
            h=self.blocks[s+1](h)
            if self.kind!='none':
                o,st=self.acc[s](self.proj[s](h),states[s] if states else None);new.append(st)
                if s<2:h=torch.cat((h,o),1)
                else:h=torch.cat((h,o),1)
        return h,new
    def forward(self,images,task):
        images=self.frames(images);B,T=images.shape[:2];states=None;feats=[]
        for t in range(T):
            h,states=self.encode_frame(images[:,t],states)
            feats.append(self.norm(F.relu(self.feat(h.flatten(1)))))
        _,hT=self.gru(torch.stack(feats,1))
        return self.heads[task](hT[-1])
