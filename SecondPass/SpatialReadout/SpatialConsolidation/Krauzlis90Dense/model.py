"""Global dense image observer: no spatial tokens or shared local projections.

Native centered RGB100 stack3 -> Linear90000x128/ReLU -> three dense
KDA residual blocks -> dense gated recurrence128 -> Linear128x128/ReLU
-> task Linear128x2. KDA state is global, not a spatial attention map.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from PreAttentiveVision.TemporalIntegration.accumulators import kda_update

TASK='krauzlis_cued_motion'
WIDTH=128


class DenseKDA(nn.Module):
    heads, key_dim, value_dim = 2, 8, 16

    def __init__(self, width=WIDTH):
        super().__init__()
        self.inputs=nn.Linear(width,82)
        self.output=nn.Linear(32,width)
        with torch.no_grad():
            for head in range(2):
                start=head*41
                self.inputs.weight[start+32:start+41].zero_()
                self.inputs.bias[start+32:start+40].fill_(math.log(9.))
                self.inputs.bias[start+40].zero_()

    def forward(self, x, state=None):
        p=self.inputs(x).reshape(x.shape[0],2,41)
        q=F.normalize(p[...,:8],dim=-1,eps=1e-6)
        k=F.normalize(p[...,8:16],dim=-1,eps=1e-6)
        v=p[...,16:32]
        alpha=p[...,32:40].sigmoid()
        beta=p[...,40:41].sigmoid()
        if state is None: state=x.new_zeros(x.shape[0],2,8,16)
        if state.shape!=(x.shape[0],2,8,16): raise ValueError('Invalid global KDA state')
        read,state=kda_update(state,q,k,v,alpha,beta)
        return (x+self.output(read.flatten(1))).relu(),state


class DenseGRU(nn.Module):
    def __init__(self,width=WIDTH):
        super().__init__()
        self.gates=nn.Linear(2*width,2*width)
        self.candidate=nn.Linear(2*width,width)
        nn.init.zeros_(self.gates.bias)
        nn.init.zeros_(self.candidate.bias)

    def forward(self,x,previous=None):
        if previous is None: previous=torch.zeros_like(x)
        write,reset=self.gates(torch.cat((x,previous),1)).sigmoid().chunk(2,1)
        candidate=self.candidate(torch.cat((x,reset*previous),1)).tanh()
        return (1-write)*previous+write*candidate


class DenseObserver(nn.Module):
    def __init__(self,task_classes):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(90000,WIDTH),nn.ReLU())
        self.kda=nn.ModuleList([DenseKDA() for _ in range(3)])
        self.dense_gru=DenseGRU()
        self.readout=nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.ReLU())
        self.heads=nn.ModuleDict({TASK:nn.Linear(WIDTH,int(task_classes[TASK]))})

    def frames(self,images):
        if images.ndim!=5 or tuple(images.shape[2:])!=(3,100,100):
            raise ValueError('Expected native [B,T,3,100,100] RGB, no resizing')
        images=images-.5
        b,t=images.shape[:2]
        pad=images.new_zeros(b,2,*images.shape[2:])
        x=torch.cat((pad,images),1)
        return torch.cat([x[:,k:k+t] for k in range(3)],2)

    def recurrent_states(self,images):
        images=self.frames(images)
        states=[None]*3; hidden=None; history=[]
        for t in range(images.shape[1]):
            x=self.encoder(images[:,t].flatten(1))
            for i,block in enumerate(self.kda): x,states[i]=block(x,states[i])
            hidden=self.dense_gru(x,hidden)
            history.append(hidden)
        return history

    def forward(self,images,task):
        return self.heads[task](self.readout(self.recurrent_states(images)[-1]))
