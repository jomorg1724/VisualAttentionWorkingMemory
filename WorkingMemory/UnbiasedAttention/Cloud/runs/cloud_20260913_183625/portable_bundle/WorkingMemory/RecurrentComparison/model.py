"""Two explicit recurrent memories downstream of per-frame opponent fields."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from WorkingMemory.model import SequenceOpponent

VERSION = 'wm_focused_recurrent_v1'
ARMS = ('lstm', 'ei_adaptive')
NEW_PREFIXES = ('memory_input.', 'memory.', 'memory_output.')


def logit(x):
    return torch.log(x)-torch.log1p(-x)


class NormalizedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(128,1024,bias=False)
        self.recurrent = nn.Linear(256,1024,bias=False)
        self.norms = nn.ModuleList([nn.LayerNorm(256,elementwise_affine=True) for _ in range(4)])
        self.bias = nn.Parameter(torch.zeros(4,256))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input.weight,gain=.5)
            for block in self.recurrent.weight.chunk(4): nn.init.orthogonal_(block,gain=.5)
            self.bias[0].fill_(-2.)
            tau=torch.logspace(math.log10(4),math.log10(128),256)
            self.bias[1].copy_(logit(torch.exp(-1/tau)))

    def forward(self,z,state=None,diagnostic=False):
        if state is None: state=(z.new_zeros(len(z),256),z.new_zeros(len(z),256))
        c,h=state
        values=(self.input(z)+self.recurrent(h)).chunk(4,1)
        i,f,o,g=[norm(v)+bias for norm,v,bias in zip(self.norms,values,self.bias)]
        i,f,o=i.sigmoid(),f.sigmoid(),o.sigmoid();g=g.tanh()
        c=f*c+i*g;h=o*c.tanh()
        stats={}
        if diagnostic:
            for name,v in [('input',i),('forget',f),('output',o)]:
                stats[name+'_saturation']=((v<.05)|(v>.95)).float().mean().detach()
                stats[name+'_mean']=v.mean().detach()
            stats['candidate_saturation']=(g.abs()>.95).float().mean().detach()
        return h,(c,h),stats


class AdaptiveEI(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(128,256,bias=False)
        self.raw_recurrent = nn.Parameter(torch.empty(256,256))
        self.bias = nn.Parameter(torch.full((256,),.05))
        self.raw_tau_r = nn.Parameter(logit((torch.logspace(math.log10(2),math.log10(8),256)-1)/31))
        self.raw_tau_a = nn.Parameter(logit((torch.logspace(math.log10(16),math.log10(64),256)-4)/124))
        self.raw_adaptation = nn.Parameter(torch.full((256,),math.log(.1/.9)))
        self.register_buffer('signs',torch.cat((torch.ones(205),-torch.ones(51))))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input.weight,gain=.4)
            # Each row starts with equal total E and I magnitudes (.6 each),
            # despite the unequal presynaptic population sizes.
            magnitude=torch.empty(256,256).uniform_(.5,1.5)
            magnitude[:,:205]*=.6/magnitude[:,:205].sum(1,keepdim=True)
            magnitude[:,205:]*=.6/magnitude[:,205:].sum(1,keepdim=True)
            self.raw_recurrent.copy_(torch.log(torch.expm1(magnitude)))

    def recurrent_weight(self):
        # F.linear uses weight[out,in]: Dale signs apply to columns.
        return F.softplus(self.raw_recurrent)*self.signs[None,:]

    def forward(self,z,state=None,diagnostic=False):
        if state is None: state=(z.new_zeros(len(z),256),z.new_zeros(len(z),256))
        old_r,old_a=state
        tau_r=1+31*self.raw_tau_r.sigmoid();tau_a=4+124*self.raw_tau_a.sigmoid()
        alpha=-torch.expm1(-1/tau_r);beta=-torch.expm1(-1/tau_a)
        gain=.5*self.raw_adaptation.sigmoid()
        drive=self.input(z)+F.linear(old_r,self.recurrent_weight())-gain*old_a+self.bias
        r=(1-alpha)*old_r+alpha*drive.relu()
        a=(1-beta)*old_a+beta*old_r
        stats={}
        if diagnostic:
            stats={k:v.detach() for k,v in dict(inactive_fraction=(r<1e-6).float().mean(),
                rectifier_off_fraction=(drive<=0).float().mean(),excitatory_mean=r[:,:205].mean(),
                inhibitory_mean=r[:,205:].mean(),tau_r_mean=tau_r.mean(),tau_a_mean=tau_a.mean(),
                adaptation_gain_mean=gain.mean()).items()}
        return r,(r,a),stats


class RecurrentOpponent(SequenceOpponent):
    def __init__(self,task_classes,arm,common_seed=59311,core_seed=59312,activation_checkpoint=True):
        if arm not in ARMS:raise ValueError(arm)
        super().__init__(task_classes,activation_checkpoint)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(common_seed)
            self.memory_input=nn.Sequential(nn.Conv2d(64,8,1),nn.Flatten(),nn.Linear(8*13*13,128),nn.LayerNorm(128),nn.SiLU())
            self.memory_output=nn.Linear(256,128)
            nn.init.normal_(self.memory_output.weight,std=.01/math.sqrt(256));nn.init.zeros_(self.memory_output.bias)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(core_seed)
            self.memory=NormalizedLSTM() if arm=='lstm' else AdaptiveEI()
        self.arm=arm
        self.config.update(implementation=VERSION,arm=arm,memory_scalars=512,attention=False)

    def _sensory(self,frame,*old):
        current=self._encode(frame)
        state=tuple((u,u) for u in current) if not old else tuple((.25*old[2*j]+.75*u,.75*old[2*j+1]+.25*u) for j,u in enumerate(current))
        emitted=self._emit(state)
        scales=[F.adaptive_avg_pool2d(layer(torch.cat((u,o),1)),(13,13)) for u,o,layer in zip(current,emitted,self.readout.local)]
        field=self.readout.fusion(torch.cat(scales,1))
        return (field,)+tuple(x for pair in state for x in pair)

    def step(self,frame,state=None,reset_memory=False,diagnostic=False):
        traces=() if state is None else state[0]
        if self.training and self.activation_checkpoint and torch.is_grad_enabled():
            result=checkpoint(self._sensory,frame,*traces,use_reentrant=False,preserve_rng_state=True)
        else: result=self._sensory(frame,*traces)
        field,traces=result[0],result[1:]
        z=self.memory_input(field)
        memory=None if state is None or reset_memory else state[1]
        h,memory,stats=self.memory(z,memory,diagnostic)
        return (field,h,z),(traces,memory),stats

    def forward(self,images,task,reset_memory_each_frame=False,diagnostic=False):
        state=None;stats=[];representations=[]
        for t in range(images.shape[1]):
            (field,h,z),state,record=self.step(images[:,t],state,reset_memory_each_frame,diagnostic)
            if diagnostic:
                if z.requires_grad:z.retain_grad()
                representations.append(z)
                for j,v in enumerate(state[1]):
                    record['state'+str(j)+'_rms']=v.square().mean().sqrt().detach()
                    record['state'+str(j)+'_max']=v.abs().max().detach()
                stats.append(record)
        sensory=self.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
        logits=self.classify(sensory+self.memory_output(h),task)
        return (logits,stats,representations) if diagnostic else logits


def load_parent(parent,arm,common_seed=59311,core_seed=59312,activation_checkpoint=True):
    if parent['version']!='wm_opponent_all_learned_v1' or parent['step']!=6860:
        raise ValueError('Expected selected WM step6860 parent')
    model=RecurrentOpponent(parent['config']['task_classes'],arm,common_seed,core_seed,activation_checkpoint)
    result=model.load_state_dict(parent['model'],strict=False)
    if result.unexpected_keys or any(not name.startswith(NEW_PREFIXES) for name in result.missing_keys):
        raise ValueError('Expanded warm-start tensor incompatibility: '+str(result))
    if set(parent['model'])!={k for k in model.state_dict() if not k.startswith(NEW_PREFIXES)}:
        raise ValueError('Parent/source tensor inventory mismatch')
    return model


def optimizer_groups(model,new_lr=.0003,parent_lr=.00003,decay=.0001):
    groups={}
    for name,p in model.named_parameters():
        new=name.startswith(NEW_PREFIXES)
        # Only unconstrained matrix/conv weights receive generic decay.
        wd=decay if p.ndim>=2 and name.endswith('weight') and 'norm' not in name else 0.
        key=(new_lr if new else parent_lr,wd)
        groups.setdefault(key,[]).append(p)
    return [dict(params=params,lr=lr,weight_decay=wd) for (lr,wd),params in groups.items()]
