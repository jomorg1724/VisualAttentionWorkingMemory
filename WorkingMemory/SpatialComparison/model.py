"""Practical dense/spatial E/I competitors, comparing previous memory every frame."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from WorkingMemory.RecurrentComparison.model import RecurrentOpponent,optimizer_groups as old_groups,logit
from WorkingMemory.stimuli import TASK_CLASSES
ARMS=('dense_comparator','spatial_ei')
VERSION='spatial_dense_ei_comparison_v1'
class LocalNorm(nn.Module):
    def __init__(self):super().__init__();self.norm=nn.LayerNorm(64)
    def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
class SpatialEI(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Conv2d(64,64,1,bias=False);self.raw_recurrent=nn.Parameter(torch.empty(64,64,3,3));self.bias=nn.Parameter(torch.full((64,),.05))
        self.raw_tau_r=nn.Parameter(logit((torch.logspace(math.log10(2),math.log10(8),64)-1)/31));self.raw_tau_a=nn.Parameter(logit((torch.logspace(math.log10(16),math.log10(64),64)-4)/124));self.raw_adaptation=nn.Parameter(torch.full((64,),math.log(.1/.9)))
        self.register_buffer('signs',torch.cat((torch.ones(51),-torch.ones(13))))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.input.weight,gain=.4);v=torch.empty_like(self.raw_recurrent).uniform_(.5,1.5)
            for a,b in ((0,51),(51,64)):v[:,a:b]*=.6/v[:,a:b].sum((1,2,3),keepdim=True)
            self.raw_recurrent.copy_(torch.log(torch.expm1(v)))
    def recurrent_weight(self):return F.softplus(self.raw_recurrent)*self.signs[None,:,None,None]
    def forward(self,z,state=None,diagnostic=False):
        if state is None:state=(torch.zeros_like(z),torch.zeros_like(z))
        r0,a0=state;tau=1+31*self.raw_tau_r.sigmoid();ta=4+124*self.raw_tau_a.sigmoid();alpha=(-torch.expm1(-1/tau))[None,:,None,None];beta=(-torch.expm1(-1/ta))[None,:,None,None];gain=(.5*self.raw_adaptation.sigmoid())[None,:,None,None]
        drive=self.input(z)+F.conv2d(r0,self.recurrent_weight(),padding=1)-gain*a0+self.bias[None,:,None,None]
        r=(1-alpha)*r0+alpha*drive.relu();a=(1-beta)*a0+beta*r0
        stats={} if not diagnostic else dict(inactive_fraction=(r<1e-6).float().mean().detach(),tau_r_mean=tau.mean().detach(),tau_a_mean=ta.mean().detach(),adaptation_gain_mean=gain.mean().detach())
        return r,(r,a),stats
class Competitor(RecurrentOpponent):
    def __init__(self,arm,seed=31973001,activation_checkpoint=True):
        if arm not in ARMS:raise ValueError(arm)
        super().__init__(TASK_CLASSES,'ei_adaptive',activation_checkpoint=activation_checkpoint);self.variant=arm
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if arm=='dense_comparator':
                self.comparator=nn.Sequential(nn.Linear(384,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
            else:
                self.memory_input=LocalNorm();self.memory=SpatialEI();self.memory_output=nn.Linear(128,128)
                nn.init.normal_(self.memory_output.weight,std=.01/math.sqrt(128));nn.init.zeros_(self.memory_output.bias)
                self.comparator=nn.Sequential(nn.Conv2d(128,64,1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU())
            # Same nonzero residual comparison scale, no oracle gating.
            torch.random.default_generator.manual_seed(seed+1)
            self.comparison_output=nn.Linear(128,128);nn.init.normal_(self.comparison_output.weight,std=.01/math.sqrt(128));nn.init.zeros_(self.comparison_output.bias)
            self.readout.heads['orientation_binding']=nn.Linear(128,2)
    def forward(self,images,task,diagnostic=False):
        traces=();state=None;records=[];states=[];first_field=None
        for t in range(images.shape[1]):
            if self.training and self.activation_checkpoint and torch.is_grad_enabled():res=checkpoint(self._sensory,images[:,t],*traces,use_reentrant=False,preserve_rng_state=True)
            else:res=self._sensory(images[:,t],*traces)
            field,traces=res[0],res[1:];z=self.memory_input(field)
            if diagnostic and t==0:
                field.retain_grad();first_field=field
            if self.variant=='dense_comparator':
                old=field.new_zeros(len(field),256) if state is None else state[0]
                comp=self.comparator(torch.cat((old,z),1))
            else:
                old=torch.zeros_like(field) if state is None else state[0]
                local=self.comparator(torch.cat((old,field),1));comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1)
            r,state,stats=self.memory(z,state,diagnostic)
            if diagnostic:
                for v in state:
                    if v.requires_grad:v.retain_grad()
                states.append(state);records.append({k:float(v) for k,v in stats.items()})
        sensory=self.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
        retained=r if self.variant=='dense_comparator' else torch.cat((r.mean((2,3)),r.amax((2,3))),1)
        logits=self.classify(sensory+self.memory_output(retained)+self.comparison_output(comp),task)
        return (logits,dict(states=states,first_field=first_field,records=records)) if diagnostic else logits
def groups(model,cfg):
    result={}
    for name,p in model.named_parameters():
        new=name.startswith(('memory.','memory_input.','memory_output.','comparator.','comparison_output.','readout.heads.orientation_binding.'))
        wd=cfg['weight_decay'] if p.ndim>=2 and name.endswith('weight') and 'norm' not in name else 0.
        result.setdefault((cfg['new_lr'] if new else cfg['parent_lr'],wd),[]).append(p)
    return [dict(params=v,lr=k[0],weight_decay=k[1]) for k,v in result.items()]
def migrate(parent,arm,cfg):
    if parent['version']!='ei_retention_core_readout_v1' or parent['step']!=14800:raise ValueError('Expected trained Retention14800')
    old=RecurrentOpponent(TASK_CLASSES,'ei_adaptive',activation_checkpoint=False);old.load_state_dict(parent['model'],strict=True)
    oldopt=torch.optim.Adam(old_groups(old,cfg['new_lr'],cfg['parent_lr'],cfg['weight_decay']),eps=cfg['adam_eps']);oldopt.load_state_dict(parent['optimizer'])
    model=Competitor(arm,cfg['model_seed'],True);state=model.state_dict();copied=[]
    excluded=('memory.','memory_input.','memory_output.') if arm=='spatial_ei' else ()
    for name,value in parent['model'].items():
        if name.startswith(excluded):continue
        if name not in state or state[name].shape!=value.shape:raise RuntimeError('Unexpected shared tensor mismatch '+name)
        state[name]=value;copied.append(name)
    model.load_state_dict(state,strict=True);model.readout.heads['orientation_binding'].load_state_dict(old.readout.heads['orientation'].state_dict())
    optimizer=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);oldnames=dict(old.named_parameters());transferred=[]
    for name,p in model.named_parameters():
        if name in copied and name in oldnames and oldnames[name] in oldopt.state:
            optimizer.state[p]=copy.deepcopy(oldopt.state[oldnames[name]]);transferred.append(name)
    return model,optimizer,dict(version=VERSION,parent_step=14800,retained_tensors=copied,adam_states_by_name=transferred,new_binding_head_initialization='copy parent orientation head weights; fresh Adam',new_sampler=True)
