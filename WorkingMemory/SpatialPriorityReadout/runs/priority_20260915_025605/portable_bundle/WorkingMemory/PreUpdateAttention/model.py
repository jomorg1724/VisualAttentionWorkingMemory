"""Joint visual/memory attention supplies the sole external E/I memory drive."""
import copy, math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from WorkingMemory.SpatialComparison.model import Competitor, groups as parent_groups
VERSION = 'spatial_preupdate_joint_attention_v1'
ARMS = ('preupdate_attention',)

class JointAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(64)
        self.key_norm = nn.LayerNorm(64)
        self.position = nn.Parameter(torch.empty(169,64))
        self.source = nn.Parameter(torch.empty(2,64))
        self.query = nn.Linear(64,64,bias=False)
        self.key = nn.Linear(64,64,bias=False)
        self.value = nn.Linear(64,64,bias=False)
        self.output = nn.Linear(64,64,bias=False)
        self.raw_locality = nn.Parameter(torch.full((2,),math.log(math.expm1(4.))))
        self.source_bias = nn.Parameter(torch.tensor([[2.,0.],[2.,0.]]))
        xy=torch.stack(torch.meshgrid(torch.arange(13),torch.arange(13),indexing='ij'),-1).reshape(169,2).float()
        d=(xy[:,None]-xy[None,:]).square().sum(-1)
        self.register_buffer('distance_squared',torch.cat((d,d),1))
        with torch.no_grad():
            self.position.normal_(0,.02);self.source.normal_(0,.02)
            nn.init.xavier_uniform_(self.query.weight,gain=.1)
            nn.init.xavier_uniform_(self.key.weight,gain=.1)
            nn.init.eye_(self.value.weight);nn.init.eye_(self.output.weight)
    def forward(self,field,old,diagnostic=False):
        b=field.shape[0]
        visual=field.flatten(2).transpose(1,2)
        memory=old.flatten(2).transpose(1,2)
        q=self.query(self.query_norm(memory)+self.position+self.source[1])
        raw=torch.cat((visual,memory),1)
        identities=torch.cat((self.position+self.source[0],self.position+self.source[1]),0)
        k=self.key(self.key_norm(raw)+identities)
        v=self.value(raw)
        q=q.reshape(b,169,2,32).transpose(1,2)
        k=k.reshape(b,338,2,32).transpose(1,2)
        v=v.reshape(b,338,2,32).transpose(1,2)
        logits=q@k.transpose(-1,-2)/math.sqrt(32)
        # Smooth learned locality/source preferences initialize useful local input.
        # No tokens are masked and no blank/phase information is supplied.
        bias=self.source_bias.repeat_interleave(169,dim=1)[:,None,:]-F.softplus(self.raw_locality)[:,None,None]*self.distance_squared
        weights=(logits+bias[None]).softmax(-1)
        u=self.output((weights@v).transpose(1,2).reshape(b,169,64))
        output=u.transpose(1,2).reshape(b,64,13,13)
        stats={} if not diagnostic else dict(memory_attention_mass=float(weights[:,:,:,169:].detach().sum(-1).mean()),locality=F.softplus(self.raw_locality).detach().tolist())
        return output,stats

class AttentionMemory(Competitor):
    def __init__(self,arm='preupdate_attention',seed=41973001,activation_checkpoint=True):
        if arm not in ARMS:raise ValueError(arm)
        super().__init__('spatial_ei',activation_checkpoint=activation_checkpoint)
        self.attention_arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.attention=JointAttention()
    def forward(self,images,task,diagnostic=False):
        traces=();state=None;records=[];states=[];first_field=None
        for t in range(images.shape[1]):
            if self.training and self.activation_checkpoint and torch.is_grad_enabled():res=checkpoint(self._sensory,images[:,t],*traces,use_reentrant=False,preserve_rng_state=True)
            else:res=self._sensory(images[:,t],*traces)
            field,traces=res[0],res[1:]
            old=torch.zeros_like(field) if state is None else state[0]
            attended,attn_stats=self.attention(field,old,diagnostic)
            # The existing memory input normalization and input projection now
            # receive attended content only. There is no raw sensory drive bypass.
            z=self.memory_input(attended)
            local=self.comparator(torch.cat((old,field),1))
            comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1)
            r,state,stats=self.memory(z,state,diagnostic)
            if diagnostic:
                if t==0:field.retain_grad();first_field=field
                for value in state:
                    if value.requires_grad:value.retain_grad()
                states.append(state);records.append(dict({k:float(v) for k,v in stats.items()},**attn_stats))
        sensory=self.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
        retained=torch.cat((r.mean((2,3)),r.amax((2,3))),1)
        logits=self.classify(sensory+self.memory_output(retained)+self.comparison_output(comp),task)
        return (logits,dict(states=states,first_field=first_field,records=records)) if diagnostic else logits

def groups(model,cfg):
    result={}
    for name,p in model.named_parameters():
        new=name.startswith(('memory.','memory_input.','memory_output.','comparator.','comparison_output.','readout.heads.orientation_binding.','attention.'))
        wd=cfg['weight_decay'] if p.ndim>=2 and name.endswith('weight') and 'norm' not in name else 0.
        result.setdefault((cfg['new_lr'] if new else cfg['parent_lr'],wd),[]).append(p)
    return [dict(params=v,lr=k[0],weight_decay=k[1]) for k,v in result.items()]

def migrate(parent,arm,cfg):
    assert parent['version']=='spatial_dense_ei_comparison_v1' and parent['arm']=='spatial_ei' and parent['step']==4400
    old=Competitor('spatial_ei',activation_checkpoint=False);old.load_state_dict(parent['model'],strict=True)
    oldopt=torch.optim.Adam(parent_groups(old,parent['config']),eps=parent['config']['adam_eps']);oldopt.load_state_dict(parent['optimizer'])
    model=AttentionMemory(arm,cfg['model_seed']);state=model.state_dict()
    for name,value in parent['model'].items():
        assert name in state and state[name].shape==value.shape,name
        state[name]=value
    missing=set(state)-set(parent['model']);assert all(n.startswith('attention.') for n in missing)
    model.load_state_dict(state,strict=True);opt=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);oldnames=dict(old.named_parameters());copied=[]
    for name,p in model.named_parameters():
        if name in oldnames and oldnames[name] in oldopt.state:opt.state[p]=copy.deepcopy(oldopt.state[oldnames[name]]);copied.append(name)
    return model,opt,dict(version=VERSION,parent_step=4400,parent_version=parent['version'],retained_tensors=list(parent['model']),adam_states_by_name=copied,new_parameters=sorted(missing),sampler='exact parent stream and RNG continuation; global schedule offset4400',extra_persistent_state_entries=0,initialization='local sensory-favoring joint attention, not functionally identical to parent')
