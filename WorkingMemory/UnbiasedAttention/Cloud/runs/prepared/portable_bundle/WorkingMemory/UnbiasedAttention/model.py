"""Remove only explicit attention source/locality logit biases from trained8400."""
import copy,math
import torch
from torch import nn
from WorkingMemory.PreUpdateAttention.model import AttentionMemory,JointAttention,groups as parent_groups
VERSION='attention_without_explicit_logit_bias_v1'
REMOVED=('attention.source_bias','attention.raw_locality','attention.distance_squared')

class UnbiasedJointAttention(JointAttention):
    def __init__(self):
        super().__init__()
        del self.source_bias;del self.raw_locality;del self.distance_squared
    def forward(self,field,old,diagnostic=False):
        b=field.shape[0];visual=field.flatten(2).transpose(1,2);memory=old.flatten(2).transpose(1,2)
        q=self.query(self.query_norm(memory)+self.position+self.source[1])
        raw=torch.cat((visual,memory),1)
        identities=torch.cat((self.position+self.source[0],self.position+self.source[1]),0)
        k=self.key(self.key_norm(raw)+identities);v=self.value(raw)
        q=q.reshape(b,169,2,32).transpose(1,2);k=k.reshape(b,338,2,32).transpose(1,2);v=v.reshape(b,338,2,32).transpose(1,2)
        weights=(q@k.transpose(-1,-2)/math.sqrt(32)).softmax(-1)
        u=self.output((weights@v).transpose(1,2).reshape(b,169,64))
        stats={} if not diagnostic else dict(memory_attention_mass=float(weights[:,:,:,169:].detach().sum(-1).mean()),memory_attention_mass_by_head=weights[:,:,:,169:].detach().sum(-1).mean((0,2)).tolist(),attention_entropy_by_head=(-(weights.clamp_min(1e-30).log()*weights).sum(-1)).detach().mean((0,2)).tolist())
        return u.transpose(1,2).reshape(b,64,13,13),stats

class UnbiasedMemory(AttentionMemory):
    def __init__(self,seed=41973001,activation_checkpoint=True,new_tasks=None):
        super().__init__(seed=seed,activation_checkpoint=activation_checkpoint)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.attention=UnbiasedJointAttention()
            if new_tasks:
                torch.random.default_generator.manual_seed(seed+100)
                for task,n in new_tasks.items():
                    if task in self.readout.heads:raise ValueError('New semantic head name collides '+task)
                    self.readout.heads[task]=nn.Linear(128,n)

def groups(model,cfg):
    result={};new_tasks=tuple('readout.heads.'+n+'.' for n in cfg.get('task_classes',{})) if cfg.get('battery')=='spatial' else ()
    for name,p in model.named_parameters():
        high=name.startswith(('memory.','memory_input.','memory_output.','comparator.','comparison_output.','readout.heads.orientation_binding.','attention.')+new_tasks)
        wd=cfg['weight_decay'] if p.ndim>=2 and name.endswith('weight') and 'norm' not in name else 0.
        result.setdefault((cfg['new_lr'] if high else cfg['parent_lr'],wd),[]).append(p)
    return [dict(params=ps,lr=key[0],weight_decay=key[1]) for key,ps in result.items()]

def migrate(parent,arm,cfg):
    assert parent['version']=='spatial_preupdate_joint_attention_v1' and parent['step']==8400
    old=AttentionMemory(activation_checkpoint=False);old.load_state_dict(parent['model'],strict=True)
    oldopt=torch.optim.Adam(parent_groups(old,parent['config']),eps=parent['config']['adam_eps']);oldopt.load_state_dict(parent['optimizer'])
    tasks=cfg['task_classes'] if cfg.get('battery')=='spatial' else None
    model=UnbiasedMemory(seed=cfg['model_seed'],new_tasks=tasks);state=model.state_dict();copied=[]
    for name,value in parent['model'].items():
        if name in REMOVED:continue
        assert name in state and state[name].shape==value.shape,name
        state[name]=value;copied.append(name)
    missing=set(state)-set(copied);assert all(name.startswith(tuple('readout.heads.'+t+'.' for t in (tasks or {}))) for name in missing)
    model.load_state_dict(state,strict=True);opt=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);oldnames=dict(old.named_parameters());inherited=[]
    for name,p in model.named_parameters():
        if name in copied and name in oldnames and oldnames[name] in oldopt.state:opt.state[p]=copy.deepcopy(oldopt.state[oldnames[name]]);inherited.append(name)
    return model,opt,dict(version=VERSION,parent_version=parent['version'],parent_step=8400,retained_tensors=copied,retained_adam_names=inherited,removed_tensors=list(REMOVED),new_tensors=sorted(missing),qkvo_and_embeddings_reinitialized=False,new_task_heads_fresh=bool(tasks),bias_removal='Only explicit source_bias and -softplus(raw_locality)*distance_squared terms removed; learned source and position embeddings remain')
