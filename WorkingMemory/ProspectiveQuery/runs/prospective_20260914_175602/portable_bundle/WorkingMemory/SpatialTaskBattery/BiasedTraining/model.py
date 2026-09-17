"""Original learned source/locality priors, fresh five semantic task heads."""
import copy,math
import torch
from WorkingMemory.PreUpdateAttention.model import AttentionMemory,JointAttention,groups as parent_groups
from WorkingMemory.UnbiasedAttention.model import UnbiasedMemory,groups
VERSION='spatial_five_task_original_attention_biases_v1'

class InstrumentedJointAttention(JointAttention):
    def forward(self,field,old,diagnostic=False):
        b=field.shape[0];visual=field.flatten(2).transpose(1,2);memory=old.flatten(2).transpose(1,2)
        q=self.query(self.query_norm(memory)+self.position+self.source[1]);raw=torch.cat((visual,memory),1)
        identities=torch.cat((self.position+self.source[0],self.position+self.source[1]),0)
        k=self.key(self.key_norm(raw)+identities);v=self.value(raw)
        q=q.reshape(b,169,2,32).transpose(1,2);k=k.reshape(b,338,2,32).transpose(1,2);v=v.reshape(b,338,2,32).transpose(1,2)
        logits=q@k.transpose(-1,-2)/math.sqrt(32)
        bias=self.source_bias.repeat_interleave(169,dim=1)[:,None,:]-torch.nn.functional.softplus(self.raw_locality)[:,None,None]*self.distance_squared
        weights=(logits+bias[None]).softmax(-1);u=self.output((weights@v).transpose(1,2).reshape(b,169,64))
        stats={} if not (diagnostic or getattr(self,'capture',False)) else dict(memory_attention_mass=float(weights[:,:,:,169:].detach().sum(-1).mean()),memory_attention_mass_by_head=weights[:,:,:,169:].detach().sum(-1).mean((0,2)).tolist(),attention_entropy_by_head=(-(weights.clamp_min(1e-30).log()*weights).sum(-1)).detach().mean((0,2)).tolist(),locality=torch.nn.functional.softplus(self.raw_locality).detach().tolist())
        if getattr(self,'capture',False):self.captured.append(stats)
        return u.transpose(1,2).reshape(b,64,13,13),stats

class BiasedMemory(UnbiasedMemory):
    def __init__(self,cfg):
        # Reuse exactly the fresh-head initialization of the cancelled arm.
        super().__init__(seed=cfg['model_seed'],new_tasks=cfg['task_classes'])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg['model_seed']);self.attention=InstrumentedJointAttention()

def migrate(parent,arm,cfg):
    assert parent['version']=='spatial_preupdate_joint_attention_v1' and parent['step']==8400 and cfg['battery']=='spatial'
    old=AttentionMemory(activation_checkpoint=False);old.load_state_dict(parent['model'],strict=True)
    oldopt=torch.optim.Adam(parent_groups(old,parent['config']),eps=parent['config']['adam_eps']);oldopt.load_state_dict(parent['optimizer'])
    model=BiasedMemory(cfg);state=model.state_dict()
    for n,v in parent['model'].items():assert n in state and state[n].shape==v.shape;state[n]=v
    missing=set(state)-set(parent['model']);assert all(n.startswith(tuple('readout.heads.'+t+'.' for t in cfg['task_classes'])) for n in missing)
    model.load_state_dict(state,strict=True);opt=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);oldnames=dict(old.named_parameters());inherited=[]
    for n,p in model.named_parameters():
        if n in oldnames and oldnames[n] in oldopt.state:opt.state[p]=copy.deepcopy(oldopt.state[oldnames[n]]);inherited.append(n)
    return model,opt,dict(version=VERSION,parent_step=8400,parent_version=parent['version'],retained_tensors=list(parent['model']),retained_adam_names=inherited,new_tensors=sorted(missing),source_and_locality_biases='Restored original trained values and compatible Adam states; both remain trainable',initialization='Original intact parent, not cancelled bias-free weights; identical five fresh semantic heads')
