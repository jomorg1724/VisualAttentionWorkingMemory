"""Inference-only interventions; ordinary trained model source remains unchanged."""
import math
import torch
from torch.nn import functional as F

def stages(family,delay):
    if family=='orientation_single':
        return dict(instruction=[0],sample=[1,2],blanks=list(range(3,3+delay)),query=[3+delay],probe=[4+delay])
    if family=='motion_direction':
        return dict(instruction=[0],reference=[1],moving=list(range(2,10)),blanks=list(range(10,10+delay)),report=[10+delay])
    raise ValueError(family)

def attention(module,field,old,exclude_memory=False):
    b=field.shape[0];visual=field.flatten(2).transpose(1,2);memory=old.flatten(2).transpose(1,2)
    q=module.query(module.query_norm(memory)+module.position+module.source[1]);raw=torch.cat((visual,memory),1)
    identities=torch.cat((module.position+module.source[0],module.position+module.source[1]),0)
    k=module.key(module.key_norm(raw)+identities);v=module.value(raw)
    q=q.reshape(b,169,2,32).transpose(1,2);k=k.reshape(b,338,2,32).transpose(1,2);v=v.reshape(b,338,2,32).transpose(1,2)
    logits=q@k.transpose(-1,-2)/math.sqrt(32)
    bias=module.source_bias.repeat_interleave(169,dim=1)[:,None,:]-F.softplus(module.raw_locality)[:,None,None]*module.distance_squared
    scores=logits+bias[None]
    if exclude_memory:scores[:,:,:,169:]=-torch.inf
    weights=scores.softmax(-1)
    u=module.output((weights@v).transpose(1,2).reshape(b,169,64))
    source_parts=[weights[:,:,:,lo:hi]@v[:,:,lo:hi,:] for lo,hi in ((0,169),(169,338))]
    stats=torch.stack((weights[:,:,:,169:].sum(-1).mean(-1),source_parts[0].square().mean((-2,-1)).sqrt(),source_parts[1].square().mean((-2,-1)).sqrt()),-1)
    return u.transpose(1,2).reshape(b,64,13,13),stats

@torch.no_grad()
def sensory_fields(model,images):
    traces=();fields=[]
    for t in range(images.shape[1]):
        res=model._sensory(images[:,t],*traces);fields.append(res[0]);traces=res[1:]
    return fields

@torch.no_grad()
def forward(model,fields,task,exclude_frames=(),bypass_frames=()):
    state=None;attn_stats=[]
    for t,field in enumerate(fields):
        old=torch.zeros_like(field) if state is None else state[0]
        if hasattr(model,'attention'):
            attended,stats=attention(model.attention,field,old,t in exclude_frames);attn_stats.append(stats)
            drive=field if t in bypass_frames else attended
        else:
            if exclude_frames or bypass_frames:raise ValueError('Intervention applies to attention model only')
            drive=field
        z=model.memory_input(drive);local=model.comparator(torch.cat((old,field),1));comp=torch.cat((local.mean((2,3)),local.amax((2,3))),1)
        r,state,_=model.memory(z,state,False)
    sensory=model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
    memory=model.memory_output(torch.cat((r.mean((2,3)),r.amax((2,3))),1));comparison=model.comparison_output(comp)
    logits=model.classify(sensory+memory+comparison,task)
    head=model.readout.heads[task]
    parts=torch.stack([F.linear(x,head.weight,None) for x in (sensory,memory,comparison)],1)
    reconstructed=parts.sum(1)+head.bias
    error=float((reconstructed-logits).abs().max())
    if not torch.allclose(reconstructed,logits,atol=3e-5,rtol=3e-5):raise RuntimeError('Branch decomposition mismatch')
    return dict(logits=logits.cpu(),parts=parts.cpu(),head_bias=head.bias.detach().cpu(),branch_reconstruction_max_abs=error,attention_stats=torch.stack(attn_stats,1).cpu() if attn_stats else None)
