"""Exact attention8400 continuation; architecture and optimizer policy unchanged."""
import torch
from WorkingMemory.PreUpdateAttention.model import AttentionMemory,groups
VERSION='attention_training_exposure_v1'
ARMS=('control_10','focused_50')
def migrate(parent,arm,cfg):
    assert arm in ARMS and parent['version']=='spatial_preupdate_joint_attention_v1' and parent['step']==8400
    model=AttentionMemory(seed=cfg['model_seed'],activation_checkpoint=True);model.load_state_dict(parent['model'],strict=True)
    opt=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);before=[{k:g[k] for k in ('lr','weight_decay','eps')} for g in opt.param_groups]
    opt.load_state_dict(parent['optimizer']);after=[{k:g[k] for k in ('lr','weight_decay','eps')} for g in opt.param_groups];assert before==after
    return model,opt,dict(version=VERSION,parent_version=parent['version'],parent_step=8400,retained_tensors=list(parent['model']),retained_optimizer_states=len(opt.state),optimizer_policy_exact=True,new_parameters=[],sampler='Exact parent family-local SpatialStream and RNG continuation; common per-family evidence prefixes; different schedules may assign different delays to the same prefix episode',global_schedule_offset=8400)
