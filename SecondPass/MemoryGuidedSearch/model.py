"""Three new location heads. Original 16 heads and shared backbone are preserved."""
import copy
from pathlib import Path
import torch
from torch import nn
from SecondPass.StabilizedAttentionVDA.model import ExpandedSuite as ParentModel
from .dataset import TASK,POPOUT,MOTION

PARENT=Path(__file__).resolve().parents[1]/'StabilizedAttentionVDA/LocalRuntime/continuation01/best.pt'

class ExpandedSuite(ParentModel):
    def __init__(self):
        super().__init__()
        for task in (TASK,POPOUT,MOTION):self.heads[task]=nn.Linear(64,4)

def initialize(checkpoint=None):
    model=ExpandedSuite();opt=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-8)
    if checkpoint is None:return model,opt
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    missing,extra=model.load_state_dict(saved['model'],strict=False)
    if set(missing)!={f'heads.{task}.{k}' for task in (TASK,POPOUT,MOTION) for k in ('weight','bias')} or extra:raise ValueError((missing,extra))
    old=ParentModel();old.load_state_dict(saved['model']);old_opt=torch.optim.Adam(old.parameters(),lr=3e-4,eps=1e-8)
    old_opt.load_state_dict(saved['optimizer'])
    if len(old_opt.param_groups)!=1:raise ValueError('Expected one parent Adam group')
    opt.param_groups[0].update({k:copy.deepcopy(v) for k,v in old_opt.param_groups[0].items() if k!='params'})
    by_name=dict(model.named_parameters())
    for name,p in old.named_parameters():
        if p in old_opt.state:opt.state[by_name[name]]=copy.deepcopy(old_opt.state[p])
    return model,opt
