"""Only three new terminal heads; no change to the learned shared backbone."""
import torch
from torch import nn
from SecondPass.StabilizedAttentionSuite.model import StabilizedAttentionSuite

class ExpandedSuite(StabilizedAttentionSuite):
    def __init__(self):
        super().__init__()
        for size in (4,9,16):self.heads[f'vda{size}']=nn.Linear(64,2)

def initialize(checkpoint=None):
    model=ExpandedSuite();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-8)
    if checkpoint is None:return model,optimizer
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    missing,extra=model.load_state_dict(saved['model'],strict=False)
    expected={f'heads.vda{s}.{k}' for s in (4,9,16) for k in ('weight','bias')}
    if set(missing)!=expected or extra:raise ValueError((missing,extra))
    # Preserve old Adam moments by name. New heads receive state on first update.
    old=StabilizedAttentionSuite();old.load_state_dict(saved['model'])
    opt=torch.optim.Adam(old.parameters(),lr=3e-4,eps=1e-8);opt.load_state_dict(saved['optimizer'])
    new_by_name=dict(model.named_parameters())
    for name,p in old.named_parameters():
        if p in opt.state:optimizer.state[new_by_name[name]]={k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state[p].items()}
    return model,optimizer
