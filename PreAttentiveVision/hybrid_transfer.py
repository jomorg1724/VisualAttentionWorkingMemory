"""Explicit parent-to-hybrid parameter and Adam-state transfer by name."""
import copy
import torch
from PreAttentiveVision.models import build_encoder
from PreAttentiveVision.decoder_multitask import MultitaskPairClassifier


NEW_PREFIXES={
    'convnext_grn':(),
    'convnext_gabor_residual':('encoder.gabor_',),
    'convnext_se_residual':('encoder.se_',),
}


def transfer_parent(model,optimizer,parent,arm):
    if parent['version']!='pav_ordered_tasks_v1' or parent['config']['model']!='convnext_grn' or parent['step']!=756:
        raise ValueError('Expected documented ConvNeXt ordered-task parent at step756')
    incompatible=model.load_state_dict(parent['model'],strict=False)
    if incompatible.unexpected_keys or any(not k.startswith(NEW_PREFIXES[arm]) for k in incompatible.missing_keys):
        raise RuntimeError('Unexpected parent/hybrid parameter or buffer mismatch')
    reference=MultitaskPairClassifier(build_encoder('convnext_grn'),parent['config']['task_classes'])
    old_names=[name for name,_ in reference.named_parameters()]
    old_groups=parent['optimizer']['param_groups']
    if len(old_groups)!=1 or len(old_groups[0]['params'])!=len(old_names):
        raise RuntimeError('Parent optimizer layout needs explicit adaptation')
    old_ids=dict(zip(old_names,old_groups[0]['params']))
    current=optimizer.state_dict();new_names=[name for name,_ in model.named_parameters()]
    new_ids=dict(zip(new_names,current['param_groups'][0]['params']))
    if not set(old_names).issubset(new_names):raise RuntimeError('Parent parameter names were lost')
    copied={}
    for name,old_id in old_ids.items():
        if old_id in parent['optimizer']['state']:
            copied[new_ids[name]]=copy.deepcopy(parent['optimizer']['state'][old_id])
    group=copy.deepcopy(old_groups[0]);group['params']=current['param_groups'][0]['params']
    optimizer.load_state_dict(dict(state=copied,param_groups=[group]))
    return dict(arm=arm,parent_parameter_names=len(old_names),inherited_adam_states=len(copied),
        new_parameter_names=[n for n in new_names if n not in old_ids],
        new_state_keys=incompatible.missing_keys,transfer='Original tensors and compatible Adam states by exact parameter name; no inherited state assigned to new parameters')
