"""Explicit name+shape architecture migration, verified full-state persistence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from SecondPass.JointTraining.core import cpu_tree, tree_equal
from SecondPass.JointTraining.worker import verify_sources
from SecondPass.TaskSuite.suite import TASKS, task_classes, SuiteStream
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline
from .model import SpatialReadout

SOURCE_SHA256='51e37b64a69a6e5e7d5b7d6be8dccee28dd9223ebe573a4c59cc67638bb8d413'
INIT_SEED=94592763


def portable_source_config(config,root=None):
    """Rebase lookup paths only; validate identical bytes in a local runtime copy.

    launchd lacks Desktop TCC access on this host. No permission is changed and
    no parent checkpoint/config is edited; its recorded digests still control.
    """
    root=Path(__file__).resolve().parents[2] if root is None else Path(root)
    original=next(Path(p).parents[2] for p in config['source_hashes'] if p.endswith('/SecondPass/JointTraining/worker.py'))
    result=copy.deepcopy(config)
    result['source_hashes']={str(root/Path(p).relative_to(original)):sha for p,sha in config['source_hashes'].items()}
    return result


def load_verified(receipt):
    path=Path(receipt['path'])
    if path.stat().st_size!=receipt['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']:
        raise ValueError('Checkpoint size/digest mismatch')
    return torch.load(path,map_location='cpu')


def verify_parent(pointer):
    receipt=json.loads(Path(pointer).read_text())
    if receipt['sha256']!=SOURCE_SHA256 or receipt['step']!=3393:
        raise ValueError('Not authorized global-GRU step3393')
    source=load_verified(receipt)
    if source['schema']!=1 or source['state']['step']!=3393 or source['state']['episodes']!=108576:
        raise ValueError('Parent schema/exposure mismatch')
    if source['scheduler']['updates']!=3393 or source['scheduler']['tasks']:
        raise ValueError('Parent must be a complete task-cycle boundary')
    verify_sources(portable_source_config(source['state']['config']))
    stream=SuiteStream('train'); stream.load_state_dict(source['stream'])
    if source['rng']['mps'] is None: raise ValueError('Parent MPS RNG missing')
    names=legacy_optimizer_names(source)
    if len(source['optimizer']['state'])!=len(names): raise ValueError('Parent Adam state missing')
    return source,receipt


def legacy_optimizer_names(source):
    # Legacy checkpoint has no names. Recover its source-defined parameter order,
    # validate every name/shape against the immutable source before mapping ids.
    with torch.random.fork_rng(devices=[]):
        model=AccumulatorBaseline(task_classes(),stack=3,center=True,accumulator='kda')
    names=[n for n,_ in model.named_parameters()]
    if list(source['model'])!=list(model.state_dict()): raise ValueError('Legacy parameter order changed')
    for n,p in model.named_parameters():
        if source['model'][n].shape!=p.shape: raise ValueError('Legacy shape changed: '+n)
    groups=source['optimizer']['param_groups']
    if len(groups)!=1 or groups[0]['params']!=list(range(len(names))):
        raise ValueError('Cannot prove legacy optimizer parameter order')
    return names


def map_adam(saved,old_names,old_model,new_names,new_model):
    if len(saved['param_groups'])!=1 or len(old_names)!=len(saved['param_groups'][0]['params']):
        raise ValueError('Optimizer group/name mismatch')
    named=dict(zip(old_names,saved['param_groups'][0]['params']))
    result={'state':{},'param_groups':[copy.deepcopy(saved['param_groups'][0])]}
    result['param_groups'][0]['params']=list(range(len(new_names)))
    for index,name in enumerate(new_names):
        if name not in named or old_model[name].shape!=new_model[name].shape: continue
        state=saved['state'].get(named[name])
        if state is None: raise ValueError('Missing inherited Adam state: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=new_model[name].shape: raise ValueError('Adam shape mismatch: '+name)
        if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
            raise ValueError('Nonfinite inherited Adam state')
        result['state'][index]=cpu_tree(state)
    return result


def migrate(source,receipt,initialization_seed=INIT_SEED):
    with torch.random.fork_rng(devices=[]):
        # CPU Generator isolation avoids touching the carried MPS RNG at all.
        torch.set_rng_state(torch.Generator().manual_seed(initialization_seed).get_state())
        model=SpatialReadout(task_classes())
    weights=cpu_tree(model.state_dict()); carried=[]
    for name,tensor in weights.items():
        if name in source['model'] and source['model'][name].shape==tensor.shape:
            weights[name]=source['model'][name].clone(); carried.append(name)
    new_names=[n for n,_ in model.named_parameters()]
    optimizer=map_adam(source['optimizer'],legacy_optimizer_names(source),source['model'],new_names,weights)
    group=optimizer['param_groups'][0]
    if (group['lr'],group['betas'],group['eps'],group['weight_decay'])!=(1e-4,(.9,.999),1e-8,0):
        raise ValueError('Parent optimizer recipe mismatch')
    state=dict(step=0,episodes=0,frames=0,optimizer_seconds=0.,best_step=None,best_key=None,best_checkpoint=None,
        selection_history=[],parent=cpu_tree(source['state']),
        exposure={t:dict(updates=0,episodes=0,frames=0,cells={c['id']:0 for c in spec['conditions']}) for t,spec in TASKS.items()})
    return dict(schema=2,model=weights,optimizer=optimizer,optimizer_names=new_names,
        scheduler=cpu_tree(source['scheduler']),stream=cpu_tree(source['stream']),rng=cpu_tree(source['rng']),state=state,
        migration=dict(source=receipt,initialization_seed=initialization_seed,carried_names=carried,
            fresh_names=[n for n in new_names if n not in carried],removed_names=[n for n in source['model'] if n not in weights],
            adam_policy='Exact parameter name plus shape; legacy IDs recovered against verified immutable constructor; destination IDs remapped',
            selection_reset=True,profile_state_inherited=False))


def save_payload(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('Immutable checkpoint already exists: '+str(path))
    payload=cpu_tree(payload)
    temporary=path.with_suffix('.pt.tmp')
    with temporary.open('wb') as f:
        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)
    receipt=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        verified=True,step=payload['state']['step'],parent_step=payload['state']['parent']['step'])
    if not tree_equal(payload,load_verified(receipt)): raise RuntimeError('Full-state readback mismatch')
    return receipt


def restore(payload,model,optimizer,scheduler,stream,device):
    if payload['schema']!=2 or payload['optimizer_names']!=[n for n,_ in model.named_parameters()]:
        raise ValueError('Architecture/optimizer names mismatch')
    model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler']); stream.load_state_dict(payload['stream'])
    torch.set_rng_state(payload['rng']['cpu']); np.random.set_state(payload['rng']['numpy']); random.setstate(payload['rng']['python'])
    if str(device)=='mps': torch.mps.set_rng_state(payload['rng']['mps'])
    return cpu_tree(payload['state'])


def snapshot(model,optimizer,scheduler,stream,state,migration,device):
    return dict(schema=2,model=model.state_dict(),optimizer=optimizer.state_dict(),
        optimizer_names=[n for n,_ in model.named_parameters()],scheduler=scheduler.state_dict(),stream=stream.state_dict(),
        state=state,migration=migration,rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
        mps=torch.mps.get_rng_state() if str(device)=='mps' else None))


def verify_updated(migration_receipt,updated_receipt):
    """Read-only CPU evidence: actual changed recurrence + full saved state."""
    initial=load_verified(migration_receipt); saved=load_verified(updated_receipt)
    if initial['schema']!=2 or saved['schema']!=2 or initial['state']['step']!=0 or saved['state']['step']<1:
        raise ValueError('Require migration and actual post-update checkpoint')
    if initial['optimizer_names']!=saved['optimizer_names']:
        raise ValueError('Post-update optimizer names changed')
    if saved['scheduler']['updates']!=saved['state']['parent']['step']+saved['state']['step']:
        raise ValueError('Saved scheduler/branch exposure mismatch')
    if not tree_equal(initial['migration'],saved['migration']): raise ValueError('Lineage changed')
    changed=[n for n in saved['model'] if n.startswith('spatial_gru.') and not torch.equal(saved['model'][n],initial['model'][n])]
    if len(changed)!=4: raise ValueError('ConvGRU weights have not all changed')
    names=saved['optimizer_names']; adam={}
    for name in initial['migration']['fresh_names']:
        index=names.index(name)
        if index in initial['optimizer']['state']: raise ValueError('Fresh Adam state was inherited')
        value=saved['optimizer']['state'][index]
        if float(value['step'])!=saved['state']['step'] or not torch.isfinite(value['exp_avg']).all():
            raise ValueError('Fresh Adam progress missing/nonfinite')
        adam[name]=float(value['step'])
    for key in ('model','optimizer','scheduler','stream','rng','state','migration'):
        if key not in saved: raise ValueError('Incomplete full state: '+key)
    return dict(verified=True,branch_step=saved['state']['step'],branch_episodes=saved['state']['episodes'],
        optimizer_seconds=saved['state']['optimizer_seconds'],parent_step=saved['state']['parent']['step'],
        changed_convgru_tensors=changed,fresh_adam_steps=adam,migration=migration_receipt,checkpoint=updated_receipt)
