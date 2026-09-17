"""Explicit platform warm start from exact, unfitted local EI tensors."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from PreAttentiveVision.train import read,sha
from WorkingMemory.RecurrentComparison import model as module
from WorkingMemory.RecurrentComparison.train import worker

job=read(sys.argv[1]);original=module.load_parent
def exact_initializer(parent,arm,common_seed,core_seed,activation_checkpoint):
    if arm!='ei_adaptive':raise RuntimeError('Remote worker is only the authorized EI arm')
    path=ROOT/'portable/fresh_ei_initialization.pt'
    if sha(path)!=job['initializer_sha256']:raise RuntimeError('Fresh initializer identity changed')
    init=torch.load(path,map_location='cpu')
    if init['version']!='fresh_ei_initialization_v1' or init['optimizer_updates']!=0 or init['parent_sha256']!=job['parent_sha256']:
        raise RuntimeError('Invalid initialization lineage')
    m=original(parent,arm,common_seed,core_seed,activation_checkpoint)
    m.load_state_dict(init['model'],strict=True)
    return m
module.load_parent=exact_initializer
worker(job)
