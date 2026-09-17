"""One CPU check of unchanged continuation, signed feedback and temporal gradients."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from WorkingMemory.SelectiveMaintenance.model import migrate
from WorkingMemory.SelectiveMaintenance.sweep import PARENT,protocol
from WorkingMemory.SpatialComparison.model import Competitor
from WorkingMemory.SpatialComparison.stimuli import SpatialStream
torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.time()
cfg=dict(model_seed=41973001,new_lr=.0003,parent_lr=.00003,adam_eps=1e-10,weight_decay=.0001)
cp=torch.load(PARENT,map_location='cpu');control,co,cl=migrate(cp,'continuation',cfg);model,opt,lineage=migrate(cp,'controller_feedback',cfg)
parent=Competitor('spatial_ei',activation_checkpoint=False);parent.load_state_dict(cp['model'],strict=True)
for n,v in cp['model'].items():assert torch.equal(control.state_dict()[n],v) and torch.equal(model.state_dict()[n],v)
control.eval();model.eval();parent.eval();stream=SpatialStream(47973001,'val');x,y,_=stream.batch(2,'orientation_single',dict(delay=4,spacing='mixed'))
with torch.no_grad():
    before=parent(x,'orientation');same=control(x,'orientation');changed=model(x,'orientation');assert torch.equal(before,same)
    off=model(x,'orientation',feedback_off_frames=set(range(x.shape[1])));assert torch.equal(off,before)
model.train();opt.zero_grad(set_to_none=True);out,d=model(x,'orientation',True);torch.nn.functional.cross_entropy(out,y).backward()
assert d['states'][0][0].grad.norm()>0 and d['controls'][0][0].grad.norm()>0
assert sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)>0
for n in ('raw_recurrent','raw_feedback','raw_basis','raw_coupling'):assert getattr(model.controller,n).grad is not None and getattr(model.controller,n).grad.norm()>0
c=model.controller;w=torch.nn.functional.softplus(c.raw_feedback)*c.signs[None,:];assert (w[:,:26]>0).all() and (w[:,26:]<0).all()
cn=dict(control.named_parameters());mn=dict(model.named_parameters());assert set(co.state.keys())==set(cn[n] for n in cl['adam_states_by_name'])
for n in cl['adam_states_by_name']:
    for key,value in co.state[cn[n]].items():
        other=opt.state[mn[n]][key]
        assert torch.equal(value,other) if torch.is_tensor(value) else value==other
cells,cycle=protocol();assert len(cycle)==80 and sum(n.startswith('motion') for n in cycle)==8
result=dict(status='passed',seconds=time.time()-start,ordinary_continuation_logits_exact=True,zero_allframe_feedback_logits_exact=True,initial_feedback_max_logit_deviation=float((changed-before).abs().max()),early_spatial_rate_gradient=float(d['states'][0][0].grad.norm()),early_controller_rate_gradient=float(d['controls'][0][0].grad.norm()),controller_gradients={n:float(getattr(c,n).grad.norm()) for n in ('raw_recurrent','raw_feedback','raw_basis','raw_coupling')},retained_adam_states_exact=len(cl['adam_states_by_name']),model_parameters={a:sum(p.numel() for p in m.parameters()) for a,m in [('continuation',control),('controller_feedback',model)]},source_signs='26E/6I; nonnegative source weights and spatial bases',teaching='unchanged imported SpatialComparison renderer and80-updatecycle',parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest())
(Path(__file__).parent/'model_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
