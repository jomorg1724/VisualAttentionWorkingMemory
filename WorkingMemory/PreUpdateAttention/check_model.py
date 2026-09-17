"""Focused CPU migration, input-path, and gradient check; no local GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from WorkingMemory.PreUpdateAttention.model import migrate
from WorkingMemory.PreUpdateAttention.remote_sweep import recipe
from WorkingMemory.SpatialComparison.model import Competitor,groups
from WorkingMemory.SpatialComparison.stimuli import SpatialStream
torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.time()
p=ROOT/'WorkingMemory/SpatialComparison/runs/spatial_20260913_100913/spatial_ei/checkpoint_004400.pt'
cp=torch.load(p,map_location='cpu');cfg=recipe();model,opt,lineage=migrate(cp,'preupdate_attention',cfg)
parent=Competitor('spatial_ei',activation_checkpoint=False);parent.load_state_dict(cp['model'],strict=True)
oldopt=torch.optim.Adam(groups(parent,cp['config']),eps=cp['config']['adam_eps']);oldopt.load_state_dict(cp['optimizer'])
oldnames=dict(parent.named_parameters());newnames=dict(model.named_parameters())
for n,v in cp['model'].items():assert torch.equal(model.state_dict()[n],v)
for n in lineage['adam_states_by_name']:
    for k,v in oldopt.state[oldnames[n]].items():
        actual=opt.state[newnames[n]][k]
        assert torch.equal(v,actual) if torch.is_tensor(v) else v==actual
stream=SpatialStream(47973001,'val');x,y,_=stream.batch(2,'orientation_single',dict(delay=4,spacing='mixed'))
model.eval();parent.eval()
with torch.no_grad():before=parent(x,'orientation');after=model(x,'orientation')
model.train();opt.zero_grad(set_to_none=True);logits,diag=model(x,'orientation',True);loss=torch.nn.functional.cross_entropy(logits,y);loss.backward()
grads={n:float(p.grad.norm()) for n,p in model.attention.named_parameters() if p.grad is not None}
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert diag['states'][0][0].grad.norm()>0 and diag['first_field'].grad.norm()>0
assert grads['query.weight']>0 and grads['key.weight']>0 and grads['value.weight']>0 and grads['output.weight']>0
assert grads['position']>0 and grads['source_bias']>0 and grads['raw_locality']>0
result=dict(status='passed',seconds=time.time()-start,retained_adam_states_exact=len(lineage['adam_states_by_name']),initial_parent_max_logit_deviation=float((after-before).abs().max()),parameters=sum(p.numel() for p in model.parameters()),new_parameters=sum(p.numel() for p in model.attention.parameters()),extra_persistent_state_entries=0,attention_gradients=grads,early_rate_gradient=float(diag['states'][0][0].grad.norm()),early_field_gradient=float(diag['first_field'].grad.norm()),initial_and_final_attention_stats=[diag['records'][0],diag['records'][-1]],cycle_matches_parent=cfg['cycle']==recipe()['cycle'])
(Path(__file__).parent/'model_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
