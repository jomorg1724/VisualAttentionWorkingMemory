"""One focused CPU check; no acquisition training or GPU context."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from WorkingMemory.RecurrentComparison.model import load_parent,ARMS,NEW_PREFIXES,optimizer_groups
from WorkingMemory.stimuli import SequenceStream,BRIDGE_CONDITIONS
from PreAttentiveVision.train import sha
start=time.monotonic()
path=ROOT/'WorkingMemory/runs/wm_20260912_181219/opponent/checkpoint_006860.pt'
assert any(i['file']==path.name and i['sha256']==sha(path) for i in map(json.loads,(path.parent/'checkpoint_index.jsonl').read_text().splitlines()))
parent=torch.load(path,map_location='cpu');stream=SequenceStream(593000,'train')
x,y,_=stream.batch(2,'orientation',BRIDGE_CONDITIONS['bridge_recall'])
rows=[];shared=None
for arm in ARMS:
    model=load_parent(parent,arm).train();assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(model.state_dict()[k],v) for k,v in parent['model'].items())
    common={k:v for k,v in model.state_dict().items() if k.startswith(('memory_input.','memory_output.'))}
    if shared is not None:assert all(torch.equal(v,shared[k]) for k,v in common.items())
    shared=common
    logits,states,z=model(x,'orientation',diagnostic=True)
    torch.nn.functional.cross_entropy(logits,y).backward()
    assert torch.isfinite(logits).all() and z[0].grad.norm()>0 and z[-1].grad.norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
    assert model.memory_output.weight.abs().sum()>0
    row=dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),new_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(NEW_PREFIXES)),early_representation_grad=float(z[0].grad.norm()),late_representation_grad=float(z[-1].grad.norm()),encoder_grad=float(sum(p.grad.square().sum() for p in model.encoder.parameters()).sqrt()))
    if arm=='ei_adaptive':
        w=model.memory.recurrent_weight();assert (w[:,:205]>=0).all() and (w[:,205:]<=0).all()
        assert all(s['state0_max'].isfinite() and s['state1_max'].isfinite() for s in states)
        row['raw_recurrent_gradient_norm']=float(model.memory.raw_recurrent.grad.norm())
        opt=torch.optim.Adam(optimizer_groups(model));opt.step()
        w=model.memory.recurrent_weight();assert (w[:,:205]>=0).all() and (w[:,205:]<=0).all()
    model.eval()
    with torch.no_grad():
        pair=x[:,:2];a=model(pair,'orientation');b=model(pair,'orientation')
        assert torch.equal(a,b)
    rows.append(row)
result=dict(status='passed',seconds=time.monotonic()-start,parent=str(path),parent_sha256=sha(path),results=rows,checks=['strict unchanged parent tensors','same common new initialization','ordinary-image encoder gradients with checkpointing','early and late representation gradients','finite real-sequence state','nonzero residual','Dale column signs before and after one discarded CPU update','deterministic reset on repeated inference'])
(ROOT/'WorkingMemory/RecurrentComparison/focused_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
