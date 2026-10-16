"""Bounded CPU checks of the changed normalization, write rule, and optimizer."""
import hashlib
import json
from pathlib import Path
import torch
from .memory import PatchNorm,SpatialGRU
from .train import build,update,source_hashes
from . import dataset

HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    norm=PatchNorm(64)
    x=torch.randn(2,64,10,10)
    y=norm(x)
    x2=x.clone();x2[:,:,0,0]*=100
    y2=norm(x2)
    assert torch.equal(y[:,:,1:,1:],y2[:,:,1:,1:])
    assert y.mean(1).abs().max()<1e-5
    cell=SpatialGRU(64)
    previous=torch.rand_like(x)*2-1
    held,_=cell(x,previous,torch.zeros(2,1,10,10))
    assert torch.equal(held,previous)
    changed,_=cell(x*100,previous,torch.ones(2,1,10,10))
    assert changed.abs().max()<=1.000001
    alpha=torch.full((2,100),.01)
    gate=alpha/(alpha+.01)
    assert torch.allclose(gate,torch.full_like(gate,.5))
    alpha[0]=0;alpha[0,0]=1
    assert (alpha/(alpha+.01)).max()<1
    model,opt=build('cpu')
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    rows=dataset.render('train',12,2,block=900001)
    values=update(model,opt,rows,[0,1],'cpu',2)
    assert values['clipped_gradient_norm']<=100.001
    assert len(opt.state)==len(before)
    assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    # Explicitly exercise the clipping branch, independently of the native sample's norm.
    for p in model.parameters(): p.grad=torch.ones_like(p)*100
    raw=torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True)
    post=torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
    assert raw>100 and post<=100.001
    with torch.no_grad():
        logits,details=model(torch.stack([r[0] for r in rows]),return_details=True)
    assert torch.isfinite(logits).all()
    assert torch.allclose(details['attention'].flatten(2).sum(-1),torch.ones(2,len(rows[0][0])),atol=1e-6)
    assert 1<float(details['attention_scale'])<16
    result=dict(passed=True,device='cpu',native_frames=len(rows[0][0]),
        parameter_count=sum(p.numel() for p in model.parameters()),parameter_tensors=len(before),
        optimizer_states=len(opt.state),changed_tensors=[k for k,p in model.named_parameters() if not torch.equal(p,before[k])],
        patch_norm_spatial_independence=True,zero_write_preserves_memory=True,bounded_state=True,
        forced_clip_raw=float(raw),forced_clip_post=float(post),native_update=values,
        weights_discarded=True)
    (HERE/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('changed_tensors','native_update')},indent=2))

if __name__=='__main__':main()
