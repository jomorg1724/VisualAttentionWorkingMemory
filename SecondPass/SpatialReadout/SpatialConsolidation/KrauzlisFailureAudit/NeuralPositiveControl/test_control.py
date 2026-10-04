"""Native input/label, batching, optimizer and full-history tests; CPU only."""
import copy
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

def test_native_end_to_end():
    path=Path(__file__).with_name('control.py')
    assert path.exists(), 'Fresh positive-control implementation not yet written'
    spec=importlib.util.spec_from_file_location('npc',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    torch.set_num_threads(1); torch.manual_seed(901)
    stream=mod.NativeStream(810001,'train')
    x,y,meta=stream.batch(2,12)
    assert x.shape==(2,29,3,100,100)
    assert all(int(m['event_type']=='target')==int(v) for v,m in zip(y,meta))
    direct=mod.SpatialBatteryStream(810001,'train').batch(2,mod.TASK,{'baseline_transitions':12})
    assert torch.equal(x,direct[0]) and torch.equal(y,direct[1])
    a=mod.Model().eval(); b=copy.deepcopy(a)
    with torch.no_grad():
        joint=a(x); separate=torch.cat([a(z[None]) for z in x])
    torch.testing.assert_close(joint,separate,atol=2e-5,rtol=2e-4)
    x.requires_grad_(True); loss=F.cross_entropy(a(x),y); loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad[:,0].abs().sum()>0 and x.grad[:,-1].abs().sum()>0
    for name,p in a.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for i in range(2): (F.cross_entropy(b(x.detach()[i:i+1]),y[i:i+1])/2).backward()
    maxerr=max((p.grad-q.grad).abs().max().item() for p,q in zip(a.parameters(),b.parameters()))
    assert maxerr<3e-4,maxerr
    before={n:p.detach().clone() for n,p in a.named_parameters()}
    opt=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=0)
    opt.step()
    assert all(not torch.equal(before[n],p) for n,p in a.named_parameters())
    print({'loss':loss.item(),'accumulation_max_gradient_error':maxerr,'first_frame_gradient':x.grad[:,0].abs().sum().item(),'last_frame_gradient':x.grad[:,-1].abs().sum().item()})
