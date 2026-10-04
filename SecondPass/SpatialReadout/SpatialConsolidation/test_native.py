import torch
from SecondPass.SpatialReadout.SpatialConsolidation import worker as w
from SecondPass.SpatialReadout.SpatialConsolidation.model import SpatialConsolidation


def test_final_only_and_no_future_input():
    torch.set_num_threads(2)
    model=SpatialConsolidation({'x':2})
    frames=torch.randn(2,3,3,112,112)
    with torch.no_grad():
        full=model.recurrent_states(frames)
        prefix=model.recurrent_states(frames[:,:2])
    assert torch.equal(full[1],prefix[-1])
    captured=[]
    model.spatial_gru.register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()))
    terminal=[]
    model.consolidation.register_forward_pre_hook(lambda m,i:terminal.append(i[0].detach().clone()))
    model(frames,'x')
    assert len(captured)==3 and len(terminal)==1
    assert torch.equal(captured[-1],terminal[0])
    # Altering readout-side transformer cannot change recurrent computation.
    with torch.no_grad():
        model.consolidation.qkv.weight.add_(10)
        changed=model.recurrent_states(frames)
    assert all(torch.equal(a,b) for a,b in zip(full,changed))


def test_all_native_tasks_full_gradient_and_update(tmp_path):
    import time
    torch.set_num_threads(2)
    session=w.Session(tmp_path,dict(device='cpu',effective_batch=2,microbatch=2,cap_started=time.time(),deadline=time.time()+300))
    before={n:p.detach().clone() for n,p in session.model.named_parameters()}
    tasks=[]
    for _ in range(13):
        task,cell=session.scheduler.next();tasks.append(task)
        row=session.train_update(task,cell)
        assert row['episodes']==2
        for n,p in session.model.named_parameters():
            if n.startswith('heads.') and not n.startswith('heads.'+task+'.'): continue
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
    assert set(tasks)==set(w.TASKS)
    after=dict(session.model.named_parameters())
    # Every learned tensor (including every task head) changes across the cycle.
    assert all(not torch.equal(before[n],p) for n,p in after.items())
