"""Disposable CPU checks of adaptation, native stream replay and head isolation."""
import json
import torch
from .model import StabilizedAttentionSuite
from .dataset import Stream,TASKS,all_cells
from .train import build,update
from SecondPass.StabilizedAttentionKrauzlis.model import StabilizedAttentionKrauzlis
from SecondPass.JointTraining.core import BalancedScheduler,tree_equal

def main():
    torch.set_num_threads(2)
    original=StabilizedAttentionKrauzlis().eval(); adapted=StabilizedAttentionSuite().eval()
    state=adapted.state_dict()
    for name,value in original.state_dict().items():
        target=name.replace('readout.2.','heads.krauzlis_cued_motion.')
        state[target]=value
    adapted.load_state_dict(state)
    stream=Stream('train');task='krauzlis_cued_motion';cell=TASKS[task]['conditions'][0]['id']
    x,y,_=stream.batch(2,task,cell)
    with torch.no_grad():
        a,ad=original(x,True);b,bd=adapted(x,task,True)
    assert torch.equal(a,b) and tree_equal(ad,bd), 'Backbone adaptation changed outputs'
    saved=stream.state_dict();expected=stream.batch(2,task,cell)
    restored=Stream('train');restored.load_state_dict(saved)
    assert tree_equal(expected,restored.batch(2,task,cell)), 'Native stream replay mismatch'
    model,opt=build('cpu');scheduler=BalancedScheduler(199212)
    for _ in range(13):
        task,cell=scheduler.next();x,y,_=stream.batch(2,task,cell)
        before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('heads.') and not n.startswith('heads.'+task+'.')}
        result=update(model,opt,x,y,task,'cpu',micro=2)
        assert all(torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)
        print(json.dumps(dict(task=task,cell=cell,frames=x.shape[1],loss=result['loss'])),flush=True)
    assert len(opt.state)==80
    for name,p in model.named_parameters():
        s=opt.state[p];assert int(s['step'])==(1 if name.startswith('heads.') else 13)
        assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
    print(json.dumps(dict(passed=True,exact_original_logits_and_details=True,native_stream_replay=True,
        task_heads_isolated=True,all13_tasks_updated=True,optimizer_states=len(opt.state),
        parameters=sum(p.numel() for p in model.parameters()),cells=len(all_cells()))),flush=True)

if __name__=='__main__':main()
