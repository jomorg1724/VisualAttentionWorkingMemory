"""One focused CPU migration/logit/next-draw check; MPS steps in launch profile."""
import importlib.util
import json
from pathlib import Path
import torch
from SecondPass.JointTraining import worker as original
from SecondPass.JointTraining.core import BalancedScheduler, tree_equal, atomic_json
from SecondPass.TaskSuite.suite import SuiteStream, task_classes
from SecondPass.SpatialReadout.model import SpatialReadout


def main(output):
    assert importlib.util.find_spec('SecondPass.SpatialComparisonReadout.experiment'), 'Missing comparison implementation'
    from .experiment import ComparisonReadout, migrate, source_payload, selection_key
    original.cpu_setup()
    source, receipt = source_payload()
    before = torch.get_rng_state().clone()
    control = migrate(source, receipt, 'control')
    candidate = migrate(source, receipt, 'candidate')
    assert torch.equal(before, torch.get_rng_state()), 'Initialization changed inherited RNG'
    assert sum(v.numel() for k,v in candidate['model'].items() if k.startswith('comparator.')) == 12416
    for key in ('scheduler', 'stream', 'rng'):
        assert tree_equal(source[key], control[key]) and tree_equal(control[key], candidate[key])
    assert tree_equal(source['optimizer'], control['optimizer'])
    for i,n in enumerate(source['optimizer_names']):
        j=candidate['optimizer_names'].index(n)
        assert tree_equal(source['optimizer']['state'][i],candidate['optimizer']['state'][j])
    for n in candidate['migration']['fresh_names']:
        assert candidate['optimizer_names'].index(n) not in candidate['optimizer']['state']
    a=SpatialReadout(task_classes()); a.load_state_dict(control['model'])
    b=ComparisonReadout(task_classes()); b.load_state_dict(candidate['model'])
    assert all(p.requires_grad and p.dtype==torch.float32 for p in b.parameters())
    schedulers=[BalancedScheduler(0),BalancedScheduler(0)]
    streams=[SuiteStream('train'),SuiteStream('train')]
    for i,p in enumerate((control,candidate)):
        schedulers[i].load_state_dict(p['scheduler']);streams[i].load_state_dict(p['stream'])
    cells=[s.next() for s in schedulers];assert cells[0]==cells[1]
    draws=[s.batch(1,*cells[0]) for s in streams]
    assert tree_equal(draws[0],draws[1])
    with torch.no_grad():
        expected=a(draws[0][0],cells[0][0]);actual=b(draws[1][0],cells[1][0])
    assert torch.equal(expected,actual), 'Zero residual must preserve exact logits'
    # Nonzero comparator must not change the recurrent state it observes.
    with torch.no_grad():
        b.comparator[-1].weight.fill_(0.01)
        native=a.recurrent_states(draws[0][0])
        observed=[]
        hook=b.spatial_gru.register_forward_hook(lambda module,args,out:observed.append(out.clone()))
        b(draws[0][0],cells[0][0]);hook.remove()
    assert all(torch.equal(x,y) for x,y in zip(native,observed))
    fake={'complete':True,'cells':[{'task':t,'auc':.6} for t in ('orientation_cued','spatial_binding') for _ in range(4)],'summary':{'equal_task_mean_auc':.7}}
    assert selection_key(fake)==[.6,.7]
    result=dict(verified=True,source=receipt,next_cell=cells[0],same_next_draw=True,initial_logits_exact=True,
        recurrent_carry_unchanged=True,adam_preserved=True,new_parameters=12416,initialization_rng_isolated=True,
        comparator_finite_grad_and_adam='verified separately by bounded actual-launcher profile')
    atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
