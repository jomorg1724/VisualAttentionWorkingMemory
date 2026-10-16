"""Independent CPU reload of initial migration and saved production progress."""
import hashlib,json,time
from pathlib import Path
import torch
from .model import ExpandedSuite,ParentModel,PARENT


def main():
    torch.set_num_threads(2)
    run=Path(__file__).parent/'LocalRuntime/run01'
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    names=list(dict(ExpandedSuite().named_parameters()))
    old_names=list(dict(ParentModel().named_parameters()))
    initial_ids=initial['optimizer']['param_groups'][0]['params']
    parent_ids=parent['optimizer']['param_groups'][0]['params']
    latest_ids=latest['optimizer']['param_groups'][0]['params']
    init_states=dict(zip(names,(initial['optimizer']['state'].get(i) for i in initial_ids)))
    old_states=dict(zip(old_names,(parent['optimizer']['state'].get(i) for i in parent_ids)))
    states=dict(zip(names,(latest['optimizer']['state'].get(i) for i in latest_ids)))
    for name,value in parent['model'].items():assert torch.equal(value,initial['model'][name]),name
    for name,state in old_states.items():
        for key,value in state.items():
            actual=init_states[name][key]
            assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
    for key,value in parent['optimizer']['param_groups'][0].items():
        if key!='params':assert initial['optimizer']['param_groups'][0][key]==value
    finite=0
    for name,state in states.items():
        if state is None:continue
        for key in ('exp_avg','exp_avg_sq'):
            assert state[key].shape==latest['model'][name].shape and torch.isfinite(state[key]).all()
        assert torch.isfinite(state['step']);finite+=1
    changed=[n for n in names if not torch.equal(initial['model'][n],latest['model'][n])]
    config=latest['config'];root=PARENT.parents[4]
    for name,sha in config['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==sha,name
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==config['parent_sha256']
    step=latest['step']
    if step>=19:
        assert finite==92
        for name in names:
            expected=float(init_states[name]['step']) if init_states[name] is not None else 0.
            assert float(states[name]['step'])>expected,name
    status=json.loads((run/'status.json').read_text())
    receipt=dict(verified_at=time.time(),saved_step=step,live_status=status,parent_step=parent['step'],
                 exact_parent_tensors_and_adam_preserved=True,finite_aligned_adam_states=finite,
                 changed_parameter_tensors=len(changed),changed_names=changed,
                 source_files_verified=len(config['source_sha256']),parent_sha256=config['parent_sha256'],
                 no_parent_mutation=True,all_tasks_updated=step>=19)
    name='independent_production_verified.json' if step>=19 else 'independent_initial_verified.json'
    (run/name).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='changed_names'},indent=2))


if __name__=='__main__':main()
