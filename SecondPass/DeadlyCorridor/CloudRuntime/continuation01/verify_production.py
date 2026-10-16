"""Read-only independent verification of resumed cloud learning."""
import hashlib,json,time
from pathlib import Path
import torch
from SecondPass.DeadlyCorridor.agent import Agent

ROOT=Path('/workspace/vawm');RUN=ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/continuation01'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    torch.set_num_threads(2)
    original=ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/run01/training'
    saved=torch.load(original/'latest.pt',map_location='cpu',weights_only=False)
    latest=torch.load(RUN/'training/latest.pt',map_location='cpu',weights_only=False)
    assert latest['steps']>saved['steps'] and latest['updates']>saved['updates']
    assert latest['episodes']>saved['episodes']
    agent=Agent();agent.load_state_dict(latest['agent'])
    counts={}
    for k in ('critic_opt','actor_opt','alpha_opt'):
        opt=getattr(agent,k);opt.load_state_dict(latest[k]);counts[k]=len(opt.state)
        old_ids=saved[k]['param_groups'][0]['params'];new_ids=latest[k]['param_groups'][0]['params']
        for group in opt.param_groups:
            for p in group['params']:
                state=opt.state[p]
                for key in ('exp_avg','exp_avg_sq'):
                    assert state[key].shape==p.shape and torch.isfinite(state[key]).all()
                assert torch.isfinite(state['step'])
        for old_id,new_id in zip(old_ids,new_ids):
            assert latest[k]['state'][new_id]['step']>saved[k]['state'][old_id]['step']
    assert all(torch.isfinite(t).all() for t in latest['agent'].values())
    changed={prefix:any(not torch.equal(t,latest['agent'][n]) for n,t in saved['agent'].items() if n.startswith(prefix))
             for prefix in ('encoder.','actor.','q1.','q2.','target_encoder.','target_q1.','target_q2.')}
    assert all(changed.values())
    config=json.loads((RUN/'deployment.json').read_text())
    for n,sha in config['original_checkpoint_sha256'].items():assert digest(original/(n+'.pt'))==sha
    source=json.loads((ROOT.parent/'source_manifest.json').read_text())
    for name,sha in source.items():assert digest(ROOT/name)==sha,name
    worker=json.loads((RUN/'worker.json').read_text());guard=json.loads((RUN/'guard.json').read_text())
    assert 'SecondPass.DeadlyCorridor.train' in Path(f"/proc/{worker['pid']}/cmdline").read_bytes().decode()
    assert 'cloud_control.py' in Path(f"/proc/{guard['pid']}/cmdline").read_bytes().decode()
    assert guard['status']=='armed'
    receipt=dict(time=time.time(),pod_id=config['pod_id'],worker_pid=worker['pid'],supervisor_pid=worker['supervisor_pid'],guard_pid=guard['pid'],
        saved_steps=latest['steps'],saved_updates=latest['updates'],saved_episodes=latest['episodes'],
        resumed_steps=saved['steps'],optimizer_states=counts,all_optimizer_states_finite_aligned_advanced=True,
        changed=changed,original_checkpoints_unchanged=True,source_files_verified=len(source),
        replay_transitions=latest['replay'].transitions,deadline=config['deadline'],
        live_status=json.loads((RUN/'training/status.json').read_text()))
    (RUN/'independent_production_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
