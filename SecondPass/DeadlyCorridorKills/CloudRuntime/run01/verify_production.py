"""Independent CPU reload of fresh initialization and actual learning state."""
import hashlib,json,time
from pathlib import Path
import torch
from SecondPass.DeadlyCorridorKills.agent import Agent

ROOT=Path('/workspace/vawm');RUN=ROOT/'SecondPass/DeadlyCorridorKills/CloudRuntime/run01'

def main():
    torch.set_num_threads(2)
    initial=torch.load(RUN/'training/initial.pt',map_location='cpu',weights_only=False)
    latest=torch.load(RUN/'training/latest.pt',map_location='cpu',weights_only=False)
    assert initial['steps']==initial['updates']==initial['episodes']==0
    assert initial['replay'].transitions==0
    assert all(not initial[k]['state'] for k in ('critic_opt','actor_opt','alpha_opt'))
    assert initial['config']['parent'] is None and initial['config']['reward_scale']==1.
    torch.manual_seed(202211);fresh=Agent()
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in initial['agent'].items())
    assert latest['steps']>=2000 and latest['updates']>0
    agent=Agent();agent.load_state_dict(latest['agent']);counts={}
    for name in ('critic_opt','actor_opt','alpha_opt'):
        opt=getattr(agent,name);opt.load_state_dict(latest[name]);counts[name]=len(opt.state)
        for group in opt.param_groups:
            for parameter in group['params']:
                state=opt.state[parameter]
                assert state['step'].item()==latest['updates']
                for key in ('exp_avg','exp_avg_sq'):
                    assert state[key].shape==parameter.shape and torch.isfinite(state[key]).all()
    assert counts==dict(critic_opt=62,actor_opt=4,alpha_opt=1)
    assert all(torch.isfinite(v).all() for v in latest['agent'].values())
    changed={prefix:any(not torch.equal(t,latest['agent'][n]) for n,t in initial['agent'].items() if n.startswith(prefix))
             for prefix in ('encoder.','actor.','q1.','q2.','target_encoder.','target_q1.','target_q2.')}
    assert all(changed.values())
    for e in latest['replay'].episodes:
        assert (e['rewards']>=0).all() and (e['rewards']==e['rewards'].astype(int)).all()
    rows=[json.loads(s) for s in (RUN/'training/episodes.jsonl').read_text().splitlines()]
    assert all(r['raw_return']==r['kills'] for r in rows)
    manifest=json.loads((ROOT.parent/'source_manifest.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    worker=json.loads((RUN/'worker.json').read_text());guard=json.loads((RUN/'guard.json').read_text())
    assert 'SecondPass.DeadlyCorridorKills.train' in Path(f"/proc/{worker['pid']}/cmdline").read_bytes().decode()
    assert 'cloud_control.py' in Path(f"/proc/{guard['pid']}/cmdline").read_bytes().decode()
    assert guard['status']=='armed'
    receipt=dict(time=time.time(),worker=worker,guard=guard,saved_steps=latest['steps'],saved_updates=latest['updates'],saved_episodes=latest['episodes'],
        fresh_initialization_exact=True,initial_replay_and_optimizers_empty=True,all_optimizer_states_finite_aligned_advanced=True,
        optimizer_states=counts,changed=changed,source_files_verified=len(manifest),replay_transitions=latest['replay'].transitions,
        all_episode_rewards_equal_kills=True,mean_kills_last_100=sum(r['kills'] for r in rows[-100:])/len(rows[-100:]),
        live_status=json.loads((RUN/'training/status.json').read_text()))
    (RUN/'independent_production_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
