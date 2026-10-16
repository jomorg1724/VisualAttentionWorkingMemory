"""Independently reload preserved final and selected checkpoints before cleanup."""
import json,time
from pathlib import Path
import torch
from SecondPass.DeadlyCorridorKills.agent import Agent
from .recover_results import digest

def main():
    torch.set_num_threads(2)
    base=Path(__file__).resolve().parent;root=base/'recovered';run=root/'vawm/SecondPass/DeadlyCorridorKills/CloudRuntime/run01';p=run/'training'
    assert json.loads((root/'download_verified.json').read_text())['all_hashes_match']
    latest=torch.load(p/'latest.pt',map_location='cpu',weights_only=False);best=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
    agent=Agent();agent.load_state_dict(latest['agent']);counts={}
    for name in ('critic_opt','actor_opt','alpha_opt'):
        opt=getattr(agent,name);opt.load_state_dict(latest[name]);counts[name]=len(opt.state)
        for param,state in opt.state.items():
            for k in ('exp_avg','exp_avg_sq'):assert state[k].shape==param.shape and torch.isfinite(state[k]).all()
            assert torch.isfinite(state['step'])
    for ck in (latest,best):assert all(torch.isfinite(x).all() for x in ck['agent'].values())
    agent.load_state_dict(best['agent'])
    exit=json.loads((p/'exit.json').read_text());assert all(latest[k]==exit[k] for k in ['steps','updates','episodes'])
    assert json.loads((run/'supervisor_exit.json').read_text())['exit_code']==0
    replay=latest['replay'];assert replay.transitions==sum(len(e['actions']) for e in replay.episodes)
    episodes=[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines()]
    assert all(x['raw_return']==x['kills'] for x in episodes)
    for e in replay.episodes:assert (e['rewards']>=0).all() and (e['rewards']==e['rewards'].astype(int)).all()
    report=dict(verified_at=time.time(),exit=exit,latest_step=latest['steps'],best_step=best['steps'],best_validation_mean=best['validation']['mean_return'],optimizer_states=counts,finite_aligned=True,replay_transitions=replay.transitions,last100_training_reward=sum(x['raw_return'] for x in episodes[-100:])/100,checkpoint_sha256={n:digest(p/(n+'.pt')) for n in ['latest','best']})
    (root/'checkpoints_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
