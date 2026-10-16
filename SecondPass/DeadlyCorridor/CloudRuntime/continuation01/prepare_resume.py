"""Explicit allocation renewal; all learning state and original files preserved."""
import copy, hashlib, json, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from SecondPass.DeadlyCorridor.agent import Agent
from SecondPass.DeadlyCorridor.environment import Corridor
from SecondPass.DeadlyCorridor.train import PARENT, configure

ROOT=Path('/workspace/vawm')
RUN=ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/continuation01'
SOURCE=ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/run01/training'
TRAIN=RUN/'training'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    configure('cuda')
    deployment=json.loads((RUN/'deployment.json').read_text())
    assert time.time()<deployment['deadline']-600
    latest=SOURCE/'latest.pt';best=SOURCE/'best.pt'
    originals={n:digest(SOURCE/(n+'.pt')) for n in ('latest','best')}
    assert originals==deployment['original_checkpoint_sha256']
    manifest=json.loads((ROOT.parent/'source_manifest.json').read_text())
    for name,sha in manifest.items():assert digest(ROOT/name)==sha,name
    saved=torch.load(latest,map_location='cpu',weights_only=False)
    assert (saved['steps'],saved['updates'],saved['episodes'])==(114931,25161,4600)
    for n,sha in saved['config']['source_sha256'].items():
        assert digest(ROOT/'SecondPass/DeadlyCorridor'/n)==sha,n
    assert digest(PARENT)==saved['config']['parent_sha256']
    env=Corridor()
    try:assert env.provenance==saved['config']['environment']
    finally:env.close()
    agent=Agent(PARENT).to('cuda');agent.load_state_dict(saved['agent'])
    for k in ('critic_opt','actor_opt','alpha_opt'):
        getattr(agent,k).load_state_dict(copy.deepcopy(saved[k]))
        assert same(getattr(agent,k).state_dict(),saved[k]),k
    assert same(agent.state_dict(),saved['agent'])
    def restore_rng():
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    restore_rng()
    assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
    replay=copy.deepcopy(saved['replay']);reference=copy.deepcopy(saved['replay'])
    assert same(replay.sample(2,8),reference.sample(2,8))
    metrics=agent.update(replay.sample(2,8))
    assert all(np.isfinite(float(v)) for v in metrics.values())
    del agent,replay,reference
    torch.cuda.empty_cache()
    TRAIN.mkdir(exist_ok=False)
    # Renew only allocation metadata, explicitly requested by the user.
    config=copy.deepcopy(saved['config'])
    config.update(started=time.time(),deadline=deployment['deadline']-60,
        resumed_from=str(SOURCE),resumed_step=saved['steps'],
        original_config=copy.deepcopy(saved['config']),original_checkpoint_sha256=originals,
        allocation='Explicit October 10 resume; new eight-hour cap from pod creation; unchanged decision target')
    saved['config']=config
    torch.save(saved,TRAIN/'latest.pt');shutil.copy2(best,TRAIN/'best.pt')
    (TRAIN/'config.json').write_text(json.dumps(config,indent=2))
    reloaded=torch.load(TRAIN/'latest.pt',map_location='cpu',weights_only=False)
    for k in ('agent','critic_opt','actor_opt','alpha_opt','steps','updates','episodes','next_eval','best','torch_rng','cuda_rng','numpy_rng','python_rng'):
        assert same(saved[k],reloaded[k]),k
    assert saved['replay'].rng.getstate()==reloaded['replay'].rng.getstate()
    assert saved['replay'].transitions==reloaded['replay'].transitions
    assert len(saved['replay'].episodes)==len(reloaded['replay'].episodes)
    assert all(same(a,b) for a,b in zip(saved['replay'].episodes,reloaded['replay'].episodes))
    receipt=dict(time=time.time(),steps=saved['steps'],updates=saved['updates'],episodes=saved['episodes'],
        optimizer_states={k:len(saved[k]['state']) for k in ('critic_opt','actor_opt','alpha_opt')},
        replay_transitions=saved['replay'].transitions,exact_model_optimizer_replay_rng_restore=True,
        environment_exact=True,source_hashes_verified=True,source_files_verified=len(manifest),original_hashes=originals,
        disposable_cuda_update=metrics,disposable_weights_discarded=True,next_eval=saved['next_eval'],
        best=saved['best'],deadline=config['deadline'],torch=torch.__version__)
    (TRAIN/'resume_prepared.json').write_text(json.dumps(receipt,indent=2))
    assert all(digest(SOURCE/(n+'.pt'))==sha for n,sha in originals.items())
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
