"""Independent CPU reload of a complete episode-boundary production checkpoint."""
import hashlib,json,os,time
from pathlib import Path
import torch
from SecondPass.DeadlyCorridor.agent import Agent

root=Path('/workspace/vawm');run=root/'SecondPass/DeadlyCorridor/CloudRuntime/run01';train=run/'training'
torch.set_num_threads(2)
# Opening pins one atomically published inode even if the trainer replaces it.
with (train/'latest.pt').open('rb') as f:saved=torch.load(f,map_location='cpu',weights_only=False)
assert saved['updates']>0
agent=Agent();agent.load_state_dict(saved['agent']);states={}
for name in ('critic_opt','actor_opt','alpha_opt'):
 opt=getattr(agent,name);opt.load_state_dict(saved[name]);assert opt.state
 for p,s in opt.state.items():
  assert all(s[k].shape==p.shape and torch.isfinite(s[k]).all() for k in ('exp_avg','exp_avg_sq'))
  assert int(s['step'])==saved['updates']
 states[name]=len(opt.state)
assert all(torch.isfinite(p).all() for p in agent.parameters())
parent=Path(saved['config']['parent']);initial=Agent(parent)
changed=sum(not torch.equal(p,dict(initial.encoder.named_parameters())[n]) for n,p in agent.encoder.named_parameters())
assert changed>0
manifest=json.loads(Path('/workspace/source_manifest.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
worker=json.loads((run/'worker.json').read_text());guard=json.loads((run/'guard.json').read_text())
for pid in (worker['pid'],worker['supervisor_pid'],guard['pid']):os.kill(pid,0)
assert guard['status']=='armed'
receipt=dict(time=time.time(),pod_id=guard['pod_id'],worker_pid=worker['pid'],supervisor_pid=worker['supervisor_pid'],guard_pid=guard['pid'],cloud_deadline=guard['deadline'],saved_steps=saved['steps'],saved_updates=saved['updates'],saved_episodes=saved['episodes'],optimizer_states=states,encoder_tensors_changed=changed,all_parameters_finite=True,all_optimizer_states_finite_aligned=True,source_files_verified=len(manifest),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),replay_transitions=saved['replay'].transitions,checkpoint_bytes=(train/'latest.pt').stat().st_size,checkpoint_cpu_reload=True)
(run/'independent_production_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
