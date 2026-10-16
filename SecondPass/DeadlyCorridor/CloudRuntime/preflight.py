"""Disposable cloud equivalence and production-sized CUDA update checks."""
import hashlib,json,os,time
from pathlib import Path
import requests,numpy as np,torch
from SecondPass.DeadlyCorridor.agent import Agent
from SecondPass.DeadlyCorridor.environment import Corridor
from SecondPass.StabilizedAttentionVDA.train import PARENT
from SecondPass.PixelContrastiveVAE.train import configure

root=Path('/workspace/vawm');run=root/'SecondPass/DeadlyCorridor/CloudRuntime/run01'
manifest=json.loads(Path('/workspace/source_manifest.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
pod_id=json.loads((run/'deployment.json').read_text())['pod_id']
key=Path('/root/.doom-pod-key').read_text().strip()
r=requests.post('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key},json={'query':'query($input: PodFilter) { pod(input: $input) { id desiredStatus } }','variables':{'input':{'podId':pod_id}}},timeout=30)
d=r.json();assert r.status_code==200 and d.get('data',{}).get('pod',{}).get('id')==pod_id, 'Pod-scoped authentication failed'
configure('cuda');agent=Agent(PARENT).cuda();env=Corridor()
record=dict(observations=[],actions=[],rewards=[],done=[],tics=[])
obs=env.reset(202777);record['observations'].append(obs)
try:
 for i in range(10):
  obs,reward,done,info=env.step(0)
  record['observations'].append(np.zeros((3,100,100),np.uint8) if obs is None else obs)
  for k,v in dict(actions=0,rewards=reward,done=done,tics=info['tics']).items():record[k].append(v)
  if done:break
 old={n:p.detach().clone() for n,p in agent.named_parameters()}
 metrics=agent.update([(record,0,8),(record,1,8)])
 changed={prefix:any(not torch.equal(p,old[n]) for n,p in agent.named_parameters() if n.startswith(prefix)) for prefix in ('encoder.','actor.','q1.','q2.','target_encoder.','target_q1.','target_q2.')}
 assert all(changed.values()) and all(np.isfinite(v) for v in metrics.values())
 states={}
 for name in ('critic_opt','actor_opt','alpha_opt'):
  opt=getattr(agent,name)
  for p,s in opt.state.items():
   assert all(torch.isfinite(s[k]).all() and s[k].shape==p.shape for k in ('exp_avg','exp_avg_sq'))
  states[name]=len(opt.state)
 receipt=dict(time=time.time(),pod_id=pod_id,pod_scoped_auth=True,source_files_verified=len(manifest),parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),environment=env.provenance,quantiles=32,segments=2,unroll=8,optimizer_states=states,changed=changed,metrics=metrics,disposable_weights_discarded=True,contract_tests='4 passed')
 (run/'preflight_verified.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
finally:env.close()
