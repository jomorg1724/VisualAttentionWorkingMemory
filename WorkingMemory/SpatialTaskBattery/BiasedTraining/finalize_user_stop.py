"""Freeze already-stopped cloud artifacts at the user's explicit request."""
import json, subprocess, shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'cloud_provisioning.json').read_text())
code=r'''
import pathlib,json,csv,time,hashlib,shutil,subprocess
r=pathlib.Path('/workspace/vawm_unbiased/biased_results')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert not gpu.strip(),gpu
rows=list(csv.DictReader((r/'spatial_biased/training/metrics.csv').open()))
last=rows[-1]
index=[json.loads(x) for x in (r/'spatial_biased/training/checkpoint_index.jsonl').read_text().splitlines()]
durable=max(index,key=lambda x:x['step'])
p=r/'spatial_biased/training'/durable['file']
assert hashlib.sha256(p.read_bytes()).hexdigest()==durable['sha256']
receipt=dict(status='stopped_by_user',reason='User requested immediate retrieval and pod termination before sleep; no further training or evaluation.',stopped_unix=time.time(),supervisor_pid=20111,worker_pid=55057,stop_order='STOP supervisor, KILL worker, KILL supervisor',gpu_processes=gpu,logged_global_step=int(last['step']),logged_added_updates=int(last['step'])-8400,logged_added_episodes=int(last['episodes']),durable_global_step=durable['step'],durable_added_updates=durable['step']-8400,durable_added_episodes=(durable['step']-8400)*40,durable_checkpoint=str(p),durable_sha256=durable['sha256'],uncheckpointed_logged_updates=int(last['step'])-durable['step'],planned_updates=4000,planned_episodes=160000,final_heldout_evaluation_performed=False)
def write(p,v):p.write_text(json.dumps(v,indent=2))
for name in ['aggregate','budget']:
 p=r/(name+'.json')
 if p.exists():
  shutil.copy2(p,r/(name+'_before_user_stop.json'))
  v=json.loads(p.read_text());v['status']='stopped_by_user';v['user_stop']=receipt
  if name=='aggregate':
   arm=v['runs']['spatial_biased'];arm['status']='stopped_by_user';arm['planned_added_episodes']=160000;arm['added_episodes']=int(last['episodes']);arm['user_stop']=receipt
  else:v['active']=None
  write(p,v)
shutil.copy2('/workspace/vawm_unbiased/biased_supervisor.log',r/'supervisor_at_user_stop.log')
write(r/'user_stop_receipt.json',receipt)
write(r/'exit.json',receipt)
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(r.rglob('*')) if p.is_file() and p.name!='artifact_index.json'}
write(r/'artifact_index.json',files)
print(json.dumps(dict(receipt=receipt,files=len(files))))
'''
args=['ssh','-n','-i',cfg['ssh_identity'],'-o','UserKnownHostsFile='+cfg['known_hosts'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-p',str(cfg['ssh_port']),'root@'+cfg['ssh_host'],'python -c '+shlex.quote(code)]
result=subprocess.run(args,capture_output=True,text=True,timeout=90,check=True)
data=json.loads(result.stdout)
(HERE/'user_stop_receipt.json').write_text(json.dumps(data,indent=2))
cfg['status']='stopped_by_user_retrieving';cfg['user_stop']=data['receipt']
(HERE/'cloud_provisioning.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps(data,indent=2))
