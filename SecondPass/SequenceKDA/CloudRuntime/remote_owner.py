"""One-shot guarded launch ON a newly created pod; never rents/restarts compute."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/vawm_sequence_kda01')
MODULE='SecondPass.SequenceKDA.worker'

def read(path): return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def validate_contract(b,q,now):
    if not all(isinstance(b.get(k),(int,float)) and math.isfinite(b[k]) for k in ('cap_started','deadline','hard_deadline')): raise ValueError('Bad clock')
    if b.get('wall_cap_seconds')!=28800 or b['hard_deadline']-b['cap_started']!=28800 or b['hard_deadline']-b['deadline']!=600 or b.get('retrieval_reserve_seconds')!=600 or b.get('max_usd')!=5: raise ValueError('Immutable8h/$5 creation cap required')
    if not b['cap_started']<=now<b['deadline']: raise ValueError('Expired/future cap')
    if q.get('gpu')!='NVIDIA A40' or q.get('gpu_count')!=1 or q.get('verified_live_rates') is not True: raise ValueError('Live singleA40 quote required')
    if not 0<=b['cap_started']-q['observed']<=900: raise ValueError('Quote was not current at creation')
    if not all(isinstance(q.get(k),(int,float)) and math.isfinite(q[k]) and q[k]>0 for k in ('compute_hourly_usd','storage_reserve_usd')): raise ValueError('Compute+storage reserve required')
    if q['compute_hourly_usd']>.49 or 8*q['compute_hourly_usd']+q['storage_reserve_usd']>5: raise ValueError('Compute plus storage exceeds$5')

def claim(run):
    run.mkdir(parents=True,exist_ok=True)
    with (run/'owner_claim.json').open('x') as f: json.dump(dict(pid=os.getpid(),observed=time.time(),restart=False),f)

def preflight(root):
    b=read(root/'assets/budget.json');q=read(root/'assets/pricing.json');validate_contract(b,q,time.time())
    g=read(root/'billing_guard.json')
    if not(g.get('state')=='armed' and g.get('authenticated_read') and g.get('publication_verified') and time.time()-g['updated']<60 and g['deadline']==b['hard_deadline']): raise ValueError('Independent authenticated/publishing guard not ready')
    os.kill(g['pid'],0)
    if g['pod']!=os.environ.get('RUNPOD_POD_ID'): raise ValueError('Wrong pod guard')
    m=read(root/'deployment_manifest.json')
    if m.get('checkpoints_included') is not False or m.get('budget_included') is not False: raise ValueError('Not a fresh bundle')
    for r in m['files']:
        p=root/r['path']
        if not p.resolve().is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']: raise ValueError('Source mismatch')
    # Secret stop authority is owned by guard only, never passed to workers.
    if 'VAWM_STOP_API_KEY' in os.environ: raise ValueError('Remove guard credential from worker environment')

def command(root,*args):
    write(root/'run'/'owner_stage.json',dict(stage=args[0],observed=time.time()))
    print('STAGE '+args[0],flush=True)
    subprocess.run([sys.executable,'-u','-m',MODULE,*args],cwd=root/'repo',check=True,
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))

def finalize(root):
    run=root/'run';p=run/'cloud_completion.json';complete=read(p)
    if complete['status']!='complete': raise ValueError('Incomplete scientific run')
    result=read(run/'supervisor_result.json')
    if result['returncode']!=0: raise ValueError('Supervisor failed')
    # Called only after subprocess.run reaps the supervisor, which reaped worker.
    for pid in (result['worker_pid'],result['supervisor_pid']):
        try: os.kill(pid,0)
        except ProcessLookupError: pass
        else: raise ValueError('Worker/supervisor still alive')
    write(run/'owner_run.json',dict(status='complete',pid=os.getpid(),observed=time.time(),no_restart=True))
    rows={r['relative_path']:r for r in complete['artifact_manifest']}
    for name in ('supervisor_result.json','run_supervisor_result.json','owner_run.json','owner_claim.json'):
        f=run/name
        if f.is_file(): rows[name]=dict(path=str(f),relative_path=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    complete['artifact_manifest']=list(rows.values());write(p,complete)
    # Guard verifies hashes and this owner's actual exit before publishing/stop.

def execute(root=ROOT):
    run=root/'run'
    claim(run)  # Duplicate invocation never marks a healthy run as failed.
    try:
        preflight(root)
        command(root,'prepare',str(run),'--budget',str(root/'assets/budget.json'))
        command(root,'supervise-profile',str(run))
        command(root,'pin',str(run))
        command(root,'supervise-run',str(run))
        finalize(root)
        return 0
    except BaseException as exc:
        import traceback
        from pod_guard import clean
        write(run/'owner_error.json',dict(type=type(exc).__name__,message=clean(str(exc)),stage=read(run/'owner_stage.json') if (run/'owner_stage.json').exists() else 'preflight',frames=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)]))
        write(run/'owner_run.json',dict(status='failed',pid=os.getpid(),error_type=type(exc).__name__,observed=time.time()))
        write(run/'DEPLOYMENT_FAILED',dict(stop=True,reason='setup_or_worker_failure',observed=time.time()))
        return 1

if __name__=='__main__': raise SystemExit(execute())
