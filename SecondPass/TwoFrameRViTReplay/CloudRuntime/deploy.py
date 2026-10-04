#!/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python
"""Parent-only guarded deployment, adapted from cloud_spatial_consolidation_01.
Import/check are offline. Only explicit `deploy --use-existing-guard-credential`
may provision. No old-pod credential lookup, restart, delete, or training resume.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import tomllib
import traceback
from pod_guard import clean

_SECRETS=()
from urllib.parse import quote

ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/vawm_two_frame_rvit_01'
KEY=Path('/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_01/ssh_key')
IMAGE='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
def frozen_archive_sha():
    return json.loads((ROOT/'runtime_manifest.json').read_text())['approved_archive_sha256']
MODULE='SecondPass.TwoFrameRViTReplay'


def api(path,method='GET',body=None):
    # Qualified file import: never shadow active guard/owner modules.
    spec=importlib.util.spec_from_file_location('prior_control_transport',Path('/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/control.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.api(path,method,body)


def save(name,value):
    path=ROOT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.chmod(tmp,0o600);tmp.replace(path)


AUTHORIZED_PARALLEL_POD='pa0ko8f2qirisy'
def require_no_parallel(pods):
    # User explicitly authorizes THIS RViT beside the existing16-head KDA.
    for p in pods:
        if p.get('status') in ('EXITED','STOPPED','TERMINATED'): continue
        if p.get('id')==AUTHORIZED_PARALLEL_POD and p.get('status')=='RUNNING': continue
        raise ValueError('Unrelated active or unknown-lifecycle pod; rental not authorized')


def list_pods():
    pods=[];cursor=None
    while True:
        suffix='?includeClusterPods=true&limit=1000'+('&cursor='+quote(cursor,safe='') if cursor else '')
        page=api('/v2/pods'+suffix)
        pods.extend(page['pods'])
        pagination=page.get('pagination',{})
        if not pagination.get('hasNextPage'): return pods
        new=pagination['nextCursor']
        if not new or new==cursor: raise ValueError('Invalid pod pagination')
        cursor=new


def price_contract(gpu,reserve,retention,observed):
    rate=gpu['price']['secure']
    if gpu['id']!='NVIDIA A40' or gpu.get('availability') not in ('LOW','MEDIUM','HIGH'): raise ValueError('A40 unavailable')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (rate,reserve,retention)): raise ValueError('Invalid price')
    # Published storage rates, conservative 28-day month: 30GB container +20GB
    # persistent while running; 20GB persistent after stop. Reserve is explicit.
    minimum=50*0.10*8/(28*24)+20*0.20*retention/(28*24)
    if rate>.49 or reserve<minimum or 8*rate+reserve>5: raise ValueError('Compute+storage exceeds $5 or reserve insufficient')
    return dict(gpu='NVIDIA A40',gpu_count=1,compute_hourly_usd=rate,
        storage_reserve_usd=reserve,retention_hours=retention,storage_estimate_usd=minimum,
        storage_rates_source='https://docs.runpod.io/pods/pricing',running_gb_month_usd=.10,
        stopped_gb_month_usd=.20,verified_live_rates=True,observed=observed)


def live_quote(reserve,retention):
    gpu=api('/v2/catalog/gpus/NVIDIA%20A40?include=AVAILABILITY&product=POD&count=1&cloud=SECURE&minCudaVersion=12.8')
    q=price_contract(gpu,reserve,retention,time.time())
    save('pricing_approval.json',q);return q


def check():
    receipt=json.loads((ROOT/'package/bundle_receipt.json').read_text())
    archive=Path(receipt['archive'])
    if receipt['sha256']!=frozen_archive_sha() or hashlib.sha256(archive.read_bytes()).hexdigest()!=frozen_archive_sha():
        raise ValueError('Not the approved immutable archive')
    with tarfile.open(archive) as tf:
        members=tf.getmembers();manifest=json.load(tf.extractfile('deployment_manifest.json'))
        if len({m.name for m in members})!=len(members) or len({r['path'] for r in manifest['files']})!=len(manifest['files']): raise ValueError('Duplicate archive paths')
        if len(members)!=len(manifest['files'])+1 or manifest.get('checkpoints_included') is not False: raise ValueError('Not fresh')
        for m in members:
            if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts or any(c in m.name for c in ('\n','\r')) or Path(m.name).suffix in ('.pt','.pth','.ckpt','.pem','.key','.safetensors'): raise ValueError('Unsafe archive')
        for row in manifest['files']:
            data=tf.extractfile(row['path']).read()
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('Manifest mismatch')
    manifest=json.loads((ROOT/'runtime_manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Runtime changed: '+name)
    for name,digest in manifest['inherited_source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest: raise ValueError('Inherited transport changed')
    runtime=json.loads((ROOT/'runtime_bundle_receipt.json').read_text())
    runtime_archive=Path(runtime['archive'])
    if hashlib.sha256(runtime_archive.read_bytes()).hexdigest()!=runtime['sha256']: raise ValueError('Runtime archive changed')
    if not KEY.is_file() or not KEY.with_suffix('.pub').is_file(): raise ValueError('Existing SSH keypair absent')
    return archive,runtime_archive,runtime['sha256']


def create_status():
    if (ROOT/'status_record_verified.json').exists():
        receipt=json.loads((ROOT/'status_record_verified.json').read_text())
        fetched=api('/v2/templates/'+receipt['id'])
        if fetched.get('public') is not False: raise ValueError('Status record not private')
        return receipt
    status=dict(state='prepared_not_launched',observed=time.time(),attempt=ROOT.name)
    body=dict(name='vawm-two-frame-rvit-01-status-DO-NOT-DEPLOY',image=IMAGE,disk=1,ports=[],
        env={'VAWM_PUBLIC_STATUS':json.dumps(status)},public=False,serverless=False,
        startSsh=False,startJupyter=False,category='NVIDIA')
    created=api('/v2/templates','POST',body);ident=created['id']
    save('status_record.json',dict(id=ident,purpose='private status only; never deploy'))
    fetched=api('/v2/templates/'+ident)
    if fetched['public'] is not False or fetched['env']!=body['env']: raise ValueError('Status create readback failed')
    status['publication_test_verified']=True
    env={'VAWM_PUBLIC_STATUS':json.dumps(status)}
    api('/v2/templates/'+ident,'PATCH',dict(public=False,env=env))
    fetched=api('/v2/templates/'+ident)
    if fetched['public'] is not False or fetched['env']!=env: raise ValueError('Status patch readback failed')
    receipt=dict(id=ident,verified=True,observed=time.time());save('status_record_verified.json',receipt);return receipt


def ssh_command(endpoint=None):
    e=endpoint or json.loads((ROOT/'ssh_endpoint.json').read_text())
    return ['ssh','-i',str(KEY),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+str(ROOT/'known_hosts'),'-o','ConnectTimeout=8',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-p',str(e['port']),'root@'+e['ip']]


def remote(command,timeout=120,script=None):
    # Source image runtime, then explicitly scrub all provider credentials.
    wrapped='if [ -f /etc/rp_environment ]; then source /etc/rp_environment; fi; unset VAWM_STOP_API_KEY RUNPOD_API_KEY; '+command
    result=subprocess.run(ssh_command()+['bash -lc '+shlex.quote(wrapped)],input=script,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        exc=RuntimeError('Remote exit='+str(result.returncode)+' '+result.stderr[-3000:]+' '+result.stdout[-1500:])
        save('remote_failure.json',diagnostic(exc));raise exc
    return result.stdout


def remote_python(code,timeout=120): return remote('python3 -',timeout,code)


def wait_ssh(ident):
    end=time.time()+240
    while time.time()<end:
        p=api('/pods/'+ident);ip=p.get('publicIp');port=p.get('portMappings',{}).get('22')
        if ip and port:
            e=dict(ip=ip,port=port,pod=ident)
            try:
                result=subprocess.run(ssh_command(e)+['true'],capture_output=True,text=True,timeout=15)
            except subprocess.TimeoutExpired:
                time.sleep(5)
                continue
            if result.returncode==0: save('ssh_endpoint.json',e);return
            if 'Permission denied' in result.stderr or 'IDENTIFICATION HAS CHANGED' in result.stderr: raise ValueError('SSH identity/authentication failed')
        time.sleep(5)
    raise TimeoutError('SSH setup exceeded240s')


def stop_and_verify(ident):
    # Stop, never delete: preserve checkpoint-bearing persistent disk.
    for _ in range(6):
        try: api('/v2/pods/'+ident+'/action','POST',{'action':'stop'})
        except Exception: pass
        p=api('/v2/pods/'+ident)
        save('deployment_stop.json',dict(id=ident,status=p['status'],observed=time.time(),disk_retained=True))
        if p['status'] in ('EXITED','STOPPED'): return
        time.sleep(5)
    raise RuntimeError('Stop not yet verified; independent pod guard remains active')


def verify_guard(budget,record):
    g=json.loads(remote_python('from pathlib import Path;print(Path('+repr(REMOTE+'/billing_guard.json')+').read_text())'))
    if not(g['state']=='armed' and g['authenticated_read'] and g['publication_verified'] and g['deadline']==budget['hard_deadline'] and time.time()-g['updated']<60): raise ValueError('Guard not verified')
    remote_python('import os;os.kill('+str(g['pid'])+',0)')
    p=api('/v2/templates/'+record['id']);s=json.loads(p['env']['VAWM_PUBLIC_STATUS'])
    if p['public'] is not False or s['guard']['deadline']!=budget['hard_deadline'] or s['guard']['state']!='armed': raise ValueError('Status publication unverified')
    save('guard_verified.json',g);save('provider_status_verified.json',s)


def diagnostic(exc):
    return dict(type=type(exc).__name__,message=clean(str(exc),_SECRETS),
        frames=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(exc.__traceback__)])


def verify_boot(pod,body):
    # RunPod omits an empty cmd in GET. Only that documented empty default is allowed.
    if pod.get('entrypoint')!=body['entrypoint'] or pod.get('cmd',[])!=body['cmd']:
        raise ValueError('Guard command readback mismatch')
    for field in ('VAWM_HARD_DEADLINE','VAWM_STATUS_TEMPLATE_ID'):
        if pod.get('env',{}).get(field)!=body['env'][field]: raise ValueError('Guard environment readback mismatch')
    return True


def snapshot():
    return json.loads(remote_python('''import json,os
from pathlib import Path
root=Path(ROOT_VALUE);r=root/'run';out={}
for n in ('DEPLOYMENT_FAILED','owner_run.json','owner_stage.json','owner_error.json','latest_checkpoint.json','persisted_progress_verification.json','first_cycle_timing.json','live_status.json','owner_claim.json','run_supervisor.json','profile/profile_supervisor.json'):
 p=r/n
 if p.exists(): out[n]=json.loads(p.read_text())
p=r/'progress.jsonl'
if p.exists():
 lines=p.read_text().splitlines()
 for line in reversed(lines[-2:]):
  try: out['progress']=json.loads(line);break
  except ValueError: pass
for n in ('owner_claim.json','live_status.json'):
 pid=out.get(n,{}).get('pid')
 if pid:
  try: os.kill(pid,0);out[n+'_alive']=True
  except ProcessLookupError: out[n+'_alive']=False
p=root/'owner.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-12000));out['owner_log_tail']=f.read().decode(errors='replace').splitlines()[-30:]
print(json.dumps(out))
'''.replace('ROOT_VALUE',repr(REMOTE)),timeout=25))


def capture_diagnostics():
    try:
        data=snapshot()
        # Preserve bounded owner logs and receipts, never credential-bearing env.
        save('failure_snapshot.json',clean(data,_SECRETS))
    except Exception as exc: save('diagnostic_capture_error.json',diagnostic(exc))


def reconcile_verification(data):
    cp=data.get('latest_checkpoint.json',{})
    v=data.get('persisted_progress_verification.json',{})
    if cp.get('step',0)<3: return False
    if v.get('verified') is True and v.get('checkpoint')==cp and v.get('step')==cp['step']: return False
    # Receipt lag is not a failed optimizer. Native verifier checks digest, initial
    # construction, full named Adam, scheduler/stream exposure and module changes.
    result=json.loads(remote('cd '+REMOTE+'/repo && CUDA_VISIBLE_DEVICES="" python3 -m '+MODULE+'.worker verify '+REMOTE+'/run',timeout=120))
    if result.get('verified') is not True: raise ValueError('Native checkpoint verification failed')
    # Verifier can observe a newer atomic checkpoint than this poll; bind both.
    data['persisted_progress_verification.json']=result
    data['latest_checkpoint.json']=result['checkpoint']
    return True


def download_checkpoint(data):
    cp=data['latest_checkpoint.json'];path=Path(cp['path'])
    if not path.is_relative_to(Path(REMOTE)/'run') or '..' in path.parts: raise ValueError('Wrong checkpoint path')
    dest=ROOT/'retrieved_early'/path.name;dest.parent.mkdir(exist_ok=True)
    ssh=ssh_command()
    subprocess.run(['rsync','-rt','--timeout=60','-e',shlex.join(ssh[:-1]),ssh[-1]+':'+str(path),str(dest)],check=True,capture_output=True,text=True,timeout=180)
    if dest.stat().st_size!=cp['bytes'] or hashlib.sha256(dest.read_bytes()).hexdigest()!=cp['sha256']: raise ValueError('Downloaded checkpoint integrity failed')
    save('downloaded_checkpoint_verified.json',verify_checkpoint_file(dest,cp))

def verify_checkpoint_file(dest,cp):
    if dest.stat().st_size!=cp['bytes'] or hashlib.sha256(dest.read_bytes()).hexdigest()!=cp['sha256']: raise ValueError('Checkpoint file integrity failed')
    import torch
    torch.set_num_threads(2)
    saved=torch.load(dest,map_location='cpu',weights_only=False)
    if saved['state']['step']!=cp['step'] or not saved['stream']['streams'] or saved['state']['optimizer_seconds']<=0: raise ValueError('Invalid persisted progress')
    if saved['scheduler']['updates']!=cp['step'] or not saved['optimizer_names'] or not saved['rng']: raise ValueError('Missing full state')
    if saved['state']['episodes'] <= 0 or saved['state']['episodes'] > 32*cp['step']: raise ValueError('Wrong replay presentation exposure')
    for tensor in saved['model'].values():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all(): raise ValueError('Nonfinite model state')
    active=saved['optimizer']['state']
    if not active or not saved['model']: raise ValueError('Missing model/Adam')
    names=saved['optimizer_names']
    flattened=names if isinstance(names,list) and all(isinstance(n,str) for n in names) else [n for group in names for n in group]
    if len(set(flattened))!=len(flattened): raise ValueError('Duplicate optimizer parameter names')
    param_ids=[i for group in saved['optimizer']['param_groups'] for i in group['params']]
    if len(active)!=len(param_ids) or set(active)!=set(param_ids) or len(flattened)!=len(param_ids): raise ValueError('Incomplete named Adam coverage')
    for opt in active.values():
        if float(opt['step'])!=cp['step'] or any(not torch.isfinite(opt[k]).all() for k in ('exp_avg','exp_avg_sq')): raise ValueError('Invalid Adam progress')
    return dict(verified=True,step=cp['step'],sha256=cp['sha256'],bytes=cp['bytes'],active_adam_states=len(active),path=str(dest))


def production_ready(data):
    cp=data.get('latest_checkpoint.json',{});v=data.get('persisted_progress_verification.json',{})
    timing=data.get('first_cycle_timing.json',{})
    return (cp.get('step',0)>=3 and data.get('progress',{}).get('step',0)>=3
        and v.get('verified') is True and v.get('step',0)==cp.get('step')
        and v.get('checkpoint')==cp and bool(cp.get('sha256'))
        and isinstance(timing.get('material_slowdown'),bool)
        and isinstance(timing.get('matched_profile_ratio'),(int,float))
        and math.isfinite(timing['matched_profile_ratio']) and timing['matched_profile_ratio']>0)


def start_mirror():
    # Start detached before profile/production; survives parent tool disconnection.
    with (ROOT/'mirror_launch_claim.json').open('x') as f:
        json.dump(dict(observed=time.time()),f)
    log=(ROOT/'mirror.log').open('ab')
    proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'mirror_watch.py')],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    save('mirror_started.json',dict(pid=proc.pid,observed=time.time(),interval_seconds=15))


def install_and_launch(ident,budget,pricing,record,archive,runtime_archive,runtime_sha):
    wait_ssh(ident);verify_guard(budget,record)
    for path,name,digest in ((archive,'fresh_bundle.tar.gz',frozen_archive_sha()),(runtime_archive,'runtime_sources.tar.gz',runtime_sha)):
        ssh=ssh_command()
        result=subprocess.run(['rsync','--partial','--inplace','--timeout=60','-e',shlex.join(ssh[:-1]),str(path),ssh[-1]+':/workspace/'+name],capture_output=True,text=True,timeout=300)
        if result.returncode: raise RuntimeError('Immutable archive transfer failed')
        remote_python('import hashlib;from pathlib import Path;assert hashlib.sha256(Path('+repr('/workspace/'+name)+').read_bytes()).hexdigest()=='+repr(digest))
    remote('mkdir -p '+REMOTE+' && tar --no-same-owner -xzf /workspace/fresh_bundle.tar.gz -C '+REMOTE+' && tar --no-same-owner -xzf /workspace/runtime_sources.tar.gz -C '+REMOTE)
    code='''import json,hashlib
from pathlib import Path
r=Path(ROOT_VALUE)
m=json.loads((r/'deployment_manifest.json').read_text())
for row in m['files']:
 p=r/row['path'];assert p.resolve().is_relative_to(r) and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
m=json.loads((r/'runtime_manifest.json').read_text())
for name,h in m['source_hashes'].items(): assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h
(r/'assets').mkdir(exist_ok=True)
for name,value in [('budget.json',BUDGET_VALUE),('pricing.json',PRICING_VALUE)]:
 with (r/'assets'/name).open('x') as f: json.dump(value,f)
'''.replace('ROOT_VALUE',repr(REMOTE)).replace('BUDGET_VALUE',repr(budget)).replace('PRICING_VALUE',repr(pricing))
    remote_python(code)
    remote('python3 -m pip install --break-system-packages --disable-pip-version-check -r '+REMOTE+'/requirements.txt',180)
    # Native CPU engineering checks already passed before rental. The actual
    # CUDA profile below exercises imports, forward/backward and persistence;
    # do not duplicate the full CPU suite on paid compute.
    save('remote_cpu_tests.json',dict(repeated=False,
        reason='User requested immediate launch; verified local native CPU evidence retained',
        observed=time.time()))
    start_mirror()
    code='''import os,subprocess,json
from pathlib import Path
r=Path(ROOT_VALUE)
env=dict(os.environ)
for k in ('VAWM_STOP_API_KEY','RUNPOD_API_KEY'): env.pop(k,None)
log=(r/'owner.log').open('ab')
p=subprocess.Popen(['python3','-u',str(r/'remote_owner.py')],cwd=r/'repo',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'observed':__import__('time').time()}))
'''.replace('ROOT_VALUE',repr(REMOTE))
    result=json.loads(remote_python(code));save('launch.json',dict(**result,pod=ident,phase='automatic_prepare_profile_pin_production',budget=budget))
    # Bounded parent handshake: first persisted production update, never profile.
    end=min(time.time()+1200,budget['deadline']-600)
    while time.time()<end:
        if api('/v2/pods/'+ident)['status']!='RUNNING': raise RuntimeError('Pod stopped before launch verification')
        data=snapshot()
        save('handshake_snapshot.json',clean(data,_SECRETS))
        if data.get('DEPLOYMENT_FAILED',{}).get('stop') or data.get('owner_run.json',{}).get('status')=='failed': raise RuntimeError('Remote setup/worker failed')
        reconcile_verification(data)
        if production_ready(data):
            timing=data['first_cycle_timing.json']
            if timing['material_slowdown']:
                warning=dict(step=data['latest_checkpoint.json']['step'],matched_profile_ratio=timing['matched_profile_ratio'],
                    material_slowdown=True,observed=time.time(),action='continue healthy production within unchanged deadline/reserve; report actual exposure')
                save('production_timing_warning.json',warning)
                data['production_timing_warning']=warning
            remote_python('import hashlib,json;from pathlib import Path;r=Path('+repr(REMOTE)+')/"run";c=json.loads((r/"latest_checkpoint.json").read_text());p=Path(c["path"]);assert p.resolve().is_relative_to(r) and p.stat().st_size==c["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==c["sha256"]')
            download_checkpoint(data)
            verify_guard(budget,record);save('production_verified.json',data);return data
        time.sleep(10)
    capture_diagnostics()
    raise TimeoutError('Bounded launch handshake unresolved; see failure_snapshot.json for worker/progress versus receipt failure')


def deploy(reserve,retention,approved=False):
    global _SECRETS
    if not approved: raise ValueError('Explicit existing-credential approval flag required')
    archive,runtime_archive,runtime_sha=check()
    # Exclusive attempt prevents retries after ambiguous create from duplicating.
    with (ROOT/'deploy_claim.json').open('x') as f: json.dump(dict(pid=os.getpid(),observed=time.time()),f)
    if (ROOT/'pod.json').exists(): raise ValueError('Pod receipt exists; no duplicate')
    require_no_parallel(list_pods());pricing=live_quote(reserve,retention)
    record=create_status()
    require_no_parallel(list_pods())
    # Key resolved ONLY in memory from the known working local backend.
    key=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['default']['api_key']
    _SECRETS=(key,)
    now=time.time();budget=dict(cap_started=now,hard_deadline=now+28800,deadline=now+28200,
        wall_cap_seconds=28800,retrieval_reserve_seconds=600,max_usd=5,
        origin='Explicit NEW8h/$5 two-frame visual-query RViT whole-model fresh; no old cap/checkpoint reuse',
        retained_storage_cleanup_due=now+28800+retention*3600)
    if now-pricing['observed']>900: raise ValueError('Live quote expired')
    boot='import base64;exec(compile(base64.b64decode('+repr(base64.b64encode((ROOT/'pod_guard.py').read_bytes()).decode())+'),"pod_guard.py","exec"))'
    body=dict(name='vawm-two-frame-rvit-01',cloud='SECURE',gpu=dict(id='NVIDIA A40',count=1,minCudaVersion='12.8',minRamPerGpu=24,minVcpuCountPerGpu=4),image=IMAGE,disk=30,
        mounts=dict(persistent=dict(size=20,path='/workspace')),ports=['22/tcp'],entrypoint=['python3','-u','-c',boot],cmd=[],startSsh=True,startJupyter=False,globalNetworking=False,
        env=dict(PUBLIC_KEY=KEY.with_suffix('.pub').read_text().strip(),VAWM_STOP_API_KEY=key,VAWM_HARD_DEADLINE=str(budget['hard_deadline']),VAWM_STATUS_TEMPLATE_ID=record['id'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    ident=None;success=False
    try:
        pod=api('/v2/pods','POST',body);ident=pod['id']
        body['env'].pop('VAWM_STOP_API_KEY',None);key=None
        save('pod.json',dict(id=ident,budget=budget,hard_deadline=budget['hard_deadline'],attempt_started=now,authorized_max_usd=5,bundle_sha256=frozen_archive_sha(),remote_root=REMOTE))
        save('budget.json',budget)
        p=api('/v2/pods/'+ident)
        if p['cost']>pricing['compute_hourly_usd'] or 8*p['cost']+reserve>5: raise ValueError('Provisioned rate exceeds quote/budget')
        verify_boot(p,body)
        if p['gpu']['id']!='NVIDIA A40' or p['gpu']['count']!=1: raise ValueError('Provisioned hardware mismatch')
        save('provisioned.json',{k:p[k] for k in ('id','cost','status','gpu')})
        result=install_and_launch(ident,budget,pricing,record,archive,runtime_archive,runtime_sha)
        success=True;return dict(pod=ident,phase='verified_production',hard_deadline=budget['hard_deadline'],evidence='production_verified.json')
    except BaseException as exc:
        save('deployment_error.json',dict(**diagnostic(exc),observed=time.time()))
        if ident: capture_diagnostics()
        if ident is None:
            # A create timeout may still have allocated. Never retry creation.
            # Match only this uniquely named attempt and exact guarded deadline.
            for candidate in list_pods():
                if candidate.get('name')==body['name']:
                    current=api('/v2/pods/'+candidate['id'])
                    if current.get('env',{}).get('VAWM_HARD_DEADLINE')==str(budget['hard_deadline']):
                        ident=current['id']
                        save('pod.json',dict(id=ident,budget=budget,hard_deadline=budget['hard_deadline'],remote_root=REMOTE,ambiguous_create_recovered=True))
                        break
        raise
    finally:
        body['env'].pop('VAWM_STOP_API_KEY',None);key=None
        if ident and not success:
            try: stop_and_verify(ident)
            except Exception as cleanup: save('cleanup_error.json',diagnostic(cleanup))
        _SECRETS=()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['check','inspect','deploy','status','retrieve'])
    p.add_argument('--use-existing-guard-credential',action='store_true')
    p.add_argument('--storage-reserve-usd',type=float,default=1.0)
    p.add_argument('--retention-hours',type=float,default=24)
    args=p.parse_args()
    if args.mode=='check':
        a,r,h=check();print(json.dumps(dict(archive=str(a),sha256=frozen_archive_sha(),runtime_sha256=h,no_cloud_calls=True)));return
    if args.mode=='inspect':
        check();pods=list_pods();q=live_quote(args.storage_reserve_usd,args.retention_hours)
        print(json.dumps(dict(pods=[{k:v for k,v in x.items() if k in ('id','name','status','cost')} for x in pods],quote=q),indent=2));require_no_parallel(pods);return
    if args.mode=='deploy': print(json.dumps(deploy(args.storage_reserve_usd,args.retention_hours,args.use_existing_guard_credential)));return
    import monitor
    if args.mode=='status': print(json.dumps(monitor.status(),indent=2))
    else:
        import retrieve_once
        retrieve_once.main()

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if isinstance(exc,SystemExit): raise
        if not (ROOT/'deployment_error.json').exists():
            save('deployment_error.json',dict(**diagnostic(exc),observed=time.time()))
        print('Failed: '+type(exc).__name__+' (see deployment_error.json)',file=sys.stderr);raise SystemExit(1)
