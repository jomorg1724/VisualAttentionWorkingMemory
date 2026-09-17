"""Provision one RunPod pod and run the accumulator program (two parallel lanes) with a hard deadline.

Usage (from PowerShell, so ssh/scp are Windows OpenSSH):
  python WorkingMemory/PlainBaseline/cloud/launch.py --hours 6 [--seeds 1,2]
Writes runs/cloud_<stamp>/cloud_provisioning.json, which the watcher and WorkingMemory/cloud_shutdown.py consume.
"""
import argparse,datetime,json,shutil,subprocess,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from WorkingMemory.cloud_shutdown import api_key,provider_request,read,sha,write

DATASET=ROOT/'WorkingMemory/ProspectiveQuery/runs/prospective_20260914_175602/bsds500_runtime.tar.gz'
DATASET_SHA='990e2bd405f6bcc4c940299f437685fc08765dcb56923ab0ab639f5f37c1883e'
SSH_KEY=Path.home()/'.runpod/ssh/runpodctl-ssh-key'
GPU_PREFERENCE=('NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 3090','NVIDIA RTX A5000')
SOURCES=('WorkingMemory/stimuli.py','WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/BatteryAudit/observers.py',
    'WorkingMemory/PlainBaseline/baseline.py','WorkingMemory/PlainBaseline/variants.py','WorkingMemory/PlainBaseline/accum.py','WorkingMemory/PlainBaseline/program.py','WorkingMemory/PlainBaseline/summarize.py',
    'PreAttentiveVision/neuroscience_stimuli.py','PreAttentiveVision/natural_stimuli.py','PreAttentiveVision/decoder.py','PreAttentiveVision/TemporalIntegration/accumulators.py')
LANES={'lane1':'plain,convgru','lane2':'opponent,kda'}

def bundle(out):
    out.mkdir(parents=True);hashes={}
    files=list(SOURCES)+[p for p in ('WorkingMemory/__init__.py','WorkingMemory/SpatialTaskBattery/__init__.py','WorkingMemory/BatteryAudit/__init__.py','WorkingMemory/PlainBaseline/__init__.py',
        'PreAttentiveVision/__init__.py','PreAttentiveVision/TemporalIntegration/__init__.py') if (ROOT/p).is_file()]
    for rel in files:
        dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst);hashes[rel]=sha(dst)
    # Self-check: the bundle must import on its own.
    check=subprocess.run([sys.executable,'-B','-c','import WorkingMemory.PlainBaseline.program,WorkingMemory.PlainBaseline.accum;print("bundle imports ok")'],cwd=str(out),capture_output=True,text=True)
    if check.returncode:raise RuntimeError('bundle self-check failed: '+check.stderr[-800:])
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(out)).replace('\\','/'))
    return hashes,archive

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=6.);ap.add_argument('--seeds',default='1,2');ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--attach',default=None,help='existing run dir whose pod was created but not set up; continue from keyscan');a=ap.parse_args()
    if sha(DATASET)!=DATASET_SHA:raise RuntimeError('dataset bundle hash mismatch')
    key=api_key();public_key=SSH_KEY.with_suffix('.pub').read_text().strip();pod=None
    if a.attach:
        run_root=Path(a.attach);stamp=run_root.name.split('_',1)[1];pod=read(run_root/'pod_create.json')['body'];started=(run_root/'pod_create.json').stat().st_mtime;deadline=started+a.hours*3600
        archive=run_root/'bundle.tar.gz';bundle_sha=sha(archive)
    else:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');run_root=HERE.parent/'runs'/f'cloud_{stamp}';run_root.mkdir(parents=True)
        started=time.time();deadline=started+a.hours*3600
        hashes,archive=bundle(run_root/'bundle');bundle_sha=sha(archive);write(run_root/'bundle_manifest.json',dict(files=hashes,archive_sha256=bundle_sha))
    for gpu in ([] if pod else GPU_PREFERENCE):
        request=dict(name=f'vawm-accum-{stamp}',imageName='runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04',computeType='GPU',cloudType='COMMUNITY',gpuTypeIds=[gpu],gpuTypePriority='custom',gpuCount=1,
            containerDiskInGb=30,volumeInGb=10,volumeMountPath='/workspace',ports=['22/tcp'],supportPublicIp=True,interruptible=False,minRAMPerGPU=16,minVCPUPerGPU=6,env=dict(PUBLIC_KEY=public_key))
        created=provider_request('POST','https://rest.runpod.io/v1/pods',key,request);write(run_root/'pod_create.json',created)
        if created['status'] in (200,201) and isinstance(created['body'],dict) and created['body'].get('id'):pod=created['body'];break
        print('creation failed for',gpu,created['status'],str(created['body'])[:300],flush=True)
    if pod is None:raise RuntimeError('no pod could be created')
    pod_id=pod['id'];print('pod',pod_id,flush=True);ready=None
    for _ in range(120):
        cur=provider_request('GET','https://rest.runpod.io/v1/pods/'+pod_id,key);body=cur['body'] if isinstance(cur['body'],dict) else {}
        if body.get('publicIp') and body.get('portMappings',{}).get('22'):ready=body;break
        time.sleep(10)
    if ready is None:
        provider_request('POST',f'https://rest.runpod.io/v1/pods/{pod_id}/stop',key);provider_request('DELETE','https://rest.runpod.io/v1/pods/'+pod_id,key);raise RuntimeError('pod never exposed SSH; deleted')
    write(run_root/'pod_ready.json',ready);host,port=ready['publicIp'],int(ready['portMappings']['22']);known=run_root/'known_hosts'
    for _ in range(0 if (known.is_file() and known.stat().st_size) else 90):
        scan=subprocess.run(['ssh-keyscan','-p',str(port),'-T','15',host],capture_output=True,text=True)
        keys=[l for l in scan.stdout.splitlines() if l.strip() and not l.startswith('#')]
        if keys:known.write_text(chr(10).join(keys)+chr(10));break
        time.sleep(10)
    else:
        if not (known.is_file() and known.stat().st_size):raise RuntimeError('ssh-keyscan failed (Windows ssh-keyscan returned nothing; create known_hosts with Git ssh-keyscan and re-attach)')
    config=dict(status='provisioned',account_email='jonathan@palladio.ai',pod_id=pod_id,pod_name=pod['name'],created_unix=started,deadline_unix=deadline,cloud='COMMUNITY',
        gpu_requested=ready.get('machine',{}).get('gpuTypeId') or GPU_PREFERENCE[0],gpu_usd_per_hour=ready.get('costPerHr'),remote_root='/workspace/vawm_accum',remote_results='/workspace/vawm_accum/results',
        remote_bundle='/workspace/accum_bundle.tar.gz',bundle_sha256=bundle_sha,remote_dataset_bundle='/workspace/bsds500_runtime.tar.gz',dataset_bundle_sha256=DATASET_SHA,venv_root='/workspace/accum-venv',
        ssh_host=host,ssh_port=port,ssh_user='root',ssh_identity=str(SSH_KEY).replace('\\','/'),known_hosts=str(known).replace('\\','/'),local_run=str(run_root).replace('\\','/'),
        lanes=LANES,seeds=a.seeds,workers=a.workers,arm='accumulator_program',retrieval_grace_seconds=900,
        cleanup_owner='WorkingMemory/PlainBaseline/cloud/watch.py pulls results every 10 min and stops+deletes the pod at completion or deadline.')
    cfg=run_root/'cloud_provisioning.json';write(cfg,config)
    common=['-i',str(SSH_KEY),'-o','UserKnownHostsFile='+str(known),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','KexAlgorithms=curve25519-sha256'];dest=f'root@{host}'
    for _ in range(30):
        probe=subprocess.run(['ssh','-n',*common,'-p',str(port),dest,'echo ready'],capture_output=True,text=True,timeout=60)
        if probe.returncode==0:break
        time.sleep(10)
    else:raise RuntimeError('SSH never became ready: '+probe.stderr[-300:])
    for local,remote in ((archive,config['remote_bundle']),(DATASET,config['remote_dataset_bundle']),(cfg,'/workspace/accum_cloud.json'),(HERE/'setup_remote.sh','/workspace/setup_accum.sh')):
        subprocess.run(['scp',*common,'-P',str(port),str(local),dest+':'+remote],check=True,timeout=1800,capture_output=True,text=True)
    print('uploaded',flush=True)
    setup=subprocess.run(['ssh','-n',*common,'-p',str(port),dest,"sed -i 's/\\r$//' /workspace/setup_accum.sh && bash /workspace/setup_accum.sh /workspace/accum_cloud.json"],capture_output=True,text=True,timeout=2400)
    (run_root/'setup_output.log').write_text(setup.stdout+'\n--- stderr ---\n'+setup.stderr)
    if setup.returncode:raise RuntimeError('remote setup failed; see setup_output.log')
    config.update(status='launched',launched_unix=time.time(),remote_pids=(setup.stdout or '').strip().splitlines()[-1:] or None);write(cfg,config)
    write(run_root/'launch_receipt.json',dict(status='launched',pod_id=pod_id,gpu=config['gpu_requested'],cost_per_hour=config['gpu_usd_per_hour'],deadline_unix=deadline,max_cost_usd=(config['gpu_usd_per_hour'] or 0)*a.hours,bundle_sha256=bundle_sha,lanes=LANES,seeds=a.seeds))
    write(HERE.parent/'runs'/'latest_cloud.json',dict(run=str(run_root),config=str(cfg),pod_id=pod_id,deadline_unix=deadline))
    print(json.dumps(dict(pod_id=pod_id,host=host,port=port,gpu=config['gpu_requested'],cost_per_hour=config['gpu_usd_per_hour'],run=str(run_root)),indent=1))

if __name__=='__main__':main()
