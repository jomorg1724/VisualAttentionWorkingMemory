"""Run-specific cloud guard and supervisor; training implementation is unchanged."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
import requests

ROOT=Path('/workspace/vawm')
RUN=ROOT/'SecondPass/DeadlyCorridor/CloudRuntime/continuation02'
TRAIN=RUN/'training'

def write(name,data):
    p=RUN/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def guard(config):
    write('guard.json',dict(pid=os.getpid(),deadline=config['deadline'],pod_id=config['pod_id'],status='armed'))
    while time.time()<config['deadline']-15:
        if (RUN/'shutdown_requested.json').exists():break
        if time.time()>config['deadline']-180 and TRAIN.exists():
            (TRAIN/'stop_requested.json').write_text('{"reason":"cloud_allocation_deadline"}')
        time.sleep(5)
    worker=RUN/'worker.json'
    if worker.exists():
        pid=json.loads(worker.read_text())['pid']
        try:os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    os.sync()
    write('guard.json',dict(pid=os.getpid(),deadline=config['deadline'],pod_id=config['pod_id'],status='requesting_pod_stop',time=time.time()))
    # Existing provider-issued pod-scoped key; never the account credential.
    key=Path('/root/.doom-pod-key').read_text().strip()
    for attempt in range(3):
        try:
            result=requests.post('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key},json={'query':'mutation($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }','variables':{'input':{'podId':config['pod_id']}}},timeout=30)
            data=result.json();pod=(data.get('data') or {}).get('podStop') or {}
            write('stop_receipt.json',dict(attempt=attempt+1,http_status=result.status_code,pod=pod,errors=data.get('errors'),time=time.time()))
            if pod.get('desiredStatus')=='EXITED':return
        except Exception as exc:
            write('stop_receipt.json',dict(attempt=attempt+1,error_type=type(exc).__name__,time=time.time()))
        time.sleep(10)
    raise RuntimeError('Pod stop failed; stop_receipt.json has provider response')

def supervise(config):
    assert (TRAIN/'resume_prepared.json').exists(),'Explicit prepared continuation required'
    assert not (RUN/'worker.json').exists(),'No automatic restart'
    remaining=int(config['deadline']-time.time()-60)
    assert remaining>300,'Insufficient remaining allocation'
    cmd=[sys.executable,'-u','-m','SecondPass.DeadlyCorridor.train','--resume','--run',str(TRAIN),'--steps',str(config['decision_target']),'--wall-seconds',str(remaining)]
    with (RUN/'worker.log').open('w') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write('worker.json',dict(pid=child.pid,supervisor_pid=os.getpid(),argv=cmd,started=time.time(),deadline=config['deadline']))
        code=child.wait()
    write('supervisor_exit.json',dict(exit_code=code,time=time.time()))
    # Allow a brief final artifact read before stopping compute. All files are
    # already on the independently persistent network volume.
    time.sleep(60)
    write('shutdown_requested.json',dict(reason='training_worker_finished',exit_code=code,time=time.time()))
    raise SystemExit(code)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['guard','supervise']);args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    config=json.loads((RUN/'deployment.json').read_text())
    # SSH sessions do not inherit container startup environment variables.
    identity=json.loads((RUN/'preflight_verified.json').read_text())
    assert identity['pod_id']==config['pod_id'] and identity['pod_scoped_auth'],'Guard must address only its verified allocated pod'
    (guard if args.mode=='guard' else supervise)(config)
