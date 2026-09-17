"""Wait for the existing local trainer; capture-only, finite previously pending allowance."""
import os,sys,time,json,subprocess
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=ROOT/'WorkingMemory/TrainingExposure/runs/exposure_20260913_163542'
WAIT_DEADLINE=1789357544.4010413
def save(v):
    p=HERE/'queue_receipt.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def blockers():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or []).replace('\\','/').lower()
            if p.pid!=os.getpid() and 'python' in (p.info['name'] or '').lower() and any(s in cmd for s in ('trainingexposure/worker.py','trainingexposure/sweep.py')):found.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found
def main():
    if (HERE/'budget.json').exists() or (HERE.parent/'LatentDynamics/budget.json').exists():raise RuntimeError('No automatic renewal of shared allowance')
    r=dict(status='queued',pid=os.getpid(),queued_at=time.time(),wait_deadline=WAIT_DEADLINE,compute_cap_seconds=1800,scope='attention visualization only; broader latent probes paused');save(r)
    while time.time()<WAIT_DEADLINE:
        blocked=blockers();ready=(HERE/'implementation_ready.json').exists()
        if (RUN/'exit.json').exists() and not blocked and not psutil.pid_exists(4272) and ready:
            r.update(status='capturing',started=time.time());save(r)
            with (HERE/'worker.log').open('w') as log:
                p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(HERE/'run.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));r['worker_pid']=p.pid;save(r)
                try:code=p.wait(timeout=1810)
                except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            r.update(status='completed' if code==0 else 'failed',returncode=code,finished=time.time());save(r);return
        r.update(last_checked=time.time(),blocked_pids=blocked,implementation_ready=ready);save(r);time.sleep(20)
    r.update(status='queue_expired',finished=time.time());save(r)
if __name__=='__main__':main()
