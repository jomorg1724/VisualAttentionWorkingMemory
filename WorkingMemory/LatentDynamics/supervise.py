"""Finite queue: do not overlap the existing TrainingExposure GPU worker."""
import os,sys,time,json,subprocess
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'WorkingMemory/TrainingExposure/runs/exposure_20260913_163542'
WAIT_DEADLINE=1789357544.4010413
def save(value):
    p=HERE/'queue_receipt.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2),encoding='utf-8');q.replace(p)
def blockers():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or []).replace('\\','/').lower()
            if p.pid!=os.getpid() and 'python' in (p.info['name'] or '').lower() and any(s in cmd for s in ('trainingexposure/worker.py','trainingexposure/sweep.py')):found.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found
def main():
    if (HERE/'budget.json').exists():raise RuntimeError('No renewed extraction allowance')
    receipt=dict(status='queued',pid=os.getpid(),queued_at=time.time(),wait_deadline=WAIT_DEADLINE,compute_cap_seconds=1800,training_run=str(RUN),condition='TrainingExposure exit receipt plus supervisor4272 and all TrainingExposure workers exited; implementation ready')
    save(receipt)
    while time.time()<WAIT_DEADLINE:
        blocked=blockers();ready=(HERE/'implementation_ready.json').exists()
        if (RUN/'exit.json').exists() and not blocked and not psutil.pid_exists(4272) and ready:
            receipt.update(status='extracting',started=time.time(),blocked_pids=[]);save(receipt)
            env=os.environ.copy()
            for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='2'
            with (HERE/'worker.log').open('w') as log:
                worker=subprocess.Popen([sys.executable,'-X','utf8','-B',str(HERE/'extract.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                receipt['worker_pid']=worker.pid;save(receipt)
                try:code=worker.wait(timeout=1810)
                except subprocess.TimeoutExpired:worker.kill();worker.wait();code=124
            receipt.update(status='completed' if code==0 else 'failed',returncode=code,finished=time.time());save(receipt);return
        receipt.update(last_checked=time.time(),blocked_pids=blocked,implementation_ready=ready);save(receipt);time.sleep(30)
    receipt.update(status='queue_expired',finished=time.time());save(receipt)
if __name__=='__main__':main()
