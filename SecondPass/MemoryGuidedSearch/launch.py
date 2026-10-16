"""One detached local allocation, no automatic restart or extension."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from SecondPass.PixelContrastiveVAE.train import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--steps',type=int,default=19000);p.add_argument('--wall-seconds',type=int,default=28800);a=p.parse_args()
    run=a.run.resolve();root=Path(__file__).resolve().parents[2]
    if run.exists():raise FileExistsError('No implicit restart or overwrite')
    if a.steps<19 or a.steps%19 or not 180<a.wall_seconds<=28800:raise ValueError('Invalid allocation')
    run.parent.mkdir(parents=True,exist_ok=True);started=time.time();deadline=started+a.wall_seconds
    receipt=run.with_name(run.name+'_supervisor.json')
    cmd=[sys.executable,'-X','utf8','-u','-m','SecondPass.MemoryGuidedSearch.train','--run',str(run),'--steps',str(a.steps),'--wall-seconds',str(a.wall_seconds-30)]
    with run.with_name(run.name+'_worker.log').open('w',encoding='utf-8') as log:
        worker=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        info=dict(supervisor_pid=os.getpid(),worker_pid=worker.pid,argv=cmd,started=started,deadline=deadline,status='running',automatic_restart=False)
        atomic_json(receipt,info)
        try:
            while worker.poll() is None:
                if time.time()>=deadline-180 and run.exists():atomic_json(run/'stop_requested.json',dict(reason='allocation_deadline',time=time.time()))
                if time.time()>=deadline:raise subprocess.TimeoutExpired(cmd,a.wall_seconds)
                time.sleep(2)
            code=worker.returncode;reason='worker_exit'
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=5)
            code=124;reason='hard_time_cap'
        info.update(status='exited',exit_code=code,reason=reason,ended=time.time());atomic_json(receipt,info)
    raise SystemExit(code)


if __name__=='__main__':main()
