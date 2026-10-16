import os,sys,time,subprocess
from .design import ROOT,OUT
from SecondPass.PixelContrastiveVAE.train import atomic_json

def main():
    OUT.mkdir(exist_ok=False);start=time.time();deadline=start+7200
    atomic_json(OUT/'budget.json',dict(started=start,deadline=deadline,wall_seconds=7200,no_automatic_extension=True))
    with (OUT/'worker.log').open('w',encoding='utf8') as log:
        child=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','SecondPass.StabilizedAttentionVDA.Analysis.experiment'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        atomic_json(OUT/'owner.json',dict(supervisor_pid=os.getpid(),worker_pid=child.pid,started=start))
        try:code=child.wait(timeout=max(.1,deadline-time.time()));reason='worker_exit'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            code=124;reason='hard_time_cap'
        atomic_json(OUT/'exit.json',dict(exit_code=code,reason=reason,time=time.time()))
    raise SystemExit(code)

if __name__=='__main__':main()
