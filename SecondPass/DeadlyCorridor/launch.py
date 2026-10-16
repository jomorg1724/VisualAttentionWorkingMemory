"""Finite local supervisor for either prepared experiment; no automatic restart."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['vda','doom']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--wall-seconds',type=int,default=28800);p.add_argument('--fresh',action='store_true');a=p.parse_args()
    if not 0<a.wall_seconds<=28800:raise ValueError('One finite local cap of at most 8 hours')
    if a.fresh and a.experiment!='vda':raise ValueError('--fresh is a VDA choice; Doom uses the requested suite checkpoint')
    if a.run.exists():raise FileExistsError('Use a new run directory; no implicit resume or replacement')
    a.run.parent.mkdir(parents=True,exist_ok=True)
    module='SecondPass.StabilizedAttentionVDA.train' if a.experiment=='vda' else 'SecondPass.DeadlyCorridor.train'
    cmd=[sys.executable,'-X','utf8','-u','-m',module,'--run',str(a.run),'--steps',str(a.steps),'--wall-seconds',str(max(1,a.wall_seconds-30))]
    if a.fresh:cmd.append('--fresh')
    receipt=a.run.with_name(a.run.name+'_supervisor.json');log=a.run.with_name(a.run.name+'_worker.log')
    started=time.time();flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with log.open('w',encoding='utf8') as output:
        child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,creationflags=flags,cwd=Path(__file__).resolve().parents[2])
        info=dict(pid=child.pid,argv=cmd,started=started,deadline=started+a.wall_seconds,status='running')
        receipt.write_text(json.dumps(info,indent=2),encoding='utf8')
        try:code=child.wait(timeout=max(.1,started+a.wall_seconds-time.time()));reason='worker_exit'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            code=124;reason='hard_time_cap'
        info.update(status='exited',exit_code=code,reason=reason,ended=time.time());receipt.write_text(json.dumps(info,indent=2),encoding='utf8')
    raise SystemExit(code)

if __name__=='__main__':main()
