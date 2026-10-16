"""One finite local frozen-analysis allocation; no automatic extension."""
import sys,time,subprocess,os
from .design import ROOT,OUT
from SecondPass.PredictiveConsistencyVAE.queue import hidden_options,process_identity,prevent_sleep,terminate_owned,atomic_json
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'budget.json').exists(),'No automatic budget renewal'
    started=time.time();atomic_json(OUT/'budget.json',dict(started=started,deadline=started+14400,wall_seconds=14400))
    atomic_json(OUT/'owner.json',process_identity(os.getpid()));prevent_sleep(True);guard=None;child=None;identity=None
    try:
        with (OUT/'guard.log').open('w') as f:
            guard=subprocess.Popen([sys.executable,'-m','SecondPass.PredictiveConsistencyVAE.queue','--guard','--queue-dir',str(OUT),'--run',str(OUT),'--owner-pid',str(os.getpid())],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,**hidden_options())
        with (OUT/'worker.log').open('w') as f:
            child=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','SecondPass.StabilizedAttentionSuite.Analysis.experiment'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,**hidden_options())
            identity=process_identity(child.pid);assert identity is not None
            atomic_json(OUT/'active_child.json',dict(identity=identity,phase='frozen_analysis'))
            code=child.wait();atomic_json(OUT/'exit.json',dict(exit_code=code,time=time.time()))
    finally:
        if child is not None and child.poll() is None:terminate_owned(identity)
        atomic_json(OUT/'guard_stop.json',dict(stopped=time.time()))
        if guard is not None:guard.wait(timeout=10)
        prevent_sleep(False)
if __name__=='__main__':main()
