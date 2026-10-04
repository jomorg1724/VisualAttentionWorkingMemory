"""Attach a transferable CPU guardian to an existing fixed-deadline worker.

No accelerator imports, worker launches, restarts, or budget writes. Keep stdout
open to Hermes: redirecting all streams triggers its premature EOF completion.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .core import atomic_json


def watch(directory):
    directory=Path(directory)
    last_print=0.
    while True:
        if (directory/'supervisor_result.json').exists() and (directory/'report.json').exists():
            result=json.loads((directory/'supervisor_result.json').read_text())
            atomic_json(directory/'guardian_result.json',dict(pid=os.getpid(),supervisor=result,finished=time.time(),
                report=str(directory/'report.json'),new_worker_started=False,budget_renewed=False))
            print(json.dumps(dict(event='completed',supervisor=result,report=str(directory/'report.json'))),flush=True)
            return result
        supervisor=json.loads((directory/'supervisor.json').read_text())
        budget=json.loads((directory/'budget.json').read_text())
        assert supervisor['deadline']==budget['deadline']
        now=time.time()
        if now>=budget['deadline']:
            # Independent duplicate backstop, exact identity and process group only.
            pid=supervisor['worker_pid']
            identity=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout.strip()
            expected='-m SecondPass.JointTraining.continuation_v3 worker '+str(directory.resolve())
            if expected in identity and Path(identity.split()[0]).name.lower().startswith('python'):
                if os.getpgid(pid)!=pid: raise RuntimeError('Worker process group identity changed')
                os.killpg(pid,signal.SIGKILL)
                atomic_json(directory/'guardian_hard_stop.json',dict(worker_pid=pid,deadline=budget['deadline'],epoch=now))
            if now>budget['deadline']+20:
                result=dict(returncode=2,deadline=budget['deadline'],hard_cap_triggered=True,
                    note='Supervisor publication missing after fixed deadline; no automatic restart')
                if not (directory/'report.json').exists():
                    atomic_json(directory/'report.json',dict(final_coverage_complete=False,stop_reason='deadline_publication_missing',supervisor=result))
                atomic_json(directory/'guardian_result.json',dict(pid=os.getpid(),supervisor=result,finished=now))
                return result
        status=json.loads((directory/'live_status.json').read_text()) if (directory/'live_status.json').exists() else {}
        record=dict(guardian_pid=os.getpid(),supervisor_pid=supervisor['pid'],worker_pid=supervisor['worker_pid'],
            deadline=budget['deadline'],utc_epoch=now,step=status.get('step'),phase=status.get('phase'),
            report=str(directory/'report.json'),new_worker_started=False,budget_renewed=False)
        atomic_json(directory/'guardian_status.json',record)
        if now-last_print>=60:
            print(json.dumps(record),flush=True); last_print=now
        time.sleep(min(5.,max(.05,budget['deadline']-time.time())))


if __name__=='__main__':
    raise SystemExit(watch(Path(sys.argv[1]).resolve())['returncode'])
