"""One-shot CPU-only queue; activate only after the predecessor fully exits."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from SecondPass.PredictiveConsistencyVAE.queue import (
    FileLock, atomic_json, hidden_options, process_identity, read_json, prevent_sleep)

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path,required=True)
    args=parser.parse_args()
    run,previous=args.run.resolve(),args.predecessor.resolve()
    owner=FileLock(run/'queue_owner.lock')
    prevent_sleep(True)
    try:
        if (run/'budget.json').exists(): raise RuntimeError('Already activated; no retry')
        budget=read_json(previous/'budget.json')
        expires=float(budget['deadline'])+900
        receipt=dict(status='waiting',pid=os.getpid(),identity=process_identity(os.getpid()),
            predecessor=str(previous),run=str(run),expires=expires,
            new_budget_started=False,local_only=True,automatic_retry=False)
        atomic_json(run/'queue_status.json',receipt)
        while time.time()<expires:
            completed=read_json(previous/'completion_verified.json')
            supervisor=read_json(previous/'supervisor/status.json')
            active=read_json(previous/'supervisor/active_child.json')
            launch=read_json(previous/'supervisor/launch.json')
            if (read_json(previous/'failure.json') or (supervisor or {}).get('status')=='failed'):
                raise RuntimeError('Predecessor failed; preserve evidence and hold next run')
            worker_live=active and process_identity(active['identity']['pid'])==active['identity']
            owner_live=launch and process_identity(launch['pid'])==launch['identity']
            if (completed or {}).get('status')=='complete' and (supervisor or {}).get('status')=='complete' and not worker_live and not owner_live:
                try: lock=FileLock(ROOT/'LocalRuntime/local_gpu_worker.lock')
                except RuntimeError:
                    time.sleep(5); continue
                lock.close()
                manifest=read_json(run/'prelaunch_source_manifest.json')
                for relative,digest in manifest.items():
                    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
                        raise RuntimeError('Queued source changed: '+relative)
                command=[sys.executable,'-u','-m','SecondPass.CorrespondencePredictiveMemory.launch','--run',str(run)]
                with (run/'launcher.stdout.log').open('a') as out, (run/'launcher.stderr.log').open('a') as err:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,**hidden_options())
                receipt.update(status='dispatched',supervisor_pid=process.pid,supervisor_identity=process_identity(process.pid),
                    dispatched=time.time(),new_budget_started='see budget.json; starts at first profile')
                atomic_json(run/'queue_status.json',receipt)
                return
            time.sleep(5)
        raise TimeoutError('Queue expired without successful predecessor release')
    except BaseException as exc:
        atomic_json(run/'queue_status.json',dict(status='held',error=repr(exc),pid=os.getpid(),no_retry=True))
        raise
    finally:
        owner.close(); prevent_sleep(False)


if __name__=='__main__': main()
