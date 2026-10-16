"""One explicit resumption using only the prior run's unused wall-time allowance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def remaining_seconds(source):
    receipt = json.loads(source.with_name(source.name + '_supervisor.json').read_text())
    if receipt['status'] != 'exited':
        raise ValueError('Original supervisor is not recorded as exited')
    return int(receipt['deadline'] - receipt['ended'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    source, run = a.source.resolve(), a.run.resolve()
    if run.exists():
        raise FileExistsError('No implicit overwrite or restart')
    remaining = remaining_seconds(source)
    if remaining <= 150:
        raise ValueError('No useful allocation remains')
    run.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-X', 'utf8', '-u', '-m', 'SecondPass.StabilizedAttentionVDA.Continuation.resume',
           '--source', str(source), '--run', str(run), '--wall-seconds', str(remaining-30)]
    receipt = run.with_name(run.name + '_supervisor.json')
    started = time.time()
    with run.with_name(run.name + '_worker.log').open('w', encoding='utf8') as log:
        child = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[3], stdout=log,
                                 stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        info = dict(pid=child.pid, supervisor_pid=os.getpid(), argv=cmd, status='running',
                    started=started, deadline=started+remaining, wall_seconds=remaining,
                    allocation='unused original local cap, relocated after explicit user resume')
        receipt.write_text(json.dumps(info, indent=2), encoding='utf8')
        try:
            code = child.wait(timeout=max(.1, started+remaining-time.time()))
            reason = 'worker_exit'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            code, reason = 124, 'hard_time_cap'
        info.update(status='exited', exit_code=code, reason=reason, ended=time.time())
        receipt.write_text(json.dumps(info, indent=2), encoding='utf8')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
