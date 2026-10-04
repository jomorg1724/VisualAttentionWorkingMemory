"""Independent deadline guard for this single local training attempt."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

MODULE = 'SecondPass.ThreeFrameConvVAE.worker'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def owned(pid, run):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
        capture_output=True, text=True)
    return MODULE in result.stdout and str(run) in result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    before = time.time()
    while not (run / 'budget.json').exists():
        if time.time() - before > 90:
            write(run / 'guard_result.json', dict(state='no_budget_created', pid=os.getpid()))
            return
        time.sleep(1)
    budget = json.loads((run / 'budget.json').read_text())
    deadline = budget['hard_deadline']
    assert deadline - budget['cap_started'] == 28800
    write(run / 'guard_status.json', dict(state='armed', pid=os.getpid(),
        hard_deadline=deadline, cap_started=budget['cap_started']))
    # Keep this authorized local job awake while allowing the display to sleep.
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())])
    try:
        while time.time() < deadline:
            if (run / 'local_supervisor_result.json').exists():
                activation = json.loads((run / 'activation.json').read_text())
                if not owned(activation['supervisor_pid'], run):
                    write(run / 'guard_result.json', dict(state='normal_owner_exit', pid=os.getpid(),
                        observed=time.time(), hard_deadline=deadline))
                    return
            time.sleep(min(3, max(.05, deadline - time.time())))
        killed = []
        for name in ('profile/profile_supervisor.json', 'run_supervisor.json', 'activation.json'):
            path = run / name
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            for field in ('worker_pid', 'supervisor_pid', 'pid'):
                pid = row.get(field)
                if pid and pid not in killed and owned(pid, run):
                    try:
                        if field == 'worker_pid':
                            os.killpg(pid, signal.SIGKILL)
                        else:
                            os.kill(pid, signal.SIGKILL)
                        killed.append(pid)
                    except ProcessLookupError:
                        pass
        write(run / 'guard_result.json', dict(state='hard_deadline', killed=killed,
            observed=time.time(), hard_deadline=deadline))
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == '__main__':
    main()
