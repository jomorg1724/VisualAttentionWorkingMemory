"""One local queued experiment; no accelerator work or new cap while waiting."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(os.environ.get('VAWM_QUEUE_REPO', str(Path(__file__).resolve().parents[3])))
PREVIOUS = Path('/Users/jonathanmorgan/VAWMRuntime/vae_rvit_local01/run')
QUEUE = Path('/Users/jonathanmorgan/VAWMRuntime/weighted_mean_rvit_queue01')
GPU_LOCK = Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock')
NEXT_BASE = Path('/Users/jonathanmorgan/VAWMRuntime/weighted_mean_rvit_local01')
PREVIOUS_MODULE = 'SecondPass.VAERViT.worker'


def read(path):
    return json.loads(path.read_text())


def status(state, **extra):
    temporary = QUEUE / 'status.tmp'
    temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), updated=time.time(),
        predecessor=str(PREVIOUS), **extra), indent=2) + '\n')
    temporary.replace(QUEUE / 'status.json')


def owned(pid):
    if not pid:
        return False
    command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
        capture_output=True, text=True, check=False).stdout
    return PREVIOUS_MODULE in command and str(PREVIOUS) in command


def predecessor_finished():
    if not (PREVIOUS / 'local_supervisor_result.json').exists():
        return False
    activation = read(PREVIOUS / 'activation.json')
    supervisor = read(PREVIOUS / 'run_supervisor.json')
    return not owned(activation['supervisor_pid']) and not owned(supervisor['worker_pid'])


def gpu_available():
    with GPU_LOCK.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def main():
    with (QUEUE / 'claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), queued_at=time.time()), stream)
    expires = read(PREVIOUS / 'budget.json')['hard_deadline'] + 300
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())])
    try:
        status('waiting_for_vae_rvit', waiting_expires=expires, new_cap_started=False)
        while time.time() < expires:
            if predecessor_finished() and gpu_available():
                result = read(PREVIOUS / 'local_supervisor_result.json')
                status('launching', predecessor_result=result, new_cap_started=False)
                with (QUEUE / 'launch.log').open('a') as log:
                    child = subprocess.run([sys.executable, '-u', '-m',
                        'SecondPass.WeightedMeanRViT.LocalRuntime.launch', '--activate-existing'],
                        cwd=NEXT_BASE / 'repo',
                        stdout=log, stderr=subprocess.STDOUT, check=False)
                status('launcher_installed' if child.returncode == 0 else 'launch_failed',
                    returncode=child.returncode, no_retry=True, predecessor_result=result)
                return child.returncode
            time.sleep(5)
        status('waiting_expired', no_training_started=True, no_retry=True)
        return 1
    except Exception as exc:
        status('queue_error', error=repr(exc), no_retry=True)
        return 1
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == '__main__':
    raise SystemExit(main())
