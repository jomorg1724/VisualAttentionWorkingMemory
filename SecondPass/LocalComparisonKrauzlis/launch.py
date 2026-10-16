"""Run one fresh single-head shared-attention native Krauzlis model under an independently enforced local cap."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from SecondPass.PredictiveConsistencyVAE.queue import (
    FileLock, hidden_options, prevent_sleep, process_identity, read_json, terminate_owned)
from SecondPass.PixelContrastiveVAE.train import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args, additional = parser.parse_known_args()
    run = args.run.resolve()
    supervisor = run / 'supervisor'
    supervisor.mkdir(parents=True, exist_ok=True)
    owner_lock = FileLock(supervisor / 'owner.lock')
    guard = child = None
    prevent_sleep(True)
    try:
        if (run / 'budget.json').exists():
            raise RuntimeError('Existing activated experiment; explicit recovery required')
        atomic_json(supervisor / 'launch.json', dict(pid=os.getpid(),
                    identity=process_identity(os.getpid()), run=str(run), started=time.time(),
                    authorized='user approved targeted local temporal comparison variant after frozen-model failure diagnostics',
                    worker_arguments=additional, cloud=False))
        command = [sys.executable, '-m', 'SecondPass.PredictiveConsistencyVAE.queue',
                   '--guard', '--queue-dir', str(supervisor), '--run', str(run),
                   '--owner-pid', str(os.getpid())]
        with (supervisor / 'guard.log').open('a', encoding='utf-8') as log:
            guard = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, **hidden_options())
        base = [sys.executable, '-u', '-m', 'SecondPass.LocalComparisonKrauzlis.train',
                '--run', str(run)] + additional
        for phase, command in [('profile', base + ['--profile-only']), ('production', base)]:
            atomic_json(supervisor / 'status.json', dict(status=phase, pid=os.getpid()))
            with (supervisor / (phase + '.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, **hidden_options())
                identity = process_identity(child.pid)
                if identity is None:
                    time.sleep(.1)
                    identity = process_identity(child.pid)
                if identity is None:
                    raise RuntimeError('Cannot verify owned worker identity')
                atomic_json(supervisor / 'active_child.json', dict(
                    phase=phase, identity=identity, command=command, started=time.time()))
                result = child.wait()
                if result:
                    raise RuntimeError('%s worker exited with code %s; preserve evidence' % (phase, result))
            if phase == 'profile':
                profile = read_json(run / 'profile.json')
                if not profile or time.time() >= float(profile['deadline']) - 120:
                    raise RuntimeError('No valid profile or finite budget exhausted')
        completion = read_json(run / 'completion_verified.json')
        if not completion or completion.get('status') != 'complete':
            raise RuntimeError('Worker exited without verified completion')
        atomic_json(supervisor / 'status.json', dict(status='complete', pid=os.getpid(), completed=time.time()))
    except BaseException as exc:
        atomic_json(supervisor / 'status.json', dict(status='failed', pid=os.getpid(),
                    error=repr(exc), failed=time.time()))
        raise
    finally:
        active = read_json(supervisor / 'active_child.json')
        if child is not None and child.poll() is None and active:
            terminate_owned(active.get('identity'))
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if guard is not None:
            atomic_json(supervisor / 'guard_stop.json', dict(stopped=time.time()))
            try:
                guard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                guard.terminate()
        prevent_sleep(False)
        owner_lock.close()


if __name__ == '__main__':
    main()
