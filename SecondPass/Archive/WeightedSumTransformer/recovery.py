"""One explicitly requested shutdown recovery, under an independent finite cap."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from SecondPass.PredictiveConsistencyVAE.queue import (
    FileLock, hidden_options, prevent_sleep, process_identity, read_json, terminate_owned)
from SecondPass.PixelContrastiveVAE.train import atomic_json
from .train import source_hashes

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recovery-dir', type=Path, required=True)
    args = parser.parse_args()
    run, recovery = args.run.resolve(), args.recovery_dir.resolve()
    if recovery.parent.parent != run or recovery.parent.name != 'Recovery':
        raise ValueError('Recovery directory must be inside this run/Recovery')
    recovery.mkdir(parents=True, exist_ok=True)
    owner_lock = FileLock(recovery / 'owner.lock')
    child = guard = None
    prevent_sleep(True)
    try:
        budget_path = recovery / 'budget.json'
        if budget_path.exists() or (run / 'completion_verified.json').exists():
            raise RuntimeError('No implicit recovery retry or completed-run extension')
        proof = read_json(recovery / 'pre_recovery_checkpoint_verified.json')
        if not proof or proof.get('unchanged_cumulative_target') != 25024:
            raise RuntimeError('Verified interrupted checkpoint required')
        checkpoint_hash = hashlib.sha256((run / 'latest.pt').read_bytes()).hexdigest()
        if checkpoint_hash != proof['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed since recovery verification')
        original = read_json(run / 'production_config.json')
        if original['speed_scale'] != 4 or original['target_updates'] != 25024:
            raise RuntimeError('This recovery is only for the authorized speed4 run')
        current = source_hashes()
        # Protect all scientific code; only recovery plumbing may differ.
        for relative, digest in original['source_hashes'].items():
            if Path(relative).name != 'train.py' or 'WeightedSumTransformer' not in relative:
                if current.get(relative) != digest:
                    raise RuntimeError('Scientific source changed: ' + relative)
        started = time.time()
        atomic_json(budget_path, dict(started=started, deadline=started + 3600.,
            wall_seconds=3600, source_hashes=current, checkpoint_sha256=checkpoint_hash,
            authorization='explicit user continue after shutdown',
            original_budget=read_json(run / 'budget.json'), target_updates=25024,
            restored_step=proof['saved_step'], recovery_only=True, cloud=False,
            recovery_version='shutdown_exact_state_resume_v1'))
        atomic_json(recovery / 'launch.json', dict(pid=os.getpid(),
            identity=process_identity(os.getpid()), run=str(run), started=started,
            requested='Go ahead and continue the training. My computer was shut down'))
        command = [sys.executable, '-m', 'SecondPass.PredictiveConsistencyVAE.queue',
                   '--guard', '--queue-dir', str(recovery), '--run', str(recovery),
                   '--owner-pid', str(os.getpid())]
        with (recovery / 'guard.log').open('a', encoding='utf-8') as log:
            guard = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, **hidden_options())
        command = [sys.executable, '-u', '-m', 'SecondPass.WeightedSumTransformer.train',
                   '--run', str(run), '--speed-scale', '4', '--resume',
                   '--recovery-budget', str(budget_path)]
        atomic_json(recovery / 'status.json', dict(status='restoring', pid=os.getpid()))
        with (recovery / 'worker.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, **hidden_options())
            identity = process_identity(child.pid)
            if identity is None:
                time.sleep(.1)
                identity = process_identity(child.pid)
            if identity is None:
                raise RuntimeError('Cannot verify owned recovery worker identity')
            atomic_json(recovery / 'active_child.json', dict(phase='recovery',
                identity=identity, command=command, started=time.time()))
            result = child.wait()
            if result:
                raise RuntimeError('Recovery exited with code %s; evidence preserved' % result)
        completion = read_json(run / 'completion_verified.json')
        if not completion or completion.get('status') != 'complete':
            raise RuntimeError('Recovery exited without verified completion')
        atomic_json(recovery / 'status.json', dict(status='complete', pid=os.getpid(),
            completed=time.time(), completion=completion))
    except BaseException as exc:
        atomic_json(recovery / 'status.json', dict(status='failed', pid=os.getpid(),
                    error=repr(exc), failed=time.time()))
        raise
    finally:
        active = read_json(recovery / 'active_child.json')
        if child is not None and child.poll() is None and active:
            terminate_owned(active.get('identity'))
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if guard is not None:
            atomic_json(recovery / 'guard_stop.json', dict(stopped=time.time()))
            try:
                guard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                guard.terminate()
        prevent_sleep(False)
        owner_lock.close()


if __name__ == '__main__':
    main()
