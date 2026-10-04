"""One explicit pre-activation recovery; the original queue and guards stay frozen.

A changed ps identity is never classified as harmless. Recovery requires all
original PIDs absent, re-verifies normal completion, and retains blocked receipts.
"""
from __future__ import annotations
import copy
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import types
from . import continuation_v4_queue as q
from .core import atomic_json


def activate(directory, manifest):
    """Use the frozen activation verbatim, redirecting only queue status receipts.

    Original blocked queue_result/status stay byte-identical in place as well as
    archived. No process, verification, planning, budget, or worker logic changes.
    """
    directory = Path(directory)
    def recovery_json(path, value):
        path = Path(path)
        if path.parent == directory and path.name in {'queue_status.json', 'queue_result.json'}:
            path = path.with_name(path.name.replace('queue_', 'recovery_'))
        atomic_json(path, value)
    scope = dict(q.activate.__globals__, atomic_json=recovery_json)
    original = q.activate
    callback = types.FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    return callback(directory, manifest)


def verify_recovery(directory, manifest):
    """CPU-only gates; no revival/reinterpretation of a changed live identity."""
    source = Path(manifest['source'])
    for name in ('budget.json', 'activation.json', 'supervisor.json', 'migration.pt', 'progress.jsonl', 'config.json'):
        if (directory/name).exists():
            raise ValueError('Refusing prior activation or renewed cap: '+name)
    if time.time() >= manifest['queue_expiry']:
        raise ValueError('Bounded predecessor queue expired')
    failure = q.load(directory/'queue_result.json')
    if failure.get('phase') != 'blocked' or failure.get('error') != 'Predecessor PID identity changed; fail closed' or failure.get('new_worker_started') is not False:
        raise ValueError('Not the authorized pre-activation identity incident')
    handoff = q.load(directory/'queue_handoff.json')
    if handoff['manifest_sha256'] != q.digest(directory/'queue_manifest.json') or handoff['queue_expiry'] != manifest['queue_expiry']:
        raise ValueError('Blocked manifest/handoff mismatch')
    sup = q.load(source/'supervisor.json')
    if sup != manifest['predecessor'] or set(manifest['predecessor_identities']) != {str(sup['pid']), str(sup['worker_pid'])}:
        raise ValueError('Original predecessor receipt/identity mismatch')
    for key, role in [('pid', 'run'), ('worker_pid', 'worker')]:
        expected = manifest['predecessor_identities'][str(sup[key])]
        if expected is None:
            raise ValueError('Missing original predecessor identity')
        q.validate_identity(expected, role, source)
    observed = {str(pid): q.identity(pid) for pid in [sup['pid'], sup['worker_pid'], handoff['pid']]}
    if any(value is not None for value in observed.values()):
        raise ValueError('Original PID still present; no identity bypass: '+json.dumps(observed, sort_keys=True))
    q.no_other_worker()
    q.verify_hashes(manifest['source_hashes'])
    if q.digest(source/'config.json') != manifest['source_config_sha256'] or q.digest(source/'budget.json') != manifest['source_budget_sha256']:
        raise ValueError('Predecessor budget/config changed')
    completion = q.verify_predecessor(source)
    return dict(predecessor=completion, observed_identities=observed, verified_epoch=time.time(),
                queue_expiry=manifest['queue_expiry'], original_manifest_sha256=q.digest(directory/'queue_manifest.json'),
                original_error=failure['error'], cause='Unobserved changed identity; normal-exit zombie transition reproduced separately, not established for original incident',
                policy='Explicit pre-activation recovery only; all original PIDs absent; no live identity accepted or normalized')


def archive_blocked(directory):
    """Create a new read-only hash-verified snapshot; never replace old evidence."""
    archive = directory/'blocked_attempt_01'
    if archive.exists():
        raise ValueError('Refusing previously attempted recovery archive')
    paths = sorted(p for p in directory.rglob('*') if p.is_file() and p.name != 'queue.lock')
    archive.mkdir()
    hashes = {}
    for path in paths:
        relative = path.relative_to(directory)
        destination = archive/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(path.read_bytes())
        hashes[str(relative)] = q.digest(path)
        if q.digest(destination) != hashes[str(relative)]:
            raise ValueError('Blocked evidence archive mismatch: '+str(relative))
        destination.chmod(0o444)
    atomic_json(archive/'archive_sha256.json', hashes)
    (archive/'archive_sha256.json').chmod(0o444)
    for path in sorted(archive.rglob('*'), reverse=True):
        if path.is_dir(): path.chmod(0o555)
    archive.chmod(0o555)
    return dict(path=str(archive), sha256=hashes)


def watch(directory):
    directory = Path(directory).resolve()
    with (directory/'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/'recovery_handoff.json').exists() or (directory/'recovery_result.json').exists():
            raise ValueError('Refusing recovery restart')
        try:
            manifest = q.load(directory/'queue_manifest.json')
            verified = verify_recovery(directory, manifest)
            archive = archive_blocked(directory)
            recovered = copy.deepcopy(manifest)
            path = str(Path(__file__).resolve())
            recovered['source_hashes'][path] = q.digest(path)
            frozen = directory/'locked_source'/Path(path).relative_to(Path.cwd())
            frozen.parent.mkdir(parents=True, exist_ok=True)
            with frozen.open('xb') as stream: stream.write(Path(path).read_bytes())
            frozen.chmod(0o444)
            atomic_json(directory/'recovery_manifest.json', recovered)
            (directory/'recovery_manifest.json').chmod(0o444)
            atomic_json(directory/'recovery_verification.json', dict(verified, archive=archive))
            atomic_json(directory/'recovery_handoff.json', dict(pid=os.getpid(), identity=q.identity(os.getpid()),
                started_epoch=time.time(), queue_expiry=manifest['queue_expiry'], stdout_stderr_attached=True,
                manifest_sha256=q.digest(directory/'recovery_manifest.json')))
            print(json.dumps(dict(phase='verified_pre_activation_recovery', **verified)), flush=True)
            return activate(directory, recovered)
        except Exception as exc:
            result = dict(phase='blocked', error=str(exc), traceback=traceback.format_exc(), epoch=time.time(),
                          new_worker_started=(directory/'supervisor.json').exists())
            atomic_json(directory/'recovery_result.json', result)
            print(json.dumps(result), flush=True)
            return 2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    raise SystemExit(watch(args.directory))


if __name__ == '__main__': main()
