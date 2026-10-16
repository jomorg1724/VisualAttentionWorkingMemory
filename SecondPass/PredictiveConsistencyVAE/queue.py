"""CPU-only queue and independent deadline guard for one authorized local run.

Pending queue imports no Torch and creates no CUDA context. Successful completion
of BOTH predecessor arms and release of the shared OS GPU lock are prerequisites.
"""
from __future__ import annotations
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    for attempt in range(9):
        try:
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return
        except OSError as exc:
            transient = getattr(exc, 'winerror', None) in (5, 32, 33)
            if transient and attempt < 8:
                time.sleep(min(.025 * 2 ** attempt, .4))
                continue
            if path.name != 'queue_status.json':
                raise
            print('WARNING skipped queue telemetry after bounded retries: ' + str(exc), file=sys.stderr, flush=True)
            return


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py')) + [
        ROOT / 'SecondPass' / 'PixelContrastiveVAE' / 'train.py',
        ROOT / 'SecondPass' / 'PixelContrastiveVAE' / 'model.py',
        ROOT / 'SecondPass' / 'PixelContrastiveVAE' / 'dataset.py',
        ROOT / 'SecondPass' / 'VariationalMotionPredictor' / 'dataset.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files))}


class FileLock:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('a+b')
        try:
            # Windows byte-range locks can reject the initial read itself.
            # Treat both the probe and locking operation as occupied-lock errors.
            self.file.seek(0)
            if self.file.read(1) == b'':
                self.file.write(b'0')
                self.file.flush()
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.file.close()
            raise RuntimeError('Lock is already held: ' + str(path))

    def close(self):
        self.file.close()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def process_identity(pid):
    """Read-only PID + creation-time identity; never use Windows os.kill(pid,0)."""
    if not pid:
        return None
    if os.name != 'nt':
        try:
            # Linux start-time tick separates reused PID identities.
            stat = Path('/proc') / str(pid) / 'stat'
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return None
            return dict(pid=int(pid), created=fields[19])
        except (OSError, IndexError):
            return None
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                      ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                      ctypes.POINTER(wintypes.FILETIME))
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            return None
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(system), ctypes.byref(user)):
            return None
        return dict(pid=int(pid), created=str((created.dwHighDateTime << 32) + created.dwLowDateTime))
    finally:
        kernel.CloseHandle(handle)


def terminate_owned(identity):
    if not identity or process_identity(identity['pid']) != identity:
        return False
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME))
        handle = kernel.OpenProcess(0x1000 | 1, False, identity['pid'])
        if not handle:
            return False
        try:
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(system), ctypes.byref(user)):
                return False
            held_creation = str((created.dwHighDateTime << 32) + created.dwLowDateTime)
            if held_creation != identity['created']:
                return False
            return bool(kernel.TerminateProcess(handle, 124))
        finally:
            kernel.CloseHandle(handle)
    import signal
    os.kill(identity['pid'], signal.SIGTERM)
    return True


def hidden_options():
    if os.name != 'nt':
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    return dict(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)


def prevent_sleep(enabled):
    if os.name == 'nt':
        # ES_SYSTEM_REQUIRED only; display may sleep normally.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (1 if enabled else 0))


def predecessor_status(path, expected_updates, accept_cancelled_control=False):
    """Require published completed full two-arm results, not a live snapshot."""
    completion = read_json(path / 'completion_verified.json')
    cancelled = read_json(path / 'vae_control' / 'cancellation_receipt.json')
    user_cancelled = bool(accept_cancelled_control and cancelled and
                          cancelled.get('status') == 'cancelled' and
                          cancelled.get('authorized_by') == 'user')
    if (completion and completion.get('status') == 'complete') or user_cancelled:
        names = ('pixel_contrastive',) if user_cancelled else ('pixel_contrastive', 'vae_control')
        for name in names:
            receipt = read_json(path / name / 'completion_verified.json')
            config = read_json(path / name / 'config.json')
            final = read_json(path / name / 'final_test.json')
            if not receipt or receipt.get('status') != 'complete' or receipt.get('stop_reason') != 'planned_complete':
                return 'blocked', 'Root completion does not have both successful arm receipts'
            if receipt.get('step') != expected_updates or not config or config.get('target_updates') != expected_updates:
                return 'blocked', 'Predecessor exposure differs from required completed allocation'
            if not final or final.get('n', 0) < 12 or not (path / name / 'best.pt').exists() or not (path / name / 'latest.pt').exists():
                return 'blocked', 'Predecessor final artifacts/checkpoints are incomplete'
        config = read_json(path / 'production_config.json') or {}
        resume = (read_json(path / 'resume_worker_receipt.json') or
                  read_json(path / 'resume_receipt.json') or {})
        live = read_json(path / 'live_status.json') or {}
        # Do not bind to the crashed original PID: a reviewed resume may have
        # completed under a new worker. Only completion plus current identity
        # exit and shared-lock availability will activate the next job.
        current_pid = resume.get('worker_pid') or resume.get('pid') or live.get('pid') or config.get('pid')
        pids = {current_pid, live.get('pid')}
        if any(process_identity(pid) is not None for pid in pids if pid):
            return 'pending', 'Completed artifacts exist; waiting for predecessor process exit'
        return 'ready', ('Joint arm complete; control explicitly cancelled by user' if user_cancelled
                         else 'Both complete predecessor arms verified')
    # Failure is durable BLOCKED unless a reviewed resumed run is clearly live.
    live = read_json(path / 'live_status.json') or {}
    current_alive = bool(process_identity(live.get('pid')))
    failures = [path / name / 'failure.json' for name in ('pixel_contrastive', 'vae_control')]
    if any(p.exists() for p in failures) and not current_alive:
        return 'blocked', 'Predecessor failed; preserve pending queue and await explicit recovery'
    return 'pending', 'Waiting for completion of BOTH current experiment arms'


def guard(args):
    root_identity = process_identity(args.owner_pid)
    atomic_json(args.queue_dir / 'guard_started.json', dict(pid=os.getpid(), owner_identity=root_identity,
                                                          independent=True))
    while True:
        if (args.queue_dir / 'guard_stop.json').exists():
            atomic_json(args.queue_dir / 'guard_result.json', dict(status='normal_stop', pid=os.getpid()))
            return
        budget = read_json(args.run / 'budget.json')
        child = read_json(args.queue_dir / 'active_child.json')
        owner_gone = root_identity is not None and process_identity(args.owner_pid) != root_identity
        expired = bool(budget and time.time() >= float(budget['deadline']))
        if owner_gone or expired:
            stopped = terminate_owned(child.get('identity')) if child else False
            atomic_json(args.queue_dir / 'guard_result.json', dict(status='terminated_owned_worker',
                       reason='owner_exited' if owner_gone else 'finite_cap', stopped=stopped,
                       child=child, budget=budget, pid=os.getpid()))
            return
        time.sleep(2)


def run_child(command, args, phase):
    log = args.queue_dir / (phase + '.log')
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, **hidden_options())
        identity = process_identity(child.pid)
        if identity is None:
            time.sleep(.1)
            identity = process_identity(child.pid)
        atomic_json(args.queue_dir / 'active_child.json', dict(phase=phase, identity=identity,
                       command=command, started=time.time()))
        return child.wait()


def queue(args):
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    owner_lock = FileLock(args.queue_dir / 'queue_owner.lock')
    prevent_sleep(True)
    guard_process = None
    try:
        config_path = args.queue_dir / 'queue.json'
        config = read_json(config_path)
        requested = dict(predecessor=str(args.predecessor.resolve()), run=str(args.run.resolve()),
                         expected_predecessor_updates=args.predecessor_updates,
                         requested_target_updates=args.target_updates, wall_seconds=args.wall_seconds,
                         microbatch_pairs=args.microbatch, source_hashes=source_hashes(),
                         accept_cancelled_control=args.accept_cancelled_control,
                         budget_origin=str(args.budget_origin.resolve()) if args.budget_origin else None,
                         budget_origin_sha256=hashlib.sha256(args.budget_origin.read_bytes()).hexdigest() if args.budget_origin else None,
                         local_only=True, no_cuda_while_pending=True, automatic_activation_once=True)
        if config:
            if any(config.get(key) != value for key, value in requested.items()):
                raise RuntimeError('Queue arguments/source changed; existing pending configuration is preserved')
            if (args.queue_dir / 'activation.json').exists():
                raise RuntimeError('Already activated; no automatic retry or budget renewal')
        else:
            atomic_json(config_path, dict(**requested, queued=time.time()))
        atomic_json(args.queue_dir / 'queue_status.json', dict(status='queued', pid=os.getpid(), budget_started=False))
        while True:
            status, reason = predecessor_status(args.predecessor, args.predecessor_updates,
                                                args.accept_cancelled_control)
            if source_hashes() != requested['source_hashes']:
                status, reason = 'blocked', 'Source changed after queue snapshot'
            if status == 'blocked':
                atomic_json(args.queue_dir / 'queue_status.json', dict(status='blocked', reason=reason,
                           pid=os.getpid(), budget_started=False, pending_preserved=True))
                return
            if status == 'ready':
                try:
                    test_lock = FileLock(args.lock_path)
                    test_lock.close()
                    break
                except RuntimeError:
                    reason = 'Predecessor complete; shared GPU lock still owned'
            atomic_json(args.queue_dir / 'queue_status.json', dict(status='queued', reason=reason,
                       pid=os.getpid(), budget_started=False, checked=time.time()))
            time.sleep(args.poll_seconds)
        args.run.mkdir(parents=True, exist_ok=True)
        if (args.run / 'budget.json').exists() or (args.run / 'production_config.json').exists():
            raise RuntimeError('New run directory already contains an activated experiment')
        if args.budget_origin:
            origin = read_json(args.budget_origin)
            if not origin or time.time() >= float(origin['deadline']) - 120:
                raise RuntimeError('Original corrective budget is exhausted; no renewal')
            if float(origin['deadline']) - float(origin['started']) > args.wall_seconds + 1e-6:
                raise RuntimeError('Budget origin exceeds authorized four-hour cap')
        atomic_json(args.queue_dir / 'activation.json', dict(pid=os.getpid(), activated=time.time(),
                   predecessor_verified=True, source_hashes=requested['source_hashes']))
        guard_command = [sys.executable, '-m', 'SecondPass.PredictiveConsistencyVAE.queue',
                         '--guard', '--queue-dir', str(args.queue_dir), '--run', str(args.run),
                         '--owner-pid', str(os.getpid())]
        with (args.queue_dir / 'guard.log').open('a', encoding='utf-8') as stream:
            guard_process = subprocess.Popen(guard_command, cwd=ROOT, stdout=stream,
                                              stderr=subprocess.STDOUT, **hidden_options())
        base = [sys.executable, '-m', 'SecondPass.PredictiveConsistencyVAE.train',
                '--run', str(args.run), '--profile-json', str(args.run / 'profile.json'),
                '--target-updates', str(args.target_updates), '--wall-seconds', str(args.wall_seconds),
                '--microbatch', str(args.microbatch), '--lock-path', str(args.lock_path)]
        if args.budget_origin:
            base += ['--budget-origin', str(args.budget_origin)]
        result = run_child(base + ['--profile-only'], args, 'profile')
        if result:
            raise RuntimeError('New profile failed; no automatic restart')
        profile = read_json(args.run / 'profile.json')
        if not profile or profile.get('source_hashes') != requested['source_hashes']:
            raise RuntimeError('Profile source proof differs from queued snapshot')
        if time.time() >= float(profile['deadline']) - 120:
            raise RuntimeError('Original cap exhausted after profile; do not launch production')
        target = int(profile['target_updates'])
        base[base.index('--target-updates') + 1] = str(target)
        atomic_json(args.queue_dir / 'queue_status.json', dict(status='training', pid=os.getpid(),
                   budget_started=True, deadline=profile['deadline'], pinned_target_updates=target))
        result = run_child(base, args, 'production')
        receipt = read_json(args.run / 'completion_verified.json')
        if result or not receipt or receipt.get('status') != 'complete' or receipt.get('step') != target:
            raise RuntimeError('Production/evaluation failed or capped; preserve artifacts, no retry')
        atomic_json(args.queue_dir / 'queue_status.json', dict(status='complete', pid=os.getpid(),
                   budget_started=True, completion=receipt))
    except BaseException as exc:
        atomic_json(args.queue_dir / 'queue_status.json', dict(status='blocked', pid=os.getpid(),
                   reason=str(exc), pending_preserved=True, budget_started=(args.run / 'budget.json').exists()))
        raise
    finally:
        if guard_process:
            active = read_json(args.queue_dir / 'active_child.json')
            if active and active.get('identity'):
                # An interrupted wait must not disarm the independent guard
                # while its own training child is still running.
                terminate_owned(active['identity'])
            atomic_json(args.queue_dir / 'guard_stop.json', dict(owner_pid=os.getpid(), stopped=time.time()))
            try:
                guard_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Only the guard created by this queue process is terminated.
                guard_process.terminate()
        prevent_sleep(False)
        owner_lock.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--budget-origin', type=Path,
                        help='Original budget JSON when a reviewed fresh correction must keep the same deadline')
    parser.add_argument('--predecessor', type=Path)
    parser.add_argument('--predecessor-updates', type=int, default=25024)
    parser.add_argument('--accept-cancelled-control', action='store_true',
                        help='Require completed joint arm plus an explicit user-authorized control cancellation receipt')
    parser.add_argument('--target-updates', type=int, default=25024)
    parser.add_argument('--wall-seconds', type=int, default=14400)
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--owner-pid', type=int)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--lock-path', type=Path, default=ROOT / 'LocalRuntime' / 'local_gpu_worker.lock')
    args = parser.parse_args()
    if args.guard and not args.owner_pid:
        parser.error('Guard requires exact owner PID')
    if not args.guard and args.predecessor is None:
        parser.error('Queue requires predecessor experiment directory')
    if args.target_updates % 64 or args.wall_seconds != 14400 or args.poll_seconds < 1:
        parser.error('Require complete64-update pools, authorized14400s cap and positive polling')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    guard(arguments) if arguments.guard else queue(arguments)
