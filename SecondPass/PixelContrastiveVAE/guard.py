"""Independent deadline guard: only terminates the original owned worker PID.

Windows uses a held process handle and creation timestamp, preventing a recycled
PID from being affected. POSIX records /proc start identity when available.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


class OwnedProcess:
    def __init__(self, pid):
        self.pid = pid
        self.handle = None
        if os.name == 'nt':
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            self.kernel.GetProcessTimes.restype = wintypes.BOOL
            self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
            self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            self.kernel.TerminateProcess.restype = wintypes.BOOL
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.CloseHandle.restype = wintypes.BOOL
            self.handle = self.kernel.OpenProcess(0x00100000 | 0x1000 | 0x0001, False, pid)
            if not self.handle:
                raise ProcessLookupError('Owned worker exited or cannot be opened')
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not self.kernel.GetProcessTimes(self.handle, ctypes.byref(created), ctypes.byref(exited),
                                               ctypes.byref(kernel), ctypes.byref(user)):
                raise OSError(ctypes.get_last_error(), 'GetProcessTimes failed')
            self.identity = (created.dwHighDateTime << 32) | created.dwLowDateTime
        else:
            self.identity = self.posix_identity()

    def posix_identity(self):
        stat = Path(f'/proc/{self.pid}/stat')
        if stat.exists():
            # Name in parentheses can contain spaces; starttime is field22.
            return stat.read_text().rsplit(')', 1)[1].split()[19]
        os.kill(self.pid, 0)
        return 'pid_without_proc_starttime'

    def alive(self):
        if os.name == 'nt':
            from ctypes import wintypes
            code = wintypes.DWORD()
            return bool(self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code))) and code.value == 259
        try:
            return self.posix_identity() == self.identity
        except (ProcessLookupError, FileNotFoundError):
            return False

    def terminate(self):
        if not self.alive():
            return False
        if os.name == 'nt':
            if not self.kernel.TerminateProcess(self.handle, 124):
                raise OSError(ctypes.get_last_error(), 'Owned worker termination failed')
        else:
            os.kill(self.pid, signal.SIGTERM)
        return True

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)


def main(args):
    receipt = args.run / 'guard_status.json'
    try:
        owned = OwnedProcess(args.pid)
    except ProcessLookupError:
        write(receipt, dict(status='worker_already_exited', pid=args.pid))
        return
    write(receipt, dict(status='armed', pid=args.pid, process_creation_identity=owned.identity,
                        deadline=args.deadline, parent_pid=args.parent_pid, guard_pid=os.getpid()))
    try:
        while owned.alive() and time.time() < args.deadline:
            time.sleep(min(5., max(.01, args.deadline - time.time())))
        if owned.alive():
            terminated = owned.terminate()
            write(receipt, dict(status='deadline_terminated_owned_worker', pid=args.pid,
                                process_creation_identity=owned.identity, terminated=terminated,
                                deadline=args.deadline, stopped=time.time()))
        else:
            write(receipt, dict(status='worker_exited', pid=args.pid, process_creation_identity=owned.identity,
                                deadline=args.deadline, stopped=time.time()))
    finally:
        owned.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int)
    main(parser.parse_args())
