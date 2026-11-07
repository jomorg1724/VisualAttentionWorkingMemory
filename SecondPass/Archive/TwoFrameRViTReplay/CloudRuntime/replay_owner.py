"""User-authorized fresh replay replacement on the already guarded RViT pod."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/vawm_two_frame_rvit_01')
MODULE = 'SecondPass.TwoFrameRViTReplay.worker'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def execute():
    run = ROOT / 'run'
    write(run / 'owner_claim.json', dict(pid=os.getpid(), observed=time.time(), restart=False,
          experiment='1000trial_10epoch_replay_fresh', inherited_checkpoint=False))
    budget = json.loads((ROOT / 'assets/budget.json').read_text())
    guard = json.loads((ROOT / 'billing_guard.json').read_text())
    if budget['hard_deadline'] != 1791004564.370006 or guard['deadline'] != budget['hard_deadline']:
        raise ValueError('Original RViT hard deadline must remain unchanged')
    if not (guard['state'] == 'armed' and guard['authenticated_read'] and time.time() < budget['deadline']):
        raise ValueError('Existing independent billing guard not armed')
    while not (ROOT / 'replay_activate.json').exists():
        if time.time() >= min(budget['deadline'], guard['updated'] + 900):
            raise TimeoutError('Bounded replacement activation expired')
        time.sleep(1)
    try:
        manifest = json.loads((ROOT / 'replay_source_manifest.json').read_text())
        import hashlib
        for row in manifest['files']:
            path = ROOT / 'repo' / row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Replay source mismatch')
        for args in [('prepare', str(run), '--budget', str(ROOT / 'assets/budget.json')),
                     ('supervise-profile', str(run)), ('pin', str(run)),
                     ('supervise-run', str(run))]:
            write(run / 'owner_stage.json', dict(stage=args[0], observed=time.time()))
            subprocess.run([sys.executable, '-u', '-m', MODULE, *args], cwd=ROOT / 'repo',
                env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'), check=True)
        sys.path.insert(0, str(ROOT))
        from remote_owner import finalize
        finalize(ROOT)
        return 0
    except BaseException as exc:
        write(run / 'owner_error.json', dict(type=type(exc).__name__, message=str(exc)[:1000], observed=time.time()))
        write(run / 'owner_run.json', dict(status='failed', pid=os.getpid(), observed=time.time()))
        write(run / 'DEPLOYMENT_FAILED', dict(stop=True, reason='setup_or_worker_failure', observed=time.time()))
        return 1


if __name__ == '__main__':
    raise SystemExit(execute())
