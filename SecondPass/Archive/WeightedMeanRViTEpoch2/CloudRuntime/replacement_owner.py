"""Fresh epoch2 production after an atomic same-pod, same-deadline handoff."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/vawm_weighted_mean_rvit_01')
STAGE = ROOT / 'epoch2_staged_run'
MODULE = 'SecondPass.WeightedMeanRViTEpoch2.cloud_worker'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    old_owner = int(sys.argv[1])
    pid = os.getpid()
    write(STAGE / 'owner_claim.json', dict(pid=pid, observed=time.time(),
          fresh_model=True, replay_epochs=2, same_pod=True, cap_extended=False))
    # A failed laptop handoff must not leave a suspended owner renting compute.
    limit = time.time() + 300
    while time.time() < limit:
        path = ROOT / 'run' / 'owner_claim.json'
        if path.exists() and json.loads(path.read_text()).get('pid') == pid:
            break
        time.sleep(.2)
    else:
        os.kill(old_owner, signal.SIGCONT)
        return 1
    # Parent kills old owner before allowing new training to start. It can never
    # awaken and write a failure marker into the replacement's run directory.
    limit = time.time() + 60
    while not (ROOT / 'epoch2_handoff_committed.json').exists():
        if time.time() >= limit:
            write(ROOT / 'run' / 'DEPLOYMENT_FAILED',
                  dict(stop=True, reason='setup_or_worker_failure'))
            return 1
        time.sleep(.2)
    run = ROOT / 'run'
    try:
        sys.path.insert(0, str(ROOT))
        from remote_owner import finalize
        guard = json.loads((ROOT / 'billing_guard.json').read_text())
        budget = json.loads((run / 'budget.json').read_text())
        assert guard['state'] == 'armed' and guard['deadline'] == budget['hard_deadline']
        assert time.time() < budget['deadline']
        write(run / 'owner_stage.json', dict(stage='fresh_epoch2_production', observed=time.time()))
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        for key in ('VAWM_STOP_API_KEY', 'RUNPOD_API_KEY'):
            env.pop(key, None)
        subprocess.run([sys.executable, '-u', '-m', MODULE, 'supervise-run', str(run)],
                       cwd=ROOT / 'repo_epoch2', env=env, check=True)
        finalize(ROOT)
        return 0
    except Exception as exc:
        write(run / 'owner_run.json', dict(status='failed', pid=pid, error_type=type(exc).__name__))
        write(run / 'DEPLOYMENT_FAILED', dict(stop=True, reason='setup_or_worker_failure'))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
