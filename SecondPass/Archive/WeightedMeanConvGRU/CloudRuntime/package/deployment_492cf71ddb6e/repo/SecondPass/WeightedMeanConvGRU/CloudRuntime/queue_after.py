"""Parent-owned one-shot queue; waits for verified predecessor retrieval/deletion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREDECESSOR = ROOT.parents[1] / 'SingleStimulusRViT' / 'CloudRuntime'
PREDECESSOR_POD = 'jxbmb44y9wamhl'


def read(path):
    return json.loads(path.read_text())


def write_status(state, **extra):
    tmp = ROOT / 'queue_status.tmp'
    tmp.write_text(json.dumps(dict(state=state, pid=os.getpid(), updated=time.time(),
        predecessor_pod=PREDECESSOR_POD, cap_started=(ROOT/'budget.json').exists(), **extra), indent=2)+'\n')
    tmp.replace(ROOT / 'queue_status.json')


def predecessor_ready(root=PREDECESSOR):
    """Consume the existing mirror's completed verification; never restart it."""
    if read(root / 'pod.json')['id'] != PREDECESSOR_POD:
        raise ValueError('Predecessor pod identity changed')
    result_path = root / 'mirror_result.json'
    if not result_path.exists():
        return False
    result = read(result_path)
    if result.get('status') != 'complete_verified_deleted':
        raise ValueError('Predecessor mirror ended without complete retrieval/deletion')
    cleanup = read(root / 'cleanup_verified.json')
    receipt = read(root / 'retrieval_verified.json')
    if not (cleanup.get('pod') == receipt.get('pod') == PREDECESSOR_POD
            and cleanup.get('deleted') is True and cleanup.get('artifacts_verified') is True
            and receipt.get('complete') is True):
        raise ValueError('Predecessor cleanup/retrieval evidence incomplete')
    checks = receipt.get('cpu_checkpoint_reload', {})
    if set(checks) != {'selected', 'terminal'} or not all(
            row.get('verified') is True and row.get('step', 0) > 0 for row in checks.values()):
        raise ValueError('Predecessor final checkpoint evidence missing')
    completion = root / 'artifacts' / 'cloud_completion.json'
    if hashlib.sha256(completion.read_bytes()).hexdigest() != receipt.get('completion_sha256'):
        raise ValueError('Predecessor completion identity mismatch')
    if read(completion).get('status') != 'complete':
        raise ValueError('Predecessor science incomplete')
    return True


def main():
    with (ROOT / 'queue_claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),queued_at=time.time(),predecessor_pod=PREDECESSOR_POD),stream)
    # Waiting spends none of the new allowance. Fail closed if cleanup remains
    # unresolved one hour beyond the predecessor's existing absolute deadline.
    expires = read(PREDECESSOR / 'budget.json')['hard_deadline'] + 3600
    write_status('waiting_predecessor_cleanup', waiting_expires=expires)
    try:
        while time.time() < expires:
            if predecessor_ready():
                import deploy
                pods = deploy.list_pods()
                if any(p.get('id') == PREDECESSOR_POD for p in pods):
                    raise ValueError('Predecessor deletion readback not confirmed')
                deploy.require_no_parallel(pods)
                write_status('activating', waiting_expires=expires)
                with (ROOT / 'queued_deploy.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, '-u', str(ROOT/'deploy.py'),
                        'deploy','--use-existing-guard-credential','--storage-reserve-usd','1',
                        '--retention-hours','24'], cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    write_status('activating', deploy_pid=child.pid)
                    code = child.wait()
                if code:
                    write_status('deployment_failed', returncode=code,
                        evidence='deployment_error.json', no_retry=True)
                    return code
                proof = read(ROOT / 'production_verified.json')
                if not deploy.production_ready(proof):
                    raise ValueError('Deployment exited without verified persisted optimizer evidence')
                pod = read(ROOT / 'pod.json')
                write_status('verified_production', pod=pod['id'],
                    budget=pod['budget'], evidence='production_verified.json')
                return 0
            time.sleep(30)
        write_status('predecessor_cleanup_unresolved', no_rental_created=True, no_retry=True)
        return 1
    except Exception as exc:
        # Never print credentials or transport exception details.
        write_status('blocked', error_type=type(exc).__name__, no_retry=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
