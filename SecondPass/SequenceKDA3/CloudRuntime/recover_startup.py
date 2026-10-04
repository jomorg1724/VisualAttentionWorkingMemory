"""Recover this unstarted fresh pod after an initial SSH timeout; keep its cap."""
import json
import subprocess
import time
import deploy as d


def wait_ssh(ident):
    end = time.time() + 240
    while time.time() < end:
        p = d.api('/pods/' + ident)
        ip, port = p.get('publicIp'), p.get('portMappings', {}).get('22')
        if ip and port:
            endpoint = dict(ip=ip, port=port, pod=ident)
            try:
                result = subprocess.run(d.ssh_command(endpoint) + ['true'],
                    capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired:
                time.sleep(5)
                continue
            if result.returncode == 0:
                d.save('ssh_endpoint.json', endpoint)
                return
            if 'Permission denied' in result.stderr or 'IDENTIFICATION HAS CHANGED' in result.stderr:
                raise ValueError('SSH authentication or host identity failed')
        time.sleep(5)
    raise TimeoutError('SSH setup exceeded original bounded startup allowance')


def main():
    archive, runtime, runtime_sha = d.check()
    pod = json.loads((d.ROOT / 'pod.json').read_text())
    budget = json.loads((d.ROOT / 'budget.json').read_text())
    failure = json.loads((d.ROOT / 'deployment_error.json').read_text())
    assert failure['type'] == 'TimeoutExpired'
    assert not (d.ROOT / 'launch.json').exists()
    assert not (d.ROOT / 'ssh_endpoint.json').exists()
    assert time.time() < budget['deadline'] - 1800
    current = d.api('/v2/pods/' + pod['id'])
    assert current['status'] in ('EXITED', 'STOPPED')
    assert float(current['env']['VAWM_HARD_DEADLINE']) == budget['hard_deadline']
    d.require_no_parallel(d.list_pods())
    with (d.ROOT / 'startup_recovery_claim.json').open('x') as f:
        json.dump(dict(pod=pod['id'], observed=time.time(),
            reason='initial SSH probe timed out before any source transfer or training',
            original_budget=budget, cap_renewed=False), f)
    pricing = json.loads((d.ROOT / 'pricing_approval.json').read_text())
    record = json.loads((d.ROOT / 'status_record_verified.json').read_text())
    success = False
    try:
        d.api('/pods/' + pod['id'] + '/start', 'POST', {})
        d.wait_ssh = wait_ssh
        result = d.install_and_launch(pod['id'], budget, pricing, record,
            archive, runtime, runtime_sha)
        success = True
        receipt = dict(pod=pod['id'], phase='verified_production',
            original_hard_deadline=budget['hard_deadline'], cap_renewed=False,
            checkpoint=result['latest_checkpoint.json'], observed=time.time())
        d.save('startup_recovery_verified.json', receipt)
        print(json.dumps(receipt))
    except BaseException as exc:
        d.save('startup_recovery_error.json', d.diagnostic(exc))
        raise
    finally:
        if not success:
            d.capture_diagnostics()
            d.stop_and_verify(pod['id'])


if __name__ == '__main__':
    main()
