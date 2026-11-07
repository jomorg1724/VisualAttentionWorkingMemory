"""Replace only the SSH-failed, never-uploaded pod, without renewing its budget."""
import base64
import json
import time
import tomllib
from pathlib import Path
import deploy as d
from recover_startup import wait_ssh


def main():
    archive, runtime, runtime_sha = d.check()
    old = json.loads((d.ROOT / 'pod.json').read_text())
    budget = json.loads((d.ROOT / 'budget.json').read_text())
    error = json.loads((d.ROOT / 'startup_recovery_error.json').read_text())
    assert 'not enough free GPUs' in error['message']
    assert not (d.ROOT / 'ssh_endpoint.json').exists()
    assert not (d.ROOT / 'launch.json').exists()
    assert time.time() < budget['deadline'] - 1800
    d.require_no_parallel(d.list_pods())
    current = d.api('/v2/pods/' + old['id'])
    assert current['status'] in ('EXITED', 'STOPPED')
    assert current['name'] == 'vawm-sequence-kda3-01'
    assert float(current['env']['VAWM_HARD_DEADLINE']) == budget['hard_deadline']
    with (d.ROOT / 'replacement_claim.json').open('x') as f:
        json.dump(dict(previous_pod=old['id'], original_budget=budget,
            observed=time.time(), no_source_transferred=True, cap_renewed=False), f)
    d.save('failed_unstarted_pod.json', old)
    # No source archive, worker or checkpoint ever reached this failed pod.
    d.api('/v2/pods/' + old['id'], 'DELETE')
    d.save('failed_unstarted_pod_deleted.json', dict(pod=old['id'],
        observed=time.time(), reason='SSH failure before any upload or optimizer'))
    quote = d.live_quote(1, 24)
    # The original quote was verified at the original cap origin. Preserve it
    # for the scientific worker, while enforcing the current replacement quote.
    pricing = json.loads((d.ROOT / 'initial_pricing_approval.json').read_text())
    record = json.loads((d.ROOT / 'status_record_verified.json').read_text())
    key = tomllib.loads((Path.home() / '.runpod/config.toml').read_text())['default']['api_key']
    d._SECRETS = (key,)
    boot = 'import base64;exec(compile(base64.b64decode(' + repr(base64.b64encode((d.ROOT / 'pod_guard.py').read_bytes()).decode()) + '),"pod_guard.py","exec"))'
    body = dict(name='vawm-sequence-kda3-01', cloud='SECURE',
        gpu=dict(id='NVIDIA A40', count=1, minCudaVersion='12.8', minRamPerGpu=24, minVcpuCountPerGpu=4),
        image=d.IMAGE, disk=30, mounts=dict(persistent=dict(size=20, path='/workspace')),
        ports=['22/tcp'], entrypoint=['python3', '-u', '-c', boot], cmd=[],
        startSsh=True, startJupyter=False, globalNetworking=False,
        env=dict(PUBLIC_KEY=d.KEY.with_suffix('.pub').read_text().strip(),
            VAWM_STOP_API_KEY=key, VAWM_HARD_DEADLINE=str(budget['hard_deadline']),
            VAWM_STATUS_TEMPLATE_ID=record['id'], OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    ident = None
    success = False
    try:
        p = d.api('/v2/pods', 'POST', body)
        ident = p['id']
        d.save('pod.json', dict(id=ident, budget=budget, hard_deadline=budget['hard_deadline'],
            attempt_started=budget['cap_started'], authorized_max_usd=5,
            bundle_sha256=d.frozen_archive_sha(), remote_root=d.REMOTE,
            replaced_unstarted_pod=old['id'], cap_renewed=False))
        current = d.api('/v2/pods/' + ident)
        assert current['cost'] <= quote['compute_hourly_usd']
        assert current['gpu']['id'] == 'NVIDIA A40' and current['gpu']['count'] == 1
        d.verify_boot(current, body)
        d.save('provisioned.json', {k:current[k] for k in ('id', 'cost', 'status', 'gpu')})
        d.wait_ssh = wait_ssh
        result = d.install_and_launch(ident, budget, pricing, record, archive, runtime, runtime_sha)
        success = True
        receipt = dict(pod=ident, phase='verified_production', cap_renewed=False,
            hard_deadline=budget['hard_deadline'], checkpoint=result['latest_checkpoint.json'])
        d.save('replacement_verified.json', receipt)
        print(json.dumps(receipt))
    except BaseException as exc:
        d.save('replacement_error.json', d.diagnostic(exc))
        if ident is None:
            # An ambiguous provider response must never trigger duplicate creation.
            for candidate in d.list_pods():
                if candidate.get('name') == body['name']:
                    p = d.api('/v2/pods/' + candidate['id'])
                    if p.get('env', {}).get('VAWM_HARD_DEADLINE') == str(budget['hard_deadline']):
                        ident = candidate['id']
                        d.save('pod.json', dict(id=ident, budget=budget, remote_root=d.REMOTE,
                            hard_deadline=budget['hard_deadline'], ambiguous_create_recovered=True))
                        break
        raise
    finally:
        body['env'].pop('VAWM_STOP_API_KEY', None)
        key = None
        if ident and not success:
            d.capture_diagnostics()
            d.stop_and_verify(ident)
        d._SECRETS = ()


if __name__ == '__main__':
    main()
