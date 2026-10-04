"""CPU-only regression for blocked exit transitions and explicit recovery."""
import importlib
import json
import os
from pathlib import Path
import time

import pytest
from SecondPass.JointTraining import continuation_v4_queue as q
from SecondPass.JointTraining.core import atomic_json
from SecondPass.JointTraining.test_continuation_v4_queue import predecessor


def recovery():
    # Before the versioned recovery exists, exercise the original rejecting path.
    try:
        return importlib.import_module('SecondPass.JointTraining.continuation_v4_recovery')
    except ModuleNotFoundError:
        return q


def blocked(tmp_path):
    source = tmp_path/'source'; source.mkdir(); predecessor(source)
    directory = tmp_path/'queue'; directory.mkdir()
    sup = q.load(source/'supervisor.json')
    identities = {str(sup[k]): f'Wed Sep 23 08:08:50 2026 /Python -u -m SecondPass.JointTraining.continuation_v3 {role} {source}'
                  for k, role in [('pid', 'run'), ('worker_pid', 'worker')]}
    manifest = dict(source=str(source), predecessor=sup, predecessor_identities=identities,
                    source_hashes={}, queue_expiry=time.time()+300,
                    source_config_sha256=q.digest(source/'config.json'),
                    source_budget_sha256=q.digest(source/'budget.json'))
    atomic_json(directory/'queue_manifest.json', manifest)
    atomic_json(directory/'queue_handoff.json', dict(pid=99999993, manifest_sha256=q.digest(directory/'queue_manifest.json'), queue_expiry=manifest['queue_expiry']))
    failure = dict(phase='blocked', error='Predecessor PID identity changed; fail closed', new_worker_started=False)
    atomic_json(directory/'queue_result.json', failure)
    atomic_json(directory/'queue_status.json', failure)
    return directory, source, manifest


@pytest.mark.parametrize('changed', ['Wed Sep 23 08:08:50 2026 <defunct>',
                                     'Wed Sep 23 08:09:50 2026 /Python -m unrelated'])
def test_original_queue_exit_transient_and_true_reuse_fail_closed(tmp_path, monkeypatch, changed):
    directory, source, manifest = blocked(tmp_path)
    (directory/'queue_handoff.json').unlink()
    original_identity = q.identity
    monkeypatch.setattr(q, 'identity', lambda pid: changed if str(pid) in manifest['predecessor_identities'] else original_identity(pid))
    assert q.watch(directory) == 2
    result = q.load(directory/'queue_result.json')
    assert result['error'] == 'Predecessor PID identity changed; fail closed'
    assert not (directory/'budget.json').exists()


def test_explicit_recovery_after_normal_completion_preserves_blocked_attempt(tmp_path, monkeypatch):
    directory, source, manifest = blocked(tmp_path)
    r = recovery()
    before = {p.name: q.digest(p) for p in directory.glob('*.json')}
    activated = []
    monkeypatch.setattr(r, 'activate', lambda d, m: activated.append(m) or 0)
    assert r.watch(directory) == 0
    assert len(activated) == 1
    assert activated[0]['queue_expiry'] == manifest['queue_expiry']
    for name, sha in before.items():
        assert q.digest(directory/name) == sha
        archived = directory/'blocked_attempt_01'/name
        assert q.digest(archived) == sha
        assert archived.stat().st_mode & 0o222 == 0
    receipt = q.load(directory/'recovery_verification.json')
    assert receipt['predecessor']['normal_completion']
    assert receipt['predecessor']['source_checkpoint']['step'] == 2860
    assert not (directory/'budget.json').exists()  # activation is the only mocked MPS boundary


@pytest.mark.parametrize('mutation', ['zombie', 'reuse', 'live_original', 'live_queue', 'budget', 'activation', 'migration',
                                     'expiry', 'missing_final', 'failed_predecessor', 'checkpoint', 'manifest', 'other_worker'])
def test_recovery_rejects_unsafe_state_without_activation(tmp_path, monkeypatch, mutation):
    directory, source, manifest = blocked(tmp_path)
    r = recovery()
    if mutation in {'zombie', 'reuse', 'live_original', 'live_queue'}:
        pid = '99999993' if mutation == 'live_queue' else str(manifest['predecessor']['worker_pid'])
        observed = {'zombie': 'Wed Sep 23 08:08:50 2026 <defunct>', 'reuse': 'Wed Sep 23 08:09:50 2026 /bin/sleep 1',
                    'live_original': manifest['predecessor_identities'][str(manifest['predecessor']['worker_pid'])],
                    'live_queue': '/Python -m queue'}[mutation]
        old_identity = q.identity
        monkeypatch.setattr(q, 'identity', lambda p: observed if str(p) == pid else old_identity(p))
    elif mutation in {'budget', 'activation', 'migration'}:
        (directory/('migration.pt' if mutation == 'migration' else mutation+'.json')).write_text('untouched')
    elif mutation == 'expiry':
        manifest['queue_expiry'] = time.time()-1
        atomic_json(directory/'queue_manifest.json', manifest)
    elif mutation == 'missing_final': (source/'test_selected.json').unlink()
    elif mutation == 'failed_predecessor':
        result = q.load(source/'supervisor_result.json'); result['returncode'] = 2
        atomic_json(source/'supervisor_result.json', result)
    elif mutation == 'checkpoint': (source/'terminal.pt').write_bytes(b'bad')
    elif mutation == 'manifest':
        manifest['queue_expiry'] += 30; atomic_json(directory/'queue_manifest.json', manifest)
    else:
        monkeypatch.setattr(q, 'no_other_worker', lambda: (_ for _ in ()).throw(ValueError('Another local joint worker exists')))
    original = q.digest(directory/'queue_result.json')
    activated = []
    monkeypatch.setattr(r, 'activate', lambda *args: activated.append(args))
    assert r.watch(directory) == 2
    assert not activated
    assert q.digest(directory/'queue_result.json') == original
    if mutation == 'budget': assert (directory/'budget.json').read_text() == 'untouched'
    else: assert not (directory/'budget.json').exists()
    failure = q.load(directory/'recovery_result.json')
    if mutation in {'zombie', 'reuse', 'live_original', 'live_queue'}:
        assert observed in failure['error']
    with pytest.raises(ValueError, match='restart'): r.watch(directory)


def test_recovery_real_activation_keeps_blocked_receipts_and_cap_before_worker(tmp_path, monkeypatch):
    import torch
    from SecondPass.JointTraining import launch
    from SecondPass.JointTraining.test_continuation_v4 import fixture_plan
    directory, source, manifest = blocked(tmp_path)
    r = recovery()
    cfg = q.load(source/'config.json')
    cfg['baseline_validation'] = str(source/'validation_002860.json')
    atomic_json(source/'config.json', cfg)
    for name in ['terminal.pt', 'validation_checkpoint_002860.pt']:
        saved = torch.load(source/name, map_location='cpu'); saved['state']['config'] = cfg
        torch.save(saved, source/name)
    report = q.load(source/'report.json')
    report['terminal_checkpoint'].update(sha256=q.digest(source/'terminal.pt'), bytes=(source/'terminal.pt').stat().st_size)
    atomic_json(source/'report.json', report)
    rows = fixture_plan()[3]
    with (source/'progress.jsonl').open('w') as stream:
        for i, step in enumerate(range(716, 2861)):
            stream.write(json.dumps(dict(rows[i % len(rows)], step=step, cumulative_episodes=step*32))+'\n')
    manifest['source_config_sha256'] = q.digest(source/'config.json')
    atomic_json(directory/'queue_manifest.json', manifest)
    handoff = q.load(directory/'queue_handoff.json'); handoff['manifest_sha256'] = q.digest(directory/'queue_manifest.json')
    atomic_json(directory/'queue_handoff.json', handoff)
    original = {name:q.digest(directory/name) for name in ['queue_manifest.json', 'queue_handoff.json', 'queue_status.json', 'queue_result.json']}
    calls = []
    def supervised(command, deadline, target):
        budget = q.load(target/'budget.json'); config = q.load(target/'config.json')
        assert budget['deadline'] == deadline == budget['cap_started']+28800
        assert budget['cap_started'] <= time.time()
        assert config['max_steps'] == 5005 and config['validation_steps'] == [3393,3926,4459,5005]
        assert config['allocation']['additional_updates'] == 2145
        assert config['allocation']['additional_episodes'] == 68640
        assert 'SecondPass.JointTraining.continuation_v4' in command
        assert q.load(target/'activation.json')['deadline'] == deadline
        calls.append(command)
        atomic_json(target/'report.json', {'synthetic_cpu_test':True})
        return dict(returncode=0, hard_cap_triggered=False)
    monkeypatch.setattr(launch, 'supervise', supervised)
    assert r.watch(directory) == 0
    assert len(calls) == 1
    assert q.load(directory/'recovery_result.json')['phase'] == 'finished'
    assert all(q.digest(directory/name) == sha for name, sha in original.items())
    cap = q.digest(directory/'budget.json')
    with pytest.raises(ValueError, match='restart'): r.watch(directory)
    assert q.digest(directory/'budget.json') == cap
