"""Executable continuation contracts; no accelerator profiling or test selection."""
import copy
import importlib
import random

import numpy as np
import pytest
import torch

from SecondPass.JointTraining import core
from SecondPass.TaskSuite.suite import SuiteStream


def implementation():
    try:
        return importlib.import_module('SecondPass.JointTraining.continuation_v2')
    except ModuleNotFoundError:
        pytest.fail('Versioned progress-preserving continuation is missing')


def test_resume_preserves_next_adam_update_stream_scheduler_rng(tmp_path):
    v = implementation()
    torch.set_num_threads(2)
    torch.manual_seed(31); np.random.seed(32); random.seed(33)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward(); opt.step(); opt.zero_grad()
    stream = SuiteStream('train'); scheduler = core.BalancedScheduler(34)
    scheduler.next(); stream.batch(2, 'orientation', 'mixed')
    cfg = dict(cap_started=100., deadline=14500., wall_cap_seconds=14400,
               max_steps=13, effective_batch=32, microbatch=4, device='cpu')
    state = dict(step=1, episodes=32, optimizer_seconds=2., deadline=14500.,
                 config=cfg, selection_history=[dict(step=1, key=[0., .5])])
    path = tmp_path/'source.pt'
    receipt = core.save_checkpoint(path, model, opt, scheduler, stream, state, 'cpu')
    expected_scheduler = scheduler.next()
    expected_batch = stream.batch(2, 'orientation', 'mixed')
    expected_rng = (torch.rand(2), np.random.rand(), random.random())
    model(torch.ones(1, 2)).sum().backward(); opt.step(); opt.zero_grad()
    expected_model = copy.deepcopy(model.state_dict())
    expected_adam = copy.deepcopy(opt.state_dict())
    amended = dict(cfg, max_steps=26, protocol_version='continuation_v2')
    resumed = v.restore_for_continuation(receipt, model, opt, scheduler, stream, amended, now=200.)
    assert resumed['step'] == 1 and resumed['episodes'] == 32
    assert resumed['optimizer_seconds'] == 2.
    assert resumed['selection_history'] == state['selection_history']
    assert resumed['config'] == amended
    assert scheduler.next() == expected_scheduler
    actual_batch = stream.batch(2, 'orientation', 'mixed')
    assert core.tree_equal(actual_batch, expected_batch)
    assert torch.equal(torch.rand(2), expected_rng[0])
    assert np.random.rand() == expected_rng[1] and random.random() == expected_rng[2]
    model(torch.ones(1, 2)).sum().backward(); opt.step()
    assert core.tree_equal(model.state_dict(), expected_model)
    assert core.tree_equal(opt.state_dict(), expected_adam)
    for bad in (dict(amended, deadline=14501.), dict(amended, effective_batch=16)):
        with pytest.raises(ValueError):
            v.restore_for_continuation(receipt, model, opt, scheduler, stream, bad, now=200.)
    with pytest.raises(ValueError, match='expired'):
        v.restore_for_continuation(receipt, model, opt, scheduler, stream, amended, now=14500.)
    with pytest.raises(ValueError, match='digest'):
        v.restore_for_continuation(dict(receipt, sha256='bad'), model, opt, scheduler, stream, amended, now=200.)


def test_planner_uses_live_costs_complete_cycles_and_original_deadline():
    v = implementation()
    assert hasattr(v, 'plan_continuation'), 'Live-cost continuation planner missing'
    rows = [dict(task=t, cell=c['id'], seconds=10.)
            for t, spec in core.TASKS.items() for c in spec['conditions']]
    validation = dict(cells=[dict(task=t, cell=c['id'], n=100 if t=='krauzlis_cued_motion' else 64, seconds=10.)
                             for t, spec in core.TASKS.items() for c in spec['conditions']])
    old = dict(deadline=14400., cap_started=0., wall_cap_seconds=14400, effective_batch=32,
               microbatch=4, val_n=64, val_krauzlis_n=100)
    scheduler = core.BalancedScheduler(23)
    exposure = {t:dict(episodes=0, updates=0, cells={c['id']:0 for c in spec['conditions']}) for t, spec in core.TASKS.items()}
    for _ in range(117):
        t, c = scheduler.next(); exposure[t]['updates'] += 1
        exposure[t]['episodes'] += 32; exposure[t]['cells'][c] += 32
    state = dict(step=117, exposure=exposure)
    cfg = v.plan_continuation(old, state, scheduler.state_dict(), rows, validation, now=3500.)
    assert cfg['deadline'] == 14400. and cfg['cap_started'] == 0.
    assert cfg['max_steps'] >= 650 and cfg['max_steps'] % 13 == 0
    assert cfg['estimated_total_seconds'] < 10900.
    assert cfg['test_n'] == 128 and cfg['test_krauzlis_n'] == 200
    assert cfg['validation_steps'] == [cfg['max_steps']]
    for t in core.TASKS:
        assert cfg['planned_exposure'][t]['episodes'] == cfg['max_steps']//13*32
        assert sum(cfg['planned_exposure'][t]['cell_episodes'].values()) == cfg['planned_exposure'][t]['episodes']
    with pytest.raises(ValueError):
        v.plan_continuation(old, state, scheduler.state_dict(), rows, validation, now=14300.)


def test_fresh_test_namespace_leaves_training_and_validation_unchanged():
    v = implementation()
    assert hasattr(v, 'FinalTestStream'), 'Fresh final-test namespace missing'
    old = SuiteStream('test'); fresh = v.FinalTestStream('test')
    assert fresh.split == 'test'
    assert fresh.stream_seed('orientation', 'mixed') != old.stream_seed('orientation', 'mixed')
    a = fresh.batch(4, 'orientation', 'mixed')
    b = v.FinalTestStream('test').batch(4, 'orientation', 'mixed')
    assert core.tree_equal(a, b)
    assert not torch.equal(a[0], old.batch(4, 'orientation', 'mixed')[0])
    with pytest.raises(ValueError):
        v.FinalTestStream('train')


def test_cpu_continuation_persists_migration_advancing_progress_and_finals(tmp_path):
    import json
    import time
    from SecondPass.JointTraining import worker as w
    from SecondPass.TaskSuite.suite import CATALOG
    from PreAttentiveVision.natural_stimuli import DATA_ROOT, sha256
    v = implementation()
    assert hasattr(v, 'run'), 'Executable continuation worker missing'
    torch.set_num_threads(2)
    model, opt = w.fresh_model(42, 'cpu')
    scheduler = core.BalancedScheduler(31); stream = SuiteStream('train')
    task, cell = scheduler.next()
    row = w.update(model, opt, stream, task, cell, 4, 2, 'cpu')
    started = time.time()
    cfg = dict(device='cpu', seed=42, scheduler_seed=31, cap_started=started,
               deadline=started+300, wall_cap_seconds=300, max_steps=1,
               effective_batch=4, microbatch=2, eval_microbatch=2,
               val_n=2, val_krauzlis_n=2, test_n=2, test_krauzlis_n=2,
               validation_steps=[1], final_reserve_seconds=90, estimated_one_test_seconds=30,
               estimated_validation_seconds=30, update_estimate_seconds=1, checkpoint_every=13,
               catalog=CATALOG, source_manifest_sha256=sha256(DATA_ROOT/'manifest.json'), source_hashes={})
    exposure = {t:dict(updates=0, episodes=0, frames=0, cells={c['id']:0 for c in s['conditions']}) for t,s in core.TASKS.items()}
    exposure[task].update(updates=1, episodes=4, frames=row['frames']); exposure[task]['cells'][cell]=4
    state = dict(step=1, episodes=4, frames=row['frames'], optimizer_seconds=row['seconds'],
                 deadline=cfg['deadline'], config=cfg, selection_history=[dict(step=1, key=[0., .5], complete=True)], best_step=1,
                 best_key=[0., .5], best_checkpoint=str(tmp_path/'source.pt'), exposure=exposure)
    receipt = core.save_checkpoint(tmp_path/'source.pt', model, opt, scheduler, stream, state, 'cpu')
    destination = tmp_path/'continuation'; destination.mkdir()
    core.append_jsonl(destination/'progress.jsonl', dict(row, step=1, cumulative_episodes=4))
    amended = dict(cfg, max_steps=2, validation_steps=[2], protocol_version=v.PROTOCOL)
    report = v.run(destination, amended, receipt)
    migrated = torch.load(destination/'migration.pt', map_location='cpu')
    original = torch.load(tmp_path/'source.pt', map_location='cpu')
    for key in ('model','optimizer','scheduler','stream','rng'):
        assert core.tree_equal(original[key], migrated[key])
    saved = torch.load(destination/'terminal.pt', map_location='cpu')
    assert saved['state']['step'] == saved['scheduler']['updates'] == 2
    assert saved['state']['episodes'] == 8
    progress = [json.loads(line) for line in (destination/'progress.jsonl').read_text().splitlines()]
    assert [r['step'] for r in progress] == [1,2]
    assert report['terminal_test']['complete_cells'] == 35
    assert report['terminal_step'] == 2
    assert json.loads((destination/'resume_integrity.json').read_text())['exact_nonallocation_state']
    assert report['final_coverage_complete']
    assert report['selected_step'] == 1
    assert report['selected_test']['complete_cells'] == 35
    for results in (report['selected_test'], report['terminal_test']):
        assert results['seed_namespace'] == v.FINAL_TEST_NAMESPACE
        for row in results['cells']:
            if row['cell'].startswith('N0'):
                assert row['balanced_accuracy'] is None and row['accuracy'] is None
                assert row['specificity'] is not None
            if row['task']=='krauzlis_cued_motion':
                assert sum(e['n'] for e in row['events'].values()) == row['n']


def test_cli_refuses_expired_cap_before_supervisor_or_worker(tmp_path):
    import json
    import subprocess
    import sys
    v = implementation()
    assert hasattr(v, 'main'), 'Bounded continuation CLI missing'
    (tmp_path/'config.json').write_text(json.dumps(dict(deadline=1.)))
    result = subprocess.run([sys.executable, '-m', 'SecondPass.JointTraining.continuation_v2',
                             'run', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'expired' in result.stderr
    assert not (tmp_path/'supervisor.json').exists()
    assert not (tmp_path/'migration.pt').exists()


def test_worker_identity_rejects_caffeinate_shell_and_python_code_echoes():
    v = implementation()
    assert hasattr(v, 'is_joint_worker'), 'Executable-aware process guard missing'
    worker = '/Python -u -m SecondPass.JointTraining.continuation_v2 worker /run'
    assert v.is_joint_worker(worker)
    assert v.is_joint_worker('/python3 -m SecondPass.JointTraining.worker run /old')
    assert not v.is_joint_worker('/usr/bin/caffeinate -i '+worker)
    assert not v.is_joint_worker('/bin/bash -c '+worker)
    assert not v.is_joint_worker('/Python -c print("'+worker+'")')
    assert not v.is_joint_worker('/Python -u -m SecondPass.JointTraining.continuation_v2 run /run')
