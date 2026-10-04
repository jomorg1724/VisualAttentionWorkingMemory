import copy
from collections import Counter
import importlib
import numpy as np
import torch


def core():
    try:
        return importlib.import_module('SecondPass.JointTraining.core')
    except ModuleNotFoundError:
        raise AssertionError('JointTraining scheduler implementation missing') from None


def test_scheduler_balance_and_exact_replay():
    c = core()
    from SecondPass.TaskSuite.suite import CATALOG
    s = c.BalancedScheduler(918273)
    rows = [s.next() for _ in range(13 * 12)]
    tasks = [t['id'] for t in CATALOG['tasks']]
    for i in range(0, len(rows), 13):
        assert Counter(t for t, cell in rows[i:i+13]) == Counter(tasks)
    for task in CATALOG['tasks']:
        counts = Counter(cell for t, cell in rows if t == task['id'])
        assert set(counts) == {x['id'] for x in task['conditions']}
        assert len(set(counts.values())) == 1
    state = copy.deepcopy(s.state_dict())
    expected = [s.next() for _ in range(61)]
    restored = c.BalancedScheduler(0)
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(61)] == expected


def test_metrics_n0_absent_class_and_krauzlis():
    c = core()
    assert hasattr(c, 'score_cell'), 'cell performance scorer missing'
    p = np.array([[.9, .1], [.1, .9], [.8, .2], [.2, .8]])
    n0 = c.score_cell('image_recognition', 'N0_H3', [0]*4, p, [{}]*4)
    assert n0['balanced_accuracy'] is None and n0['auc'] is None
    assert n0['specificity'] == .5 and n0['false_positive_rate'] == .5
    assert n0['accuracy'] is None
    m = [{'event_type': x, 'target_location': i % 2} for i, x in enumerate(['target','target','foil','catch'])]
    k = c.score_cell('krauzlis_cued_motion', 'B12', [1,1,0,0], p, m)
    assert k['events']['target']['n'] == 2
    assert k['events']['target']['positive_rate'] == .5
    assert k['events']['foil']['positive_rate'] == 0
    assert k['events']['catch']['positive_rate'] == 1
    assert k['confusion'] == [[1,1],[1,1]]
    absent = c.score_cell('motion_direction', 'mixed', [0]*4, np.tile([.7,.1,.1,.1], (4,1)), [{}]*4)
    assert absent['balanced_accuracy'] is None and absent['auc'] is None


def test_checkpoint_restores_next_sample_optimizer_scheduler_and_rng(tmp_path):
    c = core()
    assert hasattr(c, 'save_checkpoint'), 'full-state checkpoint missing'
    import random
    from SecondPass.TaskSuite.suite import SuiteStream
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model(torch.ones(1,2)).sum().backward(); opt.step()
    stream = SuiteStream('train'); sched = c.BalancedScheduler(12)
    sched.next(); stream.batch(2, 'orientation', 'mixed')
    state = {'step': 1, 'deadline': 12345, 'selection_history': [], 'config': {'seed':42}}
    path = tmp_path / 'state.pt'
    receipt = c.save_checkpoint(path, model, opt, sched, stream, state, 'cpu')
    assert receipt['verified'] and receipt['step'] == 1
    expected_sched = sched.next()
    expected_sample = stream.batch(2, 'orientation', 'mixed')
    expected_rng = (torch.rand(3), np.random.random(), random.random())
    expected_opt = copy.deepcopy(opt.state_dict())
    with torch.no_grad(): model.weight.zero_()
    opt.state.clear()
    recovered = c.restore_checkpoint(path, model, opt, sched, stream, 'cpu')
    assert recovered == state
    assert sched.next() == expected_sched
    sample = stream.batch(2, 'orientation', 'mixed')
    assert torch.equal(sample[0], expected_sample[0])
    assert sample[2] == expected_sample[2]
    assert torch.equal(torch.rand(3), expected_rng[0])
    assert np.random.random() == expected_rng[1] and random.random() == expected_rng[2]
    for key, val in expected_opt['state'].items():
        for name, tensor in val.items():
            assert torch.equal(opt.state_dict()['state'][key][name], tensor)


def test_cap_does_not_permit_extra_update():
    c = core()
    assert hasattr(c, 'may_update'), 'pre-update wall budget guard missing'
    assert c.may_update(now=50, deadline=100, reserve=20, estimate=10, step=0, max_steps=13)
    assert not c.may_update(now=70, deadline=100, reserve=20, estimate=10, step=0, max_steps=13)
    assert not c.may_update(now=10, deadline=100, reserve=20, estimate=10, step=13, max_steps=13)
    assert not c.may_update(now=100, deadline=100, reserve=0, estimate=0, step=0, max_steps=13)
    assert not c.may_update(now=0, deadline=100, reserve=20, estimate=10, step=0, max_steps=13, stopped=True)


def test_selection_equal_cells_equal_tasks_and_worst_task():
    c = core()
    assert hasattr(c, 'summarize'), 'selection aggregation missing'
    from SecondPass.TaskSuite.suite import TASKS
    rows = []
    for task, spec in TASKS.items():
        for cell in spec['conditions']:
            rows.append(dict(task=task, cell=cell['id'], balanced_accuracy=.5 if spec['classes']==2 else .25, auc=.5))
    # Excluded empty sets cannot improve/worsen checkpoint selection.
    for r in rows:
        if r['cell'].startswith('N0'): r.update(balanced_accuracy=1., auc=1.)
    out = c.summarize(rows)
    assert out['selection_key'] == [0., .5]
    assert set(out['groups']) == {'sensory','orientation_auxiliary','spatial'}
    for r in rows:
        if r['task']=='orientation': r['balanced_accuracy'] = .25
    assert c.summarize(rows)['selection_key'][0] == -.5


def test_cpu_thread_limits():
    from SecondPass.JointTraining import worker as w
    assert hasattr(w, 'cpu_setup'), 'explicit intra/inter-op thread cap missing'
    w.cpu_setup()
    assert torch.get_num_threads() == 2
    assert torch.get_num_interop_threads() == 2


def test_real_update_and_fixed_validation_draws(tmp_path):
    try:
        w = importlib.import_module('SecondPass.JointTraining.worker')
    except ModuleNotFoundError:
        raise AssertionError('executable optimizer/evaluation path missing') from None
    from SecondPass.TaskSuite.suite import SuiteStream
    torch.set_num_threads(2)
    model, opt = w.fresh_model(48291, 'cpu')
    before = model.blocks[0][0].weight.detach().clone()
    row = w.update(model, opt, SuiteStream('train'), 'orientation', 'mixed', 4, 2, 'cpu', diagnostics=True)
    assert row['episodes'] == 4 and row['frames'] == 8
    assert row['grad_norm'] > 0 and np.isfinite(row['loss'])
    assert not torch.equal(before, model.blocks[0][0].weight)
    for name in ('sensory','kda','gru','active_head'):
        assert row['diagnostics'][name]['grad_norm'] > 0
        assert row['diagnostics'][name]['update_norm'] > 0
    assert all(p.requires_grad for p in model.parameters())
    cells = [('orientation', 'mixed'), ('image_recognition', 'N0_H3')]
    a = w.evaluate(model, 'val', 4, 4, 2, 'cpu', tmp_path/'a.json', cells=cells)
    b = w.evaluate(model, 'val', 4, 4, 2, 'cpu', tmp_path/'b.json', cells=cells)
    for rows in (a['cells'], b['cells']):
        for row in rows: row.pop('seconds')
    assert a['cells'] == b['cells']
    assert a['complete_cells'] == 2
    assert b['cells'][1]['balanced_accuracy'] is None


def test_expired_production_saves_fresh_checkpoint_without_update(tmp_path):
    from SecondPass.JointTraining import worker as w
    assert hasattr(w, 'run'), 'bounded production worker missing'
    import time, json
    from SecondPass.TaskSuite.suite import CATALOG
    from PreAttentiveVision.natural_stimuli import DATA_ROOT, sha256
    cfg = dict(device='cpu', seed=92821, scheduler_seed=8271, cap_started=time.time()-15000,
        deadline=time.time()-10, wall_cap_seconds=14400, max_steps=13,
        effective_batch=4, microbatch=2, eval_microbatch=2,
        val_n=4, val_krauzlis_n=4, test_n=4, test_krauzlis_n=4,
        validation_steps=[13], final_reserve_seconds=100, update_estimate_seconds=1,
        checkpoint_every=13, catalog=CATALOG, source_manifest_sha256=sha256(DATA_ROOT/'manifest.json'),
        source_hashes={})
    w.run(tmp_path, cfg)
    status = json.loads((tmp_path/'live_status.json').read_text())
    assert status['step'] == 0 and status['phase'] == 'finished'
    assert not (tmp_path/'progress.jsonl').exists()
    checkpoint = torch.load(tmp_path/'checkpoint_000000.pt', map_location='cpu')
    assert checkpoint['state']['step'] == 0 and not checkpoint['optimizer']['state']
    assert checkpoint['state']['config'] == cfg
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['stop_reason'] == 'wall_budget_reserve'
    assert report['terminal_test'] is None


def test_budget_plan_and_supervisor_deadline(tmp_path):
    try:
        launch = importlib.import_module('SecondPass.JointTraining.launch')
    except ModuleNotFoundError:
        raise AssertionError('bounded supervisor/planner missing') from None
    import time, sys
    deadline = time.time()+.3
    result = launch.supervise([sys.executable, '-c', 'import time; time.sleep(30)'], deadline, tmp_path)
    assert result['hard_cap_triggered']
    assert time.time()-deadline < 3
    profile = {'rows':[dict(task=t, seconds=10., frames=640, episodes=64, eval_seconds=1., eval_n=8, sequence_frames=10) for t in core().TASKS]}
    cfg = launch.plan(profile, remaining=10000)
    assert cfg['max_steps'] % 13 == 0
    assert len(cfg['validation_steps']) == 4
    assert cfg['effective_batch'] in (16,32,64)
    assert cfg['final_reserve_seconds'] > 0
    assert cfg['estimated_total_seconds'] < 10000
    for task, spec in core().TASKS.items():
        expected = cfg['max_steps']//13 * cfg['effective_batch']
        assert cfg['planned_exposure'][task]['episodes'] == expected
        assert sum(cfg['planned_exposure'][task]['cell_episodes'].values()) == expected
        assert len(set(cfg['planned_exposure'][task]['cell_episodes'].values())) == 1
