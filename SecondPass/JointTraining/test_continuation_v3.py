"""New authorization migrates clocks, not acquired state; CPU-only tests."""
import copy
import hashlib
import importlib
import random

import numpy as np
import pytest
import torch

from SecondPass.JointTraining import core
from SecondPass.TaskSuite.suite import SuiteStream


def implementation():
    try:
        return importlib.import_module('SecondPass.JointTraining.continuation_v3')
    except ModuleNotFoundError:
        pytest.fail('Explicit new-budget continuation is missing')


def test_authorized_migration_preserves_exact_next_update_and_old_bytes(tmp_path):
    v = implementation()
    torch.set_num_threads(2)
    torch.manual_seed(31); np.random.seed(32); random.seed(33)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward(); opt.step(); opt.zero_grad()
    stream = SuiteStream('train'); scheduler = core.BalancedScheduler(34)
    scheduler.next(); stream.batch(2, 'orientation', 'mixed')
    old = dict(cap_started=100., deadline=14500., wall_cap_seconds=14400,
               max_steps=13, effective_batch=32, microbatch=4, device='cpu')
    state = dict(step=1, episodes=32, optimizer_seconds=2., deadline=14500.,
                 elapsed_cap_seconds=14000., config=old, selection_history=[dict(step=1, key=[0., .5])],
                 best_step=1, best_key=[0., .5], best_checkpoint='historical.pt')
    receipt = core.save_checkpoint(tmp_path/'source.pt', model, opt, scheduler, stream, state, 'cpu')
    expected_scheduler = scheduler.next(); expected_batch = stream.batch(2, 'orientation', 'mixed')
    expected_rng = (torch.rand(2), np.random.rand(), random.random())
    model(torch.ones(1, 2)).sum().backward(); opt.step(); opt.zero_grad()
    expected_model = copy.deepcopy(model.state_dict()); expected_adam = copy.deepcopy(opt.state_dict())
    budget = v.new_budget(20000., receipt)
    cfg = dict(old, **budget, max_steps=26, protocol_version=v.PROTOCOL, selection=v.SELECTION)
    baseline = dict(complete=True, summary=dict(tasks={t:dict(auc=.6, chance_normalized_ba=.1) for t in core.TASKS}))
    resumed = v.restore_for_continuation(receipt, model, opt, scheduler, stream, cfg, baseline, now=20001.)
    assert resumed['step'] == 1 and resumed['episodes'] == 32 and resumed['optimizer_seconds'] == 2.
    assert resumed['elapsed_cap_seconds'] == 1.
    assert resumed['previous_elapsed_cap_seconds'] == 14000.
    assert resumed['historical_selection']['history'] == state['selection_history']
    assert resumed['best_checkpoint'] == receipt['path']
    assert resumed['best_key'] == pytest.approx([.6, .1])
    assert scheduler.next() == expected_scheduler
    assert core.tree_equal(stream.batch(2, 'orientation', 'mixed'), expected_batch)
    assert torch.equal(torch.rand(2), expected_rng[0])
    assert np.random.rand() == expected_rng[1] and random.random() == expected_rng[2]
    model(torch.ones(1, 2)).sum().backward(); opt.step()
    assert core.tree_equal(model.state_dict(), expected_model)
    assert core.tree_equal(opt.state_dict(), expected_adam)
    assert hashlib.sha256((tmp_path/'source.pt').read_bytes()).hexdigest() == receipt['sha256']
    for bad in (dict(cfg, deadline=cfg['deadline']+1), dict(cfg, effective_batch=16),
                dict(cfg, wall_cap_seconds=30000), dict(cfg, authorization='unapproved')):
        with pytest.raises(ValueError):
            v.restore_for_continuation(receipt, model, opt, scheduler, stream, bad, baseline, now=20001.)
    with pytest.raises(ValueError, match='expired'):
        v.restore_for_continuation(receipt, model, opt, scheduler, stream, cfg, baseline, now=48800.)
    with pytest.raises(ValueError, match='digest'):
        v.restore_for_continuation(dict(receipt, sha256='bad'), model, opt, scheduler, stream, cfg, baseline, now=20001.)


def test_exact_schedule_allocation_and_new_selection():
    v = implementation()
    assert hasattr(v, 'plan_continuation'), 'New eight-hour allocation planner missing'
    rows = [dict(task=t, cell=c['id'], seconds=10.) for t,s in core.TASKS.items() for c in s['conditions']]
    validation = dict(cells=[dict(task=t, cell=c['id'], n=100 if t=='krauzlis_cued_motion' else 64, seconds=13.)
        for t,s in core.TASKS.items() for c in s['conditions']])
    scheduler = core.BalancedScheduler(34)
    exposure = {t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in core.TASKS.items()}
    for _ in range(715):
        t,c=scheduler.next(); exposure[t]['updates']+=1; exposure[t]['episodes']+=32; exposure[t]['cells'][c]+=32
    state=dict(step=715,episodes=22880,exposure=exposure)
    old=dict(effective_batch=32,val_n=64,val_krauzlis_n=100)
    cfg=v.plan_continuation(old,state,scheduler.state_dict(),rows,validation)
    assert cfg['allocation']['additional_updates'] >= 2100
    assert cfg['allocation']['additional_updates'] % 13 == 0
    assert cfg['estimated_total_seconds'] <= 28800
    assert cfg['allocation']['next_cycle_projected_seconds'] > 28800
    assert cfg['max_steps'] == 715+cfg['allocation']['additional_updates']
    assert len(cfg['validation_steps']) == 4 and cfg['validation_steps'][-1] == cfg['max_steps']
    assert sum(e['episodes'] for e in cfg['planned_exposure'].values()) == cfg['max_steps']*32
    assert all(e['episodes']==cfg['max_steps']//13*32 for e in cfg['planned_exposure'].values())
    for t,e in cfg['planned_exposure'].items():
        assert sum(e['cell_episodes'].values()) == e['episodes']
    low=dict(complete=True,summary=dict(tasks={t:dict(auc=.6,chance_normalized_ba=.5) for t in core.TASKS}))
    high=copy.deepcopy(low)
    for m in high['summary']['tasks'].values(): m.update(auc=.7,chance_normalized_ba=-.1)
    assert v.selection_key(high) > v.selection_key(low)
    tie=copy.deepcopy(high)
    for m in tie['summary']['tasks'].values(): m['chance_normalized_ba']=0.
    assert v.selection_key(tie) > v.selection_key(high)
    assert v.selection_key(dict(high,complete=False)) is None


def test_fresh_final_namespace_and_process_guard():
    v=implementation()
    assert hasattr(v,'FinalTestStream'), 'New final-only namespace missing'
    from SecondPass.JointTraining import continuation_v2 as old
    new=v.FinalTestStream()
    assert new.stream_seed('orientation','mixed') not in (
        old.FinalTestStream().stream_seed('orientation','mixed'),SuiteStream('test').stream_seed('orientation','mixed'))
    assert core.tree_equal(new.batch(4,'orientation','mixed'),v.FinalTestStream().batch(4,'orientation','mixed'))
    with pytest.raises(ValueError): v.FinalTestStream('train')
    assert v.is_joint_worker('/Python -u -m SecondPass.JointTraining.continuation_v3 worker /run')
    assert v.is_joint_worker('/Python -u -m SecondPass.JointTraining.continuation_v2 worker /old')
    assert not v.is_joint_worker('/usr/bin/caffeinate -i /Python -m SecondPass.JointTraining.continuation_v3 worker /run')


def test_cpu_run_persists_new_progress_final_coverage_and_report(tmp_path):
    import json,time
    from SecondPass.JointTraining import worker as w
    from SecondPass.TaskSuite.suite import CATALOG
    from PreAttentiveVision.natural_stimuli import DATA_ROOT,sha256
    v=implementation()
    assert hasattr(v,'run'), 'Executable new-budget continuation missing'
    torch.set_num_threads(2)
    model,opt=w.fresh_model(42,'cpu'); scheduler=core.BalancedScheduler(31); stream=SuiteStream('train')
    task,cell=scheduler.next(); row=w.update(model,opt,stream,task,cell,4,2,'cpu')
    old=dict(device='cpu',seed=42,scheduler_seed=31,cap_started=1.,deadline=14401.,wall_cap_seconds=14400,
        max_steps=1,effective_batch=4,microbatch=2,eval_microbatch=2,val_n=2,val_krauzlis_n=2,
        test_n=2,test_krauzlis_n=2,validation_steps=[1],final_reserve_seconds=90,estimated_one_test_seconds=30,
        estimated_validation_seconds=30,update_estimate_seconds=1,checkpoint_every=13,
        catalog=CATALOG,source_manifest_sha256=sha256(DATA_ROOT/'manifest.json'),source_hashes={})
    exposure={t:dict(updates=0,episodes=0,frames=0,cells={c['id']:0 for c in s['conditions']}) for t,s in core.TASKS.items()}
    exposure[task].update(updates=1,episodes=4,frames=row['frames']); exposure[task]['cells'][cell]=4
    state=dict(step=1,episodes=4,frames=row['frames'],optimizer_seconds=row['seconds'],deadline=old['deadline'],
        config=old,selection_history=[dict(step=1,key=[0.,.5],complete=True)],best_step=1,best_key=[0.,.5],
        best_checkpoint=str(tmp_path/'source.pt'),exposure=exposure)
    receipt=core.save_checkpoint(tmp_path/'source.pt',model,opt,scheduler,stream,state,'cpu')
    baseline=w.evaluate(model,'val',2,2,2,'cpu',tmp_path/'baseline.json')
    # Small draws may omit a class; selection eligibility needs complete defined metrics.
    for m in baseline['summary']['tasks'].values():
        if m['auc'] is None: m['auc']=.5
        if m['chance_normalized_ba'] is None: m['chance_normalized_ba']=0.
    core.atomic_json(tmp_path/'baseline.json',baseline)
    cfg=dict(old,**v.new_budget(time.time(),receipt),max_steps=2,validation_steps=[2],protocol_version=v.PROTOCOL,
        selection=v.SELECTION,baseline_validation=str(tmp_path/'baseline.json'),final_test_seed_namespace=v.FINAL_TEST_NAMESPACE)
    dest=tmp_path/'new'; dest.mkdir()
    report=v.run(dest,cfg,receipt)
    original=torch.load(tmp_path/'source.pt',map_location='cpu'); migrated=torch.load(dest/'migration.pt',map_location='cpu')
    assert all(core.tree_equal(original[k],migrated[k]) for k in ('model','optimizer','scheduler','stream','rng'))
    saved=torch.load(dest/'terminal.pt',map_location='cpu')
    assert saved['state']['step']==saved['scheduler']['updates']==2
    assert saved['state']['episodes']==8 and saved['state']['optimizer_seconds']>row['seconds']
    progress=[json.loads(s) for s in (dest/'progress.jsonl').read_text().splitlines()]
    assert [p['step'] for p in progress]==[2]
    assert progress[0]['additional_episodes']==4
    assert report['final_coverage_complete']
    assert report['terminal_test']['complete_cells']==35
    selected=report['selected_test'].get('results',report['selected_test'])
    assert selected['complete_cells']==35
    assert selected['seed_namespace']==v.FINAL_TEST_NAMESPACE
    assert len(json.loads((dest/'validation_curves.json').read_text())['looks'])>=2
    assert (dest/'REPORT.md').exists()
    assert hashlib.sha256((tmp_path/'source.pt').read_bytes()).hexdigest()==receipt['sha256']


def test_cli_refuses_expired_budget_without_starting_worker(tmp_path):
    import json,subprocess,sys
    v=implementation()
    assert hasattr(v,'main'), 'Bounded supervisor CLI missing'
    (tmp_path/'budget.json').write_text(json.dumps(dict(deadline=1.)))
    result=subprocess.run([sys.executable,'-m','SecondPass.JointTraining.continuation_v3','run',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'refusing' in result.stderr.lower()
    assert not (tmp_path/'supervisor.json').exists()


def test_budget_launch_guard_detects_renewal():
    v=implementation()
    assert hasattr(v,'validate_launch'), 'Immutable new budget guard missing'
    receipt=dict(sha256='unit-test')
    cfg=v.new_budget(20000.,receipt)
    v.validate_launch(cfg,cfg,receipt,now=20001.)
    for bad in (dict(cfg,deadline=50000.),dict(cfg,cap_started=21000.)):
        with pytest.raises(ValueError): v.validate_launch(bad,cfg,receipt,now=20001.)
    with pytest.raises(ValueError): v.validate_launch(cfg,cfg,receipt,now=48800.)
