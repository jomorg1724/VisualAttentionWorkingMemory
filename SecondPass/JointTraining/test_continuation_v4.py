"""CPU-only exact-exposure v4 contract tests."""
import copy
import importlib
import pytest
from SecondPass.JointTraining import core


def implementation():
    try:
        return importlib.import_module('SecondPass.JointTraining.continuation_v4')
    except ModuleNotFoundError:
        pytest.fail('Versioned exact-exposure v4 implementation missing')


def fixture_plan():
    scheduler=core.BalancedScheduler(34)
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in core.TASKS.items()}
    for _ in range(2860):
        t,c=scheduler.next(); exposure[t]['updates']+=1; exposure[t]['episodes']+=32; exposure[t]['cells'][c]+=32
    rows=[dict(task=t,cell=c['id'],seconds=8.) for t,s in core.TASKS.items() for c in s['conditions']]
    validation=dict(cells=[dict(task=t,cell=c['id'],n=100 if t=='krauzlis_cued_motion' else 64,seconds=13.) for t,s in core.TASKS.items() for c in s['conditions']])
    old=dict(effective_batch=32,microbatch=4,val_n=64,val_krauzlis_n=100,test_n=128,test_krauzlis_n=200)
    return old,dict(step=2860,episodes=91520,exposure=exposure),scheduler.state_dict(),rows,validation


def test_fixed_exposure_planner_never_maximizes_or_reduces():
    v=implementation(); args=fixture_plan(); before=copy.deepcopy(args[2])
    plan=v.plan_continuation(*args)
    assert plan['max_steps']==5005
    assert plan['allocation']['additional_updates']==2145
    assert plan['allocation']['additional_episodes']==68640
    assert plan['allocation']['additional_updates_per_task']==165
    assert plan['allocation']['cumulative_episodes']==160160
    assert plan['validation_steps']==[3393,3926,4459,5005]
    assert all(e['updates']==385 and e['episodes']==12320 for e in plan['planned_exposure'].values())
    assert core.tree_equal(before,args[2])
    assert plan['estimated_total_seconds'] <= 28800
    assert plan['final_test_seed_namespace']==94492763
    for row in args[3]: row['seconds']=20.
    with pytest.raises(ValueError,match='does not fit'):
        v.plan_continuation(*args)
    with pytest.raises(ValueError,match='Every cell'):
        v.plan_continuation(*args[:3],args[3][1:],args[4])


def test_v4_worker_guard_and_namespace():
    v=implementation()
    assert hasattr(v,'is_joint_worker'), 'v4 exact executable guard missing'
    assert v.is_joint_worker('/Python -u -m SecondPass.JointTraining.continuation_v4 worker /run')
    assert v.is_joint_worker('/Python -u -m SecondPass.JointTraining.continuation_v3 worker /old')
    assert not v.is_joint_worker('/usr/bin/caffeinate -i /Python -m SecondPass.JointTraining.continuation_v4 worker /run')
    assert not v.is_joint_worker('/Python -c "SecondPass.JointTraining.continuation_v4 worker /run"')
    from SecondPass.JointTraining import continuation_v3
    assert v.FinalTestStream().stream_seed('orientation','mixed') != continuation_v3.FinalTestStream().stream_seed('orientation','mixed')


def test_v4_migration_before_old_deadline_requires_verified_normal_completion(tmp_path,monkeypatch):
    import random, hashlib
    import numpy as np
    import torch
    from SecondPass.TaskSuite.suite import SuiteStream
    v=implementation()
    assert hasattr(v,'restore_for_continuation'), 'v4 normal-completion migration missing'
    torch.set_num_threads(2)
    model=torch.nn.Linear(2,2); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    model(torch.ones(1,2)).sum().backward(); opt.step(); opt.zero_grad()
    scheduler=core.BalancedScheduler(34); scheduler.next()
    stream=SuiteStream('train'); stream.batch(2,'orientation','mixed')
    old=dict(cap_started=100.,deadline=28900.,wall_cap_seconds=28800,max_steps=13,effective_batch=32,microbatch=4,device='cpu')
    state=dict(step=1,episodes=32,optimizer_seconds=2.,deadline=old['deadline'],elapsed_cap_seconds=1000.,config=old,
        selection_history=[dict(step=1,key=[.6,.1])],best_step=1,best_key=[.6,.1],best_checkpoint='old.pt')
    receipt=core.save_checkpoint(tmp_path/'terminal.pt',model,opt,scheduler,stream,state,'cpu')
    expected_scheduler=scheduler.next(); expected_batch=stream.batch(2,'orientation','mixed')
    expected_rng=(torch.rand(2),np.random.rand(),random.random())
    model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad()
    expected_model=copy.deepcopy(model.state_dict()); expected_opt=copy.deepcopy(opt.state_dict())
    cfg=dict(old,**v.new_budget(20000.,receipt),max_steps=26,protocol_version=v.PROTOCOL,selection=v.SELECTION)
    baseline=dict(complete=True,summary=dict(tasks={t:dict(auc=.6,chance_normalized_ba=.1) for t in core.TASKS}))
    with pytest.raises((ValueError,FileNotFoundError)):
        v.restore_for_continuation(receipt,model,opt,scheduler,stream,cfg,baseline,now=20001.)
    # Only the independently tested disk/OS predecessor boundary is injected here.
    monkeypatch.setattr(v,'verify_normal_completion',lambda directory: dict(source_checkpoint=receipt))
    resumed=v.restore_for_continuation(receipt,model,opt,scheduler,stream,cfg,baseline,now=20001.)
    assert resumed['step']==1 and resumed['episodes']==32 and resumed['optimizer_seconds']==2.
    assert resumed['historical_selection']['history']==state['selection_history']
    assert resumed['best_checkpoint']==receipt['path'] and resumed['elapsed_cap_seconds']==1.
    assert scheduler.next()==expected_scheduler
    assert core.tree_equal(stream.batch(2,'orientation','mixed'),expected_batch)
    assert torch.equal(torch.rand(2),expected_rng[0]) and np.random.rand()==expected_rng[1] and random.random()==expected_rng[2]
    model(torch.ones(1,2)).sum().backward();opt.step()
    assert core.tree_equal(model.state_dict(),expected_model) and core.tree_equal(opt.state_dict(),expected_opt)
    assert hashlib.sha256((tmp_path/'terminal.pt').read_bytes()).hexdigest()==receipt['sha256']
    for bad in (dict(cfg,deadline=cfg['deadline']+1),dict(cfg,effective_batch=16),dict(cfg,authorization='bad')):
        with pytest.raises(ValueError):v.restore_for_continuation(receipt,model,opt,scheduler,stream,bad,baseline,now=20001.)
    with pytest.raises(ValueError,match='expired'):v.restore_for_continuation(receipt,model,opt,scheduler,stream,cfg,baseline,now=48800.)


def test_v4_cpu_worker_progress_and_complete_report(tmp_path,monkeypatch):
    import types
    from SecondPass.JointTraining import test_continuation_v3 as tests
    v=implementation()
    assert hasattr(v,'run'), 'v4 full worker/report path missing'
    # Receipt-boundary fail-closed behavior is tested with complete synthetic files separately.
    monkeypatch.setattr(v,'verify_normal_completion',lambda directory: dict(source_checkpoint=__import__('json').loads((directory/'source_receipt.json').read_text())))
    original=v.restore_for_continuation
    def restore(receipt,*args,**kwargs):
        core.atomic_json(tmp_path/'source_receipt.json',receipt)
        return original(receipt,*args,**kwargs)
    monkeypatch.setattr(v,'restore_for_continuation',restore)
    test=tests.test_cpu_run_persists_new_progress_final_coverage_and_report
    rebound=types.FunctionType(test.__code__,dict(test.__globals__,implementation=lambda:v),test.__name__,test.__defaults__,test.__closure__)
    rebound(tmp_path)


def test_v4_worker_cli_rejects_missing_activation_and_expired_budget(tmp_path):
    import subprocess,sys
    v=implementation()
    assert hasattr(v,'main'), 'v4 worker CLI missing'
    core.atomic_json(tmp_path/'budget.json',dict(deadline=1.))
    result=subprocess.run([sys.executable,'-m','SecondPass.JointTraining.continuation_v4','worker',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0
    assert not (tmp_path/'migration.pt').exists()
