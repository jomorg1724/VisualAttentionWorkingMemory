import importlib.util
import json
import subprocess
import sys
import pytest
import torch
from SecondPass.SpatialReadout.SpatialConsolidation import worker as w


def test_bundle_exists_and_only_declared_sources():
    assert importlib.util.find_spec('SecondPass.SpatialReadout.SpatialConsolidation.bundle'), 'builder missing'
    from SecondPass.SpatialReadout.SpatialConsolidation import bundle as b
    sources=b.dependency_sources()
    assert 'SecondPass/SpatialReadout/SpatialConsolidation/model.py' in sources
    assert all(p.suffix in ('.py','.md','.json') for p in sources.values())
    assert not any('runtime' in k.lower() or 'checkpoint' in k.lower() for k in sources)


def test_cap_and_allocation(tmp_path):
    budget=dict(cap_started=100.,hard_deadline=43300.,deadline=42700.,wall_cap_seconds=43200.,retrieval_reserve_seconds=600.)
    w.validate_budget(budget,budget,200.)
    for bad in (dict(budget,wall_cap_seconds=28800),dict(budget,retrieval_reserve_seconds=0),dict(budget,deadline=43300)):
        with pytest.raises(ValueError): w.validate_budget(bad,bad,200.)
    with pytest.raises(ValueError): w.validate_budget(budget,dict(budget,cap_started=101),200.)
    with pytest.raises(ValueError): w.validate_budget(budget,budget,42700.)
    # Synthetic planner fixture, NOT GPU performance evidence.
    cells=w.cloud.original.all_cells()
    rows=[dict(task=t,cell=c,seconds=.5,episodes=32,repeat=r) for r in range(2) for t,c in cells]
    ev=[dict(task=t,cell=c,seconds=.08,n=8) for t,c in cells]
    w.atomic_json(tmp_path/'profile/profile.json',dict(complete=True,architecture=w.VERSION,training_cells=rows,evaluation_cells=ev))
    plan=w.measured_plan(tmp_path,budget,now=200.)
    assert plan['max_steps']==46800 and plan['validation_steps']==[23400,46800]
    assert all(e['updates']==3600 for e in plan['planned_exposure'].values())
    assert plan['total_episodes']==1497600
    reduced=w.measured_plan(tmp_path,budget,now=40000.)
    assert reduced['max_steps']%13==0 and reduced['max_steps']<46800
    with pytest.raises(RuntimeError): w.measured_plan(tmp_path,budget,now=42699.)


def test_actual_evaluator_namespace(monkeypatch,tmp_path):
    class Reached(Exception): pass
    def sentinel(split):
        assert split=='test'
        s=w.FreshStream(split)
        t,c=w.cloud.original.all_cells()[0]
        assert s.stream_seed(t,c)==w.FINAL_NAMESPACE*100000+w.TASKS[t]['stream_id']*1000+s._cell(t,c)[0]
        raise Reached
    # Fail before opening/generating any sealed draw.
    monkeypatch.setitem(w.evaluate.__globals__,'FreshStream',sentinel)
    with pytest.raises(Reached): w.evaluate(None,'test',8,8,4,'cpu',tmp_path/'none.json',1e20)


@pytest.mark.parametrize('module',['worker','bundle'])
@pytest.mark.parametrize('option',['--checkpoint','--resume','--predecessor'])
def test_no_checkpoint_cli(tmp_path,module,option):
    args=['verify',str(tmp_path)] if module=='worker' else ['--output',str(tmp_path)]
    r=subprocess.run([sys.executable,'-m','SecondPass.SpatialReadout.SpatialConsolidation.'+module,*args,option,'forbidden.pt'],capture_output=True,text=True)
    assert r.returncode==2 and 'unrecognized arguments' in r.stderr


def test_supervisor_uses_new_worker_and_immutable_deadline(tmp_path,monkeypatch):
    assert hasattr(w,'supervise'), 'new worker supervisor missing'
    budget=dict(cap_started=100.,hard_deadline=43300.,deadline=42700.,wall_cap_seconds=43200.,retrieval_reserve_seconds=600.)
    w.atomic_json(tmp_path/'budget.json',budget)
    w.atomic_json(tmp_path/'config.json',budget)
    monkeypatch.setattr(w.time,'time',lambda:200.)
    observed=[]
    monkeypatch.setattr(w.cloud,'supervise',lambda cmd,deadline,directory,prefix: observed.append((cmd,deadline,directory,prefix)) or {'returncode':0})
    w.supervise(tmp_path,'run')
    command,deadline,directory,prefix=observed[0]
    assert command[-3:]==['SecondPass.SpatialReadout.SpatialConsolidation.worker','run',str(tmp_path)]
    assert deadline==42700. and directory==tmp_path
