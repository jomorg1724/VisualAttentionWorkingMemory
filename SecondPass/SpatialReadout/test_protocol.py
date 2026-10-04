import importlib.util
import json
import sys
import time
from pathlib import Path
import pytest
from SecondPass.TaskSuite.suite import TASKS
from SecondPass.JointTraining.core import BalancedScheduler


def module():
    name='SecondPass.SpatialReadout.protocol'
    assert importlib.util.find_spec(name) is not None, 'Protocol and guards missing'
    return __import__(name,fromlist=['plan'])


def test_budget_all_cells_and_exact_cycles():
    p=module()
    rows=[dict(task=t,cell=c['id'],seconds=2.,eval_seconds=1.,eval_n=8) for t,s in TASKS.items() for c in s['conditions']]
    scheduler=BalancedScheduler(19).state_dict()
    plan=p.plan(rows,scheduler,remaining=28800)
    assert plan['max_steps']==2145 and plan['validation_steps']==[1066,2145]
    assert all(e['updates']==165 and e['episodes']==5280 for e in plan['planned_exposure'].values())
    assert sum(sum(e['cells'].values()) for e in plan['planned_exposure'].values())==68640
    reduced=p.plan(rows,scheduler,remaining=6000)
    assert 0<reduced['max_steps']<2145 and reduced['max_steps']%13==0
    with pytest.raises(ValueError,match='35'): p.plan(rows[:-1],scheduler,remaining=28800)
    with pytest.raises(ValueError,match='acquisition'): p.plan(rows,scheduler,remaining=500)
    budget=p.new_budget(100.)
    assert budget['deadline']==28900.
    p.validate_budget(budget,dict(budget),now=200)
    with pytest.raises(ValueError): p.validate_budget(budget,dict(budget,deadline=30000),now=200)
    with pytest.raises(ValueError): p.validate_budget(budget,dict(budget),now=28900)


def test_worker_detection_launchd_and_deadline_supervisor(tmp_path):
    p=module()
    assert p.is_worker('/tmp/env/bin/python -u -m SecondPass.SpatialReadout.worker run /tmp/run')
    assert p.is_worker('/tmp/env/bin/python -m SecondPass.JointTraining.continuation_v4 worker /tmp/run')
    assert not p.is_worker('/usr/bin/caffeinate -i /tmp/env/bin/python -m SecondPass.SpatialReadout.worker run /tmp/run')
    assert not p.is_worker('/bin/sh -c python -m SecondPass.SpatialReadout.worker run')
    assert not p.is_worker('/tmp/env/bin/python -m pytest SecondPass/SpatialReadout')
    assert not p.is_worker("/System/Example App/helper --text=unmatched'quote")
    assert p.is_worker("/tmp/env/bin/python -m SecondPass.SpatialReadout.worker run /tmp/run --note=unmatched'quote")
    plist=p.launchd_spec('org.vawm.test',tmp_path,sys.executable,Path.cwd())
    assert plist['RunAtLoad'] is True and not plist.get('KeepAlive',False)
    assert plist['ProcessType']=='Background' and plist['EnvironmentVariables']['OMP_NUM_THREADS']=='2'
    result=p.supervise([sys.executable,'-c','import time; time.sleep(10)'],time.time()+.2,tmp_path,'test')
    assert result['hard_cap_triggered'] and result['returncode']!=0
    assert json.loads((tmp_path/'test_supervisor_result.json').read_text())['hard_cap_triggered']


def test_review_approval_binds_config_and_sources(tmp_path):
    p=module()
    config={'source_hashes':{'source.py':'abc'}}
    path=tmp_path/'config.json'; path.write_text(json.dumps(config))
    with pytest.raises((ValueError,FileNotFoundError)): p.require_approval(tmp_path,config)
    receipt={'approved':True,'reviewer':'parent-independent-read-only','config_sha256':p.digest(path),'source_hashes':config['source_hashes']}
    (tmp_path/'REVIEW_APPROVED.json').write_text(json.dumps(receipt))
    p.require_approval(tmp_path,config)
    path.write_text('{}')
    with pytest.raises(ValueError,match='review'): p.require_approval(tmp_path,config)


def test_selection_no_historical_ranking_and_final_namespace():
    p=module()
    rows={t:{'auc':.75,'chance_normalized_ba':.1} for t in TASKS}
    key=p.selection_key({'complete':True,'summary':{'tasks':rows}})
    assert key==pytest.approx([.75,.1])
    assert p.selection_key({'complete':False}) is None
    from SecondPass.TaskSuite.suite import SuiteStream
    final=p.FinalTestStream('test')
    assert final.stream_seed('contrast','mixed')!=SuiteStream('test').stream_seed('contrast','mixed')
    with pytest.raises(ValueError): p.FinalTestStream('val')
