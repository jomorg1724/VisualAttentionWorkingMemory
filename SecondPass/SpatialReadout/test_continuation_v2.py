"""Focused CPU continuation contracts; never open final stimuli."""
import copy
import json
import time
import pytest
import torch
from SecondPass.JointTraining.core import tree_equal, BalancedScheduler
from SecondPass.SpatialReadout import worker, protocol
from SecondPass.SpatialReadout import continuation_v2 as c


@pytest.fixture(scope='module')
def source():
    return c.verify_predecessor(c.SOURCE_DIR)[0]


def test_exact_migration_and_only_authorized_config(source):
    config=c.build_config(source, c.measured_plan(c.SOURCE_DIR,source), 1000.)
    migrated=c.migrate_continuation(source,config)
    for key in source:
        if key!='state': assert tree_equal(source[key],migrated[key])
    for key in source['state']:
        if key not in ('config','deadline'): assert tree_equal(source['state'][key],migrated['state'][key])
    for key,value in [('microbatch',2),('max_steps',6759),('final_test_seed_namespace',protocol.FINAL_TEST_NAMESPACE)]:
        bad=copy.deepcopy(config);bad[key]=value
        with pytest.raises(ValueError): c.migrate_continuation(source,bad)
    bad=copy.deepcopy(config);bad['optimizer']['lr']=.001
    with pytest.raises(ValueError): c.migrate_continuation(source,bad)


def test_exact_schedule_and_native_next_draw(source):
    migrated=c.migrate_continuation(source,c.build_config(source,c.measured_plan(c.SOURCE_DIR,source),1000.))
    from SecondPass.TaskSuite.suite import SuiteStream
    a=BalancedScheduler(0);b=BalancedScheduler(0)
    a.load_state_dict(source['scheduler']);b.load_state_dict(migrated['scheduler'])
    assert [a.next() for _ in range(5070)]==[b.next() for _ in range(5070)]
    x=SuiteStream('train');y=SuiteStream('train')
    x.load_state_dict(source['stream']);y.load_state_dict(migrated['stream'])
    assert tree_equal(x.batch(2,'contrast','mixed'),y.batch(2,'contrast','mixed'))


def test_plan_fixed_exposure_and_four_cycle_looks(source):
    plan=c.measured_plan(c.SOURCE_DIR,source)
    assert plan['max_steps']==6760 and plan['additional_target_requested']==5070
    assert len(plan['cell_costs'])==35
    assert len(plan['validation_steps'])==4 and plan['validation_steps'][-1]==6760
    assert all(s%13==0 for s in plan['validation_steps'])
    assert sum(e['episodes'] for e in plan['planned_exposure'].values())==162240
    assert all(e['updates']==390 for e in plan['planned_exposure'].values())
    with pytest.raises(ValueError): c.measured_plan(c.SOURCE_DIR,source,remaining=1)


def test_real_evaluator_uses_new_stream_not_metadata(tmp_path,monkeypatch):
    observed=[]
    class Sentinel(Exception): pass
    def batch(self,n,task,cell):
        observed.append((type(self),self.stream_seed(task,cell)))
        raise Sentinel
    monkeypatch.setattr(c.FinalTestStream,'batch',batch)
    class Model:
        def eval(self): pass
    old=protocol.FinalTestStream('test').stream_seed('contrast','mixed')
    with pytest.raises(Sentinel):
        c.evaluate(Model(),'test',128,200,4,'cpu',tmp_path/'unused.json',time.time()+10,cells=[('contrast','mixed')])
    assert observed==[(c.FinalTestStream,c.FINAL_NAMESPACE*100000+3000)]
    assert observed[0][1]!=old
    assert worker.FinalTestStream is protocol.FinalTestStream


def test_budget_deadline_and_no_restart(tmp_path):
    budget=c.new_budget(100.)
    c.validate_budget(budget,budget,now=101.)
    for now in (99.,86500.):
        with pytest.raises(ValueError):c.validate_budget(budget,budget,now=now)
    altered=dict(budget,deadline=budget['deadline']+1)
    with pytest.raises(ValueError):c.validate_budget(altered,altered,now=101.)
    c.activate_once(tmp_path,100.)
    with pytest.raises(FileExistsError):c.activate_once(tmp_path,101.)
    assert json.loads((tmp_path/'budget.json').read_text())==budget


def test_launch_is_interactive_one_shot(tmp_path):
    spec=c.continuation_spec('test',tmp_path,'python','repo')
    assert spec['ProcessType']=='Interactive' and spec['KeepAlive'] is False
    assert spec['EnvironmentVariables']['OMP_NUM_THREADS']=='2'


@pytest.mark.parametrize('improve,early',[(True,False),(False,False),(False,True)])
def test_real_worker_loop_four_looks_and_baseline_fallback(source,tmp_path,monkeypatch,improve,early):
    config=c.build_config(source,c.measured_plan(c.SOURCE_DIR,source),time.time())
    if early:config['final_reserve_seconds']=c.CAP
    payload=copy.deepcopy(source)
    # CPU loop test doubles, not claimed production optimizer updates.
    weights={'weight':torch.zeros(1)}
    payload['model']=weights
    saves={};calls=[]
    class Model:
        def state_dict(self):return weights
        def load_state_dict(self,value):pass
    class Session:
        def __init__(self,directory,config,payload):
            self.state=copy.deepcopy(payload['state']);self.model=Model();self.device='cpu'
            self.scheduler=BalancedScheduler(0);self.scheduler.load_state_dict(payload['scheduler'])
        def checkpoint(self,filename):
            path=str(tmp_path/filename);saves[path]=dict(payload,state=copy.deepcopy(self.state))
            self.latest={'path':path};return self.latest
        def status(self,*args,**kwargs):pass
        def train_update(self,t,cell):
            self.state['step']+=1;self.state['episodes']+=32
            return dict(task=t,cell=cell,loss=1.,seconds=.001)
    def evaluate(model,split,n,kn,micro,device,path,deadline):
        calls.append((split,Path(path).name))
        score=(.8+len(calls)/100) if improve else .1
        return dict(complete=True,cells=[],summary={'tasks':{t:dict(auc=score,chance_normalized_ba=score) for t in c.TASKS}})
    from pathlib import Path
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'budget.json').write_text(json.dumps(c.new_budget(config['cap_started'])))
    monkeypatch.setattr(worker.original,'verify_sources',lambda config:None)
    monkeypatch.setattr(worker,'load_verified',lambda receipt:saves[receipt['path']])
    monkeypatch.setattr(torch,'load',lambda path,**kwargs:saves.get(str(path),dict(source,model=weights)))
    result=c.execute_worker(tmp_path,config,payload,{},Session,evaluate)
    report=json.loads((tmp_path/'report.json').read_text())
    if early:
        assert result==2 and report['terminal_step']==1690
        assert len(report['selection_history'])==2 and not any(split=='val' for split,_ in calls)
    else:
        assert result==0 and report['terminal_step']==6760
        assert [name for split,name in calls if split=='val']==[f'validation_{step:06d}.json' for step in config['validation_steps']]
        assert len(report['selection_history'])==6
    assert report['selected_step']==(6760 if improve else 1690)
    assert len([split for split,_ in calls if split=='test'])==(2 if not improve and not early else 1)


def test_cpu_restore_name_mapped_adam_and_next_update(source,tmp_path):
    from SecondPass.SpatialReadout.state import load_verified
    config=c.build_config(source,c.measured_plan(c.SOURCE_DIR,source),1000.)
    payload=c.migrate_continuation(source,config)
    config=dict(config,device='cpu',effective_batch=2,microbatch=1)
    session=worker.TrainingSession(tmp_path,config,payload)
    assert tree_equal(session.optimizer.state_dict(),source['optimizer'])
    assert tree_equal(session.model.state_dict(),source['model'])
    session.train_update('contrast','mixed')
    receipt=session.checkpoint('cpu_update.pt');saved=load_verified(receipt)
    i=saved['optimizer_names'].index('spatial_gru.candidate.weight')
    assert saved['optimizer']['state'][i]['step']==source['optimizer']['state'][i]['step']+1
    assert not torch.equal(saved['model']['spatial_gru.candidate.weight'],source['model']['spatial_gru.candidate.weight'])
    assert saved['state']['best_step']==1690 and len(saved['state']['selection_history'])==2
