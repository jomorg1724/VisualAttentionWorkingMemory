"""Focused state-preserving scheduling repair regression tests (CPU only)."""
import copy
import pytest
from SecondPass.SpatialReadout.qos_repair import validate_resume, repair_spec


def fixture():
    config={'deadline':1000.,'max_steps':1690}
    source={'schema':2,'state':{'step':117,'episodes':3744,'config':config},'scheduler':{'updates':3510}}
    report={'stop_reason':'signal','failure':None,'terminal_step':117,'terminal_checkpoint':{'step':117}}
    return source,config,report


def test_preserves_exact_source_and_config():
    source,config,report=fixture(); before=copy.deepcopy(source)
    assert validate_resume(source,config,report,now=500.) is source
    assert source==before


def test_rejects_deadline_reset_and_non_signal_stop():
    source,config,report=fixture()
    with pytest.raises(ValueError): validate_resume(source,dict(config,deadline=1100.),report,now=500.)
    report['stop_reason']='worker_error'
    with pytest.raises(ValueError): validate_resume(source,config,report,now=500.)


def test_rejects_expired_budget_and_wrong_sampler():
    source,config,report=fixture()
    with pytest.raises(ValueError): validate_resume(source,config,report,now=1001.)
    source['scheduler']['updates']+=1
    with pytest.raises(ValueError): validate_resume(source,config,report,now=500.)


def test_app_priority_without_restart_or_thread_expansion():
    spec=repair_spec('test','/tmp/run','/tmp/python','/tmp/repo')
    assert spec['ProcessType']=='Interactive'
    assert spec['KeepAlive'] is False
    assert spec['EnvironmentVariables']['OMP_NUM_THREADS']=='2'
