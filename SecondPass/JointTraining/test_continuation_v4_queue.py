"""CPU receipt/OS-exit gate tests; fixtures are synthetic, never production data."""
import copy
import importlib
import json
import os
import pytest
import torch
from SecondPass.JointTraining import core


def implementation():
    try:return importlib.import_module('SecondPass.JointTraining.continuation_v4_queue')
    except ModuleNotFoundError:pytest.fail('Durable bounded CPU queue missing')


def predecessor(tmp_path):
    from SecondPass.JointTraining import continuation_v3
    from SecondPass.JointTraining.test_continuation_v4 import fixture_plan
    old,state,scheduler,rows,_=fixture_plan()
    old.update(protocol_version=continuation_v3.PROTOCOL,selection=continuation_v3.SELECTION,deadline=1000.,source_hashes={},continuation_source_hashes={},device='cpu')
    state.update(config=old,deadline=1000.,selection_history=[dict(step=2860,key=[.6,.1],complete=True)],best_step=2860,optimizer_seconds=1.)
    saved=dict(schema=1,state=state,scheduler=scheduler,model={'w':torch.ones(1)},optimizer={'state':{}},stream={},rng={})
    torch.save(saved,tmp_path/'terminal.pt');torch.save(saved,tmp_path/'validation_checkpoint_002860.pt')
    q=implementation()
    receipt=dict(path=str(tmp_path/'terminal.pt'),sha256=q.digest(tmp_path/'terminal.pt'),step=2860,verified=True,bytes=(tmp_path/'terminal.pt').stat().st_size)
    sup=dict(pid=99999991,worker_pid=99999992,deadline=1000.)
    result=dict(returncode=0,hard_cap_triggered=False,supervisor_pid=sup['pid'],worker_pid=sup['worker_pid'],deadline=1000.)
    cells=[dict(task=t,cell=c['id'],n=100 if t=='krauzlis_cued_motion' else 64,seconds=1.,confusion=[[100 if t=='krauzlis_cued_motion' else 64]]) for t,s in core.TASKS.items() for c in s['conditions']]
    val=dict(complete=True,complete_cells=35,cells=cells,summary=dict(tasks={t:dict(auc=.6,chance_normalized_ba=.1) for t in core.TASKS}))
    final=copy.deepcopy(val)
    for c in final['cells']:c['n']*=2;c['confusion'][0][0]*=2
    final['seed_namespace']=continuation_v3.FINAL_TEST_NAMESPACE
    report=dict(protocol=continuation_v3.PROTOCOL,stop_reason='planned_complete_cycles',failure=None,terminal_step=2860,episodes=91520,final_coverage_complete=True,terminal_checkpoint=receipt,selected_step=2860)
    selected=dict(reused_terminal=True,identical_model_verified=True,selected_step=2860,results=final)
    for name,obj in [('config',old),('budget',dict(deadline=1000.)),('supervisor',sup),('supervisor_result',result),('report',report),('validation_002860',val),('test_terminal',final),('test_selected',selected)]:
        core.atomic_json(tmp_path/(name+'.json'),obj)
    with (tmp_path/'progress.jsonl').open('w') as f:
        for step in range(716,2861):f.write(json.dumps(dict(step=step,cumulative_episodes=step*32))+'\n')
    return q


def test_complete_receipts_require_real_os_exit(tmp_path):
    q=predecessor(tmp_path)
    verified=q.verify_predecessor(tmp_path)
    assert verified['source_checkpoint']['step']==2860
    sup=json.loads((tmp_path/'supervisor.json').read_text());sup['pid']=os.getpid()
    core.atomic_json(tmp_path/'supervisor.json',sup)
    with pytest.raises(ValueError,match='alive'):q.verify_predecessor(tmp_path)


@pytest.mark.parametrize('mutation',['returncode','hardcap','incomplete','duplicate','sourcehash','checkpoint','missing','namespace','dedup'])
def test_fail_closed_completion(tmp_path,mutation):
    q=predecessor(tmp_path)
    if mutation in ('returncode','hardcap'):
        p=tmp_path/'supervisor_result.json';o=json.loads(p.read_text());o['returncode' if mutation=='returncode' else 'hard_cap_triggered']=2 if mutation=='returncode' else True;core.atomic_json(p,o)
    elif mutation in ('incomplete','duplicate','namespace'):
        p=tmp_path/'test_terminal.json';o=json.loads(p.read_text())
        if mutation=='incomplete':o['complete']=False
        elif mutation=='namespace':o['seed_namespace']=94492763
        else:o['cells'][-1]=o['cells'][0]
        core.atomic_json(p,o)
    elif mutation=='sourcehash':
        p=tmp_path/'config.json';o=json.loads(p.read_text());o['source_hashes']={str(p):'bad'};core.atomic_json(p,o)
    elif mutation=='checkpoint':(tmp_path/'terminal.pt').write_bytes(b'broken')
    elif mutation=='missing':(tmp_path/'test_selected.json').unlink()
    else:
        p=tmp_path/'test_selected.json';o=json.loads(p.read_text());o['identical_model_verified']=False;core.atomic_json(p,o)
    with pytest.raises((ValueError,FileNotFoundError)):q.verify_predecessor(tmp_path)


def test_timing_plan_uses_all_completed_cell_costs(tmp_path):
    q=predecessor(tmp_path)
    from SecondPass.JointTraining.test_continuation_v4 import fixture_plan
    old,state,scheduler,rows,_=fixture_plan()
    with (tmp_path/'progress.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    for name in ['test_terminal.json','test_selected.json']:
        data=json.loads((tmp_path/name).read_text())
        if 'results' in data:data=data['results']
        for row in data['cells']:row['seconds']=4.
        core.atomic_json(tmp_path/name,data)
    state['config']=old
    plan,basis=q.timing_plan(tmp_path,dict(state=state,scheduler=scheduler))
    assert len(basis['evaluation_timing_paths'])==3
    assert plan['estimated_validation_seconds']==pytest.approx(35*5/3)


def test_exact_predecessor_identity_and_pid_reuse():
    q=implementation()
    assert hasattr(q,'validate_identity'), 'expected OS module/role/path guard missing'
    from pathlib import Path
    source=Path('/tmp/source')
    q.validate_identity('Wed Sep 23 08:08:50 2026 /Python -u -m SecondPass.JointTraining.continuation_v3 worker /tmp/source','worker',source)
    for bad in ['Wed Sep 23 08:08:50 2026 /bin/sleep 100','Wed Sep 23 08:08:50 2026 /Python -m SecondPass.JointTraining.continuation_v3 worker /wrong','Wed Sep 23 08:08:50 2026 /usr/bin/caffeinate -i /Python -m SecondPass.JointTraining.continuation_v3 worker /tmp/source']:
        with pytest.raises(ValueError):q.validate_identity(bad,'worker',source)


def test_real_cpu_queue_handoff_and_bounded_expiry(tmp_path):
    import subprocess,sys,time
    q=implementation()
    source=tmp_path/'source';source.mkdir();directory=tmp_path/'queue';directory.mkdir()
    sleeper=subprocess.Popen(['/bin/sleep','20'])
    try:
        core.atomic_json(source/'live_status.json',dict(phase='final_test_selected',step=2860))
        core.atomic_json(directory/'queue_manifest.json',dict(source=str(source),source_hashes={},queue_expiry=time.time()+2,predecessor_identities={str(sleeper.pid):q.identity(sleeper.pid)}))
        result=subprocess.run([sys.executable,'-u','-m','SecondPass.JointTraining.continuation_v4_queue','queue',str(directory)],capture_output=True,text=True,timeout=15)
        assert result.returncode==2
        handoff=json.loads((directory/'queue_handoff.json').read_text())
        assert handoff['stdout_stderr_attached'] and handoff['pid']
        assert 'queued_waiting_predecessor' in result.stdout
        assert json.loads((directory/'queue_result.json').read_text())['error']=='Bounded predecessor queue expired'
        assert not (directory/'budget.json').exists() and not (directory/'supervisor.json').exists()
        assert sleeper.poll() is None
    finally:sleeper.terminate();sleeper.wait()


def test_queue_decision_is_bounded_and_tracks_os_not_terminal_eof():
    q=implementation()
    assert hasattr(q,'queue_decision'), 'queue deadline/exit state machine missing'
    assert q.queue_decision(now=9,expiry=10,alive=True,result=None)=='waiting'
    assert q.queue_decision(now=9,expiry=10,alive=True,result={'returncode':0,'hard_cap_triggered':False})=='waiting'
    assert q.queue_decision(now=9,expiry=10,alive=False,result={'returncode':0,'hard_cap_triggered':False})=='verify'
    for kw in [dict(now=10,expiry=10,alive=True,result=None),dict(now=9,expiry=10,alive=False,result=None),dict(now=9,expiry=10,alive=False,result={'returncode':2})]:
        with pytest.raises(ValueError):q.queue_decision(**kw)
