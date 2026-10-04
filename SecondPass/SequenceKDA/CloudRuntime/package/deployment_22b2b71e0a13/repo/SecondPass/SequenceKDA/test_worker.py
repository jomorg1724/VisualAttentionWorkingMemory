"""Native renderer, persisted fresh state, budget and measured pin invariants."""
import copy
import json
from pathlib import Path
import time
import pytest
import torch
from SecondPass.JointTraining.core import tree_equal
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
from . import worker as w

def budget(now):
    return dict(cap_started=now-60,deadline=now-60+28200,hard_deadline=now-60+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800)

def test_native_stream_exact_and_event_counts():
    stream=w.FreshStream('test')
    for cell,frames in zip(w.CELLS,(29,37,45)):
        _,condition=stream._cell(w.TASK,cell)
        native=SpatialBatteryStream(stream.stream_seed(w.TASK,cell),'test')
        events=[]; positive=0
        for _ in range(100):
            a,y,meta=stream.batch(1,w.TASK,cell)
            b,yy,nmeta=native.batch(1,w.TASK,condition['kwargs'])
            assert torch.equal(a,b) and torch.equal(y,yy)
            assert tuple(a.shape)==(1,frames,3,100,100)
            for original,wrapped in zip(nmeta,meta): assert all(wrapped[k]==v for k,v in original.items())
            events.append(meta[0]['event_type']); positive+=int(y[0])
        assert {e:events.count(e) for e in ('target','foil','catch')}==dict(target=57,foil=29,catch=14)
        assert positive==57

def test_persisted_native_checkpoint3_rejects_tamper(tmp_path):
    w.cpu_setup(); now=time.time()
    cfg=dict(device='cpu',backend='chunk',chunk_size=32,effective_batch=1,microbatch=1,cap_started=now,deadline=now+600)
    session=w.Session(tmp_path,cfg); initial=session.checkpoint('initial.pt')
    payload=w.load_verified(initial)
    assert payload['optimizer']['state']=={} and payload['stream']['streams']==[]
    assert payload['state']['step']==0 and payload['scheduler']['updates']==0
    for _ in range(3):
        session.train_update(*session.scheduler.next())
        session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
    result=w.verify_progress(tmp_path)
    assert result['step']==3 and result['episodes']==3 and result['all_active_parameters_changed']
    assert result['stream_counts']==dict(B12=1,B20=1,B28=1)
    assert (tmp_path/'startup_ready.json').exists()
    saved=w.load_verified(result['checkpoint'])
    assert all(float(opt['step'])==3 for opt in saved['optimizer']['state'].values())
    assert all(v.dtype==torch.float32 for v in saved['model'].values())
    corrupt=copy.deepcopy(result['checkpoint']); corrupt['sha256']='0'*64
    with pytest.raises(ValueError): w.load_verified(corrupt)
    with pytest.raises(FileExistsError): session.checkpoint('checkpoint_000003.pt')

def test_stream_resume_exact_without_photos():
    first=w.FreshStream('train'); first.batch(2,w.TASK,'B20'); state=first.state_dict()
    restored=w.FreshStream('train'); restored.load_state_dict(state)
    a,y,m=first.batch(3,w.TASK,'B20'); b,yy,mm=restored.batch(3,w.TASK,'B20')
    assert torch.equal(a,b) and torch.equal(y,yy) and m==mm
    assert state['dataset_manifest_sha256'] is None
    assert tree_equal(state,restored.state_dict()) is False
    with pytest.raises(ValueError): restored.batch(1,'image_recognition','N4_H4')

def test_scheduler_equal_cycles_and_split_independence():
    s=w.Scheduler(w.SCHEDULER_SEED)
    for _ in range(10): assert {s.next()[1] for _ in range(3)}==set(w.CELLS)
    assert s.state_dict()['updates']==30
    seeds={w.FreshStream(split).stream_seed(w.TASK,c) for split in ('train','val','test') for c in w.CELLS}
    assert len(seeds)==9

def profile_document(cost):
    return dict(complete=True,architecture=w.VERSION,
        rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=cost) for _ in range(2) for c in w.CELLS],
        evaluation_cells=[dict(cell=c,n=20,seconds=1) for c in w.CELLS])

def test_plan_targets_native_equal_exposure_and_refuses_bad_timings(tmp_path):
    now=time.time(); b=budget(now); pd=tmp_path/'profile'; pd.mkdir()
    w.atomic_json(pd/'profile.json',profile_document(1))
    p=w.measured_plan(tmp_path,b,now)
    assert p['max_steps']==10000 and p['total_episodes']==320000 and p['validation_steps']==[5000,10000]
    counts=[r['updates'] for r in p['planned_exposure'].values()]
    assert max(counts)-min(counts)<=1
    assert p['estimated_total_remaining_seconds']<b['deadline']-now
    slower=profile_document(30); w.atomic_json(pd/'profile.json',slower)
    with pytest.raises(RuntimeError): w.measured_plan(tmp_path,b,now)
    incomplete=profile_document(1); incomplete['rows'][0]['episodes']=4
    w.atomic_json(pd/'profile.json',incomplete)
    with pytest.raises(ValueError): w.measured_plan(tmp_path,b,now)
    changed=dict(b,deadline=b['deadline']+1)
    with pytest.raises(ValueError): w.validate_budget(b,changed,now)
    with pytest.raises(ValueError): w.validate_budget(b,b,b['hard_deadline'])

def test_validation_selection_and_all_positive_metrics():
    rows=[]
    for cell in w.CELLS:
        labels=[1]*57+[0]*43; probs=[[.1,.9]]*100
        meta=[dict(event_type=e,target_location=i%2) for e,n in [('target',57),('foil',29),('catch',14)] for i in range(n)]
        row=w.score_cell(w.TASK,cell,labels,probs,meta); rows.append(row)
        assert row['balanced_accuracy']==.5 and row['accuracy']==.57
        assert row['target_hit_rate']==row['foil_false_alarm_rate']==row['catch_false_positive_rate']==1
    assert w.selection_key(dict(complete=True,cells=rows))==[.5,.5]
    assert w.selection_key(dict(complete=False,cells=rows)) is None
    assert w.selection_key(dict(complete=True,cells=rows[:2])) is None

def test_bundle_dependency_sources_exclude_training_artifacts():
    from .bundle import dependency_sources
    sources=dependency_sources()
    assert 'WorkingMemory/SpatialTaskBattery/stimuli.py' in sources
    assert 'SecondPass/SequenceKDA/kda_backend.py' in sources
    assert 'SecondPass/TaskSuite/catalog.json' in sources
    assert all(Path(p).suffix in ('.py','.md','.json') for p in sources)
    assert not any('package/' in p or 'artifacts/' in p or '/data/' in p for p in sources)

def test_run_selection_paired_final_and_completion_native_cpu(tmp_path,monkeypatch):
    """Execute real finalization with explicitly small native test-only sampling."""
    w.cpu_setup(); now=time.time(); b=budget(now)
    cfg=dict(b,device='cpu',backend='chunk',chunk_size=32,effective_batch=1,microbatch=1,
        source_hashes={},max_steps=2,checkpoint_every=100,validation_steps=[1,2],
        final_reserve_seconds=30,update_estimate_seconds=2,estimated_one_test_seconds=10)
    w.atomic_json(tmp_path/'budget.json',b); w.atomic_json(tmp_path/'config.json',cfg)
    w.atomic_json(tmp_path/'allocation.json',dict(max_steps=2,test_fixture=True))
    original=w.evaluate
    def reduced_eval(model,split,count,microbatch,device,output,deadline):
        result=original(model,split,10,2,device,output,deadline)
        result['test_fixture']=dict(actual_trials_per_condition=10,production_requested=count,
            purpose='CPU finalization codepath only; no production or acquisition evidence')
        w.atomic_json(output,result)
        return result
    monkeypatch.setattr(w,'evaluate',reduced_eval)
    assert w.run(tmp_path)==0
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['terminal_step']==2 and report['episodes']==2 and report['final_coverage_complete']
    assert report['pinned_updates']==2 and report['exposure_completed'] and not report['cap_limited']
    assert report['selected_step'] in (1,2) and len(report['selection_history'])==2
    terminal=report['terminal_test']; selected=report['selected_test'].get('results',report['selected_test'])
    assert all(a['trial_ids']==b['trial_ids'] and a['labels']==b['labels'] for a,b in zip(terminal['cells'],selected['cells']))
    assert {r['n'] for r in terminal['cells']}=={10}
    expected=max(report['selection_history'],key=lambda r:(*r['key'],-r['step']))['step']
    assert expected==report['selected_step']
    w.completion(tmp_path,'complete')
    completed=json.loads((tmp_path/'cloud_completion.json').read_text())
    assert completed['actual_updates']==completed['pinned_updates']==2
    assert completed['exposure_completed'] and not completed['cap_limited']
    assert Path(completed['selected_checkpoint']).exists()
    paths={r['relative_path'] for r in completed['artifact_manifest']}
    assert {'initial.pt','selected.pt','latest.pt','terminal.pt','report.json','REPORT.md','test_selected.json','test_terminal.json'}<=paths
    for row in completed['artifact_manifest']:
        assert w.digest(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes']
    with pytest.raises(RuntimeError): w.run(tmp_path)

def test_cap_limited_finalization_preserves_pinned_exposure():
    base=dict(failure=None,complete=True,actual=800,pinned=1000,reason='wall_budget_reserve')
    assert w.scientifically_finalized(**base)
    assert base['actual']==800 and base['pinned']==1000
    assert w.scientifically_finalized(**dict(base,actual=1000,reason='planned_complete'))
    for change in (dict(failure={'type':'error'}),dict(complete=False),dict(actual=0),dict(actual=1001),
        dict(reason='signal'),dict(reason='worker_error')):
        assert not w.scientifically_finalized(**dict(base,**change))
