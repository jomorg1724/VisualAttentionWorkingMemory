"""Focused implementation-only adapter checks; no native training launch."""
import time
import json
import pytest
import torch
from SecondPass.DelayedFrameGRU import worker as w
from SecondPass.DelayedFrameGRU.model import DelayedFrameGRU

def test_fresh_adapter_constructor_optimizer_coverage_and_rng(tmp_path):
    w.cpu_setup()
    cfg=dict(device='cpu',effective_batch=32,microbatch=4,cap_started=time.time(),deadline=time.time()+600)
    session=w.Session(tmp_path,cfg)
    expected=sum(p.numel() for p in DelayedFrameGRU().parameters())
    assert w.parameter_count()==expected
    names=[n for n,_ in session.model.named_parameters()]
    assert all(any(n.startswith(prefix) for n in names) for prefix in ('current_encoder.','previous_encoder.','gru.','classifier.'))
    parameters=list(session.model.parameters())
    assert len(session.optimizer.param_groups)==1 and session.optimizer.param_groups[0]['params']==parameters
    assert all(p.requires_grad and p.dtype==torch.float32 for p in parameters)
    assert not session.optimizer.state and session.stream.state_dict()['streams']==[]
    assert session.state['step']==session.scheduler.state_dict()['updates']==0
    before=torch.get_rng_state().clone()
    assert w.parameter_count()==expected and torch.equal(before,torch.get_rng_state())
    receipt=session.checkpoint('initial.pt')
    saved=w.load_verified(receipt)
    assert saved['optimizer_names']==names and not saved['optimizer']['state'] and not saved['stream']['streams']
    assert saved['provenance']['parameter_count']==expected
    assert not any(saved['provenance'][k] for k in ('inherited_weights','inherited_optimizer','inherited_rng','inherited_stream'))
    assert saved['provenance']['independent_encoders'] and saved['provenance']['cnn_stride']==1
    assert 'kda_layers' not in saved['provenance']
    assert torch.equal(saved['rng']['cpu'],torch.get_rng_state())
    # Small movie with native RGB/form, solely to verify the adapter's task API.
    with torch.no_grad(): result=session.model(torch.rand(1,2,3,100,100),w.TASK)
    assert result.shape==(1,2) and torch.isfinite(result).all()

def test_matched_native_protocol_and_source_only_bundle():
    from SecondPass.DelayedFrameGRU.bundle import dependency_sources
    assert w.TARGET==4216 and w.CELLS==['B12','B20','B28']
    assert w.MODULE=='SecondPass.DelayedFrameGRU.worker'
    assert len({w.FreshStream(split).stream_seed(w.TASK,c) for split in ('train','val','test') for c in w.CELLS})==9
    sources=dependency_sources()
    assert 'SecondPass/DelayedFrameGRU/model.py' in sources
    assert 'WorkingMemory/SpatialTaskBattery/stimuli.py' in sources
    assert not any('SequenceKDA' in p or 'CloudRuntime' in p for p in sources)

def test_local_manifest_micro1_and_reduced_allocation_without_cloud_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'ROOT',tmp_path)
    source=tmp_path/'pinned.py'; source.write_text('pinned source\n')
    w.atomic_json(tmp_path/'runtime_manifest.json',dict(source_hashes={'pinned.py':w.digest(source)}))
    now=time.time()
    budget=dict(cap_started=now-1,deadline=now-1+28200,hard_deadline=now-1+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800,execution_placement='local')
    config=w.local_config_for(budget)
    assert config['device']=='mps' and config['effective_batch']==32
    assert config['microbatch']==config['eval_microbatch']==1 and config['minimum_useful_updates']==3
    assert str(tmp_path/'runtime_manifest.json') in config['source_hashes']
    pd=tmp_path/'profile'; pd.mkdir()
    profile=dict(complete=True,architecture=w.VERSION,
        warmup_rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=1000) for c in w.CELLS],
        rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=60) for _ in range(2) for c in w.CELLS],
        evaluation_cells=[dict(cell=c,n=20,seconds=20) for c in w.CELLS])
    w.atomic_json(pd/'profile.json',profile)
    plan=w.measured_plan(tmp_path,budget,now)
    assert 3<=plan['max_steps']<1000 and plan['target_requested']==4216 and plan['exposure_reduced']
    assert plan['minimum_useful_updates']==3
    with pytest.raises(RuntimeError): w.measured_plan(tmp_path,dict(budget,execution_placement='cloud'),now)
    source.write_text('changed source\n')
    with pytest.raises(ValueError): w.verify_sources(config)

def test_mps_sync_dispatch_has_no_cuda_call(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.mps,'synchronize',lambda: calls.append('mps'))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *_: calls.append('cuda'))
    w.sync('mps')
    assert calls==['mps']
