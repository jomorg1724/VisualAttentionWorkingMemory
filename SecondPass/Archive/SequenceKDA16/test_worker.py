"""Focused cloud adapter checks; no provisioning or native training campaign."""
import json
import time
import torch
from . import worker as w
from .model import SequenceKDA
from .bundle import dependency_sources

def test_fresh_constructor_and_persisted_initial_schema(tmp_path):
    w.cpu_setup()
    now=time.time()
    session=w.Session(tmp_path,dict(device='cpu',backend='chunk',chunk_size=32,
        effective_batch=32,microbatch=4,cap_started=now,deadline=now+600))
    assert sum(p.numel() for p in session.model.parameters())==737170
    assert len(list(session.model.parameters()))==27
    assert session.model.kda.heads==16
    assert len(session.optimizer.param_groups)==1
    assert session.optimizer.param_groups[0]['params']==list(session.model.parameters())
    assert not session.optimizer.state and session.stream.state_dict()['streams']==[]
    initial=session.checkpoint('initial.pt'); saved=w.load_verified(initial)
    assert saved['state']['step']==saved['scheduler']['updates']==0
    assert saved['optimizer']['state']=={} and saved['stream']['streams']==[]
    assert len(saved['optimizer_names'])==27 and len(saved['model'])==27
    assert all(t.dtype==torch.float32 for t in saved['model'].values())
    assert saved['provenance']['kda_layers']==1 and saved['provenance']['kda_heads']==16
    assert saved['provenance']['parameter_count']==737170
    assert not any(saved['provenance'][k] for k in ('inherited_weights','inherited_optimizer','inherited_rng','inherited_stream'))
    assert torch.equal(saved['rng']['cpu'],torch.get_rng_state())
    assert json.loads((tmp_path/'constructor_equality.json').read_text())['verified']

def test_native_seed_protocol_profile_pinning_and_dependency_closure(tmp_path):
    assert w.TARGET==4216 and w.CELLS==['B12','B20','B28']
    assert w.MODULE=='SecondPass.SequenceKDA16.worker'
    seeds={w.FreshStream(s).stream_seed(w.TASK,c) for s in ('train','val','test') for c in w.CELLS}
    assert len(seeds)==9
    pd=tmp_path/'profile'; pd.mkdir()
    prof=dict(complete=True,architecture=w.VERSION,
        warmup_rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=1000) for c in w.CELLS],
        rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=1) for _ in range(2) for c in w.CELLS],
        evaluation_cells=[dict(cell=c,n=20,seconds=1) for c in w.CELLS])
    w.atomic_json(pd/'profile.json',prof)
    now=time.time()
    budget=dict(cap_started=now-1,deadline=now-1+28200,hard_deadline=now-1+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800)
    plan=w.measured_plan(tmp_path,budget,now)
    assert plan['max_steps']==4216 and plan['total_episodes']==134912
    assert plan['validation_steps']==[2108,4216] and plan['target_requested']==4216
    sources=dependency_sources()
    assert 'SecondPass/SequenceKDA16/model.py' in sources
    assert 'SecondPass/SequenceKDA/kda_backend.py' in sources
    assert 'WorkingMemory/SpatialTaskBattery/stimuli.py' in sources
    assert not any('artifacts/' in p or 'package/' in p or '/data/' in p for p in sources)
