"""Focused fresh-state and frequent-validation budget checks; no provisioning."""
import json
import time
import torch
from . import worker as w
from .bundle import dependency_sources

def test_fresh_encoder_checkpointing_and_initial_evidence(tmp_path):
    w.cpu_setup(); now=time.time()
    session=w.Session(tmp_path,dict(device='cpu',effective_batch=32,microbatch=4,
        cap_started=now,deadline=now+600))
    assert session.model.checkpoint_encoder
    params=list(session.model.parameters()); names=[n for n,_ in session.model.named_parameters()]
    assert sum(p.numel() for p in params)==7270290 and len(params)==92
    assert len(session.optimizer.param_groups)==1 and session.optimizer.param_groups[0]['params']==params
    assert not session.optimizer.state and session.stream.state_dict()['streams']==[]
    assert all(p.requires_grad and p.dtype==torch.float32 for p in params)
    receipt=session.checkpoint('initial.pt'); saved=w.load_verified(receipt)
    assert saved['state']['step']==saved['scheduler']['updates']==0 and not saved['optimizer']['state']
    assert saved['optimizer_names']==names and saved['stream']['streams']==[]
    assert saved['provenance']['checkpoint_encoder'] and saved['provenance']['parameter_count']==7270290
    assert 'kda_layers' not in saved['provenance']
    assert not any(saved['provenance'][k] for k in ('inherited_weights','inherited_optimizer','inherited_rng','inherited_stream'))
    assert torch.equal(saved['rng']['cpu'],torch.get_rng_state())
    assert json.loads((tmp_path/'constructor_equality.json').read_text())['verified']

def test_exact_frequent_validation_cost_controls_pin(tmp_path):
    now=time.time()
    budget=dict(cap_started=now-1,deadline=now-1+28200,hard_deadline=now-1+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800)
    pd=tmp_path/'profile'; pd.mkdir()
    prof=dict(complete=True,architecture=w.VERSION,
        warmup_rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=1000) for c in w.CELLS],
        rows=[dict(task=w.TASK,cell=c,episodes=32,seconds=1) for _ in range(2) for c in w.CELLS],
        evaluation_cells=[dict(cell=c,n=20,seconds=20) for c in w.CELLS])
    w.atomic_json(pd/'profile.json',prof)
    full=w.measured_plan(tmp_path,budget,now)
    assert full['max_steps']==4216 and full['planned_validation_looks']==18
    assert full['validation_steps']==[100]+list(range(250,4001,250))+[4216]
    for r in prof['rows']: r['seconds']=8
    w.atomic_json(pd/'profile.json',prof)
    reduced=w.measured_plan(tmp_path,budget,now)
    assert 1000<=reduced['max_steps']<4216 and reduced['exposure_reduced']
    count=reduced['max_steps']; looks=w.validation_steps_for(count)
    assert reduced['planned_validation_looks']==len(looks) and reduced['validation_steps']==looks
    expected=1.25*reduced['estimated_optimizer_seconds']+len(looks)*reduced['estimated_validation_seconds']+2*reduced['estimated_one_test_seconds']+900
    assert abs(reduced['estimated_total_remaining_seconds']-expected)<1e-8
    assert expected+60<budget['deadline']-now
    assert w.validation_steps_for(100)==[100] and w.validation_steps_for(250)==[100,250]
    assert w.validation_steps_for(249)==[100,249]

def test_version_native_seeds_and_checkpoint_free_dependency_closure():
    assert w.MODULE=='SecondPass.TwoFrameRViT.worker' and w.TARGET==4216
    assert w.CELLS==['B12','B20','B28']
    assert len({w.FreshStream(s).stream_seed(w.TASK,c) for s in ('train','val','test') for c in w.CELLS})==9
    sources=dependency_sources()
    assert 'SecondPass/TwoFrameRViT/model.py' in sources
    assert 'WorkingMemory/SpatialTaskBattery/stimuli.py' in sources
    assert not any('SequenceKDA' in p or 'kda_backend' in p or 'artifacts/' in p or 'package/' in p for p in sources)
