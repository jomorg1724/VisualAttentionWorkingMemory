"""CPU-only fresh-state and real optimizer persistence regression."""
import importlib.util
import json
import time
import torch


def test_fresh_session_and_real_update(tmp_path):
    module = 'SecondPass.SpatialRecurrentConvDecoder.worker'
    assert importlib.util.find_spec(module) is not None, 'fresh worker not implemented'
    from SecondPass.SpatialRecurrentConvDecoder import worker as w
    torch.set_num_threads(2)
    config = dict(device='cpu', effective_batch=4, microbatch=2,
                  cap_started=time.time(), deadline=time.time()+60)
    session = w.Session(tmp_path, config)
    assert not session.optimizer.state
    assert session.state['step'] == session.scheduler.updates == 0
    assert session.stream.state_dict()['streams'] == []
    assert session.state['best_checkpoint'] is None
    assert session.state['selection_history'] == []
    assert 'parent' not in session.state
    assert all(p.requires_grad and p.dtype == torch.float32 for p in session.model.parameters())
    initial = session.checkpoint('initial.pt')
    task, cell = session.scheduler.next()
    # Genuine renderer-generated synthetic stimulus, no photo/checkpoint reads.
    assert not w.TASKS[task]['requires_bsds500']
    row = session.train_update(task, cell)
    saved_receipt = session.checkpoint('checkpoint_000001.pt')
    saved = w.load_verified(saved_receipt)
    assert row['step'] == saved['state']['step'] == 1
    assert saved['state']['episodes'] == 4
    assert saved['scheduler']['updates'] == 1
    assert saved['stream']['streams']
    assert saved['state']['optimizer_seconds'] > 0
    evidence = w.verify_progress(tmp_path)
    assert evidence['verified'] and evidence['step'] == 1
    assert evidence['changed_parameters'] > 0
    assert json.loads((tmp_path/'progress.jsonl').read_text())['step'] == 1
    fresh = w.Session(tmp_path/'second', config)
    original = w.load_verified(initial)
    assert w.tree_equal(original['model'], fresh.model.state_dict())
    assert not fresh.optimizer.state and fresh.scheduler.updates == 0
    assert fresh.stream.state_dict()['streams'] == []
    assert w.tree_equal(original['scheduler'], fresh.scheduler.state_dict())
    assert torch.equal(original['rng']['cpu'], torch.get_rng_state())


def test_cap_allocation_and_evaluation_namespace(tmp_path):
    import pytest
    from SecondPass.SpatialRecurrentConvDecoder import worker as w
    budget=dict(cap_started=100., hard_deadline=14500., deadline=13900.,
                wall_cap_seconds=14400., retrieval_reserve_seconds=600.)
    w.validate_budget(budget,budget,now=200.)
    with pytest.raises(ValueError): w.validate_budget(budget,dict(budget,deadline=14000.),now=200.)
    with pytest.raises(ValueError): w.validate_budget(budget,budget,now=14000.)
    # Synthetic timing fixture is for allocation logic only, not measured GPU evidence.
    scheduler=w.BalancedScheduler(w.SCHEDULER_SEED)
    rows=[dict(zip(('task','cell'),scheduler.next()),seconds=1.) for _ in range(13)]
    cells=[dict(task=t,cell=c,seconds=.008,n=8) for t,c in w.cloud.original.all_cells()]
    profile=dict(complete=True,architecture=w.VERSION,rows=rows,evaluation_cells=cells)
    w.atomic_json(tmp_path/'profile/profile.json',profile)
    plan=w.measured_plan(tmp_path,budget,now=200.)
    assert plan['max_steps']==5200 and plan['total_episodes']==166400
    assert plan['validation_steps']==[2600,5200]
    assert all(r['updates']==400 for r in plan['planned_exposure'].values())
    reduced=w.measured_plan(tmp_path,budget,now=10000.)
    assert reduced['exposure_reduced'] and reduced['max_steps']%13==0
    assert reduced['estimated_total_remaining_seconds']<3900
    with pytest.raises(RuntimeError): w.measured_plan(tmp_path,budget,now=13899.)
    for task,cell in w.cloud.original.all_cells():
        seeds=[w.FreshStream(split).stream_seed(task,cell) for split in ('train','val','test')]
        assert len(set(seeds))==3
        assert seeds[0]!=w.SuiteStream('train').stream_seed(task,cell)
        assert seeds[2]==w.FreshStream('test').stream_seed(task,cell)


def test_initializer_cannot_read_checkpoint(tmp_path,monkeypatch):
    from SecondPass.SpatialRecurrentConvDecoder import worker as w
    def forbidden(*args,**kwargs): raise AssertionError('Initialization attempted checkpoint read')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(w.cloud,'trusted_load',forbidden)
    session=w.Session(tmp_path,dict(device='cpu',cap_started=time.time(),deadline=time.time()+60))
    assert not session.optimizer.state and session.state['step']==0
