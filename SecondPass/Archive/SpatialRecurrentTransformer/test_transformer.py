"""Focused architecture and migration regressions (CPU only)."""
import importlib.util
import torch
import pytest

torch.set_num_threads(2)


def test_recurrent_spatial_cls_is_causal_and_learns():
    assert importlib.util.find_spec('SecondPass.SpatialRecurrentTransformer.model'), 'Recurrent transformer missing'
    from SecondPass.SpatialRecurrentTransformer.model import SpatialRecurrentTransformer
    torch.manual_seed(73)
    model = SpatialRecurrentTransformer({'probe': 2})
    assert not any(s in n for n, _ in model.named_parameters() for s in ('gru', 'comparator'))
    x = torch.rand(1, 3, 3, 100, 100, requires_grad=True)
    history = model.recurrent_states(x)
    assert len(history) == 3
    assert all(h.shape == (1, 50, 64) for h in history)
    with torch.no_grad():
        prefix = model.recurrent_states(x[:, :2])
        changed = x.detach().clone(); changed[:, 2] = 0
        other = model.recurrent_states(changed)
        assert torch.equal(history[0], prefix[0])
        assert torch.equal(history[1], prefix[1]) and torch.equal(history[1], other[1])
        assert not torch.equal(history[-1], other[-1])
        z = torch.randn(1, 64, 7, 7)
        assert not torch.equal(model.memory(z, history[0]), model.memory(z, None))
        h = history[-1].clone(); h[:, 1:] = torch.randn_like(h[:, 1:]) * 100
        assert torch.equal(model.classify(history[-1], 'probe'), model.classify(h, 'probe'))
        assert torch.equal(model(x, 'probe'), model(x, 'probe'))
    fresh = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(('memory.', 'cls_readout.'))}
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.nn.functional.cross_entropy(model.classify(history[-1], 'probe'), torch.tensor([1])).backward()
    assert x.grad[:, 0].abs().sum() > 0
    for n, p in model.named_parameters():
        if n in fresh:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
    opt.step()
    assert all(not torch.equal(fresh[n], p) for n, p in model.named_parameters() if n in fresh)


@pytest.mark.parametrize('fixture', [True, False])
def test_full_migration_snapshot_and_fresh_stream(tmp_path, fixture):
    assert importlib.util.find_spec('SecondPass.SpatialRecurrentTransformer.worker'), 'Cloud worker missing'
    from SecondPass.SpatialRecurrentTransformer import worker as w
    from SecondPass.JointTraining.core import tree_equal
    import json
    source_path = w.Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt' if fixture else '/Users/jonathanmorgan/VAWMRuntime/cloud_comparison_03/artifacts/terminal.pt')
    pointer = tmp_path / 'source.json'
    raw = w.cloud.trusted_load(source_path)
    cumulative = raw['state']['step'] + (0 if fixture else raw['state']['parent']['step'])
    pointer.write_text(json.dumps(dict(schema=1, path=str(source_path), sha256=w.digest(source_path), bytes=source_path.stat().st_size,
        step=raw['state']['step'], cumulative_step=cumulative, scheduler_updates=raw['scheduler']['updates'], fixture_only=fixture)))
    if fixture:
        with pytest.raises(ValueError): w.source_payload(pointer)
    source, receipt = w.source_payload(pointer, allow_fixture=fixture)
    payload = w.migrate(source, receipt, 'cpu')
    assert tree_equal(source['rng'], payload['rng'])
    for name in payload['migration']['carried_names']:
        assert torch.equal(source['model'][name], payload['model'][name])
        old_id = source['optimizer_names'].index(name); new_id = payload['optimizer_names'].index(name)
        assert tree_equal(source['optimizer']['state'][old_id], payload['optimizer']['state'][new_id])
    cfg = dict(device='cpu', effective_batch=2, microbatch=1, cap_started=w.time.time(), deadline=w.time.time()+600, disposable_profile=True)
    session = w.Session(tmp_path/'session', cfg, payload)
    initial = w.load_verified(session.checkpoint('migration.pt'))
    for key in ('model', 'optimizer', 'optimizer_names', 'scheduler', 'stream', 'rng'):
        assert tree_equal(initial[key], payload[key]), key
    expected_task = session.scheduler.next()
    expected = session.stream.batch(1, *expected_task)
    w.restore(initial, session.model, session.optimizer, session.scheduler, session.stream, 'cpu')
    assert session.scheduler.next() == expected_task
    assert tree_equal(expected, session.stream.batch(1, *expected_task))
    # Restore into a genuinely fresh Session; constructor RNG draws must not leak.
    clone = w.Session(tmp_path/'clone', cfg, initial)
    assert clone.scheduler.next() == expected_task
    assert tree_equal(expected, clone.stream.batch(1, *expected_task))
    w.restore(initial, session.model, session.optimizer, session.scheduler, session.stream, 'cpu')
    for _ in range(2): session.train_update(*session.scheduler.next())
    saved = w.load_verified(session.checkpoint('checkpoint_000002.pt'))
    assert w.verify_progress(session.directory)['verified']
    progress = [json.loads(line) for line in (session.directory/'progress.jsonl').read_text().splitlines()]
    assert progress[-1]['cumulative_lineage_steps'] == cumulative + 2
    for name in payload['migration']['fresh_names']:
        assert not torch.equal(saved['model'][name], initial['model'][name]), name
        assert float(saved['optimizer']['state'][saved['optimizer_names'].index(name)]['step']) == 2


def test_fixed_exposure_budget_and_fresh_namespaces(tmp_path):
    from SecondPass.SpatialRecurrentTransformer import worker as w
    assert hasattr(w, 'measured_plan'), 'Launch allocation missing'
    import json
    import pytest
    budget = dict(cap_started=100., deadline=200000., hard_deadline=200600., retrieval_reserve_seconds=600.)
    w.validate_budget(budget, budget, now=101.)
    with pytest.raises(ValueError): w.validate_budget(budget, dict(budget,deadline=200001.), now=101.)
    with pytest.raises(ValueError): w.validate_budget(budget, budget, now=200000.)
    scheduler = w.BalancedScheduler(0)
    source = dict(scheduler=scheduler.state_dict())
    rows = [dict(task=t,cell=c,seconds=1.) for t,c in [scheduler.next() for _ in range(13)]]
    cells = [dict(task=t,cell=c['id'],seconds=.1,n=8) for t,s in w.TASKS.items() for c in s['conditions']]
    (tmp_path/'profile').mkdir()
    (tmp_path/'profile'/'profile.json').write_text(json.dumps(dict(complete=True,rows=rows,evaluation_cells=cells,architecture=w.VERSION)))
    plan = w.measured_plan(tmp_path,source,budget,now=101.)
    assert plan['max_steps']==2600 and plan['total_episodes']==83200 and plan['validation_steps']==[1300,2600]
    assert sum(e['episodes'] for e in plan['planned_exposure'].values()) == 83200
    assert all(e['updates']==200 for e in plan['planned_exposure'].values())
    with pytest.raises(RuntimeError): w.measured_plan(tmp_path,source,dict(budget,deadline=105.),now=101.)
    assert w.VAL_NAMESPACE != w.cloud.exp.VAL_NAMESPACE and w.FINAL_NAMESPACE != w.cloud.exp.FINAL_NAMESPACE


def test_bundle_dependency_closure():
    assert importlib.util.find_spec('SecondPass.SpatialRecurrentTransformer.bundle'), 'Deployment builder missing'
    from SecondPass.SpatialRecurrentTransformer.bundle import dependency_sources, ROOT
    sources = dependency_sources()
    assert 'SecondPass/SpatialRecurrentTransformer/model.py' in sources
    assert 'SecondPass/SpatialComparisonReadout/CloudRun/worker.py' in sources
    assert 'SecondPass/TaskSuite/catalog.json' in sources
    assert all(p.is_file() and p.is_relative_to(ROOT) for p in sources.values())
