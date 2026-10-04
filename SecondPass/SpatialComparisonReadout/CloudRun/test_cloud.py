"""Focused CPU regressions; CUDA replay is exercised by the disposable profile."""
import importlib
import importlib.util
import tempfile
from pathlib import Path


def test_cloud_adapter_exists_and_trusted_load():
    assert importlib.util.find_spec('SecondPass.SpatialComparisonReadout.CloudRun.worker'), 'Cloud adapter missing'
    w = importlib.import_module('SecondPass.SpatialComparisonReadout.CloudRun.worker')
    import torch
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'trusted.pt'
        torch.save({'numpy': np.random.get_state()}, path)
        assert w.trusted_load(path)['numpy'][0] == 'MT19937'
    assert w.PROTOCOL == 'spatial_comparison_cloud_candidate_v1'


def test_cpu_migration_and_saved_replay():
    w = importlib.import_module('SecondPass.SpatialComparisonReadout.CloudRun.worker')
    assert hasattr(w, 'source_payload'), 'Verified portable migration missing'
    import torch
    from SecondPass.JointTraining.core import tree_equal
    from SecondPass.SpatialReadout.model import SpatialReadout
    from SecondPass.TaskSuite.suite import task_classes
    source, receipt = w.source_payload()
    payload = w.migrate(source, receipt, 'cpu')
    assert payload['migration']['cuda_rng_policy'] == 'fresh seeded CUDA generator on MPS-to-CUDA transition; restore saved CUDA state on cloud resume'
    assert tree_equal(source['rng'], payload['rng'])
    assert sum(v.numel() for n,v in payload['model'].items() if n.startswith('comparator.')) == 12416
    assert tree_equal(source['optimizer']['state'], payload['optimizer']['state'])
    prior_path=w.ASSETS/'prior_candidate_migration.pt'
    if prior_path.exists():
        prior=w.trusted_load(prior_path)
        prior['optimizer']=w.portable_adam_schema(prior['optimizer'])
        for key in ('model','optimizer','optimizer_names','scheduler','stream','rng'):
            assert tree_equal(prior[key],payload[key]), key
    with tempfile.TemporaryDirectory() as d:
        cfg = dict(device='cpu', effective_batch=2, microbatch=1, cap_started=w.time.time(), deadline=w.time.time()+600,
                   disposable_profile=True, arm='candidate')
        session = w.Session(Path(d), cfg, payload)
        initial_payload = w.load_verified(session.checkpoint('initial_integrity.pt'))
        for key in ('model','optimizer','optimizer_names','scheduler','stream'):
            assert tree_equal(initial_payload[key],payload[key]), key
        parent = SpatialReadout(task_classes()); parent.load_state_dict(source['model'])
        task, cell = session.scheduler.next()
        x, _, _ = session.stream.batch(1,task,cell)
        with torch.no_grad(): assert torch.equal(parent(x,task),session.model(x,task))
        initial = session.checkpoint('migration.pt')
        expected = session.stream.batch(1,task,cell)
        w.restore(w.load_verified(initial),session.model,session.optimizer,session.scheduler,session.stream,'cpu')
        assert tree_equal(expected,session.stream.batch(1,task,cell))
        session.train_update('orientation','mixed')
        session.train_update('orientation','mixed')
        saved = w.load_verified(session.checkpoint('cpu_updated.pt'))
        for name in payload['migration']['fresh_names']:
            assert not torch.equal(payload['model'][name],saved['model'][name])
            assert float(saved['optimizer']['state'][saved['optimizer_names'].index(name)]['step']) == 2


def test_budget_does_not_renew():
    w = importlib.import_module('SecondPass.SpatialComparisonReadout.CloudRun.worker')
    assert hasattr(w, 'budget_from_creation'), 'Immutable cloud cap missing'
    budget = w.budget_from_creation(100.)
    assert budget['hard_deadline'] == 28900.
    assert budget['deadline'] == 28300.
    import pytest
    with pytest.raises(ValueError): w.validate_budget(budget,budget,now=28300.)
    with pytest.raises(ValueError): w.validate_budget(budget,dict(budget,deadline=28400.),now=200.)
