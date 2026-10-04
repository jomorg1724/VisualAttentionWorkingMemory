"""CPU-only fresh ConvGRU construction and persisted native-update checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch


def test_whole_model_is_direct_fresh_construction(tmp_path, monkeypatch):
    module = 'SecondPass.SpatialReadout.FreshRun.worker'
    assert importlib.util.find_spec(module) is not None, 'fresh ConvGRU worker not implemented'
    from SecondPass.SpatialReadout.FreshRun import worker as w
    from SecondPass.SpatialReadout.model import SpatialReadout
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Initialization attempted checkpoint read or state restoration')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(w.cloud, 'trusted_load', forbidden)
    monkeypatch.setattr(w, 'load_verified', forbidden)
    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', forbidden)
    config = dict(device='cpu', effective_batch=32, microbatch=4,
                  cap_started=time.time(), deadline=time.time()+60)
    session = w.Session(tmp_path, config)
    assert type(session.model) is SpatialReadout
    initialized_rng = torch.get_rng_state()
    torch.manual_seed(w.INIT_SEED)
    direct = SpatialReadout(w.task_classes())
    assert w.tree_equal(direct.state_dict(), session.model.state_dict())
    assert torch.equal(initialized_rng, torch.get_rng_state())
    assert not session.optimizer.state
    assert len(session.optimizer.param_groups) == 1
    assert session.optimizer.param_groups[0]['lr'] == 1e-4
    assert {id(p) for p in session.optimizer.param_groups[0]['params']} == {id(p) for p in session.model.parameters()}
    assert all(p.requires_grad and p.dtype == torch.float32 for p in session.model.parameters())
    assert session.state['step'] == session.state['episodes'] == session.state['frames'] == session.scheduler.updates == 0
    assert session.state['optimizer_seconds'] == 0
    assert session.state['best_checkpoint'] is None and session.state['selection_history'] == []
    assert session.stream.state_dict()['streams'] == []
    assert all(not e['updates'] and not e['episodes'] and not e['frames'] and not any(e['cells'].values()) for e in session.state['exposure'].values())
    assert not any(w.provenance()[key] for key in ('inherited_weights', 'inherited_optimizer', 'inherited_rng', 'inherited_stream', 'profile_state_inherited'))


def test_native_update_persistence_and_profile_reset(tmp_path):
    from SecondPass.SpatialReadout.FreshRun import worker as w
    torch.set_num_threads(2)
    config = dict(device='cpu', effective_batch=32, microbatch=4,
                  cap_started=time.time(), deadline=time.time()+120)
    session = w.Session(tmp_path, config)
    initial_receipt = session.checkpoint('initial.pt')
    task, cell = session.scheduler.next()
    row = session.train_update(task, cell)
    receipt = session.checkpoint('checkpoint_000001.pt')
    initial, saved = w.load_verified(initial_receipt), w.load_verified(receipt)
    assert row['step'] == saved['state']['step'] == saved['scheduler']['updates'] == 1
    assert saved['state']['episodes'] == row['episodes'] == 32
    assert saved['state']['frames'] > 0 and saved['state']['optimizer_seconds'] > 0
    assert saved['stream']['streams'] and not w.tree_equal(initial['stream'], saved['stream'])
    assert saved['provenance'] == initial['provenance'] == w.provenance()
    changed = [n for n, p in saved['model'].items() if not torch.equal(p, initial['model'][n])]
    for prefix in ('blocks.', 'acc.0.', 'acc.1.', 'acc.2.', 'spatial_input.', 'spatial_gru.', 'readout.', f'heads.{task}.'):
        assert any(n.startswith(prefix) for n in changed), prefix
    assert w.verify_progress(tmp_path)['verified']
    assert json.loads((tmp_path/'progress.jsonl').read_text())['clipping'] is None
    # A disposable training/profile session must not advance production at all.
    fresh = w.Session(tmp_path/'production', config)
    assert w.tree_equal(initial['model'], fresh.model.state_dict())
    assert not fresh.optimizer.state and fresh.scheduler.updates == fresh.state['step'] == 0
    assert fresh.stream.state_dict()['streams'] == []
    assert w.tree_equal(initial['scheduler'], fresh.scheduler.state_dict())
    assert torch.equal(initial['rng']['cpu'], torch.get_rng_state())


def test_eight_hour_allocation_and_distinct_namespaces(tmp_path):
    from SecondPass.SpatialReadout.FreshRun import worker as w
    budget = dict(cap_started=100., hard_deadline=28900., deadline=28300.,
                  wall_cap_seconds=28800., retrieval_reserve_seconds=600.)
    w.validate_budget(budget, budget, now=200.)
    for bad in (dict(budget, wall_cap_seconds=14400.), dict(budget, retrieval_reserve_seconds=0.)):
        with pytest.raises(ValueError):
            w.validate_budget(bad, bad, now=200.)
    with pytest.raises(ValueError):
        w.validate_budget(budget, dict(budget, deadline=29000.), now=200.)
    with pytest.raises(ValueError):
        w.validate_budget(budget, budget, now=28300.)
    # Synthetic fixture tests planner arithmetic only, not measured GPU costs.
    scheduler = w.BalancedScheduler(w.SCHEDULER_SEED)
    rows = [dict(zip(('task', 'cell'), scheduler.next()), seconds=1.) for _ in range(13)]
    cells = [dict(task=t, cell=c, seconds=.008, n=8) for t, c in w.cloud.original.all_cells()]
    w.atomic_json(tmp_path/'profile/profile.json', dict(complete=True, architecture=w.VERSION, rows=rows, evaluation_cells=cells))
    plan = w.measured_plan(tmp_path, budget, now=200.)
    assert plan['max_steps'] == w.TARGET == 10400 and plan['total_episodes'] == 332800
    assert plan['validation_steps'] == [5200, 10400]
    assert all(r['updates'] == 800 for r in plan['planned_exposure'].values())
    reduced = w.measured_plan(tmp_path, budget, now=25000.)
    assert reduced['exposure_reduced'] and reduced['max_steps'] % 13 == 0
    assert reduced['estimated_total_remaining_seconds'] < 3300
    with pytest.raises(RuntimeError):
        w.measured_plan(tmp_path, budget, now=28299.)
    namespaces = [w.INIT_SEED, w.CUDA_SEED, w.SCHEDULER_SEED, w.TRAIN_NAMESPACE, w.VAL_NAMESPACE, w.FINAL_NAMESPACE]
    assert len(set(namespaces)) == 6 and min(namespaces) >= 98192763
    seeds = [w.FreshStream(split).stream_seed(t, c) for split in ('train', 'val', 'test') for t, c in w.cloud.original.all_cells()]
    assert len(seeds) == len(set(seeds)) == 105


@pytest.mark.parametrize('module', ['worker', 'bundle'])
@pytest.mark.parametrize('option', ['--predecessor', '--checkpoint', '--resume'])
def test_cli_rejects_predecessor_inputs(tmp_path, module, option):
    base = 'SecondPass.SpatialReadout.FreshRun.' + module
    args = ['verify', str(tmp_path)] if module == 'worker' else ['--output', str(tmp_path)]
    result = subprocess.run([sys.executable, '-m', base, *args, option, '/forbidden/predecessor.pt'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr
    assert not list(tmp_path.iterdir())


def test_bundle_rejects_undeclared_sources(tmp_path, monkeypatch):
    from SecondPass.SpatialReadout.FreshRun import bundle as b
    local = tmp_path/'SecondPass/SpatialReadout/FreshRun'
    local.mkdir(parents=True)
    for name in ('worker.py', 'bundle.py', 'test_fresh.py', 'README.md'):
        (local/name).write_text('')
    (local.parent/'model.py').write_text('')
    (local/'credentials.py').write_text('SECRET = "do not package"')
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setattr(b, '__file__', str(local/'bundle.py'))
    sources = b.dependency_sources()
    assert 'SecondPass/SpatialReadout/FreshRun/credentials.py' not in sources
    assert 'SecondPass/SpatialReadout/model.py' in sources


@pytest.mark.parametrize('relative', ['../predecessor.pt', '/tmp/predecessor.pt', 'images/train/weights.pt', 'images/train/credentials.json'])
def test_bundle_rejects_non_photo_assets(tmp_path, relative):
    from SecondPass.SpatialReadout.FreshRun import bundle as b
    assert hasattr(b, 'photo_path'), 'explicit BSDS asset allowlist not implemented'
    with pytest.raises(ValueError):
        b.photo_path(tmp_path, relative)
