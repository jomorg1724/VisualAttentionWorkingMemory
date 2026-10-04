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
    module = 'SecondPass.SpatialReadout.SpatialConsolidation.worker'
    assert importlib.util.find_spec(module) is not None, 'fresh ConvGRU worker not implemented'
    from SecondPass.SpatialReadout.SpatialConsolidation import worker as w
    from SecondPass.SpatialReadout.SpatialConsolidation.model import SpatialConsolidation as SpatialReadout
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
    from SecondPass.SpatialReadout.SpatialConsolidation import worker as w
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
    for prefix in ('blocks.', 'acc.0.', 'acc.1.', 'acc.2.', 'spatial_input.', 'spatial_gru.', 'readout.', 'consolidation.', f'heads.{task}.'):
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


