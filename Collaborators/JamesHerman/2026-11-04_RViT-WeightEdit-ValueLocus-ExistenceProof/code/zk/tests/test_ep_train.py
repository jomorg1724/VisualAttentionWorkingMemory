"""ep_train tests (RVIT_EXISTENCE_PROOF_PLAN.md rev 2, review #12; 2026-11-04). Run from tree_zk/ (CPU):
    CUDA_VISIBLE_DEVICES= python -m pytest -q zk/tests/test_ep_train.py
1. Two-group Adam with q/k decay 1e-6 is bit-identical to the legacy single-group Adam over 4 learn() steps, also when
   the legacy optimiser state after 2 steps is converted (_convert_one_group) and training continues.
2. Frozen actor: bit-identical after 4 learn() steps; transformer moves; on one batch the transformer gradient equals
   the unfrozen agent's (freezing changes nothing upstream of the actor).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from zk import paths  # noqa: F401
from zk import agent_orig
from zk.ep_train import freeze, two_group_optimizer
from zk.tests.test_grad_route import _build, _all_state


def _run(steps, tr, ag, seed=11):
    torch.manual_seed(seed); np.random.seed(seed)
    for _ in range(steps):
        tr = ag.learn(tr)
    return tr


def test_two_group_equals_legacy():
    tr0, ag0 = _build(agent_orig); tr0 = _run(4, tr0, ag0)
    tr1, ag1 = _build(agent_orig); tr1.optimizer = two_group_optimizer(tr1, 1e-6); tr1 = _run(4, tr1, ag1)
    a, b = _all_state(tr0, ag0), _all_state(tr1, ag1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    # conversion of a legacy state mid-run
    tr2, ag2 = _build(agent_orig)
    torch.manual_seed(11); np.random.seed(11)
    for _ in range(2):
        tr2 = ag2.learn(tr2)
    sd = tr2.optimizer.state_dict(); opt = two_group_optimizer(tr2, 1e-6); opt.load_state_dict(sd); tr2.optimizer = opt
    for _ in range(2):
        tr2 = ag2.learn(tr2)
    c = _all_state(tr2, ag2)
    assert all(torch.equal(a[k], c[k]) for k in a)


def test_frozen_actor():
    tr, ag = _build(agent_orig); freeze(ag)
    a0 = {k: v.clone() for k, v in ag.actor.state_dict().items()}
    t0 = {k: v.clone() for k, v in tr.state_dict().items()}
    tr = _run(4, tr, ag)
    assert all(torch.equal(a0[k], v) for k, v in ag.actor.state_dict().items())
    assert any(not torch.equal(t0[k], v) for k, v in tr.state_dict().items())
    # same batch, transformer gradient frozen vs unfrozen
    grads = []
    for frz in (False, True):
        trx, agx = _build(agent_orig)
        if frz:
            freeze(agx)
        np.random.seed(5); torch.manual_seed(5)
        b = agx.memory.sample_buffer(agx.batch_size)
        kl, lpol, H, _ = agx.compute_losses(trx, *b[:7])
        trx.optimizer.zero_grad(); (kl - lpol - agx.entropy_coef * H).backward()
        grads.append({n: p.grad.clone() for n, p in trx.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.allclose(grads[0][k], grads[1][k], atol=0, rtol=0) for k in grads[0])
