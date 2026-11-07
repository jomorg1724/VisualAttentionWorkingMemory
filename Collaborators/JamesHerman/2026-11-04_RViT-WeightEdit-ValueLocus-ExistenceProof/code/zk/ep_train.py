#!/usr/bin/env python3
"""
Existence-proof training wrapper (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md; added 2026-11-04).

Runs zk.train_orig unchanged, with three opt-in patches applied from outside (train_orig.py / agent_orig.py are not
edited, because other running jobs execute them):

  --freeze-actor      the actor (the readout) is never stepped: its Adam step is a no-op and its parameters do not
                      require grad. Gradients still flow THROUGH it into the transformer (MPO + entropy terms), so the
                      policy loss can only act by changing the transformer state H. The target actor follows the frozen
                      online actor by the usual Polyak rule.
  --qk-decay X        the transformer optimiser is built with two parameter groups, {everything else} and
                      {W_q, W_k, W_Cq, W_Ck}, the second with weight decay X (legacy 1e-6 = numerically the legacy
                      single-group Adam). A single-group optimiser state (any legacy checkpoint) is converted to the
                      two-group layout on load, and X is re-applied after every load (load_state_dict restores the saved
                      group settings). Every existence-proof arm uses this path, so arms differ only in X.
  --route-change-ok   allow --init-from a checkpoint saved under another --grad-route (the agent's route is used);
                      ignored when the run resumes from its own ckpt_latest.pt.

Everything else is passed to zk.train_orig. Writes ep_flags.json into --out-dir.

    python -m zk.ep_train --freeze-actor --qk-decay 1e-6 --task luo2015_sensitivity ... --init-from W/ckpt_final.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from . import paths  # noqa: F401  (puts orig/ on sys.path)
from . import agent_orig, train_orig

QK_MODS = ("W_q", "W_k", "W_Cq", "W_Ck")


def _split(tr):
    qk = [p for n, p in tr.named_parameters() if n.split(".")[-2] in QK_MODS]
    ids = {id(p) for p in qk}
    rest = [p for p in tr.parameters() if id(p) not in ids]
    return rest, qk


def _convert_one_group(sd, tr):
    """Legacy single-group Adam state (params indexed in tr.parameters() order) -> [rest, qk] two-group layout."""
    rest, qk = _split(tr)
    pos = {id(p): i for i, p in enumerate(tr.parameters())}
    g = sd["param_groups"][0]
    state, groups, nid = {}, [], 0
    for plist in (rest, qk):
        ids = []
        for p in plist:
            old = pos[id(p)]
            if old in sd["state"]:
                state[nid] = sd["state"][old]
            ids.append(nid); nid += 1
        groups.append({k: v for k, v in g.items() if k != "params"} | {"params": ids})
    return {"state": state, "param_groups": groups}


def install(freeze_actor: bool, qk_decay: float, route_change_ok: bool, trunk_decay=None):
    if freeze_actor:
        _init = agent_orig.AgentOrig.__init__

        def init(self, *a, **k):
            _init(self, *a, **k)
            freeze(self)
        agent_orig.AgentOrig.__init__ = init

    if route_change_ok:
        _load = agent_orig.AgentOrig.load_state_dict

        def load(self, sd):
            sd = dict(sd); sd["grad_route"] = self.grad_route
            return _load(self, sd)
        agent_orig.AgentOrig.load_state_dict = load

    _build = train_orig.build_orig_networks

    def build(dev, *a, **k):
        tr, actor = _build(dev, *a, **k)
        tr.optimizer = two_group_optimizer(tr, qk_decay, trunk_decay)
        return tr, actor
    train_orig.build_orig_networks = build


def freeze(agent):
    """Readout frozen: no Adam step, no grad on its parameters (gradients still flow through it into H)."""
    for p in agent.actor.parameters():
        p.requires_grad_(False)
    agent.actor.optimizer.step = lambda *x, **kw: None


def two_group_optimizer(tr, qk_decay, trunk_decay=None):
    """[rest, q/k] Adam with the legacy hyper-parameters; q/k weight decay = qk_decay. load_state_dict accepts a legacy
    single-group state (converted) and re-applies the decays after loading."""
    rest, qk = _split(tr)
    assert qk, "no q/k parameters"
    d = dict(tr.optimizer.defaults)
    opt = torch.optim.Adam([{"params": rest}, {"params": qk, "weight_decay": qk_decay}], lr=d["lr"],
                           betas=d["betas"], eps=d["eps"], weight_decay=d["weight_decay"])
    _ld = opt.load_state_dict

    def opt_load(sd):
        if len(sd["param_groups"]) == 1:
            assert len(sd["param_groups"][0]["params"]) == len(list(tr.parameters())), "optimiser/param mismatch"
            sd = _convert_one_group(sd, tr)
        _ld(sd)
        opt.param_groups[1]["weight_decay"] = qk_decay
        opt.param_groups[0]["weight_decay"] = d["weight_decay"] if trunk_decay is None else trunk_decay
    if trunk_decay is not None:
        opt.param_groups[0]["weight_decay"] = trunk_decay
    opt.load_state_dict = opt_load
    return opt


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--freeze-actor", action="store_true")
    ap.add_argument("--qk-decay", type=float, default=1e-6)
    ap.add_argument("--route-change-ok", action="store_true")
    ap.add_argument("--trunk-decay", type=float, default=None, help="weight decay of the non-q/k trunk group (default legacy 1e-6)")
    ap.add_argument("--out-dir", required=True)
    a, rest = ap.parse_known_args(argv)
    assert "--qk-no-decay" not in rest, "use --qk-decay 0 (ep_train builds the optimiser)"
    resuming = os.path.exists(os.path.join(a.out_dir, "ckpt_latest.pt"))
    install(a.freeze_actor, a.qk_decay, a.route_change_ok and not resuming, a.trunk_decay)   # route override: --init-from only
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "ep_flags.json"), "w") as f:
        json.dump({"freeze_actor": a.freeze_actor, "qk_decay": a.qk_decay, "trunk_decay": a.trunk_decay, "route_change_ok": a.route_change_ok,
                   "argv": argv}, f, indent=1)
    print(f"[ep_train] freeze_actor={a.freeze_actor} qk_decay={a.qk_decay} route_change_ok={a.route_change_ok}",
          flush=True)
    return train_orig.main(rest + ["--out-dir", a.out_dir])


if __name__ == "__main__":
    raise SystemExit(main())
