#!/usr/bin/env python3
"""
Existence proof (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md rev 2 §4-5; added 2026-11-04): representation geometry and
an on-manifold readout-vs-representation decomposition, END network against a REFerence (the warm-up).

Probe trials: lm_readout_probe's make_pairs_nofix twins (600, 18 deg), noise off, frames t3 (primary) and t4. Per
network, per tested location L (role high / low by --hl, the arm's high-value location), per frame:
  auc_slot     twin-grouped 5-fold AUC (change vs no change) from H at the tested slot (= lm_readout_probe auc_H_g)
  auc_whole    the same from concat(H_0..H_3): the information available to the reader
  gap          mean dv(change) - mean dv(no change), dv = log P(respond) - log P(wait)
  J            mean actor Jacobian d dv / d H_full over the location's trials (4096-d)
  dmu          class-mean difference of H_full (change - no change)
and, END vs REF:
  lin_E, lin_R           J_E . dmu_E, J_R . dmu_R (first-order gaps)
  d_ro  = (J_E - J_R) . (dmu_E + dmu_R)/2      readout part of the change in the first-order gap
  d_rep = (J_E + J_R)/2 . (dmu_E - dmu_R)      representation part
  reader_R_on_E = J_R . dmu_E                  the reference reader's view of the end representation
  amp_slot = |dmu_E[slot L]| / |dmu_R[slot L]|, cos_slot = cosine(dmu_E[slot L], dmu_R[slot L]); amp/cos_whole likewise
  sd_slot_E / sd_slot_R  within-class SD along the REF class-mean direction at slot L (amplitude in signal-to-spread terms)

    python -m zk.ep_geom --end RUN/ckpt_final.pt --ref W/ckpt_final.pt --hl 0 --out RUN/geom_W.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import paths  # noqa: F401

FRAMES = (3, 4)


def _net(ck, device):
    from .mech_patch_orig import OrigStepper
    return OrigStepper(ck, device)


def _jac(actor, H, bs=300):
    """Mean over trials of d dv / d H_full; H (n, 4, dff) cpu."""
    tot = None
    for s in range(0, H.shape[0], bs):
        h = H[s:s + bs].reshape(H[s:s + bs].shape[0], -1).to(next(actor.parameters()).device).requires_grad_(True)
        p = actor(h).double()
        lp = torch.log(p.clamp_min(1e-12))
        (lp[:, 1] - lp[:, 0]).sum().backward()
        g = h.grad.detach().double().sum(0).cpu()
        tot = g if tot is None else tot + g
    return (tot / H.shape[0]).numpy()


def analyse(st, frames, locs, n):
    from .lm_readout_probe import run_trials
    from .lmsens_probe import auc
    R = run_trials(st, frames)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int); L2 = np.r_[locs, locs]; tw = np.r_[np.arange(n), np.arange(n)]
    out = {}
    for t in FRAMES:
        H, dv = R[t]["H"], R[t]["dv"].numpy()
        for L in (0, 3):
            idx = np.flatnonzero(L2 == L)
            Hl = H[idx]; yl = y[idx]
            Hf = Hl.reshape(len(idx), -1).double().numpy()
            dmu = Hf[yl == 1].mean(0) - Hf[yl == 0].mean(0)
            rec = {"auc_slot": auc(Hl[:, L].numpy(), yl, groups=tw[idx]),
                   "auc_whole": auc(Hf, yl, groups=tw[idx]),
                   "gap": float(dv[idx][yl == 1].mean() - dv[idx][yl == 0].mean()),
                   "J": _jac(st.ad.actor, Hl), "dmu": dmu,
                   "Hslot_c0": Hl[yl == 0, L].double().numpy(), "Hslot_c1": Hl[yl == 1, L].double().numpy()}
            out[(t, L)] = rec
    return out


def compare(E, Rf, hl, dff=1024):
    res = {}
    for (t, L), e in E.items():
        r = Rf[(t, L)]
        sl = slice(L * dff, (L + 1) * dff)
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        u = r["dmu"][sl] / (np.linalg.norm(r["dmu"][sl]) + 1e-30)
        sd = lambda rec: float(np.sqrt(0.5 * (np.var(rec["Hslot_c0"] @ u, ddof=1) + np.var(rec["Hslot_c1"] @ u, ddof=1))))
        rec = {"auc_slot_E": e["auc_slot"], "auc_slot_R": r["auc_slot"], "auc_whole_E": e["auc_whole"],
               "auc_whole_R": r["auc_whole"], "gap_E": e["gap"], "gap_R": r["gap"],
               "lin_E": float(e["J"] @ e["dmu"]), "lin_R": float(r["J"] @ r["dmu"]),
               "d_ro": float((e["J"] - r["J"]) @ (e["dmu"] + r["dmu"]) / 2),
               "d_rep": float((e["J"] + r["J"]) @ (e["dmu"] - r["dmu"]) / 2),
               "reader_R_on_E": float(r["J"] @ e["dmu"]),
               "amp_slot": float(np.linalg.norm(e["dmu"][sl]) / (np.linalg.norm(r["dmu"][sl]) + 1e-30)),
               "cos_slot": cos(e["dmu"][sl], r["dmu"][sl]),
               "amp_whole": float(np.linalg.norm(e["dmu"]) / (np.linalg.norm(r["dmu"]) + 1e-30)),
               "cos_whole": cos(e["dmu"], r["dmu"]),
               "proj_slot_E": float(e["dmu"][sl] @ u), "proj_slot_R": float(r["dmu"][sl] @ u),
               "sd_slot_E": sd(e), "sd_slot_R": sd(r)}
        res[f"t{t}_{'high' if L == hl else 'low'}"] = rec
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hl", type=int, required=True, choices=(0, 3))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"[ep_geom] exists: {a.out}"); return 0
    from .mech_patch_orig import make_pairs_nofix
    A, B, locs = make_pairs_nofix(a.n, 18.0)
    frames = torch.cat([A, B]); n = len(locs)
    E = analyse(_net(a.end, a.device), frames, locs, n); torch.cuda.empty_cache()
    Rf = analyse(_net(a.ref, a.device), frames, locs, n)
    res = {"end": a.end, "ref": a.ref, "hl": a.hl, "n": n, "compare": compare(E, Rf, a.hl)}
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["compare"].items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)
    print(f"[ep_geom] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
