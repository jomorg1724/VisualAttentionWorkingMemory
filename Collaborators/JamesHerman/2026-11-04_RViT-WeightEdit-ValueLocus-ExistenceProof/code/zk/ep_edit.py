#!/usr/bin/env python3
"""
Existence proof, Part C (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md rev 2 §3; added 2026-11-04): hand-written,
position-gated edits of a trained model's own LSTM weights at ONE location (the low-value location L of the model's
table), swept over strength. Nothing is trained.

Position direction. On the probe trials (frames 0-4, all slots), u = norm1(x + a) is the LSTM input; ubar_j its mean at
slot j. --direction pos (default, v2): w is supported on u's 4 position coordinates (128-131) and, with a scalar b0,
solves w . ubar_j + b0 = [j == L] exactly for all four slots (least norm). --direction mean (v1, 2026-11-04 first pass):
w = (ubar_L - ubar_o) / |.|^2 against the mean of the other three slots; it leaked about 0.75 of the offset onto the other
STIMULUS slot (it mostly encoded stimulus-vs-blank), so v1 results are not location-specific. The rank-1 edit
    R_g.weight += beta * 1 (x) w,     R_g.bias += beta * b0
adds beta to every unit's gate pre-activation at L on average and 0 elsewhere (the realised per-slot offset, mean and SD
over trials and frames, is reported).
Sites:  F  forget gate RF, beta = +k * SD(F_tilde at L): more weight on the stored past, the test is diluted.
        O  output gate RO, beta = -k * SD(O_tilde at L): a gain on L's memory output (H = O * C/N).
SD = the across-trial, across-frame SD of the pre-activation at slot L, averaged over units.

Per (site, k): the 18 deg lm_assay (d', c at both locations) on the edited checkpoint (scratch copy) and the
lm_readout_probe analysis on the probe trials (twin-grouped tested-slot AUC, pooled decoder, P9) plus the whole-state AUC.

    python -m zk.ep_edit --ckpt W/loc0_s42/ckpt_final.pt --hl 0 --out OUT.json [--ks 0 0.5 1 2 4 8]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from . import paths  # noqa: F401


@torch.no_grad()
def gate_stats(st, frames):
    """ubar per slot, and per-slot SD of F_tilde / O_tilde, from the probe trials (frames 0..4)."""
    from .lm_readout_probe import run_trials
    rec = {"u": [], "F": [], "O": []}
    lstm = st.blk.LSTM

    def hook(mod, inp):
        Zi, Ci, Mi, Hi, Ni, m = inp
        rec["u"].append(Zi.detach().double().cpu())
        rec["F"].append((mod.WF(Hi) + mod.RF(Zi)).detach().double().cpu())
        rec["O"].append((mod.WO(Hi) + mod.RO(Zi)).detach().double().cpu())
    h = lstm.register_forward_pre_hook(hook)
    try:
        run_trials(st, frames)
    finally:
        h.remove()
    U = torch.cat(rec["u"]); Fp = torch.cat(rec["F"]); Op = torch.cat(rec["O"])        # (N*frames, 4, .)
    ubar = U.mean(0)                                                                  # (4, 140)
    sd = {g: X.std(0).mean(-1).numpy() for g, X in (("F", Fp), ("O", Op))}            # per slot, mean over units
    return ubar.numpy(), sd, U


def edit_state(tsd, site, beta, w, b0):
    sd = {k: v.clone() for k, v in tsd.items()}
    key = {"F": "transformer_block1.LSTM.RF", "O": "transformer_block1.LSTM.RO"}[site]
    W = sd[key + ".weight"]; b = sd[key + ".bias"]
    wt = torch.as_tensor(w, dtype=W.dtype)
    sd[key + ".weight"] = W + beta * torch.ones(W.shape[0], 1, dtype=W.dtype) * wt[None, :]
    sd[key + ".bias"] = b + beta * b0
    return sd


def probe(ck, frames, locs, n, hl, device):
    from .lm_readout_probe import run_trials, analyse_frame
    from .lmsens_probe import auc
    from .mech_patch_orig import OrigStepper
    st = OrigStepper(ck, device)
    R = run_trials(st, frames)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int); L2 = np.r_[locs, locs]; tw = np.r_[np.arange(n), np.arange(n)]
    out = {}
    for t, Rt in R.items():
        res = analyse_frame(Rt, locs, hl, n, with_extras=False)
        for L in (0, 3):
            idx = np.flatnonzero(L2 == L)
            res["high" if L == hl else "low"]["auc_whole"] = auc(Rt["H"][idx].reshape(len(idx), -1).numpy(), y[idx],
                                                                 groups=tw[idx])
        out[f"t{t}"] = res
    del st; torch.cuda.empty_cache()
    return out


def assay(ck, hl, device, td, tag):
    from . import lm_assay
    o = os.path.join(td, tag + ".json")
    lm_assay.main(["--family", "orig", "--checkpoint", ck, "--condition-loc", str(hl), "--run-id", tag, "--output", o,
                   "--thetas", "18", "--no-fixation", "--task", "luo2015_sensitivity", "--device", device])
    d = json.load(open(o))
    return {("high" if e["location"] == hl else "low"): e["empirical"] for e in d["results"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hl", type=int, required=True, choices=(0, 3), help="high-value location of the model's table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", nargs="+", default=["F", "O"])
    ap.add_argument("--ks", type=float, nargs="+", default=[0, 0.5, 1, 2, 4, 8])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--scratch", default=os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    ap.add_argument("--direction", default="pos", choices=("pos", "mean"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"ckpt": a.ckpt, "hl": a.hl, "cells": {}}
    from .mech_patch_orig import OrigStepper, make_pairs_nofix
    A, B, locs = make_pairs_nofix(a.n, 18.0)
    frames = torch.cat([A, B]); n = len(locs)
    L = 3 - a.hl; others = [i for i in range(4) if i != L]
    st = OrigStepper(a.ckpt, a.device)
    ubar, sd, U = gate_stats(st, frames)
    del st; torch.cuda.empty_cache()
    if a.direction == "mean":                    # v1 (leaky: contrasts L with the blanks as well as the other stimulus)
        ubar_o = ubar[others].mean(0); dlt = ubar[L] - ubar_o; w = dlt / (dlt @ dlt); b0 = -float(w @ ubar_o)
    else:                                        # v2: w on the 4 position coordinates of u only; exact on all 4 slot means
        P = np.arange(128, 132)
        Aeq = np.c_[ubar[:, P], np.ones(4)]      # rows: slots; unknowns: w_P (4), b
        e = np.zeros(4); e[L] = 1.0
        sol = np.linalg.lstsq(Aeq, e, rcond=None)[0]
        w = np.zeros(ubar.shape[1]); w[P] = sol[:4]; b0 = float(sol[4]); dlt = ubar[L] - ubar[others].mean(0)
    proj = (U.numpy() @ w) + b0                                                       # realised offset / beta, (N, 4)
    res["direction"] = {"kind": a.direction, "w_pos": w[128:132].tolist(), "b0": b0, "L": L, "norm_delta": float(np.linalg.norm(dlt)), "sd_F": sd["F"].tolist(),
                        "sd_O": sd["O"].tolist(), "offset_per_beta_mean": proj.mean(0).tolist(),
                        "offset_per_beta_sd": proj.std(0).tolist()}
    ck0 = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        for site in a.sites:
            for k in a.ks:
                tag = f"{site}_k{k:g}"
                if tag in res["cells"] or (k == 0 and "k0" in res["cells"]):
                    continue
                beta = (k * float(sd[site][L])) * (1.0 if site == "F" else -1.0)
                p = os.path.join(td, tag + ".pt")
                torch.save({"transformer": edit_state(ck0["transformer"], site, beta, w, b0),
                            "agent": {"actor": ck0["agent"]["actor"]}, "config": ck0.get("config"),
                            "env_state": ck0.get("env_state"), "episode": ck0.get("episode")}, p)
                cell = {"site": site, "k": k, "beta": beta, "assay": assay(p, a.hl, a.device, td, tag),
                        "probe": probe(p, frames, locs, n, a.hl, a.device)}
                res["cells"]["k0" if k == 0 else tag] = cell
                b = cell["assay"]; pr = cell["probe"]["t3"]
                print(tag, f"d' hi {b['high']['dprime']:.2f} lo {b['low']['dprime']:.2f} c lo {b['low']['criterion']:+.2f}"
                      f" | AUC slot lo {pr['low']['auc_H_g']:.3f} whole lo {pr['low']['auc_whole']:.3f}"
                      f" pooled_dn hi/lo {pr['high']['pooled_dn']:.2f}/{pr['low']['pooled_dn']:.2f}", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
    print(f"[ep_edit] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
