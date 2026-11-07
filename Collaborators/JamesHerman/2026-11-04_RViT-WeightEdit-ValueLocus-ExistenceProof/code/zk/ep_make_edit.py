#!/usr/bin/env python3
"""
Existence proof, Part D (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md rev 3; added 2026-11-05): write the Part-C
hand edit into a full TRAINING checkpoint, so the standard learner can be fine-tuned from the constructed solution.

rev 3 review: --at high puts the same edit at the HIGH-value location (control: can the learner undo an edit when undoing
pays?). The run script then re-warms the critic (5k, critic only) on the edited network before any fine-tune.

Input: a warm-up checkpoint W (critic warmed on the SENS-C table; transformer and actor = the reward x10 start model).
The edit is ep_edit's position-gated, rank-1 output-gate edit (site O) at the SENS-C low-value location L = 3 - hl,
beta = -k * SD(O_tilde at L), direction = the "pos" direction (u's position coordinates, exact on the four slot means),
computed on this checkpoint's own probe trials (identical to Part C's, because the transformer is the same).
Everything else in the checkpoint (actor, critic, targets, optimiser states, env state, RNG) is copied unchanged.

    python -m zk.ep_make_edit --ckpt W/loc0_s42/ckpt_final.pt --hl 0 --k 1 --out EDIT_k1/loc0_s42/ckpt_final.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import paths  # noqa: F401
from .ep_edit import edit_state, gate_stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hl", type=int, required=True, choices=(0, 3))
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--site", default="O", choices=("O", "F"))
    ap.add_argument("--at", default="low", choices=("low", "high"), help="edit the table's low- (default) or high-value location")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"[ep_make_edit] exists: {a.out}"); return 0
    from .mech_patch_orig import OrigStepper, make_pairs_nofix
    A, B, locs = make_pairs_nofix(a.n, 18.0)
    frames = torch.cat([A, B])
    L = 3 - a.hl if a.at == "low" else a.hl
    st = OrigStepper(a.ckpt, a.device)
    ubar, sd, _ = gate_stats(st, frames)
    del st; torch.cuda.empty_cache()
    P = np.arange(128, 132)
    e = np.zeros(4); e[L] = 1.0
    sol = np.linalg.lstsq(np.c_[ubar[:, P], np.ones(4)], e, rcond=None)[0]
    w = np.zeros(ubar.shape[1]); w[P] = sol[:4]; b0 = float(sol[4])
    beta = a.k * float(sd[a.site][L]) * (1.0 if a.site == "F" else -1.0)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck["transformer"] = edit_state(ck["transformer"], a.site, beta, w, b0)
    ck["edit"] = {"site": a.site, "k": a.k, "at": a.at, "beta": beta, "L": L, "w_pos": w[P].tolist(), "b0": b0, "source": a.ckpt}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(ck, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    with open(os.path.join(os.path.dirname(a.out), "edit.json"), "w") as f:
        json.dump(ck["edit"], f, indent=1, default=str)
    print(f"[ep_make_edit] {a.out}: site {a.site} k {a.k} beta {beta:.3f} L {L}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
