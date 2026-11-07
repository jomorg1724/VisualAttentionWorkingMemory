#!/usr/bin/env python3
"""
Existence proof (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md §4-5; added 2026-11-04): where does the value effect live?

--mode cross       2x2 hybrid decomposition. For an END checkpoint and a REFerence checkpoint (W, or the stage-3 start),
                   build the two cross networks (actor from one, transformer from the other), write them as stripped
                   checkpoints to a scratch dir (derived, reproducible) and run zk.lm_assay at 18 deg on each (the same
                   banks and common random numbers as every other assay). The (ref, ref) and (end, end) cells are the
                   runs' own roc_*.json. Output JSON: {"Wactor_ENDtr": roc payload, "ENDactor_Wtr": roc payload}.
--mode transplant  probe-level localization: zk.lm_readout_probe trials (600 twin pairs, 18 deg, noise off), actor
                   P(respond) at t3 and t4. For each transformer module group G, the END network with G reverted to REF
                   ("revert_G", necessity) and the REF network with G taken from END ("graft_G", sufficiency); plus
                   "end" and "ref". Per location: HR, FA, probe d' (t3, t4) and the mean AUC of H at the tested slot (t3).

    python -m zk.ep_hybrid --mode cross --end RUN/ckpt_final.pt --ref W/ckpt_final.pt --hl 0 --out RUN/hyb_W.json
    python -m zk.ep_hybrid --mode transplant --end ... --ref ... --hl 0 --out RUN/tpl_W.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from . import paths  # noqa: F401

GROUPS = {
    "proj": ("fc_state_projection.", "ln_state_projection.", "fc2_state_projection.", "ln2_state_projection."),
    "qk": ("transformer_block1.W_q.", "transformer_block1.W_k.", "transformer_block1.W_Cq.", "transformer_block1.W_Ck."),
    "val": ("transformer_block1.W_v.", "transformer_block1.W_Cv."),
    "mem": ("transformer_block1.norm1.", "transformer_block1.LSTM."),
}


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def _strip(ck, transformer, actor):
    return {"transformer": transformer, "agent": {"actor": actor}, "config": ck.get("config"),
            "env_state": ck.get("env_state"), "episode": ck.get("episode")}


def cross(a):
    from . import lm_assay
    E, R = _load(a.end), _load(a.ref)
    out = {}
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        for tag, tsd, asd in (("REFactor_ENDtr", E["transformer"], R["agent"]["actor"]),
                              ("ENDactor_REFtr", R["transformer"], E["agent"]["actor"])):
            p = os.path.join(td, tag + ".pt"); torch.save(_strip(E, tsd, asd), p)
            o = os.path.join(td, tag + ".json")
            lm_assay.main(["--family", "orig", "--checkpoint", p, "--condition-loc", str(a.hl), "--run-id", tag,
                           "--output", o, "--thetas", "18", "--no-fixation", "--task", "luo2015_sensitivity",
                           "--device", a.device])
            out[tag] = json.load(open(o))
    out.update(end=a.end, ref=a.ref)
    json.dump(out, open(a.out, "w"), indent=1)


def transplant(a):
    from .lm_readout_probe import run_trials, dprime
    from .lmsens_probe import auc
    from .mech_patch_orig import OrigStepper, make_pairs_nofix
    E, R = _load(a.end), _load(a.ref)
    A, B, locs = make_pairs_nofix(a.n, 18.0)
    frames = torch.cat([A, B]); n = len(locs)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int); L2 = np.r_[locs, locs]; tw = np.r_[np.arange(n), np.arange(n)]
    st = OrigStepper(a.end, a.device)
    nets = {"end": E["transformer"], "ref": R["transformer"]}
    for g, pre in GROUPS.items():
        nets[f"revert_{g}"] = {k: (R["transformer"][k] if k.startswith(pre) else v) for k, v in E["transformer"].items()}
        nets[f"graft_{g}"] = {k: (E["transformer"][k] if k.startswith(pre) else v) for k, v in R["transformer"].items()}
    allpre = tuple(p for g in GROUPS.values() for p in g)
    nets["revert_all"] = {k: (R["transformer"][k] if k.startswith(allpre) else v) for k, v in E["transformer"].items()}
    res = {"end": a.end, "ref": a.ref, "hl": a.hl, "n": n, "conditions": {}}
    for actor_tag, asd in (("endactor", E["agent"]["actor"]), ("refactor", R["agent"]["actor"])):
        st.ad.actor.load_state_dict(asd)
        for name, sd in nets.items():
            st.tr.load_state_dict(sd)
            Rt = run_trials(st, frames)
            rec = {}
            for t, Rr in Rt.items():
                p = Rr["p"].numpy()
                for L in (0, 3):
                    role = "high" if L == a.hl else "low"
                    sel = L2 == L
                    HR, FA = float(p[sel & (y == 1)].mean()), float(p[sel & (y == 0)].mean())
                    d, c = dprime(HR, FA)
                    rec[f"t{t}_{role}"] = {"HR": HR, "FA": FA, "dprime": d, "c": c}
                    if t == 3 and actor_tag == "endactor":
                        idx = np.flatnonzero(sel)
                        rec[f"t{t}_{role}"]["auc_H_g"] = auc(Rr["H"][idx, L].numpy(), y[idx], groups=tw[idx])
            res["conditions"][f"{actor_tag}|{name}"] = rec
            print(actor_tag, name, {k: round(v["dprime"], 3) for k, v in rec.items()}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("cross", "transplant"))
    ap.add_argument("--end", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hl", type=int, required=True, choices=(0, 3))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--scratch", default=os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"[ep_hybrid] exists: {a.out}"); return 0
    (cross if a.mode == "cross" else transplant)(a)
    print(f"[ep_hybrid] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
