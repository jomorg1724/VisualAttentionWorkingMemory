#!/usr/bin/env python3
"""
Existence proof, Part E0 (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md rev 4 §12 + Part E review; added 2026-11-06):
what erodes the hand-built encoding route in Part D? No training. One Part-D run per call (EDREL1 / EDREL1H; the run is
named after its start model, the paired FREE run has the same name).

Measure. log A = log|dmu_L| - log|dmu_O| on the t3 probe trials (lm_readout_probe twins, 18 deg, noise off): dmu_j is the
class-mean difference (change - no change) of H at slot j on trials tested at j; L = the edited location, O = the other.
amp_rel = A / A_W (= ep_geom's amp_slot(L)/amp_slot(O)).

  book    edit coordinate c(t) = e.(theta_t - theta_W)/|dtheta_edit| over RO's position columns + bias, at every snapshot,
          for --arm and for any extra --book-arms (E1 runs); pure-decay prediction for the edited entries from the EDIT
          checkpoint's Adam v (linear Adam, lr*wd/(sqrt(vhat)+eps) per step over the run's learn steps).
  course  amp_rel at EDIT, every EDREL snapshot, and FREE 10k / 25k / final.
  local   per trunk group g: revert (EDREL end with g from EDIT) and graft (EDIT with g from the EDREL end), and the same
          two operations in the paired FREE run relative to W; amp_rel and probe DiD d' (end actor) of each.
  gate    realised output-gate offset at L on the probe trials, beta*(w.u_L + b0) averaged over trials and frames 0-3,
          at EDIT and at the EDREL end; and u's four position coordinates at L (mean), EDIT vs end.
  grad    d log A / d theta (autograd through the token projection, attention and LSTM, t3) at the EDREL 1k and 5k
          checkpoints and FREE 10k; per loss part k (kl, pol, ent, decay) the Adam-preconditioned update
          u_k = -lr * g_k / (sqrt(vhat)+eps) on on-policy batches (zk.grad_attrib rollout + compute_losses, B 512 x 8);
          predicted per-step change <grad log A, u_k>, per group and for the edited entries alone; compared with the
          observed (log A(5k) - log A(1k)) / learn steps.

    python -m zk.ep_e0 --root RUNS --arm EDREL1 --name loc0_s42 --out OUT.json [--parts book course local gate grad]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import paths  # noqa: F401

EDIT_KEYS = ("transformer_block1.LSTM.RO.weight", "transformer_block1.LSTM.RO.bias")
GROUPS = {
    "proj": ("fc_state_projection.", "ln_state_projection.", "fc2_state_projection.", "ln2_state_projection."),
    "qk": ("transformer_block1.W_q.", "transformer_block1.W_k.", "transformer_block1.W_Cq.", "transformer_block1.W_Ck."),
    "val": ("transformer_block1.W_v.", "transformer_block1.W_Cv."),
    "norm1": ("transformer_block1.norm1.",),
    "lstmR": tuple(f"transformer_block1.LSTM.R{g}." for g in "IFOZ"),
    "lstmW": tuple(f"transformer_block1.LSTM.W{g}." for g in "IFOZ"),
}
SNAPS = ("ckpt_ep1000", "ckpt_ep5000", "ckpt_ep10000", "ckpt_ep25000", "ckpt_final")


def L(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def edit_coord(T, E, W):
    num = den = 0.0
    for k in EDIT_KEYS:
        d = (E[k] - W[k]).double(); num += float(((T[k] - W[k]).double() * d).sum()); den += float((d * d).sum())
    return num / den


def edit_mask(E, W):
    """1 where the edit changed a parameter (RO position columns + RO bias)."""
    return {k: ((E[k] - W[k]).abs() > 0).to(torch.float64) for k in EDIT_KEYS}


def decay_prediction(Eck, Wck, n_steps):
    from .adapters import build_orig_networks
    from .ep_train import two_group_optimizer
    tr, _ = build_orig_networks(torch.device("cpu"), Eck["transformer"])
    tr.optimizer = two_group_optimizer(tr, 0.0); tr.optimizer.load_state_dict(Eck["transformer_opt"])
    opt = tr.optimizer; b2 = opt.defaults["betas"][1]; eps = opt.defaults["eps"]
    named = dict(tr.named_parameters()); num = den = 0.0; rates = []
    for k in EDIT_KEYS:
        p = named[k]; st = opt.state[p]; g = [gr for gr in opt.param_groups if any(q is p for q in gr["params"])][0]
        vhat = st["exp_avg_sq"] / (1 - b2 ** float(st["step"]))
        rate = (g["lr"] * g["weight_decay"] / (vhat.sqrt() + eps)).double()
        d = (Eck["transformer"][k] - Wck["transformer"][k]).double()
        num += float((d * d * torch.exp(-n_steps * rate)).sum()); den += float((d * d).sum())
        rates.append(rate[d != 0].flatten())
    return num / den, float(torch.cat(rates).median())


# ---------------------------------------------------------------- probe forward (with or without grad)
def probe_inputs(n, device):
    from .mech_patch_orig import make_pairs_nofix
    from .adapters import encode_quadrants, T_END, TOKEN, load_vae
    A, B, locs = make_pairs_nofix(n, 18.0); frames = torch.cat([A, B]); N = frames.shape[0]
    vae = load_vae(paths.VAE_PATH, device)
    obs = torch.zeros(N, 4, 4, TOKEN, device=device)                       # frames 0..3 only
    with torch.no_grad():
        for t in range(4):
            for s0 in range(0, N, 300):
                obs[s0:s0 + 300, t, :, :128] = encode_quadrants(vae, frames[s0:s0 + 300, t].to(device))
            for p in range(4):
                obs[:, t, p, 128 + p] = 1.0
            obs[:, t, :, 132 + t] = 1.0
    y = np.r_[np.zeros(len(locs)), np.ones(len(locs))].astype(int); L2 = np.r_[locs, locs]
    return obs, y, L2


def forward_t3(tr, obs, want_u=False):
    """H at t3 (N, 4, dff) through the trunk, mirroring TransformerNetwork.forward (noise off, dropout off)."""
    N = obs.shape[0]; s = obs.reshape(-1, obs.shape[-1])
    X = F.elu(s + tr.ln_state_projection(tr.fc_state_projection(s)))
    s = tr.ln2_state_projection(s + tr.fc2_state_projection(X)).view(N, 4, 4, -1)
    b = tr.transformer_block1
    C = M = H = Nn = torch.zeros(N, 4, b.dff, device=obs.device); us = []
    m = torch.ones(N, 4, 1, device=obs.device)
    for i in range(4):
        x = s[:, i]
        q = b.W_q(x) * b.W_Cq(H); k = b.W_k(x) * b.W_Ck(H); v = b.W_v(x) * b.W_Cv(H)
        A = torch.softmax(q @ k.transpose(-2, -1) / (b.d_k ** 0.5), dim=-1)
        u = b.norm1(x + A @ v); us.append(u)
        C, M, H, Nn = b.LSTM(u, C, M, H, Nn, m)
    return (H, us) if want_u else H


def logA(H, y, L2, Le):
    O = 3 - Le; out = []
    for j in (Le, O):
        idx = torch.as_tensor(L2 == j, device=H.device); yt = torch.as_tensor(y, device=H.device)
        dmu = H[idx & (yt == 1), j].mean(0) - H[idx & (yt == 0), j].mean(0)
        out.append(torch.log(dmu.norm()))
    return out[0] - out[1]


def eval_net(tr, actor, obs, y, L2, Le, hl):
    from .lm_readout_probe import dprime
    with torch.no_grad():
        H = forward_t3(tr, obs)
        la = float(logA(H, y, L2, Le))
        p = actor(H.reshape(H.shape[0], -1)).double()[:, 1].cpu().numpy()
    d = {}
    for j in (0, 3):
        idx = L2 == j
        d[j] = dprime(float(p[idx & (y == 1)].mean()), float(p[idx & (y == 0)].mean()))[0]
    return {"logA": la, "DiD_probe": d[hl] - d[3 - hl]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", required=True, choices=("EDREL1", "EDREL1H"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", default=["book", "course", "local", "gate", "grad"])
    ap.add_argument("--book-arms", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ed = {"EDREL1": "EDIT1", "EDREL1H": "EDIT1H"}[a.arm]
    P = lambda arm, ck="ckpt_final": os.path.join(a.root, arm, a.name, ck + ".pt")
    Eck, Wck, Fend = L(P(ed)), L(P("W")), L(P(a.arm))
    hl = int(Eck["config"]["high_loc"]); edit = json.load(open(os.path.join(a.root, ed, a.name, "edit.json")))
    Le = int(edit["L"]); res.update(arm=a.arm, name=a.name, hl=hl, L_edit=Le)
    suf = "1H" if a.arm == "EDREL1H" else "1"
    a.book_arms = [x for x in a.book_arms if x.startswith("FREE") or (x.endswith(suf) and not (suf == "1" and x.endswith("1H")))]
    steps = {}

    def coord_series(arm):
        out = {}
        for ck in SNAPS:
            if os.path.exists(P(arm, ck)):
                c = L(P(arm, ck)); out[ck] = edit_coord(c["transformer"], Eck["transformer"], Wck["transformer"])
                steps[f"{arm}/{ck}"] = int(c["agent"]["learn_step_cntr"])
        return out

    if "book" in a.parts:
        book = {a.arm: coord_series(a.arm), "FREE": coord_series("FREE")}
        for arm in a.book_arms:
            if os.path.exists(P(arm)):
                book[arm] = coord_series(arm)
        n_steps = int(Fend["agent"]["learn_step_cntr"]) - int(Eck["agent"]["learn_step_cntr"])
        pred, med = decay_prediction(Eck, Wck, n_steps)
        disp = {}
        for arm in [a.arm, "FREE"] + list(a.book_arms):
            if not os.path.exists(P(arm)):
                continue
            T = L(P(arm))["transformer"]; S0 = Wck["transformer"] if arm.startswith("FREE") else Eck["transformer"]
            disp[arm] = {g: float(sum(((T[k] - S0[k]).double() ** 2).sum() for k in T if k.startswith(pre)) ** 0.5)
                         for g, pre in GROUPS.items()}
        res["book"] = {"edit_coord": book, "displacement": disp, "learn_steps": n_steps, "decay_pred_retained": pred,
                       "decay_rate_median": med, "steps": steps}
        print("[e0 book]", json.dumps(res["book"])[:600], flush=True)

    obs, y, L2 = probe_inputs(a.n, dev)
    from .adapters import build_orig_networks
    tr, actor = build_orig_networks(dev, Wck["transformer"], Fend["agent"]["actor"]); tr.eval()
    for p in tr.parameters():
        p.requires_grad_(False)
    def ev(sd, act=None):
        tr.load_state_dict(sd)
        if act is not None:
            actor.load_state_dict(act)
        return eval_net(tr, actor, obs, y, L2, Le, hl)
    laW = ev(Wck["transformer"])["logA"]

    if "course" in a.parts:
        course = {"W": 0.0, ed: ev(Eck["transformer"])["logA"] - laW}
        for arm, cks in [(a.arm, SNAPS), ("FREE", ("ckpt_ep10000", "ckpt_ep25000", "ckpt_final"))] + [(x, SNAPS) for x in a.book_arms]:
            for ck in cks:
                if os.path.exists(P(arm, ck)):
                    c = L(P(arm, ck)); course[f"{arm}/{ck}"] = ev(c["transformer"])["logA"] - laW
                    steps[f"{arm}/{ck}"] = int(c["agent"]["learn_step_cntr"])
        res["course"] = {"logA_rel": course, "amp_rel": {k: float(np.exp(v)) for k, v in course.items()},
                         "steps": steps, "steps_EDIT": int(Eck["agent"]["learn_step_cntr"])}
        print("[e0 course]", {k: round(float(np.exp(v)), 3) for k, v in course.items()}, flush=True)

    if "local" in a.parts:
        Ffree = L(P("FREE")); Et, Ft, Wt, FRt = Eck["transformer"], Fend["transformer"], Wck["transformer"], Ffree["transformer"]
        loc = {"EDREL_end": ev(Ft, Fend["agent"]["actor"]), "EDIT": ev(Et, Fend["agent"]["actor"]),
               "FREE_end": ev(FRt, Ffree["agent"]["actor"]), "W": ev(Wt, Ffree["agent"]["actor"])}
        allpre = tuple(q for g in GROUPS.values() for q in g)
        for g, pre in list(GROUPS.items()) + [("all", allpre)]:
            mix = lambda base, src: {k: (src[k] if k.startswith(pre) else v) for k, v in base.items()}
            loc[f"revert_{g}"] = ev(mix(Ft, Et), Fend["agent"]["actor"])
            loc[f"graft_{g}"] = ev(mix(Et, Ft), Fend["agent"]["actor"])
            loc[f"FREE_revert_{g}"] = ev(mix(FRt, Wt), Ffree["agent"]["actor"])
            loc[f"FREE_graft_{g}"] = ev(mix(Wt, FRt), Ffree["agent"]["actor"])
        for k in loc:
            loc[k]["logA_rel"] = loc[k]["logA"] - laW
        res["local"] = loc
        print("[e0 local]", {k: round(v["logA_rel"], 3) for k, v in loc.items()}, flush=True)

    if "gate" in a.parts:
        w = torch.zeros(obs.shape[-1], dtype=torch.float64); w[128:132] = torch.tensor(edit["w_pos"], dtype=torch.float64)
        g = {}
        gl = [("EDIT", Eck["transformer"]), ("EDREL_end", Fend["transformer"]), ("W", Wck["transformer"]),
              ("FREE_end", L(P("FREE"))["transformer"])]
        gl += [(f"{arm}_end", L(P(arm))["transformer"]) for arm in a.book_arms if os.path.exists(P(arm))]
        for tag, sd in gl:
            tr.load_state_dict(sd)
            with torch.no_grad():
                _, us = forward_t3(tr, obs, want_u=True)
            U = torch.stack(us, 1)[:, :, Le].double().cpu()                     # (N, 4 frames, 140)
            g[tag] = {"offset_per_beta": float((U @ w + edit["b0"]).mean()), "u_pos_mean": U[..., 128:132].mean((0, 1)).tolist(),
                      "u_pos_norm": float(U[..., 128:132].mean((0, 1)).norm()),
                      "u_pos_norm_allslots": [float(torch.stack(us, 1)[:, :, j, 128:132].double().mean((0, 1)).norm()) for j in range(4)]}
        g["realised_offset_ratio_end_over_EDIT"] = g["EDREL_end"]["offset_per_beta"] / g["EDIT"]["offset_per_beta"]
        res["gate"] = g; print("[e0 gate]", g, flush=True)

    if "grad" in a.parts:
        from . import grad_attrib as GA
        from .ep_train import two_group_optimizer
        _b = GA.build_orig_networks

        def build(d, *x, **k):
            t, ac = _b(d, *x, **k); t.optimizer = two_group_optimizer(t, 0.0); return t, ac
        GA.build_orig_networks = build
        mask = {k: v.to(dev) for k, v in edit_mask(Eck["transformer"], Wck["transformer"]).items()}
        vae = GA.load_vae(GA.paths.VAE_PATH, dev); out = {}
        for tag, path in (("EDREL_1k", P(a.arm, "ckpt_ep1000")), ("EDREL_5k", P(a.arm, "ckpt_ep5000")),
                          ("FREE_10k", P("FREE", "ckpt_ep10000"))):
            if not os.path.exists(path):
                continue
            random.seed(20261106); np.random.seed(20261106); torch.manual_seed(20261106)
            ckd, cfg, env, agent, trn = GA.load(path, dev)
            names = [nm for nm, _ in trn.named_parameters()]; params = [p for _, p in trn.named_parameters()]
            trn.eval()
            la = logA(forward_t3(trn, obs), y, L2, Le)
            gA = torch.autograd.grad(la, params, allow_unused=True)
            gA = [torch.zeros_like(p) if x is None else x.detach() for p, x in zip(params, gA)]
            trn.train()
            data = GA.rollout(cfg, env, agent, trn, vae, dev, a.episodes)
            opt = trn.optimizer; b2 = opt.defaults["betas"][1]; eps = opt.defaults["eps"]
            lr = {id(p): gr["lr"] for gr in opt.param_groups for p in gr["params"]}
            wd = {id(p): gr["weight_decay"] for gr in opt.param_groups for p in gr["params"]}
            pre = []
            for p in params:
                s_ = opt.state.get(p, {})
                pre.append(lr[id(p)] / ((s_["exp_avg_sq"] / (1 - b2 ** float(s_["step"]))).sqrt() + eps)
                           if "exp_avg_sq" in s_ else torch.zeros_like(p))
            groups = {gn: [i for i, nm in enumerate(names) if nm.startswith(pr)] for gn, pr in GROUPS.items()}
            groups["all"] = list(range(len(params)))
            # stationary-Adam preconditioner (Part E review: the checkpoint's v is stale where gradients were ~0, giving
            # lr/eps = 1e3 gains): per element s = mean over replay-size batches (B = 32) of the TOTAL gradient squared;
            # expected step of part k = -lr * mean_b(g_k) / (sqrt(s) + eps); parts share the denominator, so they add up.
            NB = 48; gsum = {kk: [torch.zeros_like(p) for p in params] for kk in ("kl", "pol", "ent")}
            ssum = [torch.zeros_like(p) for p in params]; rng = np.random.RandomState(20261106)
            for r in range(NB):
                sel = rng.randint(0, len(data[0]), size=32); torch.manual_seed(r)
                kl, LPol, Hh, _ = agent.compute_losses(trn, *(x[sel] for x in data[:7]))
                G = {}
                for kk, Lo in (("kl", kl), ("pol", -LPol), ("ent", -agent.entropy_coef * Hh)):
                    gg = torch.autograd.grad(Lo, params, retain_graph=True, allow_unused=True)
                    G[kk] = [torch.zeros_like(p) if x is None else x.detach() for p, x in zip(params, gg)]
                for i, p in enumerate(params):
                    tot_i = G["kl"][i] + G["pol"][i] + G["ent"][i] + wd[id(p)] * p.detach()
                    ssum[i] += tot_i ** 2
                    for kk in G:
                        gsum[kk][i] += G[kk][i]
            gbar = {kk: [x / NB for x in v] for kk, v in gsum.items()}
            gbar["decay"] = [wd[id(p)] * p.detach() for p in params]
            den = [ (x / NB).sqrt() + eps for x in ssum]
            rec = {}
            for kk in gbar:
                contrib = [-(lr[id(params[i])] * gbar[kk][i] / den[i] * gA[i]).double() for i in range(len(params))]
                for gn, idx in groups.items():
                    rec[f"{gn}.{kk}"] = float(sum(contrib[i].sum() for i in idx))
                rec[f"edited.{kk}"] = float(sum((contrib[names.index(k)] * mask[k]).sum() for k in EDIT_KEYS))
                # old (checkpoint-v) linearisation kept for the record
                rec[f"ckptv_all.{kk}"] = float(sum((-(gbar[kk][i] * pre[i] * gA[i]).double()).sum() for i in range(len(params))))
            acc = [rec]
            m = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
            sd = {k: float(np.std([x[k] for x in acc])) for k in acc[0]}
            tot = sum(m[f"all.{kk}"] for kk in ("kl", "pol", "ent", "decay"))
            out[tag] = {"mean": m, "sd": sd, "pred_dlogA_per_step": tot,
                        "share": {kk: m[f"all.{kk}"] / tot for kk in ("kl", "pol", "ent", "decay")} if tot else None}
            print("[e0 grad]", tag, "pred/step %.3e" % tot, out[tag]["share"], flush=True)
            del trn, agent; torch.cuda.empty_cache()
        if "course" in res and "EDREL_1k" in out:
            c = res["course"]; s1, s5 = steps.get(f"{a.arm}/ckpt_ep1000"), steps.get(f"{a.arm}/ckpt_ep5000")
            if s1 and s5:
                obs_rate = (c["logA_rel"][f"{a.arm}/ckpt_ep5000"] - c["logA_rel"][f"{a.arm}/ckpt_ep1000"]) / (s5 - s1)
                pred = 0.5 * (out["EDREL_1k"]["pred_dlogA_per_step"] + out.get("EDREL_5k", out["EDREL_1k"])["pred_dlogA_per_step"])
                out["check"] = {"observed_dlogA_per_step_1k_5k": obs_rate, "predicted": pred,
                                "ratio": pred / obs_rate if obs_rate else float("nan")}
                print("[e0 grad check]", out["check"], flush=True)
        res["grad"] = out
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[ep_e0] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
