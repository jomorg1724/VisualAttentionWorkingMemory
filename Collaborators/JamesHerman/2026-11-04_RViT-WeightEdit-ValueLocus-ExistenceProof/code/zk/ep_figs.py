#!/usr/bin/env python3
"""
Existence-proof figures (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md rev 2; added 2026-11-04; local only).
Reads ep_stats.json; writes PNG + PDF.

  fig_partC     hand edits: d' at both locations, criterion, and the three representation losses vs edit strength
  fig_partA     constrained / unconstrained fine-tunes: behavioral value effect and representation losses per arm
  fig_locus     where the value lives: hybrid and Jacobian representation shares; readout slope (P9) by location
  fig_partB     release: DiD d' and representation losses over the stage-3 snapshots / end

    python -m zk.ep_figs --stats .../stats/ep_stats.json --out-dir .../figs
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HI, LO = "#1f77b4", "#d62728"
ARMS_A = ("FREE", "FREEREV", "FROZ", "FROZREV", "FROZNULL")
LABEL = {"FREE": "free", "FREEREV": "free,\nreversed", "FROZ": "frozen\nreadout", "FROZREV": "frozen,\nreversed",
         "FROZNULL": "frozen,\nno value\nchange", "W": "warm-up", "REL": "release", "RELK": "release +\nreward step",
         "CONT": "kept\nfrozen", "RELREV": "release\n(rev)", "RELKREV": "release +\nstep (rev)", "CONTREV": "kept\nfrozen (rev)"}


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def err(ax, x, v, **kw):
    if v is None:
        return
    ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", capsize=3, **kw)


def fig_partC(S, out):
    C = S.get("partC") or {}
    if not C:
        return
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    for col, (site, title) in enumerate((("O", "Output-gate gain at L (C-O)"), ("F", "Forget-gate bias at L (C-F)"))):
        ks = sorted(float(t.split("_k")[1]) for t in C if t.startswith(site + "_k"))
        cell = lambda k: C["k0"] if k == 0 else C[f"{site}_k{k:g}"]
        ks = [0.0] + ks
        x = np.arange(len(ks))
        ax = axs[0, col]
        for key, c, lab in (("d_high", HI, "other stimulus location"), ("d_low", LO, "edited location L")):
            m = [cell(k)[key][0] for k in ks]; lo = [cell(k)[key][1] for k in ks]; hi = [cell(k)[key][2] for k in ks]
            ax.plot(x, m, "o-", color=c, label=lab); ax.fill_between(x, lo, hi, color=c, alpha=.2)
        ax.set_title(title, fontsize=10); ax.set_ylabel("d′ (18° assay)"); ax.axhline(0, color="k", lw=.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{k:g}" for k in ks])
        ax.legend(fontsize=8)
        ax = axs[1, col]
        for key, c, lab in (("lossN", "#2ca02c", "information (tested-slot AUC)"),
                            ("lossWhole", "#9467bd", "information (whole state)"),
                            ("lossP", "#ff7f0e", "amplitude (pooled decoder)"),
                            ("loss_beh", "k", "behavior (d′)")):
            m = [cell(k)[key][0] for k in ks]
            ax.plot(x, m, "o-", color=c, label=lab)
        ax.axhline(0.077, color="grey", ls=":", lw=1)
        ax.set_ylim(-0.4, 1.5); ax.set_ylabel("proportional loss at L"); ax.set_xticks(x)
        ax.set_xticklabels([f"{k:g}" for k in ks]); ax.set_xlabel("edit strength k (pre-activation SDs)")
        if col == 0:
            ax.legend(fontsize=7)
    ax = axs[0, 2]
    for site, c in (("O", "#ff7f0e"), ("F", "#2ca02c")):
        ks = [0.0] + sorted(float(t.split("_k")[1]) for t in C if t.startswith(site + "_k"))
        cell = lambda k: C["k0"] if k == 0 else C[f"{site}_k{k:g}"]
        ax.plot([cell(k)["c_low"][0] for k in ks], [cell(k)["d_low"][0] for k in ks], "o-", color=c, label=f"C-{site}")
    ax.set_xlabel("criterion c at L"); ax.set_ylabel("d′ at L"); ax.set_title("criterion vs sensitivity at L", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    for site, c in (("O", "#ff7f0e"), ("F", "#2ca02c")):
        ks = [0.0] + sorted(float(t.split("_k")[1]) for t in C if t.startswith(site + "_k"))
        cell = lambda k: C["k0"] if k == 0 else C[f"{site}_k{k:g}"]
        ax.plot([cell(k)["loss_beh"][0] for k in ks], [cell(k)["lossN"][0] for k in ks], "o-", color=c,
                label=f"C-{site}: information")
        ax.plot([cell(k)["loss_beh"][0] for k in ks], [cell(k)["lossP"][0] for k in ks], "s--", color=c,
                label=f"C-{site}: amplitude", alpha=.7)
    xx = np.linspace(0, 1.4, 10); ax.plot(xx, xx / 2, color="grey", ls=":", lw=1, label="½ behavioral loss")
    ax.set_xlim(-0.1, 1.4); ax.set_ylim(-0.4, 1.1)
    ax.set_xlabel("behavioral loss at L"); ax.set_ylabel("representation loss at L"); ax.legend(fontsize=7)
    fig.suptitle("Part C: one rank-1, position-gated edit of the LSTM's own weights (20 start models; mean, 95% CI)",
                 fontsize=11)
    fig.tight_layout(); save(fig, out, "fig_partC")


def fig_partA(S, out):
    A = S["arms"]; arms = [a for a in ARMS_A if a in A and "from_warmup" in A[a]]
    if not arms:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(arms)); ax = axs[0]
    for i, a in enumerate(arms):
        err(ax, i, A[a]["from_warmup"]["DiD_d"], color="k")
    CR = S.get("congruent_reversed", {})
    for j, (key, lab) in enumerate((("FREE", "free: value term"), ("FROZ", "frozen: value term"))):
        if key in CR:
            err(ax, len(arms) + j, CR[key]["DiD_d"]["value_term"], color="#2ca02c")
    ax.axhline(0, color="k", lw=.5); ax.axhline(0.3, color="grey", ls=":", lw=1)
    ax.set_xticks(list(x) + [len(arms), len(arms) + 1])
    ax.set_xticklabels([LABEL[a] for a in arms] + ["free:\nvalue\nterm", "frozen:\nvalue\nterm"], fontsize=7)
    ax.set_ylabel("change in DiD d′ from the warm-up"); ax.set_title("value effect on behavior (18° assay)", fontsize=10)
    ax = axs[1]
    for j, (key, c, lab) in enumerate((("loss_beh", "k", "behavior"), ("t3_lossN", "#2ca02c", "information (tested slot)"),
                                        ("t3_lossWhole", "#9467bd", "information (whole state)"),
                                        ("t3_lossP", "#ff7f0e", "amplitude (pooled decoder)"))):
        for i, a in enumerate(arms):
            err(ax, i + (j - 1.5) * .15, A[a].get(key), color=c, label=lab if i == 0 else None)
    ax.axhline(0, color="k", lw=.5); ax.axhline(0.077, color="grey", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in arms], fontsize=7)
    ax.set_ylabel("proportional loss, low vs high value (t3)"); ax.legend(fontsize=7); ax.set_title("end state", fontsize=10)
    ax = axs[2]
    for i, a in enumerate(arms):
        v = A[a].get("hybW_rep_share")
        if v:
            ax.errorbar(i, v["share"], yerr=[[v["share"] - v["ci90"][0]], [v["ci90"][1] - v["share"]]], fmt="o", color="#8c564b",
                        capsize=3)
    ax.axhline(0, color="k", lw=.5); ax.axhline(1, color="k", lw=.5, ls=":")
    ax.set_ylim(-0.3, 1.3); ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in arms], fontsize=7)
    ax.set_ylabel("share of the change in DiD d′ carried by\nthe transformer (2×2 hybrid, 90% CI)")
    ax.set_title("where the change lives (1 = representation, 0 = actor)", fontsize=10)
    fig.suptitle("Part A: fine-tunes from the reward ×10 models to the SENS-C table (10 pairs per arm; mean, 95% CI)",
                 fontsize=11)
    fig.tight_layout(); save(fig, out, "fig_partA")


def fig_locus(S, out):
    """P9 readout slope vs pooled amplitude: which measure sees a gain placed in H (EDIT) vs in the actor (FREE)."""
    A = S["arms"]; arms = [a for a in ("W", "FREE", "FROZ", "EDIT1", "EDIT2") if a in A and "t3_readout_slope_low" in A[a]]
    if not arms:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); x = np.arange(len(arms))
    for i, a in enumerate(arms):
        err(axs[0], i - .1, A[a].get("t3_readout_slope_high"), color=HI, label="high-value location" if i == 0 else None)
        err(axs[0], i + .1, A[a].get("t3_readout_slope_low"), color=LO, label="low-value location" if i == 0 else None)
        err(axs[1], i - .1, A[a].get("JW_amp_slot_high"), color=HI)
        err(axs[1], i + .1, A[a].get("JW_amp_slot_low"), color=LO)
    lab = {"W": "warm-up", "FREE": "free\n(value in actor)", "FROZ": "frozen\nreadout", "EDIT1": "hand edit\nk = 1",
           "EDIT2": "hand edit\nk = 2"}
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in arms], fontsize=8)
    axs[0].set_ylabel("P9 readout slope (actor dv per pooled-SD unit)"); axs[0].legend(fontsize=7)
    axs[1].set_ylabel("|class-mean difference| at the slot, vs warm-up"); axs[1].axhline(1, color="k", lw=.5, ls=":")
    axs[0].set_title("P9 slope sees an actor gain", fontsize=10); axs[1].set_title("fixed-axis amplitude sees a gain in H", fontsize=10)
    fig.tight_layout(); save(fig, out, "fig_locus")


def fig_partD(S, out):
    A = S["arms"]; Rt = S.get("retention", {})
    pairs = [(s, e) for s, e in (("EDIT1", "EDREL1"), ("EDIT2", "EDREL2"), ("EDIT1H", "EDREL1H")) if s in A and e in A]
    if not pairs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    for i, (s_, e_) in enumerate(pairs):
        pts = [("start", A[s_]["DiD_d"])] + [(k, A[e_][f"snap_ep{k}"]["DiD_d"]) for k in (1000, 5000, 10000, 25000)
                                              if f"snap_ep{k}" in A[e_]] + [("50000", A[e_]["DiD_d"])]
        xs = [0] + [int(p[0]) / 1000 for p in pts[1:]]
        ax.plot(xs, [p[1][0] for p in pts], "o-", label=f"{e_} (from {s_})")
    if "FREE" in A:
        f = A["FREE"]; xs = [0, 10, 25, 50]
        ys = [A["W"]["DiD_d"][0], f["snap_ep10000"]["DiD_d"][0], f["snap_ep25000"]["DiD_d"][0], f["DiD_d"][0]]
        ax.plot(xs, ys, "k--", label="FREE (from the unedited warm-up)")
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel("fine-tune episodes (k)"); ax.set_ylabel("DiD d′ (high − low value)")
    ax.legend(fontsize=7); ax.set_title("behavior", fontsize=10)
    ax = axs[1]
    for i, (s_, e_) in enumerate(pairs):
        r = Rt.get(e_, {})
        for j, (m, c) in enumerate((("amp", "#ff7f0e"), ("lossN", "#2ca02c"))):
            v = r.get(m)
            if v:
                ax.errorbar(i + (j - .5) * .2, v["r"], yerr=[[v["r"] - v["ci90"][0]], [v["ci90"][1] - v["r"]]], fmt="o", color=c,
                            capsize=3, label=("amplitude (fixed axis)" if m == "amp" else "information (tested slot)") if i == 0 else None)
    ax.axhline(1, color="k", lw=.5, ls=":"); ax.axhline(0, color="k", lw=.5); ax.axhspan(0.7, 1.3, color="g", alpha=.06)
    ax.axhspan(-0.3, 0.3, color="r", alpha=.06)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([e for _, e in pairs], fontsize=8)
    ax.set_ylabel("retained fraction of the edit's effect at 50k\n(EDREL − FREE) / (EDIT − warm-up), 90% CI"); ax.legend(fontsize=7)
    ax.set_title("does the learner keep the hand-built route?", fontsize=10)
    fig.tight_layout(); save(fig, out, "fig_partD")


def fig_partB(S, out):
    A = S["arms"]; arms = [a for a in ("CONT", "REL", "RELK", "CONTREV", "RELREV", "RELKREV") if a in A and "DiD_d" in A[a]]
    if not arms:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for a in arms:
        pts = [(k, A[a][f"snap_ep{k}"]["DiD_d"][0]) for k in (5000, 10000, 25000) if f"snap_ep{k}" in A[a]]
        pts.append((50000, A[a]["DiD_d"][0]))
        axs[0].plot([p[0] / 1000 for p in pts], [p[1] for p in pts], "o-", label=LABEL[a].replace("\n", " "))
    axs[0].set_xlabel("episodes after the stage-3 start (k)"); axs[0].set_ylabel("DiD d′"); axs[0].legend(fontsize=7)
    x = np.arange(len(arms))
    for j, (key, c) in enumerate((("t3_lossN", "#2ca02c"), ("t3_lossP", "#ff7f0e"), ("loss_beh", "k"))):
        for i, a in enumerate(arms):
            err(axs[1], i + (j - 1) * .15, A[a].get(key), color=c, label=key if i == 0 else None)
    axs[1].set_xticks(x); axs[1].set_xticklabels([LABEL[a] for a in arms], fontsize=7); axs[1].legend(fontsize=7)
    axs[1].set_ylabel("proportional loss at the end")
    fig.tight_layout(); save(fig, out, "fig_partB")



def fig_partE(S, out):
    E = S.get("partE0") or {}
    if not E:
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [0, 1, 5, 10, 25, 50]; cks = ["ckpt_ep1000", "ckpt_ep5000", "ckpt_ep10000", "ckpt_ep25000", "ckpt_final"]
    for j, (arm, suf, title) in enumerate((("EDREL1", "1", "edit at the low-value location"),
                                           ("EDREL1H", "1H", "edit at the high-value location"))):
        ax = axs[j]; A = E.get(arm, {})
        for lab, key, c in (("standard learner (Part D)", arm, "k"), ("critic detached (actor losses only, no decay)", f"EDRELCD{suf}", "#d62728"),
                            ("actor detached (critic KL only)", f"EDRELAD{suf}", "#1f77b4")):
            ys = [A.get("amp|EDIT", [np.nan])[0]] + [A.get(f"amp|{key}/{ck}", [np.nan])[0] for ck in cks]
            ax.plot(xs, ys, "o-", color=c, label=lab)
        for key, c in (("FREE", "k"), ("FREECD", "#d62728"), ("FREEAD", "#1f77b4")):
            ys = [1.0] + [A.get(f"amp|{key}/{ck}", [np.nan])[0] for ck in cks]
            ax.plot(xs, ys, ":", color=c, alpha=.7)
        ax.axhline(1, color="grey", lw=.5); ax.set_ylim(0.4, 1.1); ax.set_xlabel("fine-tune episodes (k)")
        ax.set_ylabel("amplitude at the edited slot (warm-up = 1)"); ax.set_title(title, fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, loc="lower right")
    ax = axs[2]; G = ("proj", "qk", "val", "norm1", "lstmR", "lstmW")
    lab = {"proj": "token\nprojection", "qk": "attention\nq/k", "val": "attention\nv", "norm1": "norm1",
           "lstmR": "LSTM input\nweights", "lstmW": "LSTM recurrent\nweights"}
    for i, g in enumerate(G):
        for j, (arm, c) in enumerate((("EDREL1", "#ff7f0e"), ("EDREL1H", "#2ca02c"))):
            for k_, (kind, mk) in enumerate((("revert", "o"), ("graft", "s"))):
                v = E.get(arm, {}).get(f"{kind}|{g}")
                if v:
                    ax.errorbar(i + (j * 2 + k_ - 1.5) * .12, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, color=c, capsize=2,
                                label=f"{arm}: {kind}" if i == 0 else None)
    ax.axhline(0, color="k", lw=.5); ax.axhline(1, color="k", lw=.5, ls=":")
    ax.set_xticks(range(len(G))); ax.set_xticklabels([lab[g] for g in G], fontsize=7)
    ax.set_ylabel("share of the compensation (net of FREE)"); ax.legend(fontsize=7); ax.set_title("where the compensation sits", fontsize=10)
    fig.suptitle("Part E: what erodes the hand-built route (mean of 10 runs per placement; dotted = unedited controls)", fontsize=11)
    fig.tight_layout(); save(fig, out, "fig_partE")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    S = json.load(open(a.stats)); os.makedirs(a.out_dir, exist_ok=True)
    for f in (fig_partC, fig_partA, fig_locus, fig_partD, fig_partE):
        f(S, a.out_dir)
    print(f"[ep_figs] -> {a.out_dir}")


if __name__ == "__main__":
    main()
