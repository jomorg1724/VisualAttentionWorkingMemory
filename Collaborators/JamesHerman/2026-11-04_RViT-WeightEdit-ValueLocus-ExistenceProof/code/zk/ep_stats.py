#!/usr/bin/env python3
"""
Existence-proof statistics (documents/plans/RVIT_EXISTENCE_PROOF_PLAN.md rev 2 §4, §7; added 2026-11-04; local only).

Input: a local copy of CRC runs/existence/ (per run: config.json, roc_ckpt_*.json, lmro_final.json, hyb_W.json,
hyb_start.json, geom_W.json, geom_start.json, tpl_W.json, lm_perturb.json, training_log.csv) and partC/edit_*.json.
Unit: the counterphased pair (loc0_sN, loc3_sN averaged; named after the START model, the table's high location is read
from config.json). Roles high / low refer to the ARM's table.

    python -m zk.ep_stats --root analyses/2026-11-04_rvit-existence-proof/data/runs --out .../stats/ep_stats.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats as st

THETA = 18.0
TIER_MIN = 0.077                      # which-loss Q2 minimum effect (half the positive control's neural loss)
dn_ = lambda a: float(np.sqrt(2) * st.norm.ppf(np.clip(a, 1e-4, 1 - 1e-4)))
REF_OF = {"FREE": "W", "FROZ": "W", "FREEREV": "WREV", "FROZREV": "WREV", "FROZNULL": "WNULL",
          "REL": "W", "RELK": "W", "CONT": "W", "RELREV": "WREV", "RELKREV": "WREV", "CONTREV": "WREV",
          "EDIT1": "W", "EDIT2": "W", "EDIT1H": "W", "EDREL1": "W", "EDREL2": "W", "EDREL1H": "W",
          "EDRELCD1": "W", "EDRELCD1H": "W", "FREECD": "W", "EDRELAD1": "W", "EDRELAD1H": "W", "FREEAD": "W"}
START_OF = {"REL": "FROZ", "RELK": "FROZ", "CONT": "FROZ", "RELREV": "FROZREV", "RELKREV": "FROZREV",
            "CONTREV": "FROZREV", "EDREL1": "EDIT1", "EDREL2": "EDIT2", "EDREL1H": "EDIT1H",
            "EDRELCD1": "EDIT1", "EDRELCD1H": "EDIT1H", "EDRELAD1": "EDIT1", "EDRELAD1H": "EDIT1H"}


def ci(x, level=0.95):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None
    m = float(x.mean())
    if len(x) < 2:
        return [m, np.nan, np.nan]
    h = st.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [m, float(m - h), float(m + h)]


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    d = float(x.mean() - y.mean()); h = float(st.t.ppf(0.975, df) * np.sqrt(vx + vy))
    return {"diff": d, "lo": d - h, "hi": d + h, "p": float(st.ttest_ind(x, y, equal_var=False).pvalue)}


def signflip(x, n=20000, seed=1):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    s = rng.choice((-1, 1), size=(n, len(x))) * x
    return float((np.abs(s.mean(1)) >= abs(x.mean()) - 1e-12).mean())


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def roc_emp(path, theta=THETA):
    d = jl(path)
    if d is None:
        return None
    return {e["location"]: e["empirical"] for e in d["results"] if abs(e["evaluation_theta"] - theta) < 1e-6}


def per_run(run, th, ck="final"):
    """Everything for one run; roles by the table's high location th."""
    lo = 3 - th; rec = {}
    e = roc_emp(os.path.join(run, f"roc_ckpt_{ck}.json"))
    if e is None:
        return None
    for role, L in (("high", th), ("low", lo)):
        for k, kk in (("dprime", "d"), ("criterion", "c"), ("hit_rate", "HR"), ("false_alarm_rate", "FA")):
            rec[f"{kk}_{role}"] = e[L][k]
    rec["learned"] = all(e[L]["hit_rate"] >= 0.10 and e[L]["false_alarm_rate"] <= 0.90 and np.isfinite(e[L]["dprime"])
                         for L in (th, lo))
    if ck != "final":
        return rec
    m = jl(os.path.join(run, "lmro_final.json"))
    if m is not None:
        mm = list(m["models"].values())[0]
        for fr in ("t3", "t4"):
            f = mm[f"{fr}_noiseoff"]
            for role in ("high", "low"):
                rec[f"{fr}_auc_{role}"] = f[role]["auc_H_g"]; rec[f"{fr}_dn_{role}"] = dn_(f[role]["auc_H_g"])
                for k in ("pooled_dn", "pooled_mid", "readout_slope", "readout_intercept", "dv_ch", "dv_noch"):
                    rec[f"{fr}_{k}_{role}"] = f[role][k]
        if "attention" in mm:
            A = mm["attention"]; keys = [f"ch0_testedhigh{t}" for t in (0, 1)]
            mass = sum(np.array(A[k]["mass"]) * A[k]["n"] for k in keys) / sum(A[k]["n"] for k in keys)
            rec["alpha_hi_minus_lo"] = float(np.mean(mass[3:5, th] - mass[3:5, lo]))
    for tag in ("W", "start"):
        g = jl(os.path.join(run, f"geom_{tag}.json"))
        if g is not None:
            for k, v in g["compare"].items():
                for kk, vv in v.items():
                    rec[f"geom{tag}_{k}_{kk}"] = vv
    p = jl(os.path.join(run, "lm_perturb.json"))
    if p is not None:
        rows = {(r["condition"], r["location"]): r for r in p["rows"] if abs(r["theta"] - THETA) < 1e-6}
        for role, L in (("high", th), ("low", lo)):
            if ("none", L) in rows and ("test_gain0", L) in rows:
                rec[f"clampcost_{role}"] = rows[("none", L)]["dprime"] - rows[("test_gain0", L)]["dprime"]
    lg = os.path.join(run, "training_log.csv")
    if os.path.exists(lg):
        rows = list(csv.DictReader(open(lg)))
        rec["resume_count"] = int(rows[-1]["resume"]) if rows else None
        rec["theta_final"] = float(rows[-1]["theta"]) if rows else None
    return rec


def hybrid(run, th, ref_run, tag):
    """2x2 hybrid DiD d' cells for one run; None if missing."""
    h = jl(os.path.join(run, f"hyb_{tag}.json"))
    EE, RR = roc_emp(os.path.join(run, "roc_ckpt_final.json")), roc_emp(os.path.join(ref_run, "roc_ckpt_final.json"))
    if h is None or EE is None or RR is None:
        return None
    emp = lambda payload: {e["location"]: e["empirical"] for e in payload["results"] if abs(e["evaluation_theta"] - THETA) < 1e-6}
    RE, ER = emp(h["REFactor_ENDtr"]), emp(h["ENDactor_REFtr"])
    did = lambda c: c[th]["dprime"] - c[3 - th]["dprime"]
    out = {"EE": did(EE), "RR": did(RR), "RE": did(RE), "ER": did(ER)}
    for nm, c in (("EE", EE), ("RR", RR), ("RE", RE), ("ER", ER)):
        out[f"{nm}_d_high"], out[f"{nm}_d_low"] = c[th]["dprime"], c[3 - th]["dprime"]
    out["rep"] = 0.5 * ((out["RE"] - out["RR"]) + (out["EE"] - out["ER"]))
    out["ro"] = 0.5 * ((out["ER"] - out["RR"]) + (out["EE"] - out["RE"]))
    out["inter"] = out["EE"] - out["RE"] - out["ER"] + out["RR"]
    out["rep_givenRefActor"], out["rep_givenEndActor"] = out["RE"] - out["RR"], out["EE"] - out["ER"]
    out["ro_givenRefTr"], out["ro_givenEndTr"] = out["ER"] - out["RR"], out["EE"] - out["RE"]
    return out


def runs_of(root, arm):
    out = {}
    for r in glob.glob(os.path.join(root, arm, "loc*_s*")):
        m = re.match(r"loc(\d)_s(\d+)$", os.path.basename(r))
        if m and os.path.exists(os.path.join(r, "config.json")):
            out[(int(m.group(2)), int(m.group(1)))] = r
    return out


def table_high(run):
    return int(json.load(open(os.path.join(run, "config.json")))["high_loc"])


def pairs(root, arm, ck="final"):
    rr = runs_of(root, arm); rows = []
    for s in sorted({s for s, _ in rr}):
        recs = []
        for L in (0, 3):
            if (s, L) not in rr:
                break
            run = rr[(s, L)]; th = table_high(run); r = per_run(run, th, ck)
            if r is None:
                break
            if ck == "final":
                name = os.path.basename(run)
                if arm in REF_OF:
                    h = hybrid(run, th, os.path.join(root, REF_OF[arm], name), "W")
                    if h:
                        r.update({f"hybW_{k}": v for k, v in h.items()})
                if arm in START_OF:
                    h = hybrid(run, th, os.path.join(root, START_OF[arm], name), "start")
                    if h:
                        r.update({f"hybS_{k}": v for k, v in h.items()})
            recs.append(r)
        if len(recs) != 2:
            continue
        row = {"seed": s, "learned_both": bool(recs[0]["learned"] and recs[1]["learned"])}
        for k in recs[0]:
            if k == "learned" or k not in recs[1]:
                continue
            a, b = recs[0][k], recs[1][k]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a is not None and b is not None:
                row[k] = 0.5 * (a + b)
        row["DiD_d"] = row["d_high"] - row["d_low"]; row["DiD_c"] = row["c_high"] - row["c_low"]
        row["loss_beh"] = 1 - row["d_low"] / row["d_high"]
        if ck == "final" and all("t3_dn_low" in r for r in recs):
            for fr in ("t3", "t4"):
                row[f"{fr}_lossN"] = 1 - np.mean([r[f"{fr}_dn_low"] / r[f"{fr}_dn_high"] for r in recs])
                row[f"{fr}_lossP"] = 1 - np.mean([r[f"{fr}_pooled_dn_low"] / r[f"{fr}_pooled_dn_high"] for r in recs])
            if all("geomW_t3_low_auc_whole_E" in r for r in recs):
                row["t3_lossWhole"] = 1 - np.mean([dn_(r["geomW_t3_low_auc_whole_E"]) / dn_(r["geomW_t3_high_auc_whole_E"])
                                                   for r in recs])
                for tag in ("W", "start"):
                    if not all(f"geom{tag}_t3_low_d_rep" in r for r in recs):
                        continue
                    g = lambda r, role, k: r[f"geom{tag}_t3_{role}_{k}"]
                    rep = np.mean([g(r, "high", "d_rep") - g(r, "low", "d_rep") for r in recs])
                    ro = np.mean([g(r, "high", "d_ro") - g(r, "low", "d_ro") for r in recs])
                    row[f"J{tag}_rep"], row[f"J{tag}_ro"] = float(rep), float(ro)
                    for k in ("cos_slot", "amp_slot", "cos_whole", "amp_whole", "proj_slot_E", "proj_slot_R",
                              "sd_slot_E", "sd_slot_R", "reader_R_on_E", "lin_R", "lin_E", "gap_E", "gap_R"):
                        for role in ("high", "low"):
                            row[f"J{tag}_{k}_{role}"] = float(np.mean([g(r, role, k) for r in recs]))
        rows.append(row)
    return rows


def summarize(rows):
    if not rows:
        return None
    keys = sorted({k for r in rows for k in r if k not in ("seed", "learned_both")})
    S = {"n_pairs": len(rows), "n_pairs_learned": int(sum(r["learned_both"] for r in rows))}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if v and all(isinstance(x, (int, float)) for x in v):
            S[k] = ci(v)
    S["DiD_c_90"] = ci([r["DiD_c"] for r in rows], 0.90)
    if all("hybW_rep" in r for r in rows):
        rep, ro = np.mean([r["hybW_rep"] for r in rows]), np.mean([r["hybW_ro"] for r in rows])
        S["hybW_rep_share"] = boot_share([r["hybW_rep"] for r in rows], [r["hybW_ro"] for r in rows])
    if all("JW_rep" in r for r in rows):
        S["JW_rep_share"] = boot_share([r["JW_rep"] for r in rows], [r["JW_ro"] for r in rows])
    if all("Jstart_rep" in r for r in rows):
        S["Jstart_rep_share"] = boot_share([r["Jstart_rep"] for r in rows], [r["Jstart_ro"] for r in rows])
    if all("hybS_rep" in r for r in rows):
        S["hybS_rep_share"] = boot_share([r["hybS_rep"] for r in rows], [r["hybS_ro"] for r in rows])
    S["signflip_DiD_d"] = signflip([r["DiD_d"] for r in rows])
    return S


def boot_share(rep, ro, n=20000, seed=20261104):
    rep, ro = np.asarray(rep, float), np.asarray(ro, float)
    rng = np.random.default_rng(seed); i = rng.integers(0, len(rep), (n, len(rep)))
    br, bo = rep[i].mean(1), ro[i].mean(1)
    sh = br / (br + bo)
    return {"share": float(rep.mean() / (rep.mean() + ro.mean())), "ci90": [float(np.quantile(sh, .05)),
            float(np.quantile(sh, .95))], "rep": float(rep.mean()), "ro": float(ro.mean())}


def tier(S, null=None):
    """Tier of value-in-representation for an arm summary (plan §4), t3 primary."""
    if S is None or "t3_lossN" not in S:
        return None
    eff = S["DiD_d"][1] > 0 and S["DiD_d"][0] >= 0.3
    out = {"value_effect": bool(eff), "lossN": S["t3_lossN"], "lossP": S["t3_lossP"], "loss_beh": S["loss_beh"]}
    if "JW_cos_slot_low" in S:
        out["cos_slot_low"] = S["JW_cos_slot_low"]
    if not eff:
        out["tier"] = "no value effect"; return out
    if S["t3_lossN"][1] > TIER_MIN:
        out["tier"] = "I (information)"
    elif S["t3_lossP"][1] > TIER_MIN:
        c = S.get("JW_cos_slot_low", [np.nan])[0]
        out["tier"] = "II-gain (amplitude)" if c >= 0.9 else "II-rot (rotation/routing)"
    else:
        out["tier"] = "III (offset/other)"
    return out


def _amp_loss(row, role):
    """1 - (|dmu| at the edited slot, end/W) / (the same at the other slot): amplitude loss on a fixed axis (rev 3)."""
    other = "high" if role == "low" else "low"
    a, b = row.get(f"JW_amp_slot_{role}"), row.get(f"JW_amp_slot_{other}")
    return None if a is None or b is None else 1 - a / b


def retention(R, n=20000, seed=20261105):
    """Part D (plan rev 3 §11): r = (m_EDREL,end - m_FREE,end) / (m_EDIT - m_W), paired by seed; m = amplitude loss
    (primary), lossN, lossP (secondary). m_W = 0 for the amplitude measure by construction; W's own value for the others."""
    out = {}
    for start, end, role, free in (("EDIT1", "EDREL1", "low", "FREE"), ("EDIT2", "EDREL2", "low", "FREE"),
                                   ("EDIT1H", "EDREL1H", "high", "FREE"),
                                   ("EDIT1", "EDRELCD1", "low", "FREECD"), ("EDIT1H", "EDRELCD1H", "high", "FREECD"),
                                   ("EDIT1", "EDRELAD1", "low", "FREEAD"), ("EDIT1H", "EDRELAD1H", "high", "FREEAD")):
        if start not in R or end not in R or free not in R:
            continue
        S, E, F = ({r["seed"]: r for r in R[a]} for a in (start, end, free))
        Wr = {r["seed"]: r for r in R.get("W", [])}
        seeds = sorted(set(S) & set(E) & set(F))
        res = {"n_pairs": len(seeds), "edited_role": role}
        lossN = lambda r: (1 - r["t3_dn_low"] / r["t3_dn_high"]) if role == "low" else (1 - r["t3_dn_high"] / r["t3_dn_low"])
        lossP = lambda r: (1 - r["t3_pooled_dn_low"] / r["t3_pooled_dn_high"]) if role == "low" else             (1 - r["t3_pooled_dn_high"] / r["t3_pooled_dn_low"])
        for name, f, wbase in (("amp", lambda r: _amp_loss(r, role), lambda s: 0.0),
                               ("lossN", lossN, lambda s: lossN(Wr[s]) if s in Wr else 0.0),
                               ("lossP", lossP, lambda s: lossP(Wr[s]) if s in Wr else 0.0)):
            try:
                num = np.array([f(E[s]) - f(F[s]) for s in seeds], float)
                den = np.array([f(S[s]) - wbase(s) for s in seeds], float)
            except (KeyError, TypeError):
                continue
            rng = np.random.default_rng(seed); i = rng.integers(0, len(seeds), (n, len(seeds)))
            br = num[i].mean(1) / den[i].mean(1)
            r = float(num.mean() / den.mean())
            verdict = ("kept" if r >= 0.7 and np.quantile(br, .05) >= 0.5 else
                       "abandoned" if r <= 0.3 and np.quantile(br, .95) <= 0.5 else "partial")
            res[name] = {"r": r, "ci90": [float(np.quantile(br, .05)), float(np.quantile(br, .95))],
                         "start_minus_W": ci(den), "end_minus_FREE": ci(num), "verdict": verdict}
        # joint rule: does the end-state transformer still carry the value effect under the W actor?
        res["hyb_RE_minus_RR"] = ci([E[s]["hybW_RE"] - E[s]["hybW_RR"] for s in seeds if "hybW_RE" in E[s]])
        res["hyb_rep_share"] = boot_share([E[s]["hybW_rep"] for s in seeds if "hybW_rep" in E[s]],
                                          [E[s]["hybW_ro"] for s in seeds if "hybW_ro" in E[s]])             if all("hybW_rep" in E[s] for s in seeds) and seeds else None
        if end != "EDREL1" and end != "EDREL1H" and end != "EDREL2":
            base = {"EDIT1": "EDREL1", "EDIT1H": "EDREL1H"}[start]
            if base in R:
                B_ = {r["seed"]: r for r in R[base]}; Fb = {r["seed"]: r for r in R["FREE"]}
                ss = [x for x in seeds if x in B_ and x in Fb]
                try:
                    d = [(_amp_loss(E[x], role) - _amp_loss(F[x], role)) - (_amp_loss(B_[x], role) - _amp_loss(Fb[x], role))
                         for x in ss]
                    den_ = np.mean([_amp_loss(S[x], role) for x in ss])
                    res["vs_partD_same_seeds"] = {"n": len(ss), "delta_retained_amp_loss": ci(d),
                                                  "delta_r": ci([v / den_ for v in d])}
                except (KeyError, TypeError):
                    pass
        out[end] = res
    return out


def partE0(root):
    """Part E0 (plan rev 4 §12): per placement, edit bookkeeping, amplitude courses (incl. E1 arms), realised gate offset,
    and the revert / graft localization (net of the same operation in FREE), from partE0/e0_*.json."""
    files = sorted(glob.glob(os.path.join(root, "partE0", "e0_*.json")))
    G = ("proj", "qk", "val", "norm1", "lstmR", "lstmW", "all")
    acc = {}
    for f in files:
        d = json.load(open(f)); arm = d["arm"]; A = acc.setdefault(arm, {})
        if "book" in d:
            A.setdefault("edit_coord_end", []).append(d["book"]["edit_coord"][arm]["ckpt_final"])
            A.setdefault("decay_pred_retained", []).append(d["book"]["decay_pred_retained"])
            for x, c in d["book"]["edit_coord"].items():
                if "ckpt_final" in c:
                    A.setdefault(f"edit_coord_end|{x}", []).append(c["ckpt_final"])
        if "course" in d:
            for k, v in d["course"]["amp_rel"].items():
                A.setdefault(f"amp|{k.replace('EDIT1H', 'EDIT').replace('EDIT1', 'EDIT') if '/' not in k else k}", []).append(v)
        if "gate" in d:
            g = d["gate"]
            for tag in g:
                if isinstance(g[tag], dict):
                    A.setdefault(f"offset_ratio|{tag}", []).append(g[tag]["offset_per_beta"] / g["EDIT"]["offset_per_beta"])
        if "local" in d:
            lo = d["local"]; e, end = lo["EDIT"]["logA_rel"], lo["EDREL_end"]["logA_rel"]; comp = end - e
            for g_ in G:
                rv = lo[f"revert_{g_}"]["logA_rel"] - lo[f"FREE_revert_{g_}"]["logA_rel"] + lo["FREE_end"]["logA_rel"]
                gr = lo[f"graft_{g_}"]["logA_rel"] - lo[f"FREE_graft_{g_}"]["logA_rel"]
                A.setdefault(f"revert|{g_}", []).append((end - rv) / comp)
                A.setdefault(f"graft|{g_}", []).append((gr - e) / comp)
    return {arm: {k: ci(v) + [len(v)] for k, v in A.items()} for arm, A in acc.items()}


def partC(root):
    files = sorted(glob.glob(os.path.join(root, "partC2", "edit_loc*_s*.json")))
    cells = {}
    for f in files:
        d = json.load(open(f))
        for tag, c in d["cells"].items():
            a, p = c["assay"], c["probe"]["t3"]
            r = {"d_high": a["high"]["dprime"], "d_low": a["low"]["dprime"], "c_low": a["low"]["criterion"],
                 "c_high": a["high"]["criterion"], "HR_low": a["low"]["hit_rate"], "FA_low": a["low"]["false_alarm_rate"],
                 "auc_low": p["low"]["auc_H_g"], "auc_high": p["high"]["auc_H_g"],
                 "auc_whole_low": p["low"]["auc_whole"], "auc_whole_high": p["high"]["auc_whole"],
                 "pooled_dn_low": p["low"]["pooled_dn"], "pooled_dn_high": p["high"]["pooled_dn"],
                 "slope_low": p["low"]["readout_slope"], "slope_high": p["high"]["readout_slope"]}
            r["loss_beh"] = 1 - r["d_low"] / r["d_high"]
            r["lossN"] = 1 - dn_(r["auc_low"]) / dn_(r["auc_high"])
            r["lossWhole"] = 1 - dn_(r["auc_whole_low"]) / dn_(r["auc_whole_high"])
            r["lossP"] = 1 - r["pooled_dn_low"] / r["pooled_dn_high"]
            cells.setdefault(tag, []).append(r)
    out = {}
    for tag, rs in cells.items():
        out[tag] = {"n": len(rs)} | {k: ci([r[k] for r in rs]) for k in rs[0]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"arms": {}, "rows": {}}
    for arm in ("W", "WREV", "WNULL", "FREE", "FREEREV", "FROZ", "FROZREV", "FROZNULL",
                "REL", "RELK", "CONT", "RELREV", "RELKREV", "CONTREV", "EDIT1", "EDIT2", "EDIT1H", "EDREL1", "EDREL2", "EDREL1H",
                "EDRELCD1", "EDRELCD1H", "FREECD", "EDRELAD1", "EDRELAD1H", "FREEAD"):
        rows = pairs(a.root, arm)
        if rows:
            res["rows"][arm] = rows; res["arms"][arm] = summarize(rows)
            res["arms"][arm]["tier"] = tier(res["arms"][arm])
        for ck in ("ep1000", "ep5000", "ep10000", "ep25000"):
            rr = pairs(a.root, arm, ck)
            if rr:
                res["arms"].setdefault(arm, {})[f"snap_{ck}"] = {k: ci([r[k] for r in rr]) for k in
                                                                ("DiD_d", "DiD_c", "d_high", "d_low")} | {"n": len(rr)}
    A = res["arms"]; R = res["rows"]
    cmp = {}
    for x, y in (("FROZ", "FROZNULL"), ("FROZREV", "FROZNULL"), ("FROZ", "FREE"), ("FREEREV", "FREE"),
                 ("FROZREV", "FROZ"), ("REL", "CONT"), ("RELK", "REL"), ("RELREV", "CONTREV"), ("RELKREV", "RELREV"),
                 ("EDREL1", "FREE"), ("EDREL2", "FREE"), ("CONT", "FROZ")):
        if x in R and y in R:
            cmp[f"{x}-{y}"] = {k: welch([r[k] for r in R[x] if k in r], [r[k] for r in R[y] if k in r])
                               for k in ("DiD_d", "DiD_c", "t3_lossN", "t3_lossP", "t3_lossWhole", "loss_beh")
                               if all(k in r for r in R[x] + R[y])}
    # change from the warm-up (paired by seed) for every Part-A arm
    for arm, ref in (("FREE", "W"), ("FROZ", "W"), ("FREEREV", "WREV"), ("FROZREV", "WREV"), ("FROZNULL", "WNULL")):
        if arm in R and ref in R:
            rb = {r["seed"]: r for r in R[ref]}
            d = {k: ci([r[k] - rb[r["seed"]][k] for r in R[arm] if r["seed"] in rb and k in r and k in rb[r["seed"]]])
                 for k in ("DiD_d", "DiD_c", "d_high", "d_low", "t3_lossN", "t3_lossP", "t3_readout_slope_low",
                           "t3_readout_slope_high", "alpha_hi_minus_lo")}
            d["DiD_c_90"] = ci([r["DiD_c"] - rb[r["seed"]]["DiD_c"] for r in R[arm] if r["seed"] in rb], 0.90)
            A[arm]["from_warmup"] = d
    # reader-asymmetry-free value effect (plan rev 2 §2): per seed, the mean of the congruent and reversed arms' changes
    # from their warm-ups; the half-difference is the reader-asymmetry term
    cr = {}
    for x, xr, w, wr in (("FROZ", "FROZREV", "W", "WREV"), ("FREE", "FREEREV", "W", "WREV")):
        if all(a in R for a in (x, xr, w, wr)):
            Rx, Rxr, Rw, Rwr = ({r["seed"]: r for r in R[a]} for a in (x, xr, w, wr))
            seeds = sorted(set(Rx) & set(Rxr) & set(Rw) & set(Rwr))
            blk = {"n_pairs": len(seeds)}
            for k in ("DiD_d", "DiD_c", "t3_lossN", "t3_lossP", "t3_lossWhole", "loss_beh"):
                try:
                    dx = np.array([Rx[s][k] - Rw[s][k] for s in seeds]); dr = np.array([Rxr[s][k] - Rwr[s][k] for s in seeds])
                except KeyError:
                    continue
                blk[k] = {"value_term": ci((dx + dr) / 2), "reader_term": ci((dr - dx) / 2),
                          "signflip_p_value_term": signflip((dx + dr) / 2)}
            cr[x] = blk
    res["congruent_reversed"] = cr
    res["compare"] = cmp
    res["retention"] = retention(R)
    res["partC"] = partC(a.root)
    res["partE0"] = partE0(a.root)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    g = lambda v: "—" if v is None else f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    for arm, S in A.items():
        if "DiD_d" not in S:
            continue
        print(f"== {arm} ({S['n_pairs']} pairs, learned {S['n_pairs_learned']}): DiD d' {g(S['DiD_d'])}  DiD c {g(S['DiD_c'])}"
              f"  d' hi/lo {S['d_high'][0]:.2f}/{S['d_low'][0]:.2f}")
        for k in ("t3_lossN", "t3_lossP", "t3_lossWhole", "loss_beh", "t3_readout_slope_high", "t3_readout_slope_low",
                  "alpha_hi_minus_lo", "hybW_rep", "hybW_ro", "hybW_inter", "hybW_rep_givenRefActor",
                  "hybW_rep_givenEndActor", "hybW_ro_givenRefTr", "hybW_ro_givenEndTr", "hybS_rep", "hybS_ro", "hybS_inter",
                  "hybS_rep_givenRefActor", "hybS_rep_givenEndActor", "JW_rep", "JW_ro", "JW_cos_slot_low",
                  "JW_cos_slot_high", "JW_amp_slot_low", "JW_amp_slot_high", "clampcost_high", "clampcost_low"):
            if k in S:
                print(f"   {k:24s} {g(S[k])}")
        for k in ("hybW_rep_share", "JW_rep_share", "hybS_rep_share", "Jstart_rep_share"):
            if k in S:
                print(f"   {k:24s} {S[k]}")
        if S.get("tier"):
            print("   tier:", S["tier"]["tier"])
    for x, blk in res["congruent_reversed"].items():
        print(f"congruent+reversed {x} ({blk['n_pairs']} pairs):", {k: f"value {g(v['value_term'])} reader {g(v['reader_term'])}"
                                                                    for k, v in blk.items() if k != "n_pairs"})
    for k, v in cmp.items():
        print(k, {kk: f"{vv['diff']:+.3f} [{vv['lo']:+.3f}, {vv['hi']:+.3f}]" for kk, vv in v.items()})
    for tag, c in sorted(res["partC"].items()):
        print("partC", tag, {k: round(c[k][0], 3) for k in ("d_high", "d_low", "c_low", "lossN", "lossWhole", "lossP",
                                                             "loss_beh")})
    print(f"[ep_stats] -> {a.out}")


if __name__ == "__main__":
    main()
