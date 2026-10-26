"""Stages 1-3 of the structure-discovery study.

1. Nuisance discovery (no labels): coherence |E z_i conj z_j| between complex field channels (cell, time).
   Channels that stay coherent share one unknown rotation (a U(1) nuisance); degree-1 moments E z
   vanish when that rotation is random. The coherent unit found here becomes the unit of
   canonicalization: w = z * conj(R)/|R|, with R the causal running sum of that unit's own past.
2. Plateau oracle (labels, train split only): label correlation of feature families in raw vs
   canonical coordinates, ungated vs gated, and the samples needed to detect each (25/rho^2).
3. Attention-gate discovery: every per-frame appearance map is a candidate gate. A candidate breaks
   the exchangeability of locations if pooling canonical motion weighted by it predicts the label.
   Each candidate is scored by a closed-form probe (shrinkage LDA, 5-fold CV on n_probe trials).

usage: python discover.py [TASK_DIR [FIELD]]   FIELD = flow (default) | orient
Reads TASK_DIR/data_{train,val}.npz; writes TASK_DIR/discovery_FIELD.json and
TASK_DIR/features_FIELD_{train,val}.npz (the original Krauzlis run used '.', flow, unsuffixed names).

October 25 generalisation (no change on Krauzlis, where every movie covers the windows used):
statistics are computed only over trials valid at the frames involved, anchored gate candidates
fall back to a uniform gate where the anchored frame does not exist, and labels may be multiclass
(balanced accuracy = mean per-class recall; oracle correlation = best one-vs-rest).

October 25b additions: 1b cross-location references (unlabeled covariance of each cell with every cell's
causal history -> regression-weighted transport; Wx with, Wxo without the self pair); 3' gate candidates
displaced by up to +/-3 cells; 3b a discrete sign nuisance (unlabeled discreteness screen) acting by
negation or conjugation, selected jointly with the 20 best gates by the same probe; per-cell probes."""
import json, os, sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold


def bal_acc(y, p):
    return float(np.mean([(p[y == k] == k).mean() for k in np.unique(y)]))


def cv_ba(F, y, k=5, seed=0):
    out = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(F, y):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit((F[tr] - mu) / sd, y[tr])
        out.append(bal_acc(y[te], m.predict((F[te] - mu) / sd)))
    return float(np.mean(out))


def corr(f, y):
    if len(np.unique(y)) > 2:
        return max((corr(f, (y == k).astype(float)) for k in np.unique(y)), key=abs)
    f = f - f.mean()
    yc = (y - y.mean()) / y.std()
    return float((f * yc).mean() / (f.std() + 1e-12))


def canonical(Z, M):
    """w_{c,t} = z_{c,t} conj(R_{c,t})/|R_{c,t}|, R = causal sum of earlier valid z in the same cell."""
    valid = M[:, 1:, None]
    Zv = np.where(valid, Z, 0)
    R = np.cumsum(Zv, 1) - Zv
    return np.where(valid, Z * np.conj(R) / (np.abs(R) + 1e-6), 0).astype(np.complex64)


def gate_maps(A, M):
    """z-score each frame's appearance map across cells, keep the positive part, normalise to sum 1."""
    mu = A.mean(-1, keepdims=True)
    sd = A.std(-1, keepdims=True) + 1e-9
    G = np.maximum((A - mu) / sd, 0) * M[..., None]
    return (G / (G.sum(-1, keepdims=True) + 1e-9)).astype(np.float32)


def components(adj):
    comp = -np.ones(len(adj), int)
    k = 0
    for i in range(len(adj)):
        if comp[i] < 0:
            stack = [i]
            comp[i] = k
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u])[0]:
                    if comp[v] < 0:
                        comp[v] = k
                        stack.append(v)
            k += 1
    return comp, k


def history(Z, M):
    """Causal running sum of earlier valid field values, per cell."""
    Zv = np.where(M[:, 1:, None], Z, 0)
    return np.cumsum(Zv, 1) - Zv


def transported(Z, P, M, beta):
    """w_i = z_i conj(R_i)/|R_i| with R_i = sum_j beta_ij P_j (cross-location reference)."""
    R = np.einsum('ntj,ij->nti', P, beta)
    return np.where(M[:, 1:, None], Z * np.conj(R) / (np.abs(R) + 1e-6), 0).astype(np.complex64)


def z2_act(Wc, s, action):
    """Z2 group action of a discovered sign s (+1/-1 per trial) on a complex field."""
    s = s[:, None, None]
    if action == 'negate':
        return (Wc * s).astype(np.complex64)
    return np.where(s > 0, Wc, np.conj(Wc)).astype(np.complex64)


def shift_map(g, dx, dy, G9=9):
    """Translate gate maps (n, 81) by (dx, dy) cells with zero fill; renormalise (uniform if empty)."""
    m = g.reshape(-1, G9, G9)
    out = np.zeros_like(m)
    ys, yd = (slice(0, G9 - dy), slice(dy, G9)) if dy >= 0 else (slice(-dy, G9), slice(0, G9 + dy))
    xs, xd = (slice(0, G9 - dx), slice(dx, G9)) if dx >= 0 else (slice(-dx, G9), slice(0, G9 + dx))
    out[:, yd, xd] = m[:, ys, xs]
    out = out.reshape(len(g), -1)
    tot = out.sum(1, keepdims=True)
    return np.where(tot > 0, out / np.maximum(tot, 1e-12), 1 / out.shape[1]).astype(np.float32)


def otsu(q):
    """Best two-cluster split of a scalar: (threshold, between-class variance / total variance)."""
    best = (np.median(q), 0.0)
    tot = q.var() + 1e-12
    for th in np.quantile(q, np.linspace(0.05, 0.95, 91)):
        lo, hi = q[q <= th], q[q > th]
        if len(lo) and len(hi):
            r = len(lo) * len(hi) / len(q) ** 2 * (lo.mean() - hi.mean()) ** 2 / tot
            if r > best[1]:
                best = (float(th), float(r))
    return best


def coh(a, b, v):
    """Normalised coherence |E a conj b| / sqrt(E|a|^2 E|b|^2) over trials v (nan if too few)."""
    if v.sum() < 20:
        return np.nan
    a, b = a[v], b[v]
    return float(np.abs(np.mean(a * np.conj(b))) / (np.sqrt(np.mean(np.abs(a) ** 2) * np.mean(np.abs(b) ** 2)) + 1e-12))


if __name__ == '__main__':
    tdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    field = sys.argv[2] if len(sys.argv) > 2 else 'flow'
    sfx = '' if tdir == '.' and field == 'flow' else f'_{field}'
    tr = np.load(os.path.join(tdir, 'data_train.npz'))
    va = np.load(os.path.join(tdir, 'data_val.npz'))
    pick = (lambda d: d['Z']) if field == 'flow' else (lambda d: d['O'][:, 1:])
    Z, A, M, y = pick(tr), tr['A'], tr['M'], tr['y'].astype(float)
    Zva = pick(va)
    n, T1, C = Z.shape
    V = M[:, 1:]                                  # validity of each field step
    rep = dict(field=field, n_train=int(n), steps=int(T1), classes=int(len(np.unique(y))))
    # ---------- 1. nuisance discovery (labels unused) ----------
    act = (np.abs(Z) * V[..., None]).sum((0, 1)) / V.sum()
    cells = np.argsort(-act)[:12]
    L = min(20, T1)
    tv = slice(T1 - L, T1)                        # last L steps (valid in the longest movies)
    Zl, Vl = Z[:, tv][:, :, cells], V[:, tv]
    P = np.array([[np.mean(np.abs(Zl[Vl[:, t], t, j]) ** 2) if Vl[:, t].any() else np.nan for j in range(len(cells))] for t in range(L)])
    deg1 = np.array([[np.abs(Zl[Vl[:, t], t, j].mean()) / np.sqrt(P[t, j]) if Vl[:, t].sum() >= 20 else np.nan for j in range(len(cells))] for t in range(L)])
    lags = [g for g in (1, 2, 4, 8, 12) if g < L]
    within = [float(np.nanmean([coh(Zl[:, t, j], Zl[:, t + g, j], Vl[:, t] & Vl[:, t + g])
                                for t in range(L - g) for j in range(len(cells))])) for g in lags]
    across = np.array([[np.nanmean([coh(Zl[:, t, i], Zl[:, t, j], Vl[:, t]) for t in range(L)])
                        for j in range(len(cells))] for i in range(len(cells))])
    # diagnostic only (not used for the sector decision): best coherence of a cell with ANOTHER cell at an earlier step
    lagged = np.array([[np.nanmax([coh(Zl[:, t, i], Zl[:, t - g, j], Vl[:, t] & Vl[:, t - g]) for g in range(1, L) for t in range(g, L)])
                        if i != j else np.nan for j in range(len(cells))] for i in range(len(cells))])
    tau = 4 / np.sqrt(n)
    adj = across > tau
    np.fill_diagonal(adj, True)
    comp, nsec = components(adj)
    off = ~np.eye(len(cells), dtype=bool)
    rep['nuisance'] = dict(top_cells=cells.tolist(), threshold=tau, degree1_moment_over_rms_mean=float(np.nanmean(deg1)),
                           within_cell_coherence_by_lag={str(g): v for g, v in zip(lags, within)},
                           across_cell_coherence_offdiag_mean=float(np.nanmean(across[off])),
                           across_cell_coherence_offdiag_max=float(np.nanmax(across[off])),
                           across_cell_coherence_matrix=np.round(across, 3).tolist(),
                           across_cell_lagged_coherence_max_diagnostic=float(np.nanmax(lagged)),
                           across_cell_lagged_coherence_mean_diagnostic=float(np.nanmean(lagged)),
                           sectors=comp.tolist(), n_sectors=int(nsec),
                           selected_canonicalization='global' if nsec == 1 else 'per_cell')
    W = canonical(Z, M)
    Wv = canonical(Zva, va['M'])
    ph = np.exp(1j * np.random.default_rng(0).uniform(0, 2 * np.pi, (1, 1, C)))
    rep['nuisance']['canonical_invariance_max_abs_diff'] = float(np.abs(canonical(Z[:50] * ph, M[:50]) - W[:50]).max())
    # ---------- 1b. cross-location references (labels unused; October 25b) ----------
    # Covariance of each cell's field with every cell's causal history, pooled over the last L steps.
    # Significant pairs define a transport: R_i = sum_j beta_ij P_j (regression weights, phase included).
    # With only self-pairs significant this is the per-cell canonicalization above; Wxo drops the self-pair.
    Ptr, Pva = history(Z, M), history(Zva, va['M'])
    sel = V[:, tv]
    zt, pt = Z[:, tv][sel], Ptr[:, tv][sel]
    zt, pt = zt - zt.mean(0), pt - pt.mean(0)
    cov = zt.T @ pt.conj() / len(zt)                                  # C x C, E[z_i conj p_j]
    vz, vp = (np.abs(zt) ** 2).mean(0), (np.abs(pt) ** 2).mean(0)
    cohx = np.abs(cov) / (np.sqrt(vz[:, None] * vp[None]) + 1e-12)
    live = (vz > 1e-3 * vz.max())[:, None] & (vp > 1e-3 * vp.max())[None]
    sig = (cohx > tau) & live
    beta = np.where(sig, cov / (vp[None] + 1e-12), 0)
    beta_o = np.where(np.eye(C, dtype=bool), 0, beta)
    Wx, Wxv = transported(Z, Ptr, M, beta), transported(Zva, Pva, va['M'], beta)
    Wxo, Wxov = transported(Z, Ptr, M, beta_o), transported(Zva, Pva, va['M'], beta_o)
    offd = sig & ~np.eye(C, dtype=bool)
    best_src = {int(i): int(np.argmax(np.where(np.eye(C, dtype=bool)[i], 0, cohx[i]))) for i in np.flatnonzero(offd.any(1))}
    rep['cross_location'] = dict(threshold=tau, significant_self_pairs=int(np.diag(sig).sum()),
                                 significant_cross_pairs=int(offd.sum()), cells_with_cross_reference=int(offd.any(1).sum()),
                                 self_coherence_mean_live=float(np.nanmean(np.where(np.diag(live), np.diag(cohx), np.nan))),
                                 cross_coherence_max=float(np.max(np.where(offd, cohx, 0))),
                                 best_cross_source_cell=best_src,
                                 global_phase_invariance_max_abs_diff=float(np.abs(
                                     transported(Z[:50] * ph[:, :, :1], Ptr[:50] * ph[:, :, :1], M[:50], beta) - Wx[:50]).max()))
    # ---------- gate candidates ----------
    # Two generic time anchors: movie onset (start-aligned frames 0..K-1) and the report (end-aligned
    # last K frames), K = min(29, longest movie). Candidates are appearance maps at each anchored frame;
    # a trial whose movie has no such frame gets a uniform gate.
    K = min(29, A.shape[1])

    def anchored(Ax, Mx):
        s = Mx.argmax(1)
        f = s[:, None] + np.arange(K)[None]
        ok = f < Ax.shape[1]
        start = Ax[np.arange(len(Ax))[:, None], np.minimum(f, Ax.shape[1] - 1)]
        end, okend = Ax[:, -K:], Mx[:, -K:]
        return np.concatenate([start, end], 1), np.concatenate([ok, okend], 1)

    def gates(Ax, Mx):
        Aa, Ma = anchored(Ax, Mx)
        G = gate_maps(Aa, Ma)
        G[~Ma] = 1 / C
        return G

    names = [f'start+{k}' for k in range(K)] + [f'end-{K - 1 - k}' for k in range(K)]
    G = gates(A, M)
    Gv = gates(va['A'], va['M'])
    # ---------- 2. plateau oracle ----------
    o = {}
    ReZ = Z.real[:, tv]
    o['raw_deg1_max_single_channel'] = float(np.max(np.abs([[corr(ReZ[:, t, c], y) for c in cells] for t in range(L)])))
    o['raw_deg1_pooled_over_cells'] = float(np.max(np.abs([corr(Z.real[:, t].sum(-1), y) for t in range(T1 - L, T1)])))
    pairs = [(t, s) for t in range(T1 - L, T1) for s in range(t + 1, T1)]
    o['raw_deg2_neutral_pooled_change_signal'] = float(np.nanmax(np.abs([corr(np.real(Z[:, t] * np.conj(Z[:, s])).sum(-1), y) for t, s in pairs])))
    o['canonical_deg1_pooled_ungated'] = float(np.max(np.abs([corr(W.real[:, t].sum(-1), y) for t in range(T1 - L, T1)])))
    gated = np.abs([[corr((G[:, kk] * W.real[:, t]).sum(-1), y) for t in range(T1 - L, T1)] for kk in range(2 * K)])
    o['canonical_deg1_gated_best'] = float(np.nanmax(gated))
    o['canonical_deg1_gated_best_candidate'] = names[int(np.unravel_index(np.nanargmax(gated), gated.shape)[0])]
    o['samples_to_detect_5sd'] = {k2: (25 / v ** 2 if v > 0 else None) for k2, v in o.items() if isinstance(v, float)}
    rep['oracle'] = o
    # ---------- 3. gate discovery by closed-form probes ----------
    n_probe = 400
    idx = np.random.default_rng(1).choice(n, n_probe, replace=False)

    def feats(Wc, g):
        return np.concatenate([(g[:, None, :] * Wc.real[:, tv]).sum(-1), (g[:, None, :] * Wc.imag[:, tv]).sum(-1)], 1)

    uni = np.full((n, C), 1 / C, np.float32)
    scores = {'none': cv_ba(feats(W, uni)[idx], y[idx])}
    for kk in range(2 * K):
        scores[names[kk]] = cv_ba(feats(W, G[:, kk])[idx], y[idx])
    best = max((v, k2) for k2, v in scores.items() if v is not None)
    rep['gate_probe'] = dict(n_probe=n_probe, scores=scores, best=best[1], best_cv_ba=best[0])
    kstar = names.index(best[1]) if best[1] != 'none' else None
    if kstar is not None:
        rep['gate_probe']['mean_gate_map_9x9'] = np.round(G[:, kstar].mean(0).reshape(9, 9), 4).tolist()
    # ---------- 3'. displaced gates (October 25b) ----------
    # A cue may point at a location instead of marking it: every anchored map is also tried translated
    # by up to +/-3 cells (30 px) in x and y, scored by the same probe. Shift (0, 0) is stage 3 above.
    SH = [(dx, dy) for dy in range(-3, 4) for dx in range(-3, 4) if (dx, dy) != (0, 0)]
    shifted = {}
    Wp, Gp = W[idx], G[idx]
    for kk in range(2 * K):
        for dx, dy in SH:
            shifted[(names[kk], dx, dy)] = cv_ba(feats(Wp, shift_map(Gp[:, kk], dx, dy)), y[idx])
    (sk, sdx, sdy), sbest = max(shifted.items(), key=lambda kv: kv[1])
    rep['gate_probe']['displaced_best'] = dict(candidate=sk, dx_cells=sdx, dy_cells=sdy, cv_ba=sbest)
    allg = {(k2, 0, 0): v for k2, v in scores.items()}
    allg.update(shifted)
    ukey, ubest = max(allg.items(), key=lambda kv: kv[1])          # best gate without a sign

    def gate_of(key, Gx):
        if key[0] == 'none':
            return np.full((len(Gx), C), 1 / C, np.float32)
        return shift_map(Gx[:, names.index(key[0])], key[1], key[2])
    # ---------- 3b. discrete sign-nuisance discovery, joint with the gate (October 25b) ----------
    # A sign nuisance hides the gate's value (without it the right location need not score best), so
    # the 20 best gates (shifted or not) are each tried with every discrete sign candidate.
    # Sign candidates: appearance energy at each anchored frame present in every movie, measured under
    # the gate or under its unshifted cue map. Unlabeled step (all training trials): keep candidates that
    # are discrete (two-cluster split explains >= 75% of variance). Probe step (same 400 trials): a
    # discrete s acts on the canonical field by negation or by complex conjugation (reversal of rotation).
    # The best (gate, sign) pair is kept only if it beats the best unsigned gate by >= 0.03 CV BA.
    Aa, Ma = anchored(A, M)
    Aav, _ = anchored(va['A'], va['M'])
    top = sorted(((k2, v) for k2, v in allg.items() if k2[0] != 'none'), key=lambda kv: -kv[1])[:20]
    cands = []
    for key, _ in top:
        g = gate_of(key, G)
        unders = {'gate': g}
        if key[1:] != (0, 0):
            unders['cue'] = G[:, names.index(key[0])]
        for kk in range(2 * K):
            if not Ma[:, kk].all():
                continue
            for u, gu in unders.items():
                q = (Aa[:, kk] * gu).sum(-1)
                th, sep = otsu(q)
                if sep < 0.75:
                    continue
                sg = np.where(q > th, 1.0, -1.0)
                for a in ('negate', 'conjugate'):
                    cands.append(dict(gate=list(key), source=names[kk], under=u, action=a, threshold=th, separation=sep,
                                      balance=float((sg > 0).mean()), cv_ba=cv_ba(feats(z2_act(W[idx], sg[idx], a), g[idx]), y[idx])))
    win = max(cands, key=lambda d: d['cv_ba']) if cands else None
    use = win is not None and win['cv_ba'] >= ubest + 0.03
    rep['sign_nuisance'] = dict(baseline_cv_ba=ubest, n_candidates=len(cands),
                                top_candidates=sorted(cands, key=lambda d: -d['cv_ba'])[:30], selected=win if use else None)
    gkey = tuple(win['gate']) if use else ukey
    rep['gate_probe']['selected'] = dict(candidate=gkey[0], dx_cells=gkey[1], dy_cells=gkey[2],
                                         cv_ba=win['cv_ba'] if use else ubest, with_sign=bool(use))
    kstar = names.index(gkey[0]) if gkey[0] != 'none' else None
    gstar, gstar_v = gate_of(gkey, G), gate_of(gkey, Gv)
    rep['gate_probe']['selected_mean_gate_map_9x9'] = np.round(gstar.mean(0).reshape(9, 9), 4).tolist()
    if use:
        kk = names.index(win['source'])
        gu, gu_v = (gstar, gstar_v) if win['under'] == 'gate' else (G[:, kstar], Gv[:, kstar])
        s_tr = np.where((Aa[:, kk] * gu).sum(-1) > win['threshold'], 1.0, -1.0).astype(np.float32)
        s_va = np.where((Aav[:, kk] * gu_v).sum(-1) > win['threshold'], 1.0, -1.0).astype(np.float32)
        Ws, Wsv = z2_act(W, s_tr, win["action"]), z2_act(Wv, s_va, win["action"])
    else:
        s_tr, s_va = np.zeros(n, np.float32), np.zeros(len(Zva), np.float32)
        Ws, Wsv = W, Wv
    # ---------- location-aware probes (October 25b) ----------
    # Pooling over cells cannot express a location answer, so also probe per-cell features
    # (real/imag summed over the last L steps; 2C features) for each coordinate system.
    def cellf(Wc):
        return np.concatenate([Wc.real[:, tv].sum(1), Wc.imag[:, tv].sum(1)], 1)
    rep['cell_probe'] = {k2: cv_ba(cellf(Wc)[idx], y[idx]) for k2, Wc in
                         (('raw', Z), ('canonical', W), ('cross_location', Wx), ('cross_location_only', Wxo), ('sign_canonical', Ws))}
    rep['oracle']['cross_location_deg1_max_single_channel'] = float(np.max(np.abs([[corr(Wx.real[:, t, c], y) for c in range(C)] for t in range(T1 - L, T1)])))
    rep['oracle']['cross_location_only_deg1_max_single_channel'] = float(np.max(np.abs([[corr(Wxo.real[:, t, c], y) for c in range(C)] for t in range(T1 - L, T1)])))
    for part in ('real', 'imag'):
        rep['oracle'][f'sign_canonical_deg1_gated_{part}'] = float(np.nanmax(np.abs(
            [corr((gstar[:, None, :] * getattr(Ws, part))[:, t].sum(-1), y) for t in range(T1 - L, T1)])))
    o['samples_to_detect_5sd'] = {k2: (25 / v ** 2 if v > 0 else None) for k2, v in o.items() if isinstance(v, float)}
    json.dump(rep, open(os.path.join(tdir, f'discovery{sfx}.json'), 'w'), indent=1)
    for name, Zc, Wc, disc, Mc, d, wx, wxo, ws, sg in (('train', Z, W, gstar, M, tr, Wx, Wxo, Ws, s_tr), ('val', Zva, Wv, gstar_v, va['M'], va, Wxv, Wxov, Wsv, s_va)):
        extra = {'loc': d['loc']} if 'loc' in d.files else {}
        np.savez_compressed(os.path.join(tdir, f'features{sfx}_{name}.npz'), Z=Zc, W=Wc, gate=disc, A=d['A'], M=Mc, y=d['y'], c=d['c'],
                            Wx=wx, Wxo=wxo, Ws=ws, s=sg, **extra)
    print(json.dumps({k2: rep[k2] for k2 in ('nuisance', 'oracle')}, indent=1, default=str)[:4000])
    print('gate best', best)
