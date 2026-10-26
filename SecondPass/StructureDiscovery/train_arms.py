"""Stage 4: train ordinary gradient networks (Adam) with and without the discovered structure.

Arms (same data, same optimizer, same budget):
  raw_flat            plain GRU over time on all cells' raw inputs concatenated (the repo's default style)
  cell_raw            one GRU shared across cells (location sharing), raw field, summed over cells
  cell_canon          shared GRU on discovered canonical (nuisance-free) coordinates, summed over cells
  cell_raw_gate       shared GRU on raw field, cells weighted by the DISCOVERED attention gate
  cell_canon_gate     discovered canonicalization + discovered gate (full discovered structure)
  cell_gcanon_gate    as above with the literal one-global-sector canonicalization
  cell_canon_hand     discovered canonicalization + HAND gate (Krauzlis cue frames; upper bound)
  cell_canon_oracle   discovered canonicalization + ORACLE gate on the true target location (upper bound)
October 25b arms (discovered sign nuisance and cross-location references, see discover.py 1b/3b):
  cell_canon_gate_sign    discovered gate on the sign-canonical field (discovered Z2 action applied)
  cell_canon_gate_ctx     discovered gate on the canonical field, discovered sign broadcast as an input
  cell_canon_oracle_sign  sign-canonical field + oracle target gate (upper bound)
  cell_xcanon[_gate]      cross-location reference (self pair included), uniform [or discovered] gate
  cell_xocanon[_gate]     cross-location reference, self pair excluded
Each arm: n labelled training trials, fixed number of sample passes, threshold chosen on training
data (binary) or argmax (multiclass), balanced accuracy and AUC on the held-out movies.

usage: python train_arms.py ARMS NS TAG [TASK_DIR [FIELD]]
Multiclass tasks (October 25): cross-entropy over K outputs per cell, summed over gated cells, and each
cell's input carries its normalised (x, y) position so that a location answer is expressible."""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from discover import gate_maps

torch.set_num_threads(2)
GRID = np.array([(x, y) for y in range(10, 91, 10) for x in range(10, 91, 10)], float)   # cell centres, row-major
TARGET_CENTERS = dict(orientation_cued=[(27, 27), (73, 27), (27, 73), (73, 73)],
                      spatial_binding=[(27, 27), (73, 27), (27, 73), (73, 73)],
                      memory_guided_search=[(25, 25), (75, 25), (25, 75), (75, 75)])
TARGET_CENTERS['memory_guided_search_rich'] = TARGET_CENTERS['memory_guided_search']


def bal_acc(y, p):
    return float(np.mean([(p[y == k] == k).mean() for k in np.unique(y)]))


def best_threshold(s, y):
    qs = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    return qs[int(np.argmax([bal_acc(y, (s > q).astype(int)) for q in qs]))]


def hand_gate(A, M):
    s = M.argmax(1)
    start = np.stack([A[np.arange(len(A)), s + k] for k in (0, 1)], 1)
    return gate_maps(start, np.ones(start.shape[:2], bool)).mean(1)


def oracle_gate(loc, task):
    xy = np.array(TARGET_CENTERS[task], float)[loc]                    # n, 2
    d2 = ((GRID[None] - xy[:, None]) ** 2).sum(-1)
    g = np.exp(-d2 / (2 * 10.0 ** 2))
    return g / g.sum(1, keepdims=True)


class Lazy:
    """npz arrays loaded on first use and cached; drop() frees the large canonical fields between arms."""
    def __init__(self, path):
        self.f, self.cache = np.load(path), {}
        self.stem = path[:-4]
        # large multi-channel fields are stored as memory-mapped PATH_KEY.npy files next to the npz
        self.npy = {k: f'{self.stem}_{k}.npy' for k in ('Z', 'W', 'Wx', 'Wxo', 'Ws') if os.path.exists(f'{self.stem}_{k}.npy')}
        self.files = list(self.f.files) + list(self.npy)

    def __getitem__(self, k):
        if k not in self.cache:
            self.cache[k] = self.f[k] if k in self.f.files else np.load(self.npy[k], mmap_mode='r')
        return self.cache[k]

    def drop(self, keys=('W', 'Wx', 'Wxo', 'Ws')):
        for k in keys:
            self.cache.pop(k, None)


def cellify(d, key):
    """(n, T, K*C) channel-major units -> (n, T, C, K); single-channel fields -> (n, T, C, 1)."""
    x = d[key]
    K = int(d['K']) if 'K' in d.files else 1
    n, T, U = x.shape
    return x.reshape(n, T, K, U // K).transpose(0, 1, 3, 2)


def inputs(d, arm, zscale, pos):
    Z, A = cellify(d, 'Z') / zscale, d['A'][:, 1:]
    a = (A / (A.std() + 1e-9))[..., None]
    if arm in ('raw_flat', 'cell_raw', 'cell_raw_gate'):
        X = np.concatenate([Z.real, Z.imag, a], -1)
    elif arm == 'cell_gcanon_gate':
        # literal discovered quotient: ONE global sector, so every cell is referenced to the causal
        # running sum of all cells' past field (a single shared rotation is removed)
        Zr = d['Z']
        valid = d['M'][:, 1:, None]
        S = np.where(valid, Zr, 0).sum(-1, keepdims=True)
        R = np.cumsum(S, 1) - S
        Wg = np.where(valid, Zr * np.conj(R) / (np.abs(R) + 1e-6), 0)
        X = np.stack([Wg.real / zscale, Wg.imag / zscale, np.abs(Zr) / zscale, a[..., 0]], -1)
    else:
        # canonical coordinates: per-cell (W), sign-canonical (Ws), cross-location (Wx) or cross-only (Wxo)
        key = 'Ws' if '_sign' in arm else 'Wx' if arm.startswith('cell_xcanon') else 'Wxo' if arm.startswith('cell_xocanon') else 'W'
        Wc = cellify(d, key)
        X = np.concatenate([Wc.real / zscale, Wc.imag / zscale, np.abs(Z), a], -1)
        if arm.endswith('_ctx'):                     # discovered discrete context broadcast to every cell
            X = np.concatenate([X, np.broadcast_to(d['s'][:, None, None, None], X.shape[:3] + (1,))], -1)
    if pos:
        n, T, C, _ = X.shape
        side = int(round(np.sqrt(C)))
        cc = np.linspace(10, 90, side)
        grid = np.array([(x, y) for y in cc for x in cc])
        P = np.broadcast_to(((grid - 50) / 40)[None, None], (n, T, C, 2))
        X = np.concatenate([X, P], -1)
    return X.astype(np.float32)                      # n, T, C, F


class CellNet(nn.Module):
    def __init__(self, F, H=16, K=1):
        super().__init__()
        self.gru = nn.GRU(F, H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, K))
        self.bias = nn.Parameter(torch.zeros(K))
        self.K = K

    def forward(self, X, g):                          # X: n,T,C,F ; g: n,C
        n, T, C, F = X.shape
        h, _ = self.gru(X.permute(0, 2, 1, 3).reshape(n * C, T, F))
        s = self.head(h[:, -1]).reshape(n, C, self.K)
        out = (g[..., None] * s).sum(1) * C + self.bias   # g sums to 1 -> scale back to an ungated-sum scale
        return out.squeeze(-1) if self.K == 1 else out


class FlatNet(nn.Module):
    def __init__(self, D, H=64, K=1):
        super().__init__()
        self.gru = nn.GRU(D, H, batch_first=True)
        self.head = nn.Linear(H, K)
        self.K = K

    def forward(self, X, g):
        n, T, C, F = X.shape
        h, _ = self.gru(X.reshape(n, T, C * F))
        out = self.head(h[:, -1])
        return out.squeeze(-1) if self.K == 1 else out


def run(arm, n, seed, tr, va, task, passes=24000, batch=32, lr=3e-3):
    K = int(len(np.unique(tr['y'])))
    multi = K > 2
    zscale = float(np.abs(tr['Z']).std())
    Xtr_all, Xva = inputs(tr, arm, zscale, multi), inputs(va, arm, zscale, multi)
    C = Xtr_all.shape[2]
    gsel = 'oracle' if 'oracle' in arm else 'hand' if arm.endswith('_hand') else 'gate' if '_gate' in arm else None

    def gates(d):
        if gsel == 'gate': return d['gate'].astype(np.float32)
        if gsel == 'hand': return hand_gate(d['A'], d['M']).astype(np.float32)
        if gsel == 'oracle': return oracle_gate(d['loc'], task).astype(np.float32)
        return np.full((len(d['y']), C), 1 / C, np.float32)
    gtr_all, gva = gates(tr), gates(va)
    idx = np.random.default_rng(100 + seed).choice(len(tr['y']), n, replace=False)
    ytr_np = tr['y'][idx].astype(int)
    Xtr, gtr = torch.tensor(Xtr_all[idx]), torch.tensor(gtr_all[idx])
    ytr = torch.tensor(ytr_np, dtype=torch.long if multi else torch.float32)
    torch.manual_seed(seed)
    Ko = K if multi else 1
    net = FlatNet(C * Xtr.shape[-1], K=Ko) if arm == 'raw_flat' else CellNet(Xtr.shape[-1], K=Ko)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss() if multi else nn.BCEWithLogitsLoss()
    steps = max(1, passes // batch)
    rng = np.random.default_rng(seed)
    for st in range(steps):
        b = rng.choice(n, min(batch, n), replace=False)
        loss = lossf(net(Xtr[b], gtr[b]), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s_tr = torch.cat([net(Xtr[i:i + 256], gtr[i:i + 256]) for i in range(0, n, 256)]).numpy()
        Xv, gv = torch.tensor(Xva), torch.tensor(gva)
        s_va = torch.cat([net(Xv[i:i + 256], gv[i:i + 256]) for i in range(0, len(Xv), 256)]).numpy()
    yv = va['y'].astype(int)
    if multi:
        pv = s_va.argmax(1)
        prob = np.exp(s_va - s_va.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
        auc = roc_auc_score(yv, prob, multi_class='ovr')
    else:
        th = best_threshold(s_tr, ytr_np)
        pv = (s_va > th).astype(int)
        auc = roc_auc_score(yv, s_va)
    return dict(arm=arm, n=n, seed=seed, ba=bal_acc(yv, pv), auc=float(auc),
                ba_by_cond={str(b): bal_acc(yv[va['c'] == b], pv[va['c'] == b]) for b in np.unique(va['c'])})


if __name__ == '__main__':
    arms = sys.argv[1].split(',')
    ns = [int(v) for v in sys.argv[2].split(',')]
    tdir = sys.argv[4] if len(sys.argv) > 4 else '.'
    field = sys.argv[5] if len(sys.argv) > 5 else 'flow'
    sfx = '' if tdir == '.' and field == 'flow' else f'_{field}'
    task = os.path.basename(os.path.normpath(os.path.abspath(tdir)))
    tr = Lazy(os.path.join(tdir, f'features{sfx}_train.npz'))
    va = Lazy(os.path.join(tdir, f'features{sfx}_val.npz'))
    out = []
    for arm in arms:
        for n in ns:
            for seed in (0, 1):
                t0 = time.time()
                r = run(arm, n, seed, tr, va, task); r['secs'] = round(time.time() - t0, 1)
                out.append(r); print(json.dumps(r), flush=True)
                json.dump(out, open(os.path.join(tdir, f'results{sfx}_{sys.argv[3]}.json'), 'w'), indent=1)
        tr.drop(); va.drop()
