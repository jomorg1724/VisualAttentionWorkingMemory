"""Render movies with the repo renderers and extract generic per-cell fields (no labels, no simulator metadata).

Per grid cell c (9x9 overlapping 20-px boxes) and frame t:
  z[t, c] = local optical flow u + i v (multi-lag Lucas-Kanade from summed-area tables)
  o[t, c] = local orientation phasor (sxx - syy) + 2i sxy of the structure tensor (doubled angle)
  a[t, c] = appearance energy, mean |I - frame mean| in the box
Sequences are left-padded to the task's longest movie so they end together (the report frame), with
a mask. Labels, condition ids and target locations are saved for evaluation / oracle-gate use only.

Tasks: krauzlis (original study), orientation_cued, spatial_binding, memory_guided_search.
The orientation field was added for the October 25 follow-up; the Krauzlis study used flow only."""
import sys, time
import numpy as np
REPO = r'C:/Users/jomor/Documents/VisualAttentionWorkingMemory'
sys.path.insert(0, REPO)

WIN, STRIDE = 20, 10
CENTERS = list(range(WIN // 2, 100 - WIN // 2 + 1, STRIDE))  # 9 per axis -> 81 cells
TMAXES = dict(krauzlis=45, orientation_cued=28, spatial_binding=29, memory_guided_search=29)


def box_sums(img, centers=CENTERS):
    S = np.cumsum(np.cumsum(img, -1), -2)
    S = np.pad(S, [(0, 0)] * (img.ndim - 2) + [(1, 0), (1, 0)])
    rows = []
    for cy in centers:
        row = []
        for cx in centers:
            y0, y1, x0, x1 = cy - WIN // 2, cy + WIN // 2, cx - WIN // 2, cx + WIN // 2
            row.append(S[..., y1, x1] - S[..., y0, x1] - S[..., y1, x0] + S[..., y0, x0])
        rows.append(np.stack(row, -1))
    return np.stack(rows, -2)


def lk_flow(I, k):
    Iavg = 0.5 * (I[:, k:] + I[:, :-k])
    Ix = np.gradient(Iavg, axis=-1); Iy = np.gradient(Iavg, axis=-2); It = I[:, k:] - I[:, :-k]
    sxx, syy, sxy = box_sums(Ix * Ix), box_sums(Iy * Iy), box_sums(Ix * Iy)
    sxt, syt = box_sums(Ix * It), box_sums(Iy * It)
    det = sxx * syy - sxy ** 2 + 1e-3
    return -(syy * sxt - sxy * syt) / det / k, -(sxx * syt - sxy * sxt) / det / k


def orientation(I):
    Ix = np.gradient(I, axis=-1); Iy = np.gradient(I, axis=-2)
    sxx, syy, sxy = box_sums(Ix * Ix), box_sums(Iy * Iy), box_sums(Ix * Iy)
    return ((sxx - syy) + 2j * sxy) / WIN ** 2


def fields(x, TMAX):
    I = x.mean(2); n, T = I.shape[:2]; G = len(CENTERS)
    U = np.zeros((n, T - 1, G, G)); V = np.zeros_like(U); W = np.zeros(T - 1)
    for k in (1, 2, 4):
        if k >= T:
            continue
        u, v = lk_flow(I, k); U[:, :T - k] += u; V[:, :T - k] += v; W[:T - k] += 1
    U /= W[None, :, None, None]; V /= W[None, :, None, None]
    z = (U + 1j * V).reshape(n, T - 1, G * G)
    o = orientation(I).reshape(n, T, G * G)
    a = box_sums(np.abs(I - I.mean((-1, -2), keepdims=True))).reshape(n, T, G * G) / WIN ** 2
    Z = np.zeros((n, TMAX - 1, G * G), np.complex64); O = np.zeros((n, TMAX, G * G), np.complex64)
    A = np.zeros((n, TMAX, G * G), np.float32); M = np.zeros((n, TMAX), bool)
    Z[:, TMAX - T:] = z; O[:, TMAX - T:] = o; A[:, TMAX - T:] = a; M[:, TMAX - T:] = True
    return Z, O, A, M


RICH_STRIDE = 5
RICH_CENTERS = list(range(WIN // 2, 100 - WIN // 2 + 1, RICH_STRIDE))  # 17 per axis -> 289 cells
RICH_SIGMAS = (0.0, 1.5, 3.0)


def rich_fields(x, TMAX):
    """Multi-channel complex field on a 5-px grid (October 25c), channel-major units (k * C + c):
    orientation phasors of the structure tensor at three Gaussian scales, and a colour-opponency
    phasor (R - G) + i (B - (R + G)/2) averaged per cell. Appearance energy on the same grid."""
    from scipy.ndimage import gaussian_filter
    I = x.mean(2); n, T = I.shape[:2]; G = len(RICH_CENTERS)
    chans = []
    for s in RICH_SIGMAS:
        Is = gaussian_filter(I, (0, 0, s, s)) if s > 0 else I
        Ix = np.gradient(Is, axis=-1); Iy = np.gradient(Is, axis=-2)
        sxx, syy, sxy = (box_sums(v, RICH_CENTERS) for v in (Ix * Ix, Iy * Iy, Ix * Iy))
        chans.append(((sxx - syy) + 2j * sxy) / WIN ** 2)
    R, Gc, B = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    chans.append((box_sums(R - Gc, RICH_CENTERS) + 1j * box_sums(B - 0.5 * (R + Gc), RICH_CENTERS)) / WIN ** 2)
    o = np.stack([c.reshape(n, T, G * G) for c in chans], 2).reshape(n, T, len(chans) * G * G)
    a = box_sums(np.abs(I - I.mean((-1, -2), keepdims=True)), RICH_CENTERS).reshape(n, T, G * G) / WIN ** 2
    O = np.zeros((n, TMAX, o.shape[-1]), np.complex64); A = np.zeros((n, TMAX, G * G), np.float32); M = np.zeros((n, TMAX), bool)
    O[:, TMAX - T:] = o; A[:, TMAX - T:] = a; M[:, TMAX - T:] = True
    return O, A, M


def render_rich(task, n, seed, split, out):
    TMAX = TMAXES[task]
    parts = {k: [] for k in ('R', 'A', 'M', 'y', 'c', 'loc')}
    for x, y, c, loc in batches(task, n, seed, split):
        for st in range(0, len(x), 50):
            O, A, M = rich_fields(x[st:st + 50], TMAX)
            for k, v in zip(('R', 'A', 'M', 'y', 'c', 'loc'), (O, A, M, y[st:st + 50], c[st:st + 50], loc[st:st + 50])):
                parts[k].append(v)
    np.savez(out, K=len(RICH_SIGMAS) + 1, stride=RICH_STRIDE, **{k: np.concatenate(v) for k, v in parts.items()})


def batches(task, n, seed, split):
    """Yield (x, y, condition id, target location) in chunks, balanced over the task's conditions."""
    if task == 'memory_guided_search':
        from SecondPass.MemoryGuidedSearch.dataset import Stream, CONDITIONS
        # The stream reuses scenes across timing conditions for equal trial ids, and its targets are a
        # permutation over each block of 4 consecutive ids. So every trial gets its own id (unique scene)
        # and ids are drawn in complete blocks of 4 (balanced targets); conditions are spread over blocks.
        s = Stream(split)
        cells = [c['id'] for c in CONDITIONS]
        order = np.random.default_rng(seed).permutation(np.resize(np.arange(len(cells)), n // 4))
        for ci in np.unique(order):
            rows = [s.trial(cells[ci], 4 * b + j) for b in np.flatnonzero(order == ci) for j in range(4)]
            x = np.stack([r[0].numpy() for r in rows]); y = np.array([int(r[1]) for r in rows])
            yield x, y, np.array([r[2]['delay'] for r in rows]), y
        return
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    s = SpatialBatteryStream(seed, split=split)
    if task == 'krauzlis':
        name, key, levels = 'krauzlis_cued_motion', 'baseline_transitions', (12, 20, 28)
    else:
        name, key, levels = task, 'delay', (0, 4, 12, 24)
    for lv in levels:
        for st in range(0, n // len(levels), 50):
            m = min(50, n // len(levels) - st)
            x, y, meta = s.batch(m, name, {key: lv})
            yield x.numpy(), y.numpy(), np.full(m, lv), np.array([r['target_location'] for r in meta])


def render(task, n, seed, split, out):
    TMAX = TMAXES[task]
    parts = {k: [] for k in ('Z', 'O', 'A', 'M', 'y', 'c', 'loc')}
    for x, y, c, loc in batches(task, n, seed, split):
        Z, O, A, M = fields(x, TMAX)
        for k, v in zip(('Z', 'O', 'A', 'M', 'y', 'c', 'loc'), (Z, O, A, M, y, c, loc)):
            parts[k].append(v)
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in parts.items()})


if __name__ == '__main__':
    t0 = time.time()
    # usage: python features.py TASK N SEED SPLIT OUT   (legacy: python features.py N SEED SPLIT OUT -> krauzlis)
    # rich multi-channel field: python features.py TASK N SEED SPLIT OUT rich
    args = sys.argv[1:] if not sys.argv[1].isdigit() else ['krauzlis'] + sys.argv[1:]
    (render_rich if args[-1] == 'rich' else render)(args[0], int(args[1]), int(args[2]), args[3], args[4])
    print('saved', args[4], round(time.time() - t0, 1), 's')
