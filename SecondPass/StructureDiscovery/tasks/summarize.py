import json, os, sys
from collections import defaultdict
import numpy as np
base = os.path.dirname(os.path.abspath(__file__))
out = {}
for t in ('spatial_binding', 'orientation_cued', 'memory_guided_search', 'krauzlis', 'memory_guided_search_rich'):
    for f, tag in (('orient', 'main'), ('flow', 'main'), ('orient', 'v2'), ('flow', 'v2'), ('rich', 'main')):
        p = os.path.join(base, t, f'results_{f}_{tag}.json')
        if not os.path.exists(p): continue
        R = json.load(open(p)); g = defaultdict(list)
        for r in R: g[(r['arm'], r['n'])].append(r)
        print(f'== {t} [{f} {tag}]')
        tab = {}
        for (arm, n), rs in g.items():
            ba = [r['ba'] for r in rs]; au = [r['auc'] for r in rs]
            conds = {c: float(np.mean([r['ba_by_cond'][c] for r in rs])) for c in rs[0]['ba_by_cond']}
            tab[f'{arm}|{n}'] = dict(ba=float(np.mean(ba)), lo=min(ba), hi=max(ba), auc=float(np.mean(au)), by_cond=conds, seeds=len(rs))
            print(f'{arm:18s} n={n:5d} BA {100*np.mean(ba):5.1f} [{100*min(ba):.1f}-{100*max(ba):.1f}] AUC {np.mean(au):.3f}  by cond ' + ' '.join(f'{c}:{100*v:.0f}' for c, v in conds.items()))
        out[f'{t}|{f}' + ('' if tag == 'main' else f'|{tag}')] = tab
json.dump(out, open(os.path.join(base, 'results_summary.json'), 'w'), indent=1)
