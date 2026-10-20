"""Scene-paired checkpoint robustness within static-search stimulus families."""
import json
import numpy as np
from .design import OUT,write

def analyze():
    root=OUT/'memory_guided_search';groups={}
    for path in (root/'native').glob('*.npz'):
        if path.stem.endswith('_latest'):continue
        a=np.load(path);b=np.load(path.with_name(path.stem+'_latest.npz'));ma=json.loads(path.with_suffix('.json').read_text());key=ma[0]['family']+'_'+ma[0]['similarity']
        assert np.array_equal(a['y'][:32],b['y'])
        assert [m['trial_id'] for m in ma[:32]]==list(range(32))
        groups.setdefault(key,[]).append(np.stack((a['p'][:32].argmax(1)==a['y'][:32],b['p'].argmax(1)==b['y']),1))
    rows=[]
    for key,values in sorted(groups.items()):
        v=np.array(values,dtype=float);d=(v[:,:,1]-v[:,:,0]).mean(0);rng=np.random.default_rng(1414);bounds=np.quantile(d[rng.integers(32,size=(3000,32))].mean(1),[.025,.975])
        rows.append(dict(family=key,conditions=len(v),matched_scenes=32,presentations=int(v.shape[0]*v.shape[1]),selected_accuracy=float(v[:,:,0].mean()),final_accuracy=float(v[:,:,1].mean()),delta_accuracy=float(d.mean()),ci95=bounds.tolist()))
    write(root/'checkpoint_family_comparison.json',dict(results=rows,method='The same first 32 native scenes in each family are compared at selected checkpoint 204289 and final checkpoint 220637. Differences are averaged across all timing conditions within scene; 3000 bootstrap resamples preserve those scene pairs. This compares two training-stage checkpoints, not randomized sampling-regime interventions.'))
    return rows
