"""Independent saved-result readback; CPU only; never refits or selects."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .run import OUT,TASKS,CKPT,EXPECTED,load,predict_relation,predict_structured,predict_ridge,local_feature,cue_feature,metric,dump,sha

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);threadpool_limits(2)
    budget=json.loads((OUT/'budget.json').read_text());assert time.time()<budget['deadline_unix'],'No computation outside authorized cap'
    frozen=json.loads((OUT/'selection_frozen.json').read_text());receipt=json.loads((OUT/'verification.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(CKPT)==EXPECTED
    assert all(sha(p)==h for p,h in receipt['source_hashes'].items())
    assert all(sha(OUT/p)==h for p,h in frozen['model_hashes'].items())
    checks=[];ids=set();n=0
    for rec in manifest['features']:
        assert sha(rec['path'])==rec['sha256']
        meta=json.loads((OUT/'features'/f'{rec["task"]}_{rec["split"]}_metadata.json').read_text())
        assert len(meta['metadata'])==manifest['counts'][rec['split']]
        for m in meta['metadata']:
            assert m['trial_id'] not in ids;ids.add(m['trial_id']);n+=1
        assert sum(m['label'] for m in meta['metadata'])==rec['n']//2
    for task in TASKS:
        a,y=load(task,'test');pred=np.load(OUT/'predictions'/f'{task}_test.npz');res=json.loads((OUT/f'{task}_comparator_results.json').read_text())
        replay={}
        for tag,name in [('true','label_only_relation'),('shuffle','shuffled_label_relation')]:
            state=torch.load(OUT/'models'/f'{task}_relation_{tag}.pt',map_location='cpu');replay[name]=predict_relation(state,a,task)
        state=torch.load(OUT/'models'/f'{task}_structured.pt',map_location='cpu');assert not set(state['decoder_fit_indices'])&set(state['calibrator_fit_indices']);replay['aux_supervised_structured']=predict_structured(state,a,task)
        for name,p in replay.items():np.testing.assert_allclose(p,pred['comparator_'+name],rtol=1e-6,atol=1e-6)
        for name,r in res.items():
            p=pred['comparator_'+name];m=metric(y['label'],p)
            for k in ['accuracy','balanced_accuracy','auc']:assert abs(m[k]-r[k])<1e-12
            assert sum(map(sum,r['confusion']))==manifest['counts']['test']
        probes=torch.load(OUT/'models'/f'{task}_probes.pt',map_location='cpu')
        for name,m in probes.items():
            if m['kind']=='angle':
                p=predict_ridge(m['state'],local_feature(a,m['layer'],m['t'],m['loc']));z=np.arctan2(p[:,1],p[:,0])/2
            else:z=predict_ridge(m['state'],cue_feature(a,m['layer'],m['t']))
            np.testing.assert_allclose(z,pred[name],atol=1e-6,rtol=1e-6)
        checks.append(dict(task=task,n=manifest['counts']['test'],replayed_probe_models=len(probes),replayed_comparators=len(replay),all_prediction_replays_match=True,metrics_recomputed_match=True))
    assert n==receipt['n_base_episodes'];assert time.time()<budget['deadline_unix']
    dump('independent_verification.json',dict(status='passed',tasks=checks,n_base_episodes=n,all_feature_hashes_match=True,all_model_hashes_match=True,source_checkpoint_unchanged=True,fitA_fitB_disjoint=True,verified_unix=time.time(),elapsed_from_first_mps_seconds=time.time()-budget['start_unix'],within_cap=True))
    print(json.dumps(checks))
if __name__=='__main__':main()
