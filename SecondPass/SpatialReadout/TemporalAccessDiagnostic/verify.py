"""Saved-artifact replay, CPU only, no fitting or model selection."""
import json
import time
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from . import run as r


def verify():
    budget=json.loads((r.OUT/'budget.json').read_text())
    assert time.time()<budget['deadline_unix'],'No computation outside original cap'
    frozen=json.loads((r.OUT/'selection_frozen.json').read_text())
    start=json.loads((r.OUT/'test_start.json').read_text())
    verification=json.loads((r.OUT/'verification.json').read_text())
    protocol=json.loads((r.OUT/'protocol.json').read_text())
    capacities=json.loads((r.OUT/'capacity.json').read_text())
    oldhash=json.loads((r.OUT/'prior_artifacts.json').read_text())
    assert frozen['unix']<start['unix']
    assert r.sha(r.OUT/'selection_frozen.json')==start['selection_frozen_sha256']
    assert r.sha(r.OUT/'protocol.json')==frozen['protocol_sha256']
    assert r.sha(r.prior.CKPT)==r.prior.EXPECTED
    assert all(r.sha(p)==h for p,h in protocol['source_hashes'].items())
    assert all(r.sha(p)==h for p,h in oldhash.items())
    assert all(r.sha(r.OUT/p)==h for p,h in frozen['model_hashes'].items())
    for rec in verification['features']:assert r.sha(rec['path'])==rec['sha256']
    allids=[];allrasters=[];checks=[]
    oldids=set();oldrasters=set()
    for p in (r.OLD/'features').glob('*_metadata.json'):
        m=json.loads(p.read_text());oldids.update(z['trial_id'] for z in m['metadata']);oldrasters.update(m['raster_sha256'])
    for task in r.TASKS:
        a,y,m=r.load(r.OUT,task,'test')
        assert len(y['label'])==512 and int(y['label'].sum())==256
        assert m['seed']==protocol['test_seeds'][task]
        allids.extend(z['trial_id'] for z in m['metadata']);allrasters.extend(m['raster_sha256'])
        saved=np.load(r.OUT/'predictions'/f'{task}_test.npz')
        for key in ('label','location','sign'):np.testing.assert_array_equal(saved[key],y[key])
        np.testing.assert_array_equal(saved['trial_id'],[z['trial_id'] for z in m['metadata']])
        np.testing.assert_array_equal(saved['raster_sha256'],m['raster_sha256'])
        results=json.loads((r.OUT/f'{task}_results.json').read_text())
        scores={'deployed':r.prior.softmax(a['logits'])[:,1]};count=0
        for layer in r.LAYERS:
            pair=[]
            for access in r.ACCESS:
                name=layer+'_'+access
                state=torch.load(r.OUT/'models'/f'{task}_{name}.pt',map_location='cpu',weights_only=False)
                assert state['decoder_fit_indices']==list(range(512))
                assert state['calibrator_fit_indices']==list(range(512,1024))
                assert not set(state['decoder_fit_indices'])&set(state['calibrator_fit_indices'])
                assert state['selections']==frozen['selections'][task][name]
                cap=r.capacity(state);assert cap==capacities[task][layer];pair.append(cap)
                assert all(c['n_fit']==512 and c['alpha_grid']==[.1,1.,10.,100.] for c in cap['components'])
                p,comp=r.predict(state,a);scores[name]=p
                np.testing.assert_allclose(p,saved['score__'+name],atol=1e-7,rtol=1e-7)
                for key,z in comp.items():np.testing.assert_allclose(z,saved[name+'__'+key],atol=1e-7,rtol=1e-7)
                if access=='final_only':assert r.access_invariance(state,a)==results['invariance'][name]
                for content in ('sample','probe'):
                    angle=np.arctan2(comp[content][...,1],comp[content][...,0])/2
                    summary=r.prior.angle_summary(y[content],angle,y['location'])
                    assert all(summary[k]==results['components'][name][content][k] for k in summary)
                for label in ('location','sign'):
                    if label in comp:
                        target=y[label] if label=='location' else (y[label]>0).astype(int)
                        assert float(balanced_accuracy_score(target,comp[label].argmax(1)))==results['components'][name][label]['balanced_accuracy']
                count+=1
            assert pair[0]==pair[1]
        metrics,pairs=r.bootstrap_metrics(y['label'],scores)
        assert pairs==results['pairs_time_separated_minus_final_only']
        for name,p in scores.items():
            np.testing.assert_array_equal(p,saved['score__'+name])
            for key,val in metrics[name].items():assert val==results['metrics'][name][key],(task,name,key)
            assert float(balanced_accuracy_score(y['label'],p>=.5))==metrics[name]['balanced_accuracy']
            assert float(roc_auc_score(y['label'],p))==metrics[name]['auc']
        checks.append(dict(task=task,replayed_structured_models=count,component_outputs_match=True,metrics_and_intervals_match=True,invariance_retested=True,capacity_rechecked=True))
        saved.close();del a
    assert len(allids)==len(set(allids))==1024
    assert len(allrasters)==len(set(allrasters))==1024
    assert not set(allids)&oldids and not set(allrasters)&oldrasters
    assert len(oldids)==len(oldrasters)==3584
    assert len(frozen['model_hashes'])==sum(z['replayed_structured_models'] for z in checks)==12
    assert time.time()<budget['deadline_unix']
    receipt=dict(status='passed',tasks=checks,all_model_feature_source_checkpoint_hashes_match=True,all_prior_artifacts_unchanged=True,decoder_calibrator_split_disjoint=True,fresh_vs_prior_disjoint=True,selections_frozen_before_test=True,elapsed_seconds=time.time()-budget['start_unix'],verified_unix=time.time(),within_cap=True)
    r.dump('independent_verification.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);r.threadpool_limits(2)
    verify()
