"""Verify saved audit counts/predictions, replay native rasters, add controls.
No refitting on test. Controls reuse the calibration-selected threshold.
"""
import json
from pathlib import Path
import numpy as np
import pixel_observer as p

def wilson(k,n):
    z=1.959963984540054
    v=k/n;den=1+z*z/n
    mid=(v+z*z/(2*n))/den
    half=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def main():
    import torch
    torch.set_num_threads(1)
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    load=lambda f:[json.loads(line) for line in (p.OUT/f).read_text().splitlines()]
    d=load('pixel_results_duration.jsonl');c=load('pixel_results_k_calibration.jsonl');k=load('pixel_results_k_test.jsonl')
    assert [len(d),len(c),len(k)]==[128,100,200]
    config=json.loads((p.OUT/'pixel_results_config.json').read_text())
    assert config['observer_sha256']==p.hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()
    assert all(p.hashlib.sha256((p.ROOT/name).read_bytes()).hexdigest()==sha for name,sha in config['sources'].items())
    threshold=json.loads((p.OUT/'pixel_results_threshold.json').read_text())['threshold']
    for group in (d,c,k):
        assert len({r['trial_id'] for r in group})==len(group)
        assert len({r['raster_sha256'] for r in group})==len(group)
    assert not {r['raster_sha256'] for r in c}&{r['raster_sha256'] for r in k}
    for r in d:
        scores=np.asarray(r['transition_scores']);directions=scores.argmax(-1)
        assert directions.tolist()==r['directions']
        counts=np.array([np.bincount(x,minlength=4) for x in directions]);t=r['decoded_target']
        tied=np.flatnonzero(counts[t]==counts[t].max());evidence=(scores/(scores.sum(-1,keepdims=True)+1e-12)).sum(1)
        pred=int(tied[evidence[t,tied].argmax()]);assert pred==r['prediction']
    for r in c+k:
        assert r['prediction']==int(r['score']>threshold)
        angles=[]
        for flows in r['flow_vectors']:
            a=[]
            for part in (flows[:20],flows[20:]):
                vectors=np.asarray([v for transition in part for v in transition])
                mean=vectors.mean(0) if len(vectors) else np.zeros(2)
                a.append(float(np.rad2deg(np.arctan2(mean[1],mean[0]))))
            angles.append(a)
        assert np.allclose(angles,r['estimated_angles_degrees'])
        diff=(np.diff(angles,axis=1).ravel()+180)%360-180
        assert abs(abs(diff[r['decoded_target']])-r['score'])<1e-10
    for group,task,seed,cfg in ((d,'motion_duration_cued',config['duration_seed'],{'delay':0}),(k,'krauzlis_cued_motion',config['k_test_seed'],{'baseline_transitions':20})):
        stream=SpatialBatteryStream(seed,'test')
        for r in group[:3]:
            x,y,m=stream.batch(1,task,cfg);a=x[0].numpy()
            assert p.digest(a)==r['raster_sha256'] and int(y[0])==r['label']
            obs=p.duration_observe(a) if task=='motion_duration_cued' else p.krauzlis_observe(a,20)
            if task=='motion_duration_cued':assert obs['prediction']==r['prediction']
            else:assert abs(obs['score']-r['score'])<1e-10
    controls={}
    for name,predict in [('duration_last_direction',lambda r:r['directions'][r['decoded_target']][-1]),('duration_fixed_patch_zero',lambda r:int(np.argmax(r['counts'][0])))]:
        controls[name]=p.metrics([dict(r,prediction=predict(r)) for r in d],4)
    for name,predict in [('krauzlis_wrong_cue',lambda r:int(abs(r['signed_changes_degrees'][1-r['decoded_target']])>threshold)),('krauzlis_any_patch',lambda r:int(max(abs(v) for v in r['signed_changes_degrees'])>threshold))]:
        rows=[dict(r,prediction=predict(r)) for r in k]
        controls[name]=p.metrics(rows,2)
        # AUC in metrics would use the original score; omit for these controls.
        controls[name].pop('auc')
    duration_strata={}
    for field in ('step_pixels','true_count_margin'):
        values=sorted({r['metadata'][field] if field=='step_pixels' else r[field] for r in d})
        duration_strata[field]={}
        for v in values:
            rows=[r for r in d if (r['metadata'][field] if field=='step_pixels' else r[field])==v]
            duration_strata[field][str(v)]={'transition_correct':sum(r['transition_correct'] for r in rows),'transition_n':8*len(rows),'class_coverage':len({r['label'] for r in rows})}
    flow_counts=[n for r in k for patch in r['flow_match_counts'] for n in patch]
    report=dict(verified_counts={'duration':len(d),'calibration':len(c),'test':len(k)},
                prediction_replay_verified=428,native_raster_replay_verified=6,source_hashes_unchanged=True,
                unique_rasters=True,calibration_test_rasters_disjoint=True,
                duration_accuracy_wilson95=wilson(sum(r['prediction']==r['label'] for r in d),len(d)),
                krauzlis_accuracy_wilson95=wilson(sum(r['prediction']==r['label'] for r in k),len(k)),
                krauzlis_calibration_cue_correct=sum(r['decoded_target']==r['metadata']['target_location'] for r in c),
                krauzlis_flow_matches_per_patch_transition=dict(mean=float(np.mean(flow_counts)),minimum=min(flow_counts),zero_count=flow_counts.count(0),n=len(flow_counts)),
                controls=controls,duration_strata=duration_strata,
                limitations=['margin-4/5 strata contain only one observed class: BA is macro recall of represented classes, not four-class BA',
                             'temporal raster differences include ordinary motion/resets, not isolated event-causal differences',
                             'average-pooling surrogate is not measured CNN information loss'])
    p.dump('pixel_results_replay.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
