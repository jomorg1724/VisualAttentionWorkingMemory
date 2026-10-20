"""Independently reconcile published attention traces to frozen recordings."""
import csv,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from .design import OUT,TASKS,write

def main():
    conditions=frames=trials=0
    for task in TASKS:
        root=OUT/task;data=json.loads((root/'attention_dynamics.json').read_text())
        cache={}
        for group in data['groups']:
            recorded=[]
            for ref in group['refs']:
                if ref['cell'] not in cache:
                    with np.load(root/'native'/f"{ref['cell']}.npz") as z:
                        cache[ref['cell']]=z['attention'].astype(np.float64)
                recorded.append(cache[ref['cell']][ref['index']])
            recorded=np.array(recorded);mean=recorded.mean(0);weights=np.array(group['region_weights'])
            assert len(recorded)==group['n']
            assert np.allclose(mean,group['mean_map'],atol=1e-6)
            assert np.allclose(recorded[0],group['example_map'],atol=1e-7)
            assert np.allclose(mean.sum((1,2)),1,atol=1e-6)
            assert np.allclose(mean.reshape(len(mean),100)@weights.T,group['mean_mass'],atol=1e-6)
            assert np.allclose(weights@np.full(100,.01),group['uniform_mass'])
            assert np.array_equal(np.array(group['time'])-group['anchor'],group['aligned_time'])
            assert len(group['phases'])==len(mean)
            lo=np.array(group['ci_low']);hi=np.array(group['ci_high'])
            assert lo.shape==hi.shape==np.array(group['mean_mass']).shape
            assert np.all(lo<=hi) and np.all(lo>=-1e-7) and np.all(hi<=1+1e-7)
            with Image.open(root/'figures'/group['sprite']) as im:assert im.size==(100*len(mean),100)
            conditions+=1;frames+=len(mean);trials+=len(recorded)
        fs=data['figures']
        for name in fs['maps']+fs['landmarks']+fs['curves']+[fs['comparison']]:
            for ext in ('.png','.pdf'):assert (root/'figures'/(name+ext)).is_file()
        expected=sum(len(g['time'])*len(g['region_names']) for g in data['groups'])
        with (root/'source_tables/attention_condition_timecourses.csv').open(newline='',encoding='utf8') as f:
            assert len(list(csv.DictReader(f)))==expected
        tex=(OUT/'papers'/f'{task}.tex').read_text(encoding='utf8')
        assert tex.count('% BEGIN CONDITION ATTENTION')==1
        assert tex.count('% BEGIN FULL CONDITION ATTENTION')==1
    manifest=json.loads((OUT/'attention/manifest.json').read_text())
    assert manifest['conditions']==conditions and manifest['frames']==frames and manifest['selected_presentations']==trials
    write(OUT/'attention/verification.json',dict(time=time.time(),tasks=19,conditions=conditions,condition_frames=frames,
        selected_presentations=trials,recording_means_and_region_masses='passed',uniform_allocation='passed',
        confidence_interval_bounds='passed',event_alignment='passed',sprite_geometry='passed',figures_and_tables='passed',
        browser_preview='not_verified_file_protocol_blocked',new_model_inference=False))
    print(json.dumps(json.loads((OUT/'attention/verification.json').read_text()),indent=2))

if __name__=='__main__':main()
