"""Bounded CPU-only native renderer audit. No neural model/checkpoint/train code.
Run from repo root with the dedicated task-suite Python; writes audit artifacts only.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from SecondPass.TaskSuite.suite import CATALOG, SuiteStream
from PreAttentiveVision.neuroscience_stimuli import TaskStream
from WorkingMemory.PlainBaseline.variants import VariantStream
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream, CENTERS, blank, local_cue, frame_count
from WorkingMemory.stimuli import visual_cues

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNTIME = Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def angle(image, target):
    x, y = CENTERS[target].astype(int)
    patch = image[0,y-12:y+13,x-12:x+13].astype(float)
    gx = gaussian_filter(patch, .65, order=(0,1))
    gy = gaussian_filter(patch, .65, order=(1,0))
    return .5*np.arctan2(2*np.sum(gx*gy), np.sum(gx*gx-gy*gy)) % np.pi

sources = ['SecondPass/TaskSuite/suite.py', 'SecondPass/TaskSuite/catalog.json',
           'WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/PlainBaseline/variants.py',
           'WorkingMemory/stimuli.py','PreAttentiveVision/neuroscience_stimuli.py',
           'PreAttentiveVision/natural_stimuli.py','WorkingMemory/PlainBaseline/accum.py',
           'SecondPass/SpatialReadout/model.py']
provenance = {p:dict(repository_sha256=digest(ROOT/p), runtime_sha256=digest(RUNTIME/'repo'/p),
                       identical=(ROOT/p).read_bytes()==(RUNTIME/'repo'/p).read_bytes()) for p in sources}
assert all(r['identical'] for r in provenance.values())
for name, record in provenance.items():
    locked=RUNTIME/'run_continuation_v2/locked_source'/name
    record['locked_source_present']=locked.is_file()
    record['locked_source_sha256']=digest(locked) if locked.is_file() else None
    if locked.is_file():
        assert record['locked_source_sha256']==record['repository_sha256']
expected={(t['id'],c['id']) for t in CATALOG['tasks'] for c in t['conditions']}
results={}
for filename in ['test_selected.json','test_terminal.json']:
    path=RUNTIME/'run_continuation_v2'/filename
    d=json.loads(path.read_text())
    assert d['complete'] and len(d['cells'])==35 and {(r['task'],r['cell']) for r in d['cells']}==expected
    for r in d['cells']:
        assert np.array(r['confusion']).sum()==r['n']
    results[filename]=dict(path=str(path),sha256=digest(path),cells=d['cells'],namespace=d['final_test_namespace'])
save('task_results_extract.json',results)
with (OUT/'task_results_cells.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['checkpoint','task','cell','n','balanced_accuracy','auc','specificity','confusion'])
    for name,d in results.items():
        for r in d['cells']:w.writerow([name,r['task'],r['cell'],r['n'],r['balanced_accuracy'],r['auc'],r.get('specificity'),json.dumps(r['confusion'])])

report=dict(cpu_threads=torch.get_num_threads(),neural_forwards=0,training_updates=0,split='val',source_parity=provenance,cells=[],pixel_oracles={})
for spec in CATALOG['tasks']:
    for cell in spec['conditions']:
        task,cid=spec['id'],cell['id']
        suite=SuiteStream('val');seed=suite.stream_seed(task,cid)
        klass=TaskStream if spec['group']=='sensory' else VariantStream if task=='orientation_ring' else SpatialBatteryStream
        native=klass(seed,'val')
        n=200 if task=='krauzlis_cued_motion' else 32
        labels=[];events=Counter();case_counts=Counter();margins=[];first=last=0;oracle=[];angle_errors=[];recognition_min_dist=[]
        for start in range(0,n,4):
            x,y,meta=suite.batch(4,task,cid)
            nx,ny,nm=native.batch(4,task) if spec['group']=='sensory' else native.batch(4,task,cell['kwargs'])
            assert torch.equal(x,nx) and torch.equal(y,ny)
            assert all(all(row[k]==v for k,v in old.items()) for row,old in zip(meta,nm))
            assert x.device.type=='cpu' and x.dtype==torch.float32 and torch.isfinite(x).all() and 0<=x.min()<=x.max()<=1
            expected_frames=2 if spec['group']=='sensory' else 4 if task=='orientation_ring' else frame_count(task,cell['kwargs'])
            assert x.shape==(4,expected_frames,3,100,100)
            for im,label,m in zip(x.numpy(),y.tolist(),meta):
                labels.append(label)
                target=m.get('target_location')
                case_counts[str((label,target,m.get('cue_sign')))]+=1
                if task.startswith('orientation_'):
                    assert label==int(m['rotations_degrees'][target]*m['cue_sign']>0)
                    if cid=='D0':
                        # Decode the cue from pixels with exact native templates, not supplied metadata.
                        candidates=[(k,s) for k in range(4) for s in ([-1,1] if task=='orientation_cued' else [None])]
                        losses=[np.square(im[0]-local_cue(blank(),'recall','sample',k,s)).sum() for k,s in candidates]
                        inferred,sign=candidates[int(np.argmin(losses))]
                        assert min(losses)==0 and inferred==target
                        a=angle(im[2],inferred);b=angle(im[-1],inferred)
                        delta=(b-a+np.pi/2)%np.pi-np.pi/2
                        pred=int(delta>0) if task=='orientation_ring' else int(delta*sign>np.deg2rad(7.5))
                        oracle.append(pred==label)
                        for frame,truth in [(2,m['sample_angles_radians']),(-1,m['probe_angles_radians'])]:
                            est=angle(im[frame],inferred)
                            angle_errors.append(float(np.rad2deg(abs((est-truth[inferred]+np.pi/2)%np.pi-np.pi/2))))
                elif task=='spatial_binding':
                    assert label==int(target in m['swapped_locations']) and len(m['swapped_locations'])==2
                    assert np.allclose(sorted(m['sample_angles_radians']),sorted(m['probe_angles_radians']))
                    if cid=='D0':
                        q=m['cue_frames'][0]
                        losses=[np.square(im[q]-local_cue(blank(),'recall','query',k)).sum() for k in range(4)]
                        inferred=int(np.argmin(losses));assert min(losses)==0 and inferred==target
                        a=angle(im[2],inferred);b=angle(im[-1],inferred)
                        delta=abs((b-a+np.pi/2)%np.pi-np.pi/2)
                        oracle.append(int(delta>np.pi/8)==label)
                elif task=='motion_duration_cued':
                    dirs=np.array(m['directions_by_patch']);counts=np.array(m['duration_counts_by_patch'])
                    assert np.array_equal(counts,np.stack([np.bincount(d,minlength=4) for d in dirs]))
                    assert label==counts[target].argmax() and (counts[target]==counts[target].max()).sum()==1
                    margins.append(int(np.sort(counts[target])[-1]-np.sort(counts[target])[-2]))
                    first+=int(dirs[target,0]==label);last+=int(dirs[target,-1]==label)
                elif task=='krauzlis_cued_motion':
                    events[m['event_type']]+=1;assert label==int(m['event_type']=='target')
                    assert np.array_equal(im[0],im[1])
                    assert np.array_equal(im[2],im[-1])
                elif task=='image_recognition':
                    study=im[m['study_frames']];probe=im[m['probe_frames'][0]]
                    hits=[np.array_equal(frame,probe) for frame in study]
                    assert any(hits)==bool(label)
                    assert all(np.array_equal(im[p],probe) for p in m['probe_frames'])
                    if len(study):recognition_min_dist.append(dict(label=label,min_pixel_mse=float(np.square(study-probe).mean((1,2,3)).min())))
        counts=np.bincount(labels,minlength=spec['classes']).tolist()
        wanted=[86,114] if task=='krauzlis_cued_motion' else [32,0] if task=='image_recognition' and cell['kwargs']['load']==0 else [n//spec['classes']]*spec['classes']
        assert counts==wanted
        row=dict(task=task,cell=cid,n=n,frames=expected_frames,class_counts=counts,native_exact=True,native_metadata_preserved=True,case_counts=dict(case_counts))
        if events:row['events']=dict(events);assert events==dict(target=114,foil=58,catch=28)
        if margins:row.update(count_margin_histogram=dict(Counter(margins)),first_equals_winner=first,last_equals_winner=last)
        if recognition_min_dist:row['recognition_min_pixel_mse']=recognition_min_dist
        if oracle:
            report['pixel_oracles'][task]=dict(n=len(oracle),correct=sum(oracle),accuracy=float(np.mean(oracle)),target_decoded_from_pixels=True,angle_error_max_degrees=max(angle_errors) if angle_errors else None)
        report['cells'].append(row)
        save('task_native_checks.json',report)
        print(task,cid,'passed',counts,flush=True)

# Cue/phase precision in unscaled native input pixels (one spatial pixel, not RGB elements).
ref=visual_cues(blank(),'recall','sample')
minus=local_cue(blank(),'recall','sample',0,-1);plus=local_cue(blank(),'recall','sample',0,1)
ring=local_cue(blank(),'recall','sample',0)
report['cue_pixels']=dict(plus_minus_different_spatial_pixels=int(np.any(plus!=minus,axis=0).sum()),
    minus_vs_no_cue=int(np.any(minus!=ref,axis=0).sum()),plus_vs_no_cue=int(np.any(plus!=ref,axis=0).sum()),
    ring_vs_no_cue=int(np.any(ring!=ref,axis=0).sum()),
    ring_range=[float(ring.min()),float(ring.max())],signed_glyph_xy_bbox=[22,4,31,13])
report['phase_pixels']={phase:dict(non_gray_spatial_pixels=int(np.any(visual_cues(blank(),'recall',phase)!=.5,axis=0).sum()),
    differences_from_sample=int(np.any(visual_cues(blank(),'recall',phase)!=ref,axis=0).sum())) for phase in ['sample','ignore','query','report']}
# Same latent sample angle is not the same raster: independent nuisance draws.
suite=SuiteStream('val');im,y,m=suite.batch(4,'orientation_cued','D0')
report['sample_repeats']=dict(exactly_identical=[bool(torch.equal(row[1],row[2])) for row in im],mean_squared_difference=[float((row[1]-row[2]).square().mean()) for row in im])
report['expected_cells']=len(expected);report['passed_cells']=len(report['cells'])
assert {(r['task'],r['cell']) for r in report['cells']}==expected and len(report['cells'])==35
save('task_native_checks.json',report)
print(json.dumps({k:report[k] for k in ['passed_cells','pixel_oracles','cue_pixels','phase_pixels','sample_repeats']},indent=2))
