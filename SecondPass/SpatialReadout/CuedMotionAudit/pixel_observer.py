"""CPU-only image observers; no model imports, training, checkpoint or network IO.
Fixed geometry, native speeds, and task event timing are diagnostic privileges.
True metadata is accepted only by scoring code, never observer functions.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
D_CENTERS = ((27,27),(73,27),(27,73),(73,73))
K_CENTERS = ((20,50),(80,50))
VECTORS = ((1,0),(0,-1),(-1,0),(0,1))

def duration_scores(a,b):
    """Overlap correlation averaged across 1/2-pixel candidates; no true speed."""
    scores=[]
    for dx,dy in VECTORS:
        c=[]
        for distance in (1,2):
            shifted=ndimage.shift(a,(dy*distance,dx*distance),order=0,mode='constant',prefilter=False)
            c.append(float((shifted*b).sum()))
        scores.append(max(c))
    return np.asarray(scores)

def decode_cue(image, centers, radius, width):
    yy,xx=np.mgrid[:100,:100]
    scores=[float(image[np.abs(np.hypot(xx-x,yy-y)-radius)<width].mean()) for x,y in centers]
    return int(np.argmax(scores)), scores

def duration_observe(frames):
    gray=frames[:,0]
    target,cue=decode_cue(gray[0],D_CENTERS,14,.8)
    scores=[]
    for x,y in D_CENTERS:
        patch=gray[1:10,y-13:y+14,x-13:x+14]-.5
        yy,xx=np.mgrid[-13:14,-13:14]
        patch=patch*(xx*xx+yy*yy<=12.3**2) # excludes every static ring pixel
        scores.append([duration_scores(a,b).tolist() for a,b in zip(patch[:-1],patch[1:])])
    scores=np.asarray(scores)
    directions=scores.argmax(-1)
    counts=np.array([np.bincount(d,minlength=4) for d in directions])
    # Ties in estimated counts: summed within-transition normalized evidence,
    # then lowest class ID. Renderer ground truth itself never permits ties.
    evidence=(scores/(scores.sum(-1,keepdims=True)+1e-12)).sum(1)
    tied=np.flatnonzero(counts[target]==counts[target].max())
    winner=int(tied[np.argmax(evidence[target,tied])])
    return dict(prediction=winner,decoded_target=target,cue_scores=cue,
                transition_scores=scores.tolist(),directions=directions.tolist(),
                counts=counts.tolist(),estimated_tie=len(tied)>1)

def isolated_centroids(a):
    labels,n=ndimage.label(a>1e-6,structure=np.ones((3,3)))
    if not n:return np.empty((0,2))
    indices=np.arange(1,n+1)
    masses=ndimage.sum(a,labels,indices)
    good=indices[(masses>.475)&(masses<.485)]
    if not len(good):return np.empty((0,2))
    return np.asarray(ndimage.center_of_mass(a,labels,good))[:,::-1]

def patch_flow(a,b):
    """Sparse optical flow via isolated bilinear-dot centroid matching.
    Known .375-pixel speed rejects birth/death/reset matches; no true dot IDs.
    """
    p,q=isolated_centroids(a),isolated_centroids(b)
    if not len(p) or not len(q):return np.empty((0,2))
    dist=np.linalg.norm(q[None]-p[:,None],axis=2)
    j=dist.argmin(1)
    i=np.arange(len(p))
    good=(dist.argmin(0)[j]==i)&(abs(dist[i,j]-.375)<.015)
    return q[j[good]]-p[good]

def krauzlis_observe(frames,baseline):
    gray=frames[:,0]
    target,cue=decode_cue(gray[0],K_CENTERS,9.925,.7)
    angles=[];counts=[];velocities=[]
    for x,y in K_CENTERS:
        patch=gray[7:8+baseline+8,y-11:y+12,x-11:x+12]-.5
        flow=[patch_flow(a,b) for a,b in zip(patch[:-1],patch[1:])]
        angles.append([]); counts.append([len(v) for v in flow]);velocities.append([v.tolist() for v in flow])
        for section in (flow[:baseline],flow[baseline:]):
            v=np.concatenate(section)
            mean=v.mean(0) if len(v) else np.array([0.,0.])
            angles[-1].append(float(np.rad2deg(np.arctan2(mean[1],mean[0]))))
    angles=np.asarray(angles)
    changes=(angles[:,1]-angles[:,0]+180)%360-180
    return dict(decoded_target=target,cue_scores=cue,estimated_angles_degrees=angles.tolist(),
                signed_changes_degrees=changes.tolist(),score=float(abs(changes[target])),
                flow_match_counts=counts,flow_vectors=velocities)

def metrics(rows,classes):
    cm=np.zeros((classes,classes),int)
    for r in rows:cm[r['label'],r['prediction']]+=1
    denom=cm.sum(1)
    recalls=np.divide(cm.diagonal(),denom,out=np.full(classes,np.nan),where=denom>0)
    result=dict(n=len(rows),correct=int(cm.trace()),accuracy=float(cm.trace()/len(rows)),
                balanced_accuracy=float(np.nanmean(recalls)),confusion=cm.tolist(),class_n=denom.tolist())
    if classes==2:
        pos=np.array([r['score'] for r in rows if r['label']==1])
        neg=np.array([r['score'] for r in rows if r['label']==0])
        result['auc']=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos)*len(neg) else None
    return result

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False))

def append(name,row):
    with (OUT/name).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def raster_stats(frames,task,baseline=None):
    gray=frames[:,0]
    selected=gray[1:10] if task=='motion_duration_cued' else gray[7:8+baseline+8]
    diff=np.diff(selected,axis=0)
    stats=dict(transition_count=len(diff),changed_pixels_per_transition=(diff!=0).sum((1,2)).tolist(),
               mean_absolute_change=float(abs(diff).mean()),max_absolute_change=float(abs(diff).max()))
    # A disclosed nonlearned averaging surrogate, NOT the actual strided CNN.
    stats['average_pool_difference_rms']={str(s):float(np.sqrt(np.mean(np.diff(selected[:,:96,:96].reshape(len(selected),96//s,s,96//s,s).mean((2,4)),axis=0)**2))) for s in (1,2,4,8,16)}
    return stats

def run():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    from SecondPass.TaskSuite.suite import SuiteStream
    start=time.monotonic()
    config=dict(duration_n=128,duration_seed=92026092901,k_calibration_n=100,k_calibration_seed=92026092902,
                k_test_n=200,k_test_seed=92026092903,baseline=20,threads=1,device='cpu',torch=torch.__version__,
                threshold_grid_degrees=list(range(1,41)),threshold_selection='max calibration BA; ties smallest threshold',
                privileges=['fixed geometry','native isolated-dot mass/speed','reference and event boundary timing','external storage of all frames'],
                duration_tie_rule='max sum of transition-normalized evidence among count ties, then lowest ID',
                krauzlis_flow='8-connected components mass .475-.485; mutual nearest-centroid matches speed .375 +/- .015; mean pre/post flow angle')
    sources=['WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py','PreAttentiveVision/neuroscience_stimuli.py','SecondPass/TaskSuite/suite.py','SecondPass/TaskSuite/catalog.json','WorkingMemory/PlainBaseline/accum.py','SecondPass/SpatialReadout/model.py','ANALYSIS_SOP.md']
    config['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    config['observer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump('pixel_results_config.json',config)
    verification={}
    for task,cell,cfg in [('motion_duration_cued','D0',{'delay':0}),('motion_duration_cued','D24',{'delay':24}),('krauzlis_cued_motion','B20',{'baseline_transitions':20})]:
        adapter=SuiteStream('test');native=SpatialBatteryStream(adapter.stream_seed(task,cell),'test')
        a,ay,am=adapter.batch(4,task,cell);b,by,bm=native.batch(4,task,cfg)
        assert torch.equal(a,b) and torch.equal(ay,by)
        assert all(all(ar[k]==v for k,v in br.items()) for ar,br in zip(am,bm))
        verification[task+'/'+cell]=dict(n=4,pixels_equal=True,labels_equal=True,native_metadata_equal=True)
    dump('pixel_results_verification.json',verification)
    duration=[]
    s0=SpatialBatteryStream(config['duration_seed'],'test');s24=SpatialBatteryStream(config['duration_seed'],'test')
    for index in range(config['duration_n']):
        x,y,meta=s0.batch(1,'motion_duration_cued',{'delay':0});xx,yy,mm=s24.batch(1,'motion_duration_cued',{'delay':24})
        a=x[0].numpy();b=xx[0].numpy();m=meta[0]
        assert np.array_equal(a[:10],b[:10]) and np.array_equal(a[-1],b[-1]) and int(y[0])==int(yy[0])
        obs=duration_observe(a) # No metadata crossing the observer interface.
        truth=np.asarray(m['directions_by_patch']);counts=np.array([[(d==k).sum() for k in range(4)] for d in truth])
        target=m['target_location'];independent=int(np.argmax(counts[target]))
        assert independent==int(y[0]) and (counts[target]==counts[target].max()).sum()==1
        assert counts.tolist()==m['duration_counts_by_patch']
        row=dict(index=index,trial_id=m['trial_id'],label=int(y[0]),**obs,metadata=m,
                 independent_label=independent,paired_D24_equal=True,raster_sha256=digest(a),
                 true_count_margin=int(np.sort(counts[target])[-1]-np.sort(counts[target])[-2]),
                 transition_correct=int((np.array(obs['directions'])[target]==truth[target]).sum()),
                 all_patch_transition_correct=int((np.array(obs['directions'])==truth).sum()),raster=raster_stats(a,'motion_duration_cued'))
        duration.append(row);append('pixel_results_duration.jsonl',row)
    dsum=metrics(duration,4)
    dsum.update(cue_correct=sum(r['decoded_target']==r['metadata']['target_location'] for r in duration),
                transition_correct=sum(r['transition_correct'] for r in duration),transition_n=len(duration)*8,
                all_patch_transition_correct=sum(r['all_patch_transition_correct'] for r in duration),all_patch_transition_n=len(duration)*32,
                count_ties=sum(r['estimated_tie'] for r in duration),D24_pairs_equal=len(duration),independent_labels_verified=len(duration))
    for key in ('step_pixels','true_count_margin'):
        values=sorted({r['metadata'][key] if key=='step_pixels' else r[key] for r in duration})
        dsum[key]={str(v):metrics([r for r in duration if (r['metadata'][key] if key=='step_pixels' else r[key])==v],4) for v in values}
    summary=dict(duration=dsum,verification=verification,status='duration_complete')
    dump('pixel_results_summary.json',summary)
    print(json.dumps({'duration':dsum}),flush=True)
    calibration=[];test=[];threshold=None
    for split,n,seed,rows in [('calibration',config['k_calibration_n'],config['k_calibration_seed'],calibration),('test',config['k_test_n'],config['k_test_seed'],test)]:
        stream=SpatialBatteryStream(seed,'test')
        for index in range(n):
            x,y,meta=stream.batch(1,'krauzlis_cued_motion',{'baseline_transitions':20});a=x[0].numpy();m=meta[0]
            obs=krauzlis_observe(a,20)
            delta=(np.array(m['postevent_means_degrees'])-m['baseline_means_degrees']+180)%360-180
            independent=int(abs(delta[m['target_location']])>1e-6)
            assert independent==int(y[0])
            row=dict(index=index,trial_id=m['trial_id'],label=int(y[0]),**obs,metadata=m,independent_label=independent,
                     prediction=int(obs['score']>threshold) if threshold is not None else 0,
                     raster_sha256=digest(a),raster=raster_stats(a,'krauzlis_cued_motion',20))
            rows.append(row);append('pixel_results_k_'+split+'.jsonl',row)
        if split=='calibration':
            curve=[]
            for t in config['threshold_grid_degrees']:
                for r in rows:r['prediction']=int(r['score']>t)
                curve.append(dict(threshold=t,**metrics(rows,2)))
            best=max(curve,key=lambda z:z['balanced_accuracy']);threshold=best['threshold']
            dump('pixel_results_threshold.json',dict(threshold=threshold,curve=curve,selection_split='calibration_only',frozen_before_test=True))
            print(json.dumps({'threshold':threshold,'calibration':best}),flush=True)
    for rows in (calibration,test):
        for r in rows:r['prediction']=int(r['score']>threshold)
    # Rewrite calibration predictions after threshold selection; preserve raw scores.
    (OUT/'pixel_results_k_calibration.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in calibration))
    ksum=metrics(test,2);ksum['threshold']=threshold;ksum['calibration']=metrics(calibration,2)
    ksum['cue_correct']=sum(r['decoded_target']==r['metadata']['target_location'] for r in test)
    ksum['independent_labels_verified']=len(test)+len(calibration)
    ksum['events']={event:dict(n=len(group),positive_predictions=sum(r['prediction'] for r in group),rate=float(np.mean([r['prediction'] for r in group]))) for event in ('target','foil','catch') if (group:=[r for r in test if r['metadata']['event_type']==event])}
    ksum['target_side']={str(t):metrics([r for r in test if r['metadata']['target_location']==t],2) for t in (0,1)}
    ksum['magnitude']={str(t):metrics([r for r in test if r['metadata']['event_magnitude_degrees']==t],2) for t in (26,28)}
    ksum['angles_mae_degrees']={phase:float(np.mean([abs((np.asarray(r['estimated_angles_degrees'])[:,i]-np.asarray(r['metadata'][truth])+180)%360-180).mean() for r in test])) for i,phase,truth in ((0,'pre','baseline_means_degrees'),(1,'post','postevent_means_degrees'))}
    summary.update(krauzlis=ksum,status='complete',elapsed_seconds=time.monotonic()-start)
    for task,rows in [('duration',duration),('krauzlis',test)]:
        summary[task]['raster']={key:float(np.mean([r['raster'][key] for r in rows])) for key in ('mean_absolute_change','max_absolute_change')}
        summary[task]['raster']['mean_changed_pixels_per_transition']=float(np.mean([np.mean(r['raster']['changed_pixels_per_transition']) for r in rows]))
        summary[task]['raster']['average_pool_difference_rms']={str(s):float(np.mean([r['raster']['average_pool_difference_rms'][str(s)] for r in rows])) for s in (1,2,4,8,16)}
    dump('pixel_results_summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    if any(OUT.glob('pixel_results_duration.jsonl')):raise SystemExit('Existing results: refusing overwrite/re-run into same namespace')
    run()
