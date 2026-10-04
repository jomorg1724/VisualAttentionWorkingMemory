"""Finite frozen-checkpoint analysis. Run as module with --prepare or --execute.
One nonrenewable 3600 s allowance includes profiling, scoring and rendering.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import argparse,copy,hashlib,json,platform,subprocess,time,traceback,sys,fcntl
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr
from SecondPass.SpatialReadout.model import SpatialReadout
from SecondPass.TaskSuite.suite import task_classes
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
from WorkingMemory.PlainBaseline.variants import VariantStream
from WorkingMemory.PlainBaseline.analysis.psych_stream import PsychOrientationStream
from PreAttentiveVision.neuroscience_stimuli import TaskStream,TASK_CLASSES
from .core import observe,masks,epoch_frames
from .stimuli import OrientationSweep,relocate
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
CHECKPOINT=Path('/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_continuation_01/artifacts/checkpoint_035039.pt')
EXPECTED='93762f7968a29092b8acce7ea9632937a23965160822fe98bda9b3e5844531e7'
SEED=986529730
SENSORY=list(TASK_CLASSES)
DEADLINE=None
COUNTS={'scored_presentations':0,'scored_frames':0,'calibration_presentations':0,'map_presentations':0,'profile_presentations':0}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def log(**row):
    row['time']=time.time();print(json.dumps(row),flush=True)
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def guard(reserve=300):
    if DEADLINE and time.time()>DEADLINE-reserve:raise TimeoutError(f'Fixed deadline reached (reserve {reserve}s)')
def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load_model():
    assert CHECKPOINT.stat().st_size==16764891 and sha(CHECKPOINT)==EXPECTED
    d=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    m=SpatialReadout(task_classes());m.load_state_dict(d['model'],strict=True);m.eval().requires_grad_(False)
    return m,d['provenance']
def cpu_smoke(out=OUT):
    torch.set_num_threads(2);model,provenance=load_model();before=model_hash(model)
    x,y,meta=SpatialBatteryStream(SEED,'test').batch(2,'orientation_cued',{'delay':4})
    with torch.inference_mode():
        direct=model(x,'orientation_cued');z,data=observe(model,x,'orientation_cued',meta,record=True)
        zero,_=observe(model,x,'orientation_cued',meta,intervention=dict(kind='inhibit',site='cued',epoch='encoding',dose=0))
    errors=[float(data[f's{s}_reconstruction_error'].max()) for s in range(3)]
    assert max(errors)<1e-4
    st=OrientationSweep(SEED);xx,yy,_=st.batch(2,'orientation_cued',{'delay':4});assert torch.equal(x,xx) and torch.equal(y,yy)
    old=PsychOrientationStream(seed=SEED);xo,yo,_=old.batch(2,'orientation_cued',{'delay':4})
    result=dict(checkpoint_sha256=EXPECTED,checkpoint_bytes=CHECKPOINT.stat().st_size,direct_wrapper_max_error=float((direct-z).abs().max()),zero_dose_max_error=float((direct-zero).abs().max()),implicit_reconstruction_max_errors=errors,model_immutable=before==model_hash(model),native_sweep_bit_parity=True,historical_psych_stream_bit_parity=torch.equal(x,xo),historical_note='Extra keep-site permutation changes RNG; not reused for native parity',state_sha256=before,provenance=provenance)
    assert result['direct_wrapper_max_error']==0 and result['zero_dose_max_error']==0 and result['model_immutable']
    dump(Path(out)/'cpu_smoke.json',result);return result


def process_check():
    lines=subprocess.check_output(['ps','-axo','pid,comm,args'],text=True).splitlines();blocked=[];workers=[]
    for line in lines:
        parts=line.strip().split(None,2)
        if len(parts)<3 or 'python' not in parts[1].lower():continue
        if 'CuedMotionAudit.model_diagnostic' in parts[2] or 'CuedMotionAudit/model_diagnostic.py' in parts[2]:blocked.append(line)
        if ('NeuroscienceAnalysis.run' in parts[2]) and int(parts[0])!=os.getpid():workers.append(line)
    result=dict(unix=time.time(),blocked_exact_motion_workers=blocked,other_neuroscience_workers=workers,pid=os.getpid(),cpu_threads=2)
    dump(OUT/'process_check.json',result)
    if blocked or workers:raise RuntimeError('Another local analysis worker active; no accelerator operation permitted')
    return result


def score(model,x,y,meta,task,group,condition='sham',intervention=None,extra=None):
    output=[]
    for start in range(0,len(x),8):
        guard();xx=x[start:start+8].to('mps');mm=meta[start:start+8]
        z,data=observe(model,xx,task,mm,intervention=intervention)
        zz=z.float().cpu().numpy();prob=z.softmax(-1).cpu().numpy();pred=zz.argmax(1)
        for i,m in enumerate(mm):
            row=dict(group=group,condition=condition,task=task,base_id=m.get('analysis_base_id',m['trial_id']),trial_id=m['trial_id'],label=int(y[start+i]),prediction=int(pred[i]),prob1=float(prob[i,1]),logits=zz[i].tolist(),correct=bool(pred[i]==y[start+i]),metadata=m)
            if extra:row.update(extra)
            output.append(row)
        if data.get('effects'):
            with (OUT/'data'/'achieved_interventions.jsonl').open('a') as f:f.write(json.dumps(dict(group=group,condition=condition,base_ids=[m.get('analysis_base_id',m['trial_id']) for m in mm],effects=data['effects']))+'\n')
        COUNTS['scored_presentations']+=len(xx);COUNTS['scored_frames']+=len(xx)*xx.shape[1]
    with (OUT/'data'/'trials.jsonl').open('a') as f:
        for r in output:f.write(json.dumps(r)+'\n')
    return output


def native(task,n,delay=0,seed=SEED):
    if task in SENSORY:return TaskStream(seed,'test').batch(n,task)
    if task=='orientation_ring':return VariantStream(seed,'test').batch(n,task,{'delay':delay})
    return SpatialBatteryStream(seed,'test').batch(n,task,{'delay':delay})


def save_maps(model,x,y,meta,task,name):
    arrays={};errors=[]
    for i in range(0,len(x),4):
        guard();z,data=observe(model,x[i:i+4].to('mps'),task,meta[i:i+4],record=True)
        data['logits']=z.cpu().numpy()
        for k,v in data.items():
            if k.endswith('masks') or k.endswith('frame'):arrays[k]=v
            else:arrays.setdefault(k,[]).append(v)
        errors.extend([float(data[f's{s}_reconstruction_error'].max()) for s in range(3)])
    data={k:np.concatenate(v,0) if isinstance(v,list) else v for k,v in arrays.items()}
    assert max(errors)<1e-4
    data['images']=x.numpy();data['labels']=y.numpy();data['targets']=np.array([m.get('target_location',-1) for m in meta]);data['cue_sign']=np.array([m.get('cue_sign',1) for m in meta])
    np.savez_compressed(OUT/'maps'/f'{name}.npz',**data)
    dump(OUT/'maps'/f'{name}.json',dict(task=task,n=len(x),metadata=meta,arrays={k:list(v.shape) for k,v in data.items()},axes='trial,time,y,x,head; final_read coefficient time is SOURCE; source2 coefficient time is READ',maximum_reconstruction_error=max(errors),spatial_coordinate_note='Cell centers mapped over 100px approximate receptive-field positions; not biological coordinates',allocation_comparator='N/A: no spatial cue' if task in SENSORY else 'target/future query, each foil separately, central matched background'))
    COUNTS['map_presentations']+=len(x);log(stage='maps',name=name,n=len(x),reconstruction_error=max(errors))


def calibration(model,n):
    features=[];rms=[];angles=[];ids=[]
    for i in range(0,n,8):
        guard();x,y,meta=native('orientation_cued',8,seed=SEED+10000+i)
        z,data=observe(model,x.to('mps'),'orientation_cued',meta,localizer=True)
        features.append(data['local_features']);rms.append(data['local_rms']);angles.extend([m['sample_angles_radians'] for m in meta]);ids.extend([m['trial_id'] for m in meta])
    X=np.concatenate(features).astype('float64');R=np.concatenate(rms);a=np.array(angles);Y=np.stack([np.cos(2*a),np.sin(2*a)],-1);half=n//2
    design=np.c_[Y[:half].reshape(-1,2),np.ones(half*4)]
    coeff=np.linalg.lstsq(design,X[:half].reshape(-1,32),rcond=None)[0]
    direction=coeff[1];direction=direction/np.sqrt(np.mean(direction**2));random=np.random.default_rng(SEED+128).normal(size=32);random-=direction*(random@direction)/(direction@direction);random/=np.sqrt(np.mean(random**2))
    # Independent validation by trial: location means remain clustered in saved rows.
    projection=np.einsum('bsc,c->bs',X[half:]-coeff[2],direction)/32
    corr=float(np.corrcoef(projection.ravel(),Y[half:,:,1].ravel())[0,1]);bytrial=np.mean(projection*Y[half:,:,1],1)
    rng=np.random.default_rng(717);boot=np.mean(bytrial[rng.integers(len(bytrial),size=(2000,len(bytrial)))],1)
    receipt=dict(n=n,fit_trials=half,validation_trials=n-half,seed=SEED+10000,feature_definition='25x25 emitted32 channels, last sample t2, mask-weighted mean per location',fit='OLS emitted feature = Bcos*cos(2theta)+Bsin*sin(2theta)+intercept; 32 features, 3 predictors; split by trial before pooling sites',direction='positive sin(2theta) encoding direction, RMS(channel)=1; does not encode task label or cue sign',validation_correlation=corr,validation_signed_product_mean=float(bytrial.mean()),validation_trial_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),independent_validation_passed=bool(corr>0 and np.quantile(boot,.025)>0),activation_rms=float(np.sqrt(np.mean(R[:half]**2))),amplitude_rule='dose * fit-set activation RMS * channel-RMS-one direction * smooth mask; positive and negative doses',random_direction='Gaussian, orthogonal to chosen feature direction, channel RMS1, fixed seed',fit_ids=ids[:half],validation_ids=ids[half:])
    np.savez_compressed(OUT/'data'/'calibration.npz',local_features=X,local_rms=R,angles=a,coefficient=coeff,direction=direction,random_direction=random,validation_projection=projection)
    dump(OUT/'calibration.json',receipt);COUNTS['calibration_presentations']+=n;log(stage='calibration',validation_r=corr,passed=receipt['independent_validation_passed'])
    return direction,random,receipt['activation_rms']


def causal_batch(n,task):
    if task=='spatial_binding':
        x,y,m=native(task,n,delay=12,seed=SEED+20000)
        for row in m:row['analysis_base_id']='binding-causal/'+row['trial_id']
        return x,y,m
    xs=[];ys=[];ms=[];schedule=[0,3,6,10,15,30,45,6]
    for i in range(n//8):
        mag=schedule[i%len(schedule)];st=OrientationSweep(SEED+30000+i,magnitude=mag);x,y,m=st.batch(8,task,{'delay':12})
        for row in m:row['analysis_base_id']=f'cued-causal/M{mag}/'+row['trial_id']
        xs.append(x);ys.append(y);ms.extend(m)
    return torch.cat(xs),torch.cat(ys),ms


def execute():
    global DEADLINE
    torch.set_num_threads(2)
    for p in ['data','maps','figures']:(OUT/p).mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'budget.json').exists():raise RuntimeError('Budget already established; refusing renewal or accidental repeated production')
    smoke=cpu_smoke();model,_=load_model();before=model_hash(model)
    # Exact process check immediately before first accelerator operation.
    check=process_check();start=time.time();DEADLINE=start+3600
    dump(OUT/'budget.json',dict(start_unix=start,deadline_unix=DEADLINE,cap_seconds=3600,origin='Immediately before first MPS model transfer, includes all profiles/statistics/render/report',pid=os.getpid(),nonrenewable=True))
    completion={'status':'running','errors':[],'completed_stages':[]}
    try:
        model.to('mps');x,y,m=native('orientation_cued',8,delay=12,seed=SEED+999)
        timing=[]
        for repeat in range(2):
            tick=time.perf_counter();z,_=observe(model,x.to('mps'),'orientation_cued',m);torch.mps.synchronize();timing.append(time.perf_counter()-tick)
        COUNTS['profile_presentations']=16
        # Forecast includes ~100k presented frames, recording and render reserve.
        seconds_per_frame=max(timing)/(8*16);full_projection=110000*seconds_per_frame*1.6+600
        factor=1 if full_projection<3000 else .5
        plan=dict(profile_seconds=timing,seconds_per_presented_frame=seconds_per_frame,conservative_full_projection_seconds=full_projection,baseline_n=int(128*factor),causal_n=int(64*factor),calibration_n=128,map_primary_n=16 if factor==1 else 8,sensory_map_n=2,chunk=8,delays=[0,4,12,24],magnitudes=[0,3,6,10,15,30,45],causal_delay=12,inhibition_sites=['cued','foil','background'],inhibition_epochs=['encoding','retention','probe'],inhibition_doses=[.5,1.],stimulation_doses=[-.5,.5,-1.,1.],stimulation_directions=['feature','random'],stimulation_epochs=['encoding','retention','probe'],binding='sham and site x encoding/retention/query/probe x suppression1, plus feature +/-1 at query/probe',budget_reserve_seconds=300,pinned_unix=time.time(),note='64 independent paired trials TOTAL per causal condition across a prespecified magnitude mixture, not64 per magnitude; no test-based selection of grid')
        # Harder fallback pinned before production, never decided on measured behavior.
        if full_projection*factor>3200:plan['stimulation_doses']=[-1.,1.]
        dump(OUT/'plan.json',plan);log(stage='profile_and_pin',plan=plan)
        with torch.inference_mode():
            direct=model(x.to('mps'),'orientation_cued');wrapped,data=observe(model,x.to('mps'),'orientation_cued',m,record=True);zero,_=observe(model,x.to('mps'),'orientation_cued',m,intervention=dict(kind='inhibit',site='cued',epoch='encoding',dose=0))
        parity=dict(direct_wrapper_max_error=float((direct-wrapped).abs().max().cpu()),zero_dose_max_error=float((direct-zero).abs().max().cpu()),reconstruction_errors=[float(data[f's{s}_reconstruction_error'].max()) for s in range(3)])
        assert parity['direct_wrapper_max_error']==0 and parity['zero_dose_max_error']==0 and max(parity['reconstruction_errors'])<1e-4
        dump(OUT/'accelerator_parity.json',parity)
        # Native diagnostic baseline and actual representative maps for ALL ten.
        for task in SENSORY+['orientation_ring','orientation_cued','spatial_binding']:
            ds=plan['delays'] if task in ('orientation_cued','spatial_binding','orientation_ring') else [0]
            for d in ds:
                guard();x,y,m=native(task,plan['baseline_n'],delay=d,seed=SEED+100+SENSORY.index(task) if task in SENSORY else SEED+110)
                score(model,x,y,m,task,'native',extra={'delay':d})
                if d==0:save_maps(model,x[:2],y[:2],m[:2],task,f'{task}_D0')
                if d==12 and task in ('orientation_cued','spatial_binding'):save_maps(model,x[:plan['map_primary_n']],y[:plan['map_primary_n']],m[:plan['map_primary_n']],task,f'{task}_D12')
                log(stage='native',task=task,delay=d,n=len(x))
        completion['completed_stages'].append('native_all10_and_maps')
        # Full magnitude-delay curves. Same native draw/RNG across magnitudes/delays.
        for d in plan['delays']:
            for mag in plan['magnitudes']:
                st=OrientationSweep(SEED+1000,magnitude=mag);x,y,m=st.batch(plan['baseline_n'],'orientation_cued',{'delay':d})
                for row in m:row['analysis_base_id']='psych/'+row['trial_id']
                score(model,x,y,m,'orientation_cued','psychometric',extra={'delay':d,'magnitude':mag})
                if d==12:
                    rx,ry,rm=relocate(x,m,st.raw)
                    score(model,rx,ry,rm,'orientation_cued','matched_relocation',extra={'delay':d,'magnitude':mag})
                log(stage='psychometric',delay=d,magnitude=mag,n=len(x))
        completion['completed_stages'].append('psychometrics_and_matched_cues')
        direction,random,rms=calibration(model,plan['calibration_n']);completion['completed_stages'].append('independent_calibration')
        grid=[dict(kind='inhibit',site=site,epoch=epoch,dose=dose) for site in plan['inhibition_sites'] for epoch in plan['inhibition_epochs'] for dose in plan['inhibition_doses']]
        grid.extend(dict(kind='stimulate',site=site,epoch=epoch,dose=dose,direction_name=name,direction=vec.tolist(),rms=rms) for site in plan['inhibition_sites'] for epoch in plan['stimulation_epochs'] for dose in plan['stimulation_doses'] for name,vec in [('feature',direction),('random',random)])
        binding=[dict(kind='inhibit',site=site,epoch=epoch,dose=1.) for site in plan['inhibition_sites'] for epoch in ['encoding','retention','query','probe']]
        binding.extend(dict(kind='stimulate',site=site,epoch=epoch,dose=dose,direction_name='feature',direction=direction.tolist(),rms=rms) for site in plan['inhibition_sites'] for epoch in ['query','probe'] for dose in [-1.,1.])
        dump(OUT/'intervention_grid.json',dict(orientation_cued=grid,spatial_binding=binding,substrate='acc[0].output post1x1 conv emission, before concat;32 channels,25x25; no underlying state change',mask=masks(25).tolist(),mask_mass=masks(25).sum((1,2)).tolist(),mask_support=(masks(25)>0).sum((1,2)).tolist(),timing='one update: encoding2, late pure blank last; probe last; binding query after retention',labels='binding cued means subsequently queried; unknown during retention'))
        for task,conditions in [('orientation_cued',grid),('spatial_binding',binding)]:
            x,y,m=causal_batch(plan['causal_n'],task)
            np.savez_compressed(OUT/'data'/f'{task}_causal_stimuli.npz',images=x.numpy(),labels=y.numpy())
            dump(OUT/'data'/f'{task}_causal_metadata.json',m)
            score(model,x,y,m,task,'causal',extra={'delay':12})
            for j,intervention in enumerate(conditions):
                guard();condition='_'.join(str(intervention.get(k,'')) for k in ['kind','site','epoch','dose','direction_name']).rstrip('_')
                score(model,x,y,m,task,'causal',condition=condition,intervention=intervention,extra={'delay':12,'intervention':{k:v for k,v in intervention.items() if k!='direction'}})
                log(stage='causal',task=task,done=j+1,total=len(conditions),condition=condition,n=len(x))
            completion['completed_stages'].append(task+'_causal_grid')
        completion['status']='measured'
    except Exception as exc:
        completion['status']='partial';completion['errors'].append(str(exc));traceback.print_exc();log(stage='exception',error=str(exc))
    finally:
        completion.update(counts=COUNTS,model_immutable=model_hash(model)==before,checkpoint_immutable=sha(CHECKPOINT)==EXPECTED,checkpoint_sha256=EXPECTED,model_state_sha256=before,evaluation_end_unix=time.time(),elapsed_before_render=time.time()-start,remaining_before_render=DEADLINE-time.time(),torch_version=torch.__version__,python=sys.version,platform=platform.platform(),pid=os.getpid())
        assert completion['model_immutable'] and completion['checkpoint_immutable']
        dump(OUT/'execution.json',completion)
        del model
        torch.mps.empty_cache()
        # Separate existing plotting interpreter: no installation or training env edits.
        left=max(1,DEADLINE-time.time()-5)
        try:
            result=subprocess.run(['/Library/Developer/CommandLineTools/usr/bin/python3',str(OUT/'render.py')],cwd=ROOT,timeout=left,check=True)
        except Exception as exc:
            completion['errors'].append('Rendering: '+str(exc));completion['status']='partial';dump(OUT/'completion.json',completion);raise
        log(stage='finished',total_seconds=time.time()-start,deadline=DEADLINE)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.prepare:print(json.dumps(cpu_smoke(),indent=2))
    elif args.execute:execute()
    else:parser.error('Choose --prepare or --execute')
