"""Bounded CPU-only frozen diagnostic. Nothing here updates deployed parameters."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
import sys, time, json, hashlib, inspect, signal, argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/Users/jonathanmorgan/Desktop/VisualAttentionWorkingMemory')
sys.path.insert(0,str(ROOT))
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
from SecondPass.SpatialReadout.SpatialConsolidation.model import SpatialConsolidation
TASK='krauzlis_cued_motion'
RUN=Path('/Users/jonathanmorgan/VAWMRuntime/krauzlis_wholemodel_fresh01/run')
CHECKPOINT=RUN/'validation_checkpoint_002297.pt'
PHASES=['cue','gap','baseline','post','final']
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def captured_batch(stream,b):
    """Read actual per-dot angles immediately before native subpixel updates.
    Python tracing does not replace renderer operations or consume any RNG.
    """
    fn=SpatialBatteryStream._krauzlis
    lines,start=inspect.getsourcelines(fn)
    line=start+next(i for i,s in enumerate(lines) if 'directions=means[k]+offsets[k]' in s)
    records=[]
    def trace(frame,event,arg):
        if frame.f_code is fn.__code__ and event=='line' and frame.f_lineno==line:
            v=frame.f_locals;k=v['k']
            records.append((v['t'],k,(v['means'][k]+v['offsets'][k]).copy()))
        return trace
    prior=sys.gettrace();sys.settrace(trace)
    try:x,y,m=stream.batch(1,TASK,dict(baseline_transitions=b))
    finally:sys.settrace(prior)
    angles=np.empty((b+8,2,16),np.float64)
    assert len(records)==2*(b+8)
    for t,k,a in records:angles[t,k]=a
    return x,y,m,angles

def circular_targets(angles,b):
    return np.stack([np.angle(np.exp(1j*a).mean(axis=(0,2))) for a in (angles[:b],angles[b:])])

def make_pair(x,m):
    swap=x.clone();target=1-m['target_location'];yy,xx=np.mgrid[:100,:100]
    ring=np.abs(np.sqrt((xx-(20 if target==0 else 80))**2+(yy-50)**2)-(8.125+1.8))<.7
    cue=x[0,2].clone();cue[:,ring]=.95;swap[0,:2]=cue
    labels=np.array([m['label'],int(m['changed_patch']==target)],dtype=np.int64)
    return torch.cat((x,swap),0),labels

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

class Extractor:
    """Native forward + observational hooks. Fixed geometry, no cue selection."""
    def __init__(self,model):
        self.model=model;self.handles=[];self.cache={};self.t=-1
        self.handles.append(model.blocks[0].register_forward_pre_hook(self.tick))
        for name,module in [('cnn25',model.blocks[1]),('kda25',model.acc[0]),('kda13',model.acc[1]),('kda7',model.acc[2]),('memory',model.spatial_gru),('terminal',model.consolidation)]:
            self.handles.append(module.register_forward_hook(self.hook(name)))
    def tick(self,*args):self.t+=1
    def hook(self,name):
        def capture(module,args,out):
            if self.t not in self.times:return
            value=out[0] if isinstance(out,tuple) else out
            if name in ('memory','terminal'):value=value.flatten(1)
            else:
                # Original pixel centers:20/80,50. Encoder strides4/8/16.
                stride={'cnn25':4,'kda25':4,'kda13':8,'kda7':16}[name]
                ry=int(np.floor(50/stride+.5));cols=[int(np.floor(x/stride+.5)) for x in (20,80)]
                value=torch.cat([value[:,:,ry-1:ry+2,c-1:c+2].flatten(1) for c in cols],1)
            self.cache[(name,self.times[self.t])]=value.cpu().numpy().copy()
        return capture
    def __call__(self,x):
        b=x.shape[1]-17;self.times={1:'cue',6:'gap',b+7:'baseline',b+15:'post',b+16:'final'}
        self.t=-1;self.cache={}
        with torch.inference_mode():logits=self.model(x,TASK).cpu().numpy().copy()
        return dict(self.cache),logits
    def close(self):
        for h in self.handles:h.remove()

def ba(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
def auc(y,s):
    a=s[y==1];b=s[y==0]
    return float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))
def class_metrics(y,s):
    p=s.argmax(1)
    r={'balanced_accuracy':ba(y,p),'accuracy':float(np.mean(y==p)),'n':len(y)}
    if s.shape[1]==2:r['auc']=auc(y,s[:,1]-s[:,0])
    r['confusion']=[[int(np.sum((y==i)&(p==j))) for j in range(s.shape[1])] for i in range(s.shape[1])]
    return r

def ridge_candidates(x,y,alpha):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
    z=(x-mean)/scale;ym=y.mean(0);yc=y-ym
    # Use the smaller exact ridge system. No optimizer, no sklearn dependency.
    dual=z.shape[0]<z.shape[1]
    gram=z@z.T if dual else z.T@z
    eig,q=np.linalg.eigh(gram);eig=np.maximum(eig,0)
    rhs=q.T@(yc if dual else z.T@yc)
    result=[]
    for a in alpha:
        w=q@(rhs/(eig[:,None]+a));w=z.T@w if dual else w
        result.append(dict(mean=mean,scale=scale,weight=w,intercept=ym,alpha=a))
    return result

def predict(fit,x):return ((x-fit['mean'])/fit['scale'])@fit['weight']+fit['intercept']
def angular_error(y,s):
    pred=np.arctan2(s[:,1],s[:,0]);return np.abs(np.angle(np.exp(1j*(pred-y))))*180/np.pi

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args();OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    # Refuse cap renewal, including after failure. New run requires explicit authorization.
    budget_path=OUT/'budget.json'
    if budget_path.exists():raise RuntimeError('Existing nonrenewable budget: refusing a new run')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(19381)
    sources=['WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py','SecondPass/SpatialReadout/model.py','SecondPass/SpatialReadout/SpatialConsolidation/model.py']
    cfg=json.loads((RUN/'config.json').read_text());source_receipt={}
    for f in sources:
        p=ROOT/f;ref=RUN.parent/'repo'/f
        assert p.read_bytes()==ref.read_bytes();assert sha(p)==cfg['source_hashes'][str(ref)]
        source_receipt[f]=sha(p)
    checkpoint_sha=sha(CHECKPOINT);c=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert c['state']['step']==2297
    classes={k.split('.')[1]:v.shape[0] for k,v in c['model'].items() if k.startswith('heads.') and k.endswith('.weight')}
    model=SpatialConsolidation(classes);model.load_state_dict(c['model'],strict=True);model.eval();model.requires_grad_(False)
    before=state_hash(model)
    dump(OUT/'identity.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_sha,step=2297,episodes=c['state']['episodes'],source_hashes=source_receipt,extractor_sha256=sha(__file__),state_sha256=before,device='cpu',threads=torch.get_num_threads(),provenance=c['provenance']))
    del c
    started=time.time();deadline=started+1800
    dump(budget_path,dict(started=started,deadline=deadline,total_seconds=1800,renewable=False,includes='profile, parity, extraction, fits, test scoring, verification, report'))
    def expired(signum,frame):raise TimeoutError('Nonrenewable 1800-second total cap expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,1800)
    ex=Extractor(model);profile=[];parity=[];renderer=[]
    for b in (12,20,28):
        t=time.time();s=SpatialBatteryStream(98100+b,'test');x,y,m,angles=captured_batch(s,b)
        native=SpatialBatteryStream(98100+b,'test');xx,yy,mm=native.batch(1,TASK,dict(baseline_transitions=b))
        assert torch.equal(x,xx) and torch.equal(y,yy) and s.state_dict()==native.state_dict()
        pair,labels=make_pair(x,m[0]);features,hooked=ex(pair)
        ex.close()
        with torch.inference_mode():direct=model(pair,TASK).numpy()
        assert np.array_equal(direct,hooked)
        parity.append(float(np.max(np.abs(direct-hooked))));renderer.append(True)
        ex=Extractor(model)
        # Time one representative paired extraction, including render and metadata.
        t=time.time();x,y,m,angles=captured_batch(s,b);pair,labels=make_pair(x,m[0]);ex(pair);profile.append(time.time()-t)
    # Conservative profile; maximum700 groups, minimum suitable pilot200groups.
    capacity=int((deadline-time.time()-450)/(max(profile)*1.5))
    total=min(700,(capacity//100)*100)
    assert total>=200, 'CPU profile cannot support minimal grouped study inside cap'
    ntest=max(50,(total//4)//25*25);nval=max(50,(total//7)//25*25);ntrain=total-ntest-nval
    counts={'train':ntrain,'val':nval,'test':ntest}
    spec=dict(counts=counts,paired_rows={k:2*v for k,v in counts.items()},profile_pair_seconds=profile,allocation_max_pair_seconds=max(profile)*1.5,reserve_seconds=450,pinned_at=time.time(),baseline_schedule=[12,20,28],seeds={'train':71003001,'val':71004001,'test':71005001},alphas=[1.,100.,10000.],phases=PHASES,parity_max_abs=parity,renderer_rng_parity=renderer,selection='validation only; test generated only after fits frozen',feature_scope='all fixed spatial sites; no cue-dependent input selection',bootstrap_groups=500)
    dump(OUT/'allocation.json',spec);print('PINNED',json.dumps(spec),flush=True)
    def extract(split):
        stream=SpatialBatteryStream(spec['seeds'][split],split);data={};meta=[];all_angles=[];logits=[]
        for i in range(counts[split]):
            if time.time()>deadline-240:raise TimeoutError('Reserve reached before extraction complete')
            b=(12,20,28)[i%3];x,y,m,angles=captured_batch(stream,b);pair,labels=make_pair(x,m[0]);f,log=ex(pair)
            assert torch.equal(pair[0,2:],pair[1,2:])
            direction=circular_targets(angles,b)
            for key,v in f.items():data.setdefault('__'.join(key),[]).append(v)
            gid=f'{split}/{i}'
            for variant in (0,1):
                mm=dict(m[0]);mm.update(group=gid,variant=variant,label=int(labels[variant]),target_location=m[0]['target_location'] if variant==0 else 1-m[0]['target_location'],actual_circular_angles_radians=direction.tolist(),movie_sha256=hashlib.sha256(pair[variant].numpy().tobytes()).hexdigest(),noncue_sha256=hashlib.sha256(pair[variant,2:].numpy().tobytes()).hexdigest())
                if variant:mm['event_type']='catch' if mm['changed_patch'] is None else 'target' if labels[variant] else 'foil'
                meta.append(mm)
            padded=np.full((36,2,16),np.nan);padded[:len(angles)]=angles;all_angles.append(padded);logits.append(log)
            if (i+1)%25==0:print('EXTRACT',split,i+1,'elapsed',round(time.time()-started,1),flush=True)
        data={k:np.concatenate(v) for k,v in data.items()};data['native_logits']=np.concatenate(logits)
        np.savez(OUT/f'{split}_features.npz',**data);np.savez(OUT/f'{split}_dot_angles.npz',angles=np.stack(all_angles))
        dump(OUT/f'{split}_metadata.json',meta)
        return data,meta
    train,tm=extract('train');val,vm=extract('val')
    def targets(meta):
        return {'cue':np.array([m['target_location'] for m in meta]),'changed':np.array([2 if m['changed_patch'] is None else m['changed_patch'] for m in meta]),'label':np.array([m['label'] for m in meta]),'angles':np.array([m['actual_circular_angles_radians'] for m in meta])}
    ty,vy=targets(tm),targets(vm)
    projections={}
    rng=np.random.default_rng(4123)
    # Both access conditions have equal 384 linear /96 quadratic input dimensions.
    for dim in (128,32):projections[dim]=rng.normal(size=(3136,dim))/np.sqrt(dim)
    np.savez(OUT/'projections.npz',**{str(k):v for k,v in projections.items()})
    def access(data,kind,quadratic=False):
        p=projections[32 if quadratic else 128]
        phases=['final']*3 if kind=='final' else ['cue','baseline','post']
        # Final access uses three disjoint fixed coordinate permutations of final state;
        # same number of coordinates and polynomial coefficients as phase access.
        blocks=[]
        for j,phase in enumerate(phases):
            a=data['memory__'+phase]
            blocks.append(np.roll(a,1045*j,axis=1)@p)
        z=np.concatenate(blocks,1)
        if quadratic:
            i,j=np.triu_indices(z.shape[1]);z=np.concatenate([z,z[:,i]*z[:,j]],1)
        return z
    for data in (train,val):
        for kind in ('final','phases'):
            for quad in (False,True):data[f'access_{kind}_{"quadratic" if quad else "linear"}']=access(data,kind,quad)
    tasks=[]
    for site in ('cnn25','kda25','kda13','kda7','memory'):
        for phase in PHASES:tasks.append((f'{site}__{phase}','cue',None))
        for phase in ('baseline','post','final'):
            for patch in (0,1):tasks.append((f'{site}__{phase}','angle',(0 if phase=='baseline' else 1,patch)))
    for site in ('memory__final','terminal__final','access_final_linear','access_phases_linear','access_final_quadratic','access_phases_quadratic'):
        for target in ('changed','label'):tasks.append((site,target,None))
    # Shuffled-label controls for the exact same matched temporal access probes.
    tasks += [(f'access_{kind}_{q}','shuffled_label',None) for kind in ('final','phases') for q in ('linear','quadratic')]
    fits={};selection={};cache={};fitarr={}
    shuffled=np.random.default_rng(7721).permutation(counts['train'])
    perm=(2*shuffled[:,None]+np.arange(2)).reshape(-1)
    for index,(feature,target,arg) in enumerate(tasks):
        key=f'{feature}/{target}'+('' if arg is None else '/'+str(arg[1]))
        if target=='angle':
            tr=ty['angles'][:,arg[0],arg[1]];va=vy['angles'][:,arg[0],arg[1]];y=np.c_[np.cos(tr),np.sin(tr)]
        else:
            name='label' if target=='shuffled_label' else target
            tr=ty[name][perm] if target=='shuffled_label' else ty[name];va=vy[name];nclass=3 if target=='changed' else 2;y=np.eye(nclass)[tr]
        candidates=ridge_candidates(train[feature].astype(np.float64),y,spec['alphas'])
        scores=[]
        for fit in candidates:
            pred=predict(fit,val[feature]);scores.append(-float(angular_error(va,pred).mean()) if target=='angle' else ba(va,pred.argmax(1)))
        best=int(np.argmax(scores));fits[key]=(candidates[best],feature,target,arg)
        selection[key]=dict(alpha=spec['alphas'][best],validation_scores=scores,features=train[feature].shape[1],target=target)
        for k,v in candidates[best].items():fitarr[f'{index}_{k}']=np.asarray(v)
        selection[key]['archive_prefix']=str(index)
    np.savez(OUT/'fitted_probes.npz',**fitarr);dump(OUT/'selection.json',selection)
    dump(OUT/'test_freeze.json',dict(time=time.time(),selection_sha256=sha(OUT/'selection.json'),fits_sha256=sha(OUT/'fitted_probes.npz'),test_generated=False))
    print('FITS FROZEN',len(fits),'elapsed',time.time()-started,flush=True)
    test,mm=extract('test');ey=targets(mm)
    for kind in ('final','phases'):
        for quad in (False,True):test[f'access_{kind}_{"quadratic" if quad else "linear"}']=access(test,kind,quad)
    # Input-only final-state API: erase all earlier arrays and preserve actual scores.
    final_only={'memory__final':test['memory__final']}
    invariance={}
    for quad in (False,True):
        baseline=access(test,'final',quad);removed=access(final_only,'final',quad)
        poisoned={k:(v if k=='memory__final' else np.full_like(v,np.nan)) for k,v in test.items()}
        assert np.array_equal(baseline,removed) and np.array_equal(baseline,access(poisoned,'final',quad))
        invariance[str(quad)]=True
    metrics={};predictions={};rr=np.random.default_rng(9223);boot=rr.integers(0,counts['test'],size=(500,counts['test']));bootrows=(2*boot[:,:,None]+np.arange(2)).reshape(500,-1)
    for index,(key,(fit,feature,target,arg)) in enumerate(fits.items()):
        s=predict(fit,test[feature]);predictions[str(index)]=s
        if target=='angle':
            truth=ey['angles'][:,arg[0],arg[1]];err=angular_error(truth,s);means=err[bootrows].mean(1)
            result=dict(mean_absolute_circular_error_degrees=float(err.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),chance_reference_degrees=90,n=len(err))
        else:
            truth=ey['label' if target=='shuffled_label' else target];result=class_metrics(truth,s)
            bs=np.array([ba(truth[r],s[r].argmax(1)) for r in bootrows]);result['ba_ci95']=np.quantile(bs,[.025,.975]).tolist()
        result.update(selection[key]);metrics[key]=result
    native=test['native_logits'];base=np.arange(0,len(mm),2);metrics['native_original']=class_metrics(ey['label'][base],native[base]);metrics['native_paired']=class_metrics(ey['label'],native)
    native_margin=native[:,1]-native[:,0];delta=native_margin[1::2]-native_margin[::2];changed=ey['label'][1::2]!=ey['label'][::2]
    aligned=(2*ey['label'][1::2]-1)*delta
    metrics['paired_cue_effect']=dict(n_groups=counts['test'],changed_label_groups=int(changed.sum()),mean_absolute_margin_shift=float(np.abs(delta).mean()),mean_task_aligned_margin_shift_changed=float(aligned[changed].mean()),argmax_flips=int(np.sum(native[::2].argmax(1)!=native[1::2].argmax(1))),both_members_correct=int(np.sum(np.all((native.argmax(1)==ey['label']).reshape(-1,2),axis=1))),both_members_correct_changed=int(np.sum(np.all((native.argmax(1)==ey['label']).reshape(-1,2),axis=1)&changed)))
    for event in ('target','foil','catch'):
        ix=np.array([i for i in base if mm[i]['event_type']==event]);metrics['native_original'][event]=dict(n=len(ix),positive_rate=float(np.mean(native[ix].argmax(1)==1)))
    # Paired gain interval, with the same grouped resamples for temporal-access pairs.
    keylist=list(fits)
    for q in ('linear','quadratic'):
        a=predictions[str(keylist.index(f'access_final_{q}/label'))];b=predictions[str(keylist.index(f'access_phases_{q}/label'))]
        gain=[ba(ey['label'][r],b[r].argmax(1))-ba(ey['label'][r],a[r].argmax(1)) for r in bootrows]
        metrics[f'phase_minus_final_{q}']=dict(ba_gain=ba(ey['label'],b.argmax(1))-ba(ey['label'],a.argmax(1)),ci95=np.quantile(gain,[.025,.975]).tolist())
    predictions['native_logits']=native;predictions['labels']=ey['label'];predictions['cue']=ey['cue'];predictions['changed']=ey['changed'];predictions['angles']=ey['angles']
    np.savez(OUT/'test_predictions.npz',**predictions);dump(OUT/'metrics.json',metrics)
    # Replay every serialized fitted probe against held-out features.
    fa=np.load(OUT/'fitted_probes.npz');pa=np.load(OUT/'test_predictions.npz')
    replay=0.
    for index,(key,(fit,feature,target,arg)) in enumerate(fits.items()):
        loaded={k:fa[f'{index}_{k}'] for k in ('mean','scale','weight','intercept')}
        replay=max(replay,float(np.max(np.abs(predict(loaded,test[feature])-pa[str(index)]))))
    assert replay==0
    assert sha(CHECKPOINT)==checkpoint_sha and state_hash(model)==before
    assert all(not p.requires_grad for p in model.parameters())
    groups=[set(m['group'] for m in rows) for rows in (tm,vm,mm)];assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    receipt=dict(complete=True,elapsed_seconds=time.time()-started,deadline=deadline,within_cap=time.time()<deadline,checkpoint_hash_unchanged=True,state_hash_unchanged=True,probe_prediction_replay_max_abs=replay,final_access_erase_nan_invariance=invariance,counts=counts,zero_model_optimizer_steps=True,no_cloud_access=True,artifacts={p.name:sha(p) for p in OUT.glob('*.npz')})
    dump(OUT/'receipt.json',receipt)
    write_report(OUT,metrics,spec,receipt)
    journal=ROOT/'LabJournal/krauzlis-frozen-diagnostic.md'
    journal.write_text('# Native-angle frozen diagnostic — selected2297\n\n'+(OUT/'REPORT.md').read_text().split('## Protocol')[0]+'\nFull report: `SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/FrozenDiagnostic/REPORT.md`.\nNo deployed training or cloud access.\n')
    receipt['elapsed_seconds']=time.time()-started;receipt['within_cap']=time.time()<deadline;dump(OUT/'receipt.json',receipt)
    print('COMPLETE',json.dumps(receipt),flush=True)
    signal.setitimer(signal.ITIMER_REAL,0)

def write_report(out,m,s,r):
    lines=['# Frozen diagnostic: native-angle Krauzlis selected2297','',
    '**Measured, exploratory, one frozen trained observer.** No deployed optimizer updates or cloud access. Native task: report a change only in the ring-cued patch; foil and catch reports are errors. This is not a cue-validity experiment.','',
    '## Behavioral result',
    f"Fresh native-original held-out episodes: n={s['counts']['test']}, BA={m['native_original']['balanced_accuracy']:.3f}, AUC={m['native_original']['auc']:.3f}. Paired variants are native-valid cue counterfactuals with altered event frequencies, not independent episodes.",
    '```json',json.dumps(m['paired_cue_effect'],indent=2),'```','',
    '## Cue accessibility through time','| Site | Cue | Gap | Baseline | Post | Final |','|---|---:|---:|---:|---:|---:|']
    for site in ('cnn25','kda25','kda13','kda7','memory'):
        lines.append('| '+site+' | '+' | '.join(f"{m[f'{site}__{p}/cue']['balanced_accuracy']:.3f}" for p in PHASES)+' |')
    lines += ['', 'Values are held-out balanced accuracy; detailed group-bootstrap intervals and all validation choices in metrics.json. Cue at frame1 is still visible; gap at6 has no cue in stack3. Later values test retained accessibility, not attention allocation.','',
    '## Physical change and correct-label accessibility','| Access / probe | Changed patch BA (left/right/catch) | Correct label BA | Label AUC |','|---|---:|---:|---:|']
    for site in ('memory__final','terminal__final','access_final_linear','access_phases_linear','access_final_quadratic','access_phases_quadratic'):
        lines.append(f"| {site} | {m[site+'/changed']['balanced_accuracy']:.3f} | {m[site+'/label']['balanced_accuracy']:.3f} | {m[site+'/label']['auc']:.3f} |")
    lines += ['', '## Actual per-dot circular direction','Mean absolute circular error in degrees; each pair is left / right. Uniform-angle reference is90°, and native change magnitude26/28°.','| Site | Baseline direction at baseline | Post direction at post | Post direction at final |','|---|---:|---:|---:|']
    for site in ('cnn25','kda25','kda13','kda7','memory'):
        vals=[' / '.join(f"{m[f'{site}__{p}/angle/{k}']['mean_absolute_circular_error_degrees']:.1f}" for k in (0,1)) for p in ('baseline','post','final')]
        lines.append('| '+site+' | '+' | '.join(vals)+' |')
    lines += ['', 'Targets are circular means over the actual16 per-dot movement angles over each complete baseline/post phase, captured before native .375px updates. They are not latent categorical proxies or patch means without dot offsets. Replacement jumps are not treated as motion vectors; offsets of reset dots enter their subsequent native movements. Dot-angle schedules are saved.','',
    '## Temporal-access controls','```json',json.dumps({k:v for k,v in m.items() if k.startswith('phase_minus') or '/shuffled_label' in k},indent=2),'```','',
    '## Protocol',
    f"Grouped counts pinned after profiling: {s['counts']}; two cue variants per group, all phases in same split. Native cycle proportions57/29/14 are retained for original trials; incomplete cycles may deviate. B12/B20/B28 are interleaved, not separate resampled copies. Train/val/test seeds: {s['seeds']}.",
    f"Profile paired extraction seconds: {s['profile_pair_seconds']}. One nonrenewable1800s cap covers profile through final report; completed in approximately{r['elapsed_seconds']:.1f}s (see receipt for final elapsed). CPU only, torch threads2/inter-op1, numerical libraries capped2, one process.",
    'Exact production sources matched archived runtime files and recorded hashes. Checkpoint2297 had73504 training episodes; parent training completed4595updates/147040episodes. Terminal4595 was not used. identity.json records full hashes and provenance.',
    'Native forward is untouched: centered stack3 → strided CNN → KDAs at25/13/7 → spatial projection/ConvGRU64×7×7 → terminal49-token spatial transformer → dense head. Observational hooks matched unhooked logits bit-for-bit for B12/B20/B28; instrumented renderer matched native pixels, labels and final RNG state. All weights eval/frozen; file and tensor hashes unchanged.',
    'Early CNN/KDA inputs are fixed3×3 neighborhoods around BOTH patch centers (stride4/8/16); no true cue chooses a patch. Feature dimensions: CNN1152, KDA576, full memory/terminal3136. Receptive fields and GroupNorm mix space; these sites do not prove localized coding. Early restricted sampling versus full memory prevents strong layerwise information-loss conclusions.',
    'Cue/gap/baseline/post/final endpoints are1/6/B+7/B+15/B+16. Final stack contains two post frames plus fixation, not cue/baseline. Mean directions aggregate full phase; endpoint decoding is an imperfect measurement of that aggregate.',
    'Ridge with intercept and train-only per-coordinate scaling; regularization1/100/10000 selected on validation BA (angle: minimum circular error), first candidate wins ties. No refit on validation, no held-out tuning. All models and choices frozen/hash-saved before generating test. Fits/predictions replayed exactly. Final-only representations passed actual removal and NaN poisoning of every earlier representation.',
    'Temporal comparisons: fixed random projections →384 dimensions for linear ridge;96 dimensions plus all upper-triangle quadratic products =4752 for quadratic ridge. Final condition uses3 fixed coordinate permutations of final memory; phase condition uses stored cue/baseline/post memory with identical projection/permutation rule. Same train groups, target supervision, candidate grid and learned coefficient counts. Storage of earlier states is external diagnostic memory, not available to deployed head. Fixed quadratic terms permit interactions but are not a supplied true-cue comparator. Shuffled controls permute entire training episode-pairs; val/test truth remains intact.',
    'Intervals:500 resamples of independent held-out groups, keeping paired variants together. One seed, modest catch counts, multiple exploratory probes, no multiplicity-adjusted claims. Exact margins retained in test_predictions.npz/native_logits.',
    '', '## Interpretation and limits',
    'Probe success establishes accessibility to that analysis-only probe, not deployed learning, causal use, or biological attention. Failure does not establish erasure: weak supervision, sample size, feature extraction, endpoint choice and ridge capacity can limit decoding. Compare actual angular precision with26/28° events before claiming useful fine-motion encoding. Even successful phase access would not show a new final head suffices.',
    'Classical validity effects, allocation maps, inhibition, microstimulation, causal localization and response times are not measured in this bounded diagnostic. No task teaching or architecture change was made.',
    '', '## Artifacts and reproducibility',
    '`diagnostic.py`, `test_diagnostic.py`; identity/allocation/budget/test_freeze/selection/receipt JSON; train/val/test metadata, features and per-dot schedules; fitted_probes/projections/test_predictions NPZ; metrics.json. Group IDs and movie/noncue hashes are included. Large local artifacts remain in this directory.',
    'Run with `/Users/jonathanmorgan/VAWMRuntime/recurrent_transformer_cpu_env/bin/python diagnostic.py --out NEW_AUTHORIZED_EMPTY_DIRECTORY` from this directory. Existing budget refuses renewal. Re-execution requires its own explicit authorization; saved features and fits permit audit without retraining.',
    '']
    (out/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
