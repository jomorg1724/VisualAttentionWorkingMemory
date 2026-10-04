"""One local worker. Resume/renewal and test retuning intentionally unsupported."""
from upstream import *
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
# Thread limits are set before numpy/torch imports by the launcher/upstream.
# Do not require optional threadpoolctl in this existing offline environment.

def targets(meta):
    angles=np.array([m['actual_circular_angles_radians'] for m in meta])
    side=np.array([-1 if m['changed_patch'] is None else m['changed_patch'] for m in meta])
    return dict(angles=angles,delta=circular_change(angles[:,0],angles[:,1]),event=(side>=0).astype(int),side=side,label=np.array([m['label'] for m in meta]),cue=np.array([m['target_location'] for m in meta]))

def change_from_scores(s):
    a=s.reshape(-1,2,2)
    return np.rad2deg(np.arctan2(a[:,:,1],a[:,:,0]))

def change_error(truth,pred):
    return np.abs((pred-truth+180)%360-180)

def metrics_binary(y,s,threshold,boot):
    p=(s>threshold).astype(int)
    result=dict(n=len(y),balanced_accuracy=d.ba(y,p),auc=d.auc(y,s),confusion=[[int(np.sum((y==i)&(p==j))) for j in (0,1)] for i in (0,1)])
    bs=[]
    for r in boot:
        if len(np.unique(y[r]))==2:bs.append([d.ba(y[r],p[r]),d.auc(y[r],s[r])])
    result['ba_ci95']=np.quantile(np.array(bs)[:,0],[.025,.975]).tolist();result['auc_ci95']=np.quantile(np.array(bs)[:,1],[.025,.975]).tolist()
    return result

def main():
    budget=json.loads((OUT/'budget.json').read_text());deadline=budget['deadline'];start=budget['started']
    assert time.time()<deadline and not (OUT/'started.json').exists(), 'no renewal/re-run permitted'
    def expire(*args):raise TimeoutError('Immutable 1200 second budget exhausted')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    d.dump(OUT/'started.json',dict(time=time.time(),pid=os.getpid(),deadline=deadline))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(920134)
    spec=json.loads((OUT/'protocol.json').read_text());sites=spec['sites'];ident=json.loads((OLD/'identity.json').read_text());receipt=json.loads((OLD/'receipt.json').read_text())
    assert d.sha(d.CHECKPOINT)==ident['checkpoint_sha256']
    assert d.sha(OLD/'diagnostic.py')==ident['extractor_sha256']
    for p,h in ident['source_hashes'].items():assert d.sha(ROOT/p)==h and (ROOT/p).read_bytes()==(d.RUN.parent/'repo'/p).read_bytes()
    reused={}
    for split in ('train','val'):
        for suffix in ('features.npz','dot_angles.npz'):
            f=f'{split}_{suffix}';assert d.sha(OLD/f)==receipt['artifacts'][f];reused[f]=d.sha(OLD/f)
        reused[split+'_metadata.json']=d.sha(OLD/(split+'_metadata.json'))
    d.dump(OUT/'identity.json',dict(**ident,reused=reused,upstream_sha256=d.sha(OUT/'upstream.py'),runner_sha256=d.sha(__file__),pixel_source_sha256=d.sha(px.__file__),protocol_sha256=d.sha(OUT/'protocol.json')))
    train=dict(np.load(OLD/'train_features.npz'));val=dict(np.load(OLD/'val_features.npz'))
    tm=json.loads((OLD/'train_metadata.json').read_text());vm=json.loads((OLD/'val_metadata.json').read_text());ty=targets(tm);vy=targets(vm)
    for split,meta in [('train',tm),('val',vm)]:
        aa=np.load(OLD/f'{split}_dot_angles.npz')['angles']
        for i,m in enumerate(meta[::2]):
            b=m['baseline_transitions'];assert np.array_equal(d.circular_targets(aa[i,:b+8],b),np.array(m['actual_circular_angles_radians']))
    # Re-render exact ORIGINAL validation streams for pixel calibration; verify actual hashes.
    stream=d.SpatialBatteryStream(71004001,'val');pixel_val=[]
    for i in range(100):
        b=(12,20,28)[i%3];x,y,mm=stream.batch(1,d.TASK,dict(baseline_transitions=b))
        assert hashlib.sha256(x[0].numpy().tobytes()).hexdigest()==vm[2*i]['movie_sha256']
        pixel_val.append(pixel_summary(x[0].numpy()))
    np.savez(OUT/'pixel_validation.npz',angles=np.stack([r['angles'] for r in pixel_val]),cue=np.array([r['cue'] for r in pixel_val]))
    thresholds={}
    for w,name in enumerate(('full','endpoint')):
        a=np.stack([r['angles'][w] for r in pixel_val]);delta=np.abs(circular_change(a[:,0],a[:,1]));cue=np.array([r['cue'] for r in pixel_val]);scores={'event':delta.max(1),'label':delta[np.arange(100),cue]}
        for task,s in scores.items():
            yy=vy[task][::2];curve=[d.ba(yy,(s>t).astype(int)) for t in range(1,41)];thresholds[name+'/'+task]=dict(threshold=int(np.argmax(curve)+1),validation_ba=curve)
    d.dump(OUT/'pixel_thresholds.json',thresholds)
    # Deterministic, unsupervised projection shared across accesses and tasks.
    rng=np.random.default_rng(143021);projections={site:rng.normal(size=(train[site+'__final'].shape[1],32))/np.sqrt(32) for site in sites}
    np.savez(OUT/'projections.npz',**projections)
    fitarr={};selection={};permgroups=np.random.default_rng(22761).permutation(425);perm=(2*permgroups[:,None]+np.arange(2)).ravel()
    eventgroups=np.flatnonzero(ty['event'][::2]);shuffevents=np.random.default_rng(22762).permutation(eventgroups);eventperm=(2*shuffevents[:,None]+np.arange(2)).ravel()
    model_index=0
    for site in sites:
        for kind in ('phases','final'):
            x=access(train,site,kind,projections[site]);v=access(val,site,kind,projections[site])
            for task in ('change','event','side'):
                ix=np.flatnonzero(ty['event']) if task=='side' else np.arange(len(tm));vi=np.flatnonzero(vy['event']) if task=='side' else np.arange(len(vm))
                for shuffle in (False,True):
                    ti=(eventperm if task=='side' else perm) if shuffle else ix
                    if task=='change':
                        r=np.deg2rad(ty['delta'][ti]);target=np.stack([np.cos(r),np.sin(r)],axis=-1).reshape(-1,4)
                    else:target=np.eye(2)[ty[task][ti]]
                    cand=d.ridge_candidates(x[ix],target,[1.,100.,10000.]);scores=[]
                    for f in cand:
                        s=d.predict(f,v[vi]);scores.append(-float(change_error(vy['delta'][vi],change_from_scores(s)).mean()) if task=='change' else d.ba(vy[task][vi],s.argmax(1)))
                    best=int(np.argmax(scores));key=f'{site}/{kind}/{task}'+('/shuffled' if shuffle else '')
                    for k,z in cand[best].items():fitarr[f'{model_index}_{k}']=np.asarray(z)
                    selection[key]=dict(index=model_index,alpha=cand[best]['alpha'],validation_scores=scores,feature_dimensions=x.shape[1],training_rows=len(ix),training_groups=len(ix)//2,site=site,access=kind,target=task,shuffled=shuffle)
                    model_index+=1
            print('FIT',site,kind,'elapsed',time.time()-start,flush=True)
    np.savez(OUT/'fitted_models.npz',**fitarr);d.dump(OUT/'selection.json',selection)
    # Reuse previously fitted FULL-dimensional direction probes; no new choices.
    oldsel=json.loads((OLD/'selection.json').read_text());oldfits=np.load(OLD/'fitted_probes.npz');assert d.sha(OLD/'fitted_probes.npz')==receipt['artifacts']['fitted_probes.npz']
    d.dump(OUT/'test_freeze.json',dict(time=time.time(),test_generated=False,seed=spec['fresh_test_seed'],groups=spec['fresh_test_groups'],hashes={p:d.sha(OUT/p) for p in ('protocol.json','selection.json','fitted_models.npz','projections.npz','pixel_thresholds.json')},old_direction_selection_hash=d.sha(OLD/'selection.json'),old_direction_fits_hash=d.sha(OLD/'fitted_probes.npz')))
    c=torch.load(d.CHECKPOINT,map_location='cpu',weights_only=False);assert c['state']['step']==2297
    classes={k.split('.')[1]:v.shape[0] for k,v in c['model'].items() if k.startswith('heads.') and k.endswith('.weight')}
    model=d.SpatialConsolidation(classes);model.load_state_dict(c['model'],strict=True);model.eval();model.requires_grad_(False);del c
    before=d.state_hash(model);assert before==ident['state_sha256'];ex=d.Extractor(model);parity=[]
    for b in (12,20,28):
        stream=d.SpatialBatteryStream(81007173+b,'test');x,y,mm,a=d.captured_batch(stream,b);native=d.SpatialBatteryStream(81007173+b,'test');xx,yy,mn=native.batch(1,d.TASK,dict(baseline_transitions=b))
        assert torch.equal(x,xx) and torch.equal(y,yy) and mm==mn and stream.state_dict()==native.state_dict()
        pair,_=d.make_pair(x,mm[0]);f,l=ex(pair);ex.close()
        with torch.inference_mode():ll=model(pair,d.TASK).numpy()
        assert np.array_equal(l,ll);parity.append(dict(b=b,renderer_pixels_labels_metadata_rng_exact=True,hook_max_abs=float(np.max(abs(l-ll)))))
        ex=d.Extractor(model)
    stream=d.SpatialBatteryStream(spec['fresh_test_seed'],'test');data={};meta=[];pixels=[];angles=[];examples={};profile=[]
    for i in range(spec['fresh_test_groups']):
        assert time.time()<deadline-140,'report/verification reserve reached'
        t=time.time();b=(12,20,28)[i%3];x,y,mm,a=d.captured_batch(stream,b);pair,labels=d.make_pair(x,mm[0]);f,log=ex(pair);obs=pixel_summary(x[0].numpy());pixels.append(obs)
        for k,v in f.items():data.setdefault('__'.join(k),[]).append(v)
        data.setdefault('native_logits',[]).append(log)
        direction=d.circular_targets(a,b);pad=np.full((36,2,16),np.nan);pad[:b+8]=a;angles.append(pad)
        for variant in (0,1):
            m=dict(mm[0]);m.update(group=f'fresh/{i}',variant=variant,label=int(labels[variant]),target_location=mm[0]['target_location'] if variant==0 else 1-mm[0]['target_location'],actual_circular_angles_radians=direction.tolist(),movie_sha256=hashlib.sha256(pair[variant].numpy().tobytes()).hexdigest(),noncue_sha256=hashlib.sha256(pair[variant,2:].numpy().tobytes()).hexdigest());meta.append(m)
        assert torch.equal(pair[0,2:],pair[1,2:])
        if i<3:examples[str(i)]=pair.numpy()
        profile.append(time.time()-t)
        if (i+1)%25==0:print('FRESH',i+1,'elapsed',time.time()-start,flush=True)
    ex.close();data={k:np.concatenate(v) for k,v in data.items()};ey=targets(meta)
    np.savez(OUT/'test_features.npz',**data);np.savez_compressed(OUT/'example_frames.npz',**examples);np.savez(OUT/'test_dot_angles.npz',angles=np.stack(angles));d.dump(OUT/'test_metadata.json',meta)
    sums=np.zeros((len(pixels),36,2,2));counts=np.zeros((len(pixels),36,2),int)
    for i,r in enumerate(pixels):sums[i,:len(r['counts'])]=r['flow_sums'];counts[i,:len(r['counts'])]=r['counts']
    pixangles=np.stack([r['angles'] for r in pixels]);pixcue=np.array([r['cue'] for r in pixels]);bs=np.array([m['baseline_transitions'] for m in meta[::2]])
    np.savez(OUT/'pixel_test.npz',angles=pixangles,cue=pixcue,flow_sums=sums,counts=counts,baseline=bs)
    # Predictions before any scoring. Every function reads only its assigned layer.
    predictions={};invariance={};xs={}
    for site in sites:
        for kind in ('phases','final'):xs[site+'/'+kind]=access(data,site,kind,projections[site])
        final={site+'__final':data[site+'__final']};poison={k:(v if k==site+'__final' else np.full_like(v,np.nan)) for k,v in data.items()}
        assert np.array_equal(xs[site+'/final'],access(final,site,'final',projections[site])) and np.array_equal(xs[site+'/final'],access(poison,site,'final',projections[site]));invariance[site]=True
    for key,z in selection.items():
        fit={k:fitarr[f"{z['index']}_{k}"] for k in ('mean','scale','weight','intercept')};predictions[key]=d.predict(fit,xs[z['site']+'/'+z['access']])
    for site in sites:
        for phase in ('baseline','post','final'):
            for k in (0,1):
                p=oldsel[f'{site}__{phase}/angle/{k}']['archive_prefix'];fit={q:oldfits[p+'_'+q] for q in ('mean','scale','weight','intercept')}
                predictions[f'{site}/direction/{phase}/{k}']=d.predict(fit,data[site+'__'+phase])
    for w,name in enumerate(('full','endpoint')):predictions['pixel/'+name+'/delta']=circular_change(pixangles[:,w,0],pixangles[:,w,1])
    predictions.update(truth_delta=ey['delta'],truth_angles=ey['angles'],truth_event=ey['event'],truth_side=ey['side'],truth_label=ey['label'],native_logits=data['native_logits'])
    np.savez(OUT/'test_predictions.npz',**predictions)
    # Grouped uncertainty: originals are one per physical group. Pairs never double n.
    n=len(pixels);boot=np.random.default_rng(38113).integers(0,n,(500,n));np.savez(OUT/'bootstrap_groups.npz',indices=boot)
    orig=np.arange(0,len(meta),2);truth={k:v[orig] for k,v in ey.items()};ev=truth['event'].astype(bool);eventidx=np.flatnonzero(ev)
    # Resample all groups, then select event rows: preserves matched cross-method samples.
    eventboot=[r[ev[r]] for r in boot];eventmap=np.full(n,-1);eventmap[eventidx]=np.arange(len(eventidx));eb=[eventmap[r] for r in eventboot]
    metrics={};cache={}
    for key,z in selection.items():
        s=predictions[key][orig];task=z['target']
        if task=='change':
            pred=change_from_scores(s);err=change_error(truth['delta'],pred);changederr=err[eventidx,truth['side'][ev]]
            res=dict(mae_degrees=err.mean(0).tolist(),mean_mae=float(err.mean()),ci95=np.quantile(err[boot].mean((1,2)),[.025,.975]).tolist(),changed_patch_mae=float(changederr.mean()),changed_patch_ci95=np.quantile([err[r,truth['side'][r]].mean() for r in eventboot],[.025,.975]).tolist(),catch_mae=float(err[~ev].mean()),predicted_change_std=pred.std(0).tolist())
            metrics[key]=res
            side_score=np.abs(pred[:,1])-np.abs(pred[:,0]);metrics[key+'/derived_side']=metrics_binary(truth['side'][ev],side_score[ev],0,eb)
        else:
            score=s[:,1]-s[:,0];mask=ev if task=='side' else np.ones(n,bool);metrics[key]=metrics_binary(truth[task][mask],score[mask],0,eb if task=='side' else boot);cache[key]=(truth[task],score,mask)
    # Phase direction accuracy and subtracting independently decoded phases.
    for site in sites:
        aa={}
        for phase in ('baseline','post','final'):
            aa[phase]=np.stack([np.arctan2(predictions[f'{site}/direction/{phase}/{k}'][orig,1],predictions[f'{site}/direction/{phase}/{k}'][orig,0]) for k in (0,1)],1)
            tr=truth['angles'][:,0 if phase=='baseline' else 1];err=np.abs(circular_change(tr,aa[phase]));metrics[site+'/direction/'+phase]=dict(mae_degrees=err.mean(0).tolist(),mean_mae=float(err.mean()),ci95=np.quantile(err[boot].mean((1,2)),[.025,.975]).tolist())
        delta=circular_change(aa['baseline'],aa['post']);err=change_error(truth['delta'],delta)
        metrics[site+'/direction_subtraction']=dict(mae_degrees=err.mean(0).tolist(),changed_patch_mae=float(err[eventidx,truth['side'][ev]].mean()),side=metrics_binary(truth['side'][ev],(abs(delta[:,1])-abs(delta[:,0]))[ev],0,eb))
    for w,name in enumerate(('full','endpoint')):
        delta=predictions['pixel/'+name+'/delta'];absdelta=abs(delta);scores=dict(side=absdelta[:,1]-absdelta[:,0],event=absdelta.max(1),label=absdelta[np.arange(n),pixcue]);err=change_error(truth['delta'],delta)
        metrics['pixel/'+name+'/change']=dict(mae_degrees=err.mean(0).tolist(),mean_mae=float(err.mean()),changed_patch_mae=float(err[eventidx,truth['side'][ev]].mean()),catch_mae=float(err[~ev].mean()),ci95=np.quantile(err[boot].mean((1,2)),[.025,.975]).tolist())
        for task,s in scores.items():
            threshold=0 if task=='side' else thresholds[name+'/'+task]['threshold'];mask=ev if task=='side' else np.ones(n,bool);key='pixel/'+name+'/'+task
            metrics[key]=metrics_binary(truth[task][mask],s[mask],threshold,eb if task=='side' else boot);metrics[key]['threshold']=threshold;cache[key]=(truth[task],s-threshold,mask)
        for ph in (0,1):
            err=np.abs(circular_change(truth['angles'][:,ph],pixangles[:,w,ph]));metrics['pixel/'+name+'/direction/'+str(ph)]=dict(mae_degrees=err.mean(0).tolist(),ci95=np.quantile(err[boot].mean((1,2)),[.025,.975]).tolist())
    metrics['zero_change_baseline']=dict(mae_degrees=np.abs(truth['delta']).mean(0).tolist(),changed_patch_mae=float(abs(truth['delta'][eventidx,truth['side'][ev]]).mean()),catch_mae=float(abs(truth['delta'][~ev]).mean()))
    metrics['native']=metrics_binary(truth['label'],data['native_logits'][orig,1]-data['native_logits'][orig,0],0,boot)
    metrics['counts']=dict(groups=n,events=int(ev.sum()),catches=int((~ev).sum()),left=int((truth['side']==0).sum()),right=int((truth['side']==1).sum()),cue_decode_correct=int((pixcue==truth['cue']).sum()),targets=int(truth['label'].sum()),foil=int(((truth['label']==0)&ev).sum()))
    metrics['paired_gains']={}
    comparisons=[(f'{s}/phases/{t}',f'{s}/final/{t}') for s in sites for t in ('event','side')]+[(f'pixel/{w}/side',f'{s}/phases/side') for w in ('full','endpoint') for s in sites]
    for a,b in comparisons:
        yy,sa,mask=cache[a];_,sb,_=cache[b];rset=eventboot if a.endswith('side') else boot;g=[d.ba(yy[r],(sa[r]>0).astype(int))-d.ba(yy[r],(sb[r]>0).astype(int)) for r in rset];metrics['paired_gains'][a+' MINUS '+b]=dict(ba_gain=d.ba(yy[mask],(sa[mask]>0).astype(int))-d.ba(yy[mask],(sb[mask]>0).astype(int)),ci95=np.quantile(g,[.025,.975]).tolist())
    d.dump(OUT/'metrics.json',metrics)
    # Serialized prediction replay, including derived image-only flow estimates.
    fa=np.load(OUT/'fitted_models.npz');pa=np.load(OUT/'test_predictions.npz');stored=dict(np.load(OUT/'test_features.npz'));pp=np.load(OUT/'projections.npz');replay=0.
    for key,z in selection.items():
        fit={k:fa[f"{z['index']}_{k}"] for k in ('mean','scale','weight','intercept')};calc=d.predict(fit,access(stored,z['site'],z['access'],pp[z['site']]))
        replay=max(replay,float(abs(calc-pa[key]).max()))
    pix=np.load(OUT/'pixel_test.npz')
    for i,b in enumerate(pix['baseline']):assert np.array_equal(flow_angles(pix['flow_sums'][i],pix['counts'][i],b),pix['angles'][i])
    for w,name in enumerate(('full','endpoint')):assert np.array_equal(circular_change(pix['angles'][:,w,0],pix['angles'][:,w,1]),pa['pixel/'+name+'/delta'])
    # Re-run pixels on three actual saved raster pairs, including image-decoded cue swap.
    ef=np.load(OUT/'example_frames.npz')
    for i in range(3):
        a=pixel_summary(ef[str(i)][0]);b=pixel_summary(ef[str(i)][1]);assert np.array_equal(a['angles'],b['angles']) and a['cue']==1-b['cue'];assert np.array_equal(a['angles'],pixangles[i])
    oldhashes=set(m['noncue_sha256'] for split in ('train','val','test') for m in json.loads((OLD/f'{split}_metadata.json').read_text()))
    assert not oldhashes.intersection(m['noncue_sha256'] for m in meta);assert len(set(m['noncue_sha256'] for m in meta))==n
    assert replay==0 and d.sha(d.CHECKPOINT)==ident['checkpoint_sha256'] and d.state_hash(model)==before and all(not p.requires_grad for p in model.parameters())
    for p,h in ident['source_hashes'].items():assert d.sha(ROOT/p)==h
    verification=dict(prediction_replay_max_abs=replay,pixel_flow_replay_exact=True,example_pixel_replay_exact=True,independent_noncue_hashes=True,checkpoint_and_state_unchanged=True,source_hashes_unchanged=True,parity=parity,final_only_removal_nan_invariance=invariance,zero_deployed_optimizer_steps=True,worker_count=1,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),profile_seconds=dict(mean=float(np.mean(profile)),max=float(np.max(profile))),elapsed_seconds=time.time()-start,within_cap=time.time()<deadline)
    d.dump(OUT/'verification.json',verification)
    # Report numerical tables immediately; interpretation can append within same cap.
    lines=['# Upstream motion diagnostic — frozen selected2297','', '**Fresh-test diagnostic, not deployed training or a causal lesion test.**', '', '## Same-trial physical comparison',f"Native originals: {n} independent groups; {int(ev.sum())} events, {int((~ev).sum())} catches. Each has a separate cue-swapped extraction; the table counts groups once.",'','| Observer/access | Event BA [95% CI] | Event AUC | Event-side BA [95% CI] | Side AUC |','|---|---:|---:|---:|---:|']
    def fmt(r):return f"{r['balanced_accuracy']:.3f} [{r['ba_ci95'][0]:.3f},{r['ba_ci95'][1]:.3f}]"
    for key in ['pixel/full','pixel/endpoint']+[s+'/'+a for s in sites for a in ('phases','final')]:
        e=metrics[key+'/event'];s=metrics[key+'/side'];lines.append(f"| {key} | {fmt(e)} | {e['auc']:.3f} | {fmt(s)} | {s['auc']:.3f} |")
    lines+=['','## Direct signed circular change','MAE in degrees against actual per-dot phase-mean change; not latent event magnitude. No-change predictions are a strong imbalanced baseline.','| Access | Left / right MAE | Changed-patch MAE, events only | Catch MAE |','|---|---:|---:|---:|']
    for key in ['pixel/full','pixel/endpoint']+[s+'/'+a for s in sites for a in ('phases','final')]:
        q=metrics[key+'/change'];lines.append(f"| {key} | {q['mae_degrees'][0]:.2f} / {q['mae_degrees'][1]:.2f} | {q['changed_patch_mae']:.2f} | {q['catch_mae']:.2f} |")
    q=metrics['zero_change_baseline'];lines.append(f"| Always zero | {q['mae_degrees'][0]:.2f} / {q['mae_degrees'][1]:.2f} | {q['changed_patch_mae']:.2f} | {q['catch_mae']:.2f} |")
    lines+=['','## Direction precision versus temporal comparison','Frozen prior full-dimensional direction fits, never reselected using this test.','| Site | Pre direction MAE L/R | Post direction MAE L/R | Subtracted directions: changed-patch MAE | Subtracted direction side BA |','|---|---:|---:|---:|---:|']
    for site in sites:
        aa=metrics[site+'/direction/baseline']['mae_degrees'];bb=metrics[site+'/direction/post']['mae_degrees'];q=metrics[site+'/direction_subtraction'];lines.append(f"| {site} | {aa[0]:.1f}/{aa[1]:.1f} | {bb[0]:.1f}/{bb[1]:.1f} | {q['changed_patch_mae']:.1f} | {fmt(q['side'])} |")
    lines+=['','## Protocol and limits', '- 425 reused train /100 reused validation groups; 300 newly generated test groups, B12/B20/B28 interleaved. Validation reuse and earlier hypothesis generation are exploratory; new test was generated only after the saved hash freeze. No old test used for fitting/selection.', '- 48 compact fits: four sites ×two access modes ×three targets ×true/shuffled supervision. Fixed random projection32 per slot →64 coordinates plus2080 quadratic products =2144 features. Same train-only scaling, coefficient counts, groups and alpha grid1/100/10000. Native input dimensions CNN1152, KDA576, memory3136; only425 independent training groups despite850 cue-variant rows. Side fits use event groups only. This limits power and neural information-loss claims.', '- Physical pre/post endpoints B+7/B+15 versus final B+16. All sites read both fixed patch neighborhoods; true cue never selects features. Final uses two fixed permutations of the SAME final array. External pre/post storage is unavailable to the deployed head. KDA emitted fields include recurrent history; CNN endpoint sees stack3, so pixel last-two-transition access is the direct instantaneous-window comparison, not equal information or capacity.', '- Pixel observer privileges: exact fixed patch geometry, bilinear dot mass/speed, reference/event timing from sequence length, visible image-decoded cue, full externally stored history or designated endpoint windows. It matches isolated centroids, not true dot identity; replacements/overlap reduce accepted matches. It is diagnostic, not deployable. Full and endpoint event/report thresholds selected only on exact re-rendered validation movies (hash verified).', '- Native renderer untouched, captured movement angles before each update. Actual circular targets include per-dot offsets, not reset displacement jumps. Means start90°apart: all-site fits cannot prove independent local coding. Signed change differs slightly from nominal26/28° because phase-averaged dot offsets evolve.', '- 500 resamples of independent test groups, paired across methods; event-only side and event/catch scored separately. Intervals conditional on fixed fits, no multiple-comparison correction or checkpoint replication. Shuffled controls and paired gain intervals are in metrics.json. Failed probes do not establish neural erasure. Pooling attenuation is NOT a measurement made here.', '- Exact selected2297 checkpoint/source hashes checked against prior receipts and archived runtime. Renderer pixels/labels/metadata/final RNG and hooked/unhooked logits are exact at all three B values. No model training, task/architecture changes, cloud work, or optimizer creation.', '- Prediction replay error0; pixel flows replayed from serialized image-derived sums; saved example rasters replay exactly; final-only predictions pass earlier-frame removal and NaN poisoning. Fresh noncue movie hashes are disjoint from ALL prior splits.', f"- Elapsed through initial report: {time.time()-start:.1f}s under immutable1200s budget, CPU threads2/inter-op1, one worker.", '', 'Artifacts: protocol/budget/identity/test_freeze/selection/metrics/verification JSON; fitted_models, projections, fresh test_features/dot_angles/predictions, pixel_validation/pixel_test, bootstrap_groups, example_frames NPZ; test_metadata; upstream.py/run_upstream.py/test_upstream.py. Original artifacts untouched.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'LabJournal/krauzlis-upstream-motion-diagnostic.md').write_text('# Upstream frozen motion diagnostic — selected2297\n\n'+ '\n'.join(lines[2:18])+'\n\nFull report: `SecondPass/SpatialReadout/SpatialConsolidation/KrauzlisFailureAudit/UpstreamMotionDiagnostic/REPORT.md`. No model, task or cloud changes.\n')
    d.dump(OUT/'completion.json',dict(complete=True,elapsed_seconds=time.time()-start,within_cap=time.time()<deadline,deadline=deadline,artifacts={p.name:d.sha(p) for p in OUT.glob('*.npz')}))
    print('COMPLETE',json.dumps(metrics['counts']),'elapsed',time.time()-start,flush=True)
    signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
