"""Small frozen terminal6760 diagnostic. No optimizer, backward, or fits.
Run --prepare first (CPU), then --run once (900s nonrenewable MPS cap).
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import argparse, copy, hashlib, json, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
from SecondPass.SpatialReadout.model import SpatialReadout
from SecondPass.TaskSuite.suite import task_classes
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream, blank, local_cue
from PreAttentiveVision.neuroscience_stimuli import TaskStream

OUT=Path(__file__).resolve().parent
RUN=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2')
CKPT=RUN/'terminal.pt'
SHA='1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841'
TASKS=['orientation_cued','spatial_binding','motion_duration_cued']
N=32

def dump(name,obj):
    p=OUT/name; tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Capture(SpatialBatteryStream):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw); self.gabors=[]; self.dots=[]
    def _gabors(self,rng,angles):
        state=copy.deepcopy(rng.bit_generator.state)
        v=super()._gabors(rng,angles); self.gabors.append((v.copy(),state)); return v
    def _dots_raster(self,*a,**kw):
        v=super()._dots_raster(*a,**kw); self.dots.append(v.copy()); return v

def prepare():
    records=[]; summaries=[]
    for ti,task in enumerate(TASKS):
        seed=97676001+1000*ti
        st=Capture(seed,'test'); x,y,meta=st.batch(N,task,{'delay':0})
        direct=SpatialBatteryStream(seed,'test').batch(N,task,{'delay':0})
        assert torch.equal(x,direct[0]) and torch.equal(y,direct[1]) and meta==direct[2]
        xl,yl,ml=SpatialBatteryStream(seed,'test').batch(N,task,{'delay':24})
        for i,m in enumerate(ml):
            keep=[t for t in range(xl.shape[1]) if t not in m['blank_frames']]
            assert torch.equal(x[i],xl[i,keep]) and yl[i]==y[i]
        records.extend([dict(task=task,condition='native_D0',x=x,y=y,metadata=meta),dict(task=task,condition='native_D24',x=xl,y=yl,metadata=ml)])
        # Retarget cue, preserving every non-cue pixel; label recomputed by native rule.
        xc=x.clone(); yc=y.clone(); mc=copy.deepcopy(meta)
        for i,m in enumerate(meta):
            old=m['target_location']
            if task=='orientation_cued':
                candidates=[j for j in range(4) if int(m['rotations_degrees'][j]*m['cue_sign']>0)!=int(y[i])]
            elif task=='spatial_binding':
                candidates=[j for j in range(4) if int(j in m['swapped_locations'])!=int(y[i])]
            else:
                candidates=[j for j in range(4) if m['winner_by_patch'][j]!=int(y[i])]
            new=candidates[i%len(candidates)] if candidates else (old+1)%4
            if task=='orientation_cued':
                for t in m['cue_frames']:
                    raw=blank() if t==0 else st.gabors[3*i+t-1][0]
                    assert np.array_equal(local_cue(raw,'recall','sample',old,m['cue_sign']),x[i,t].numpy())
                    xc[i,t]=torch.from_numpy(local_cue(raw,'recall','sample',new,m['cue_sign']))
                yc[i]=int(m['rotations_degrees'][new]*m['cue_sign']>0)
            elif task=='spatial_binding':
                t=m['cue_frames'][0]
                assert np.array_equal(local_cue(blank(),'recall','query',old),x[i,t].numpy())
                xc[i,t]=torch.from_numpy(local_cue(blank(),'recall','query',new)); yc[i]=int(new in m['swapped_locations'])
            else:
                for t in m['cue_frames']:
                    raw=blank() if t==0 else st.dots[9*i+t-1]
                    assert np.array_equal(local_cue(raw,'integration','sample',old),x[i,t].numpy())
                    xc[i,t]=torch.from_numpy(local_cue(raw,'integration','sample',new))
                yc[i]=m['winner_by_patch'][new]
            mc[i].update(target_location=new,label=int(yc[i]),original_target=old,paired_trial_id=m['trial_id'])
            untouched=[t for t in range(x.shape[1]) if t not in m['cue_frames']]
            assert torch.equal(x[i,untouched],xc[i,untouched])
        records.append(dict(task=task,condition='cue_retarget_D0',x=xc,y=yc,metadata=mc))
        if task=='orientation_cued':
            xs=x.clone(); ys=y.clone(); ms=copy.deepcopy(meta)
            for i,m in enumerate(meta):
                for t in m['cue_frames']:
                    raw=blank() if t==0 else st.gabors[3*i+t-1][0]
                    xs[i,t]=torch.from_numpy(local_cue(raw,'recall','sample',m['target_location'],-m['cue_sign']))
                ys[i]=int(-m['rotations_degrees'][m['target_location']]*m['cue_sign']>0)
                ms[i].update(cue_sign=-m['cue_sign'],label=int(ys[i]),paired_trial_id=m['trial_id'])
            records.append(dict(task=task,condition='cue_sign_flip_D0',x=xs,y=ys,metadata=ms))
        if task in ('orientation_cued','spatial_binding'):
            xe=x.clone(); ye=y.clone(); me=copy.deepcopy(meta)
            for i,m in enumerate(meta):
                angles=np.array(m['sample_angles_radians'])
                if task=='orientation_cued':
                    delta=-np.array(m['rotations_degrees']); probe=(angles+np.deg2rad(delta))%np.pi
                    ye[i]=int(delta[m['target_location']]*m['cue_sign']>0)
                    me[i]['rotations_degrees']=delta.tolist()
                else:
                    others=[j for j in range(4) if j!=m['target_location']]
                    pair=[m['target_location'],others[i%3]] if not int(y[i]) else others[:2]
                    probe=angles.copy(); probe[pair]=probe[pair[::-1]]; ye[i]=1-y[i]
                    me[i]['swapped_locations']=pair
                rng=np.random.default_rng(); rng.bit_generator.state=st.gabors[3*i+2][1]
                raster=SpatialBatteryStream._gabors(st,rng,probe)
                xe[i,m['probe_frame']]=torch.from_numpy(local_cue(raster,'recall','report',m['target_location'],visible=False))
                me[i].update(probe_angles_radians=probe.tolist(),label=int(ye[i]),paired_trial_id=m['trial_id'])
            records.append(dict(task=task,condition='native_law_evidence_flip_D0',x=xe,y=ye,metadata=me))
        else:
            xr=x.clone(); yr=(y+2)%4; mr=copy.deepcopy(meta)
            for i,m in enumerate(meta):
                ts=[m['reference_frame']]+m['moving_frames']; xr[i,ts]=x[i,list(reversed(ts))]
                mr[i].update(directions_by_patch=((np.array(m['directions_by_patch'])[:,::-1]+2)%4).tolist(),duration_counts_by_patch=np.array(m['duration_counts_by_patch'])[:,[2,3,0,1]].tolist(),winner_by_patch=((np.array(m['winner_by_patch'])+2)%4).tolist(),label=int(yr[i]),paired_trial_id=m['trial_id'],diagnostic='OOD time reversal; reverse displacement flips all direction labels by2, duration counts preserved under direction remapping. Backward dot replacements differ from native law.')
            records.append(dict(task=task,condition='OOD_motion_time_reverse_D0',x=xr,y=yr,metadata=mr))
        summaries.append(dict(task=task,seed=seed,n=N,metadata_first=meta[0],metadata_long_first=ml[0],native_capture_exact=True,matched_delay_nonblank_exact=True))
    # Successful sensory anchors are independent native draws, not task replacements.
    for task in ['orientation','motion_direction']:
        x,y,m=TaskStream(97679999,'test').batch(N,task)
        records.append(dict(task=task,condition='native_mixed',x=x,y=y,metadata=m))
    dump('renderer_inspection.json',dict(n=N,cells=summaries,conditions=[dict(task=r['task'],condition=r['condition'],shape=list(r['x'].shape),class_counts=np.bincount(r['y'].numpy(),minlength=task_classes()[r['task']]).tolist()) for r in records],notes=['No cue blanking used. Cue edits use native local_cue and captured underlying rasters.','Counterfactual cue targets chosen to disagree where possible: native-valid scenes, NOT natural random condition frequencies or cue-validity benefit.','D0 and D24 share exact evidence/labels; only native ignore blanks differ.','Metadata is analysis-only; only images and external task-head name enter model.']))
    return records

class UnmodifiedWrapper(torch.nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x,task):
        frames=self.model.frames(x); states=None; hidden=None
        for t in range(frames.shape[1]):
            field,states=self.model.encode_frame(frames[:,t],states)
            hidden=self.model.spatial_gru(self.model.spatial_input(field),hidden)
        return self.model.heads[task](self.model.readout(hidden.flatten(1)).relu())

def metrics(y,z):
    y=np.array(y); z=np.array(z); p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    pred=z.argmax(1); k=z.shape[1]; cm=np.zeros((k,k),int)
    for a,b in zip(y,pred):cm[a,b]+=1
    recall=[float(cm[c,c]/cm[c].sum()) if cm[c].sum() else None for c in range(k)]
    auc=[]
    for c in range(k):
        pos=p[y==c,c]; neg=p[y!=c,c]; auc.append(float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean()) if len(pos) and len(neg) else None)
    acc=float((pred==y).mean()); n=len(y); zz=1.959963984540054; d=1+zz*zz/n
    mid=(acc+zz*zz/(2*n))/d; half=zz*np.sqrt(acc*(1-acc)/n+zz*zz/(4*n*n))/d
    return dict(n=n,accuracy=acc,accuracy_wilson95=[mid-half,mid+half],balanced_accuracy=float(np.mean([v for v in recall if v is not None])),auc_macro=float(np.mean([v for v in auc if v is not None])),confusion=cm.tolist(),recall=recall,prediction_counts=np.bincount(pred,minlength=k).tolist(),mean_probabilities=p.mean(0).tolist())

def paired(base,other):
    z=np.array(base['logits']); q=np.array(other['logits']); y=np.array(base['labels']); v=np.array(other['labels'])
    def soft(a):
        e=np.exp(a-a.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    p,r=soft(z),soft(q); delta=(q-q.mean(1,keepdims=True))-(z-z.mean(1,keepdims=True))
    changed=y!=v; bp=z.argmax(1); op=q.argmax(1)
    benefit=(op==v).astype(float)-(bp==y).astype(float)
    rng=np.random.default_rng(9981); boots=benefit[rng.integers(0,len(y),(2000,len(y)))].mean(1)
    out=dict(label_changes=int(changed.sum()),argmax_changes=int((bp!=op).sum()),mean_abs_centered_logit_change=float(abs(delta).mean()),mean_abs_probability_change=float(abs(r-p).mean()),max_abs_probability_change=float(abs(r-p).max()),paired_accuracy_difference=float(benefit.mean()),paired_accuracy_difference_bootstrap95=np.quantile(boots,[.025,.975]).tolist())
    if changed.any():
        ids=np.flatnonzero(changed); a=(q[ids,v[ids]]-q[ids,y[ids]])-(z[ids,v[ids]]-z[ids,y[ids]])
        out.update(changed_label_margin_shift_mean=float(a.mean()),changed_label_margin_shift_positive_fraction=float((a>0).mean()),both_correct_on_label_changed=int(((bp==y)&(op==v)&changed).sum()))
    return out

def run():
    assert not (OUT/'budget.json').exists(),'Budget exists: cannot renew/restart this diagnostic'
    predecessor=json.loads((RUN/'production_supervisor_result.json').read_text()); assert predecessor['returncode']==0
    for pid in [predecessor['worker_pid'],predecessor['supervisor_pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'Predecessor PID {pid} alive')
    procs=subprocess.check_output(['ps','-axo','pid,ppid,command'],text=True)
    competitors=[s for s in procs.splitlines() if ('python' in s.lower()) and any(v in s for v in ('SpatialReadout.worker','SpatialReadout.continuation_v2','SpatialReadout.qos_repair')) and '-c ' not in s]
    assert not competitors,competitors
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    records=prepare(); assert sha(CKPT)==SHA
    cp=torch.load(CKPT,map_location='cpu',weights_only=False)
    model=SpatialReadout(task_classes()); model.load_state_dict(cp['model'],strict=True); model.eval(); model.requires_grad_(False)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    source_paths=['SecondPass/SpatialReadout/model.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py','WorkingMemory/SpatialTaskBattery/stimuli.py','SecondPass/TaskSuite/suite.py']
    source_hashes={p:sha(p) for p in source_paths}
    # Persist hard deadline before the FIRST MPS activation, never reset it.
    start=time.time(); monotonic=time.monotonic(); deadline=start+900
    dump('budget.json',dict(start_unix=start,deadline_unix=deadline,cap_seconds=900,pid=os.getpid(),cpu_threads=2,interop_threads=1,predecessor=predecessor,competing_workers=competitors,first_mps_operation='model.to(mps)',renewable=False))
    def alarm(*unused):
        dump('cap_reached.json',dict(time=time.time(),elapsed=time.monotonic()-monotonic)); os._exit(124)
    signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,900)
    model=model.to('mps'); wrapper=UnmodifiedWrapper(model).eval(); outputs=[]; parity=[]
    with torch.inference_mode():
        for record in records:
            if time.time()>deadline-75:break
            task=record['task']; xs=record['x']; zs=[]
            # Exact independent implementation parity check for every condition.
            xx=xs[:2].to('mps'); direct=model(xx,task); wrapped=wrapper(xx,task)
            error=float((direct-wrapped).abs().max().cpu()); assert error<=1e-6
            parity.append(dict(task=task,condition=record['condition'],max_abs_error=error))
            for b in range(0,len(xs),4):
                assert time.time()<deadline-30,'Reserve exhausted'
                zs.extend(model(xs[b:b+4].to('mps'),task).cpu().tolist())
            row=dict(task=task,condition=record['condition'],labels=record['y'].tolist(),logits=zs,metadata=record['metadata'],metrics=metrics(record['y'].tolist(),zs),elapsed_seconds=time.monotonic()-monotonic)
            outputs.append(row); dump('trial_results.json',outputs)
            print(json.dumps(dict(task=task,condition=record['condition'],metrics=row['metrics'],elapsed=row['elapsed_seconds'])),flush=True)
    torch.mps.synchronize()
    final=model.state_dict(); immutable=all(torch.equal(v,final[k].cpu()) for k,v in initial.items()); assert immutable
    assert sha(CKPT)==SHA and all(sha(p)==h for p,h in source_hashes.items())
    comparisons=[]
    for row in outputs:
        if row['condition'] not in ('native_D0','native_mixed'):
            base=next(v for v in outputs if v['task']==row['task'] and v['condition']=='native_D0')
            comparisons.append(dict(task=row['task'],comparison=row['condition']+' vs native_D0',**paired(base,row)))
    result=dict(checkpoint=str(CKPT),checkpoint_sha256=SHA,checkpoint_schema=cp.get('schema'),step=cp.get('step'),eval=True,all_requires_grad_false=all(not p.requires_grad for p in model.parameters()),all_state_tensors_bitwise_unchanged=immutable,checkpoint_file_unchanged=True,source_hashes_unchanged=source_hashes,wrapper_parity=parity,completed_conditions=len(outputs),planned_conditions=len(records),training_updates=0,probe_fits=0,conditions=[dict(task=r['task'],condition=r['condition'],**r['metrics']) for r in outputs],paired_comparisons=comparisons,elapsed_seconds=time.monotonic()-monotonic,finished_unix=time.time(),deadline_unix=deadline)
    dump('summary.json',result)
    lines=['# Frozen terminal6760 failure diagnostic','',f'Completed {len(outputs)}/{len(records)} conditions; n=32 per condition. One MPS worker, CPU thread cap2. No training or probe fitting.','', '|Task|Condition|BA|AUC|Predicted class counts|','|---|---|---:|---:|---|']
    for r in result['conditions']:lines.append(f"|{r['task']}|{r['condition']}|{r['balanced_accuracy']:.4f}|{r['auc_macro']:.4f}|{r['prediction_counts']}|")
    lines+=['','## Paired effects','Centered logit differences remove common-mode offsets. Margins are toward the newly correct vs formerly correct class, only when the label changes.']
    for r in comparisons:lines.append(f"- {r['task']} {r['comparison']}: {json.dumps(r)}")
    lines+=['','## Interpretation limits','Exploratory single-checkpoint small balanced native draws, not a population estimate or a full suite sweep. Accuracy Wilson intervals and paired accuracy bootstrap intervals are saved; no psychometric/probe fits. Cue edits retain exactly matched noncue evidence and recompute labels. Retargeting deliberately seeks discordant answers, hence changes condition frequencies but not task semantics. Sign-flip unchanged trials remain negative. Evidence edits use native Gabor law and matched noise/phase; binding pair selection is controlled, not original random mixture. Motion time reversal is explicitly OOD: reversal of replacement events is not native motion generation. No cue deletion or valid/neutral/invalid cue-benefit claims. No internal state erasure, spatial inhibition, microstimulation, attention maps, reaction times, or biological claims. Stack3 means early inserted blanks still contain old images; D0 is an acquisition/selection/comparison check, not a clean isolated-memory assay. Sensory anchors differ in geometry and demands; success does not prove four-patch motion encoding.','',f'All frozen tensors, checkpoint file and source files unchanged; direct/wrapper error <=1e-6. Finished {result["elapsed_seconds"]:.2f}s after first MPS activation, under fixed900s cap.']
    (OUT/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    dump('completion.json',dict(status='complete' if len(outputs)==len(records) else 'partial_budget_reserve',elapsed_seconds=time.monotonic()-monotonic,within_cap=time.time()<deadline,exit_code=0,immutable=immutable,worker_pid=os.getpid()))
    assert time.time()<deadline; signal.setitimer(signal.ITIMER_REAL,0)
    print('COMPLETE '+json.dumps(result['paired_comparisons']),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--run',action='store_true'); args=ap.parse_args()
    if args.prepare:prepare(); print('Renderer metadata and paired raster checks passed; no MPS work.')
    elif args.run:run()
    else:ap.error('choose --prepare or --run')
