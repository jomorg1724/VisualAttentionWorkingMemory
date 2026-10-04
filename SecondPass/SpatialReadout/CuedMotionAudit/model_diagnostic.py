"""Frozen current-checkpoint motion audit. No optimizer or backward call.
Run with PYTHONPATH=. python -m SecondPass.SpatialReadout.CuedMotionAudit.model_diagnostic
A persisted nonrenewable 1200s budget starts before the first MPS operation.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import copy, hashlib, json, signal, time
from pathlib import Path
import numpy as np
import torch
from SecondPass.SpatialReadout.model import SpatialReadout
from SecondPass.TaskSuite.suite import task_classes
from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream,blank,local_cue
from PreAttentiveVision.neuroscience_stimuli import TaskStream
from SecondPass.SpatialReadout.FailureAnalysis.diagnostic import metrics,paired
OUT=Path(__file__).resolve().parent
CKPT=Path('/Users/jonathanmorgan/VAWMRuntime/cloud_convgru_continuation_01/artifacts/checkpoint_035039.pt')
SHA='93762f7968a29092b8acce7ea9632937a23965160822fe98bda9b3e5844531e7'
# Historical module is used only for pure summary functions, never its checkpoint or run().
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'

def dump(name,obj):
    path=OUT/name; temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class Capture(SpatialBatteryStream):
    def __init__(self,*args,**kw):super().__init__(*args,**kw);self.dots=[]
    def _dots_raster(self,*args,**kw):
        v=super()._dots_raster(*args,**kw);self.dots.append(v.copy());return v

def duration_records(n,seed=92635039):
    task='motion_duration_cued'; st=Capture(seed,'test'); x,y,m=st.batch(n,task,{'delay':0})
    direct=SpatialBatteryStream(seed,'test').batch(n,task,{'delay':0})
    assert torch.equal(x,direct[0]) and torch.equal(y,direct[1]) and m==direct[2]
    xl,yl,ml=SpatialBatteryStream(seed,'test').batch(n,task,{'delay':24})
    xc=x.clone();yc=y.clone();mc=copy.deepcopy(m)
    for i,meta in enumerate(m):
        keep=[t for t in range(xl.shape[1]) if t not in ml[i]['blank_frames']]
        assert torch.equal(x[i],xl[i,keep]) and int(y[i])==int(yl[i])
        old=meta['target_location']; candidates=[j for j in range(4) if meta['winner_by_patch'][j]!=int(y[i])]
        new=candidates[i%len(candidates)] if candidates else (old+1)%4
        for t in meta['cue_frames']:
            raw=blank() if t==0 else st.dots[9*i+t-1]
            assert np.array_equal(local_cue(raw,'integration','sample',old),x[i,t].numpy())
            xc[i,t]=torch.from_numpy(local_cue(raw,'integration','sample',new))
            # Rings radius14 lie outside dot apertures radius11.5; verify all actual dot pixels preserved.
            dots=(raw!=.5); assert np.array_equal(xc[i,t].numpy()[dots],x[i,t].numpy()[dots])
        yc[i]=meta['winner_by_patch'][new]
        assert torch.equal(xc[i,10:],x[i,10:])
        checks=dict(native_capture_exact=True,matched_delay_nonblank_exact=True,cue_edit_preserves_dot_pixels=True)
        meta['checks']=checks;ml[i]['checks']=checks;mc[i].update(target_location=new,label=int(yc[i]),original_target=old,checks=checks)
    return [dict(task=task,condition=c,x=a,y=b,metadata=d) for c,a,b,d in [('native_D0',x,y,m),('native_D24',xl,yl,ml),('cue_retarget_D0',xc,yc,mc)]]

def krauzlis_records(n,seed=92735039,stream=None,baseline=20):
    task='krauzlis_cued_motion';st=stream if stream is not None else SpatialBatteryStream(seed,'test')
    x,y,m=st.batch(n,task,{'baseline_transitions':baseline});xc=x.clone();yc=y.clone();mc=copy.deepcopy(m)
    yy,xx=np.mgrid[:100,:100]
    for i,meta in enumerate(m):
        new=1-meta['target_location'];raw=blank();raw[:,48:52,49:51]=.1;raw[:,49:51,48:52]=.1
        def ring(target):
            a=raw.copy();cx,cy=meta['positions_xy'][target];mask=np.abs(np.sqrt((xx-cx)**2+(yy-cy)**2)-(meta['aperture_radius_pixels']+1.8))<.7;a[:,mask]=.95;return a
        for t in (0,1):
            assert np.array_equal(x[i,t].numpy(),ring(meta['target_location']));xc[i,t]=torch.from_numpy(ring(new))
        yc[i]=int(meta['changed_patch']==new)
        event='catch' if meta['changed_patch'] is None else 'target' if int(yc[i]) else 'foil'
        mc[i].update(target_location=new,label=int(yc[i]),event_type=event,original_event_type=meta['event_type'],original_target=meta['target_location'])
        meta['checks']=mc[i]['checks']=dict(movie_exact_after_cue=True,cue_reconstruction_exact=True)
    assert torch.equal(x[:,2:],xc[:,2:])
    return [dict(task=task,condition=f'{c}_B{baseline}',x=a,y=b,metadata=d) for c,a,b,d in [('native',x,y,m),('cue_swap',xc,yc,mc)]]

@torch.inference_mode()
def trajectory(model,x,task):
    frames=model.frames(x);states=None;hidden=None;zs=[]
    for t in range(frames.shape[1]):
        field,states=model.encode_frame(frames[:,t],states)
        hidden=model.spatial_gru(model.spatial_input(field),hidden)
        zs.append(model.heads[task](model.readout(hidden.flatten(1)).relu()))
    return torch.stack(zs,1)

def bootstrap(a):
    a=np.asarray(a,float);r=np.random.default_rng(935039);b=a[r.integers(0,len(a),(4000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),bootstrap95=np.quantile(b,[.025,.975]).tolist(),n=len(a))
def probs(z):
    z=np.asarray(z);p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
def summarize(rows):
    result={'conditions':[],'paired':[],'temporal':[],'duration_relations':{}}
    groups={}
    for row in rows:
        key=(row['task'],row['condition']);g=groups.setdefault(key,dict(task=key[0],condition=key[1],labels=[],logits=[],trajectory=[],metadata=[]))
        for name in ('labels','logits','trajectory','metadata'):g[name].extend(row[name])
    for (task,condition),g in groups.items():
        z=np.array(g['trajectory']);y=np.array(g['labels']);pred=z.argmax(2);m=g['metadata'];n=len(y)
        result['conditions'].append(dict(task=task,condition=condition,**metrics(y,g['logits'])))
        if task=='motion_duration_cued':
            for label,t in [('last_moving',9),('report',z.shape[1]-1)]:
                result['temporal'].append(dict(task=task,condition=condition,epoch=label,timestep=t,**metrics(y,z[:,t])))
            correct=(pred==y[:,None]).astype(float)
            result['temporal'].append(dict(task=task,condition=condition,report_minus_last_moving_accuracy=bootstrap(correct[:,-1]-correct[:,9])))
            if condition=='native_D0':
                first=np.array([a['directions_by_patch'][a['target_location']][0] for a in m]);last=np.array([a['directions_by_patch'][a['target_location']][-1] for a in m]);winner=y
                uncued=np.array([[a['winner_by_patch'][j] for j in range(4) if j!=a['target_location']] for a in m])
                margin=np.array([np.diff(np.sort(a['duration_counts_by_patch'][a['target_location']])[-2:])[0] for a in m]);speeds=np.array([a['step_pixels'] for a in m])
                relations={}
                for t in (9,10):
                    p=pred[:,t];d=dict(first_agreement=bootstrap(p==first),last_agreement=bootstrap(p==last),majority_agreement=bootstrap(p==winner),mean_uncued_agreement=bootstrap((p[:,None]==uncued).mean(1)),first_equals_majority=float((first==winner).mean()),last_equals_majority=float((last==winner).mean()))
                    for name,target in [('first',first),('last',last)]:
                        mask=target!=winner;d[name+'_discordant_n']=int(mask.sum());d[name+'_agreement_when_discordant']=float((p[mask]==target[mask]).mean());d['majority_agreement_when_'+name+'_discordant']=float((p[mask]==winner[mask]).mean())
                    d['by_speed']={str(v):metrics(y[speeds==v],z[speeds==v,t]) for v in np.unique(speeds)}
                    d['by_duration_margin']={str(v):metrics(y[margin==v],z[margin==v,t]) for v in np.unique(margin)}
                    relations[str(t)]=d
                result['duration_relations']=relations
        elif task=='krauzlis_cued_motion':
            event=m[0]['virtual_event_frame'];epochs={'pre_event':event-1,'first_postevent':event,'last_moving':z.shape[1]-2,'report':z.shape[1]-1};p=probs(z)[:,:,1];events=np.array([a['event_type'] for a in m])
            for label,t in epochs.items():
                rates={e:dict(n=int((events==e).sum()),positive_rate=float((pred[events==e,t]==1).mean()),mean_p1=float(p[events==e,t].mean())) for e in ('target','foil','catch') if (events==e).any()}
                result['temporal'].append(dict(task=task,condition=condition,epoch=label,timestep=t,event_groups=rates,**metrics(y,z[:,t])))
            result['temporal'].append(dict(task=task,condition=condition,postevent_minus_pre_event_p1={e:bootstrap(p[events==e,-2]-p[events==e,event-1]) for e in ('target','foil','catch') if (events==e).any()},report_minus_last_moving_p1=bootstrap(p[:,-1]-p[:,-2]),report_minus_last_moving_accuracy=bootstrap((pred[:,-1]==y).astype(float)-(pred[:,-2]==y).astype(float))))
    for (task,condition),g in groups.items():
        baseline='native_D0' if task=='motion_duration_cued' else 'native_B20'
        if condition not in ('native_D24','cue_retarget_D0','cue_swap_B20'):continue
        b=groups[(task,baseline)];comparison=paired(b,g);z=np.array(b['logits']);q=np.array(g['logits']);y=np.array(b['labels']);v=np.array(g['labels']);ids=np.flatnonzero(y!=v)
        if len(ids):
            shift=(q[ids,v[ids]]-q[ids,y[ids]])-(z[ids,v[ids]]-z[ids,y[ids]])
            comparison['new_vs_old_label_margin_shift']=bootstrap(shift)
        result['paired'].append(dict(task=task,condition=condition,**comparison))
    return result

def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert sha(CKPT)==SHA
    cp=torch.load(CKPT,map_location='cpu',weights_only=False)
    model=SpatialReadout(task_classes());model.load_state_dict(cp['model'],strict=True)
    original_grad={k:p.requires_grad for k,p in model.named_parameters()};model.eval();model.requires_grad_(False)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    names=cp['optimizer_names'];opt=cp['optimizer'];head_states=[]
    # Serialized names correspond explicitly to optimizer groups; no optimizer instantiated.
    if isinstance(names,dict):name_groups=names.get('param_groups',names)
    else:name_groups=names
    dump('model_checkpoint_inspection.json',dict(schema=cp['schema'],state=cp['state'],optimizer_names=names,model_constructor_requires_grad=original_grad,checkpoint_sha256=SHA))
    if isinstance(name_groups,list):
        if name_groups and isinstance(name_groups[0],str):name_groups=[name_groups]
        for ns,group in zip(name_groups,opt['param_groups']):
            if isinstance(ns,dict):ns=ns.get('params',ns.get('names'))
            for name,idx in zip(ns,group['params']):
                if 'heads.' in name:
                    s=opt['state'].get(idx,{});head_states.append(dict(name=name,optimizer_step=float(s['step']) if 'step' in s else None,exp_avg_norm=float(s['exp_avg'].norm()) if 'exp_avg' in s else None))
    dump('model_optimizer_head_states.json',head_states)
    sources=['SecondPass/SpatialReadout/model.py','WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py','WorkingMemory/PlainBaseline/accum.py','PreAttentiveVision/TemporalIntegration/accumulators.py'];hashes={p:sha(p) for p in sources}
    budget_path=OUT/'model_budget.json'
    if budget_path.exists():
        budget=json.loads(budget_path.read_text());deadline=budget['deadline_unix'];assert time.time()<deadline-150,'Nonrenewable budget expired'
    else:
        start=time.time();deadline=start+1200;budget=dict(start_unix=start,deadline_unix=deadline,cap_seconds=1200,renewable=False,cpu_threads=1,interop_threads=1,pid=os.getpid(),first_accelerator_operation='model.to(mps)');dump('model_budget.json',budget)
    def alarm(*args):
        dump('model_cap_reached.json',dict(time=time.time(),deadline=deadline));os._exit(124)
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    attempt=OUT/'model_attempts.jsonl'
    with attempt.open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),started=time.time()))+'\n')
    model.to('mps');parity=[];rows=[];times={}
    def evaluate(records,production):
        for record in records:
            task=record['task'];c=record['condition'];xx=record['x'];zs=[];t0=time.time()
            for b in range(0,len(xx),8):
                assert time.time()<deadline-90,'Finalization reserve reached'
                x=xx[b:b+8].to('mps');z=trajectory(model,x,task)
                if not production:
                    direct=model(x,task);err=float((direct-z[:,-1]).abs().max().cpu());assert err<=1e-6;parity.append(dict(task=task,condition=c,max_abs_error=err))
                zs.extend(z.cpu().tolist())
            elapsed=time.time()-t0
            if not production:times[c]=elapsed;continue
            row=dict(task=task,condition=c,labels=record['y'].tolist(),logits=[a[-1] for a in zs],trajectory=zs,metadata=record['metadata'])
            rows.append(row)
            with (OUT/'model_trials.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(task=task,condition=c,n=len(xx),seconds=elapsed,elapsed=time.time()-budget['start_unix'])),flush=True)
    # Cheap first-eight timing, disjoint seeds; these draws are not analysis rows.
    evaluate(duration_records(8,93635039),False);evaluate(krauzlis_records(8,93735039),False)
    rate_d=sum(times[c] for c in ('native_D0','native_D24','cue_retarget_D0'))/8
    rate_k=sum(times[c] for c in ('native_B20','cue_swap_B20'))/8
    n_d,n_k=128,200;available=deadline-time.time()-180
    estimate=n_d*rate_d+n_k*rate_k
    if estimate>available:n_d,n_k=64,100
    assert n_d*rate_d+n_k*rate_k<available,'Too slow even for preregistered reduced panel'
    dump('model_sample_plan.json',dict(timing8_seconds=times,duration_n=n_d,krauzlis_n=n_k,full_requested=[128,200],estimated_seconds=n_d*rate_d+n_k*rate_k,remaining_before_reserve=available,chosen_before_production=True))
    assert not (OUT/'model_trials.jsonl').exists(),'Do not silently mix a resumed attempt with old trials'
    for b in range(0,n_d,16):evaluate(duration_records(min(16,n_d-b),92635039+b),True)
    st=SpatialBatteryStream(92735039,'test')
    for b in range(0,n_k,8):evaluate(krauzlis_records(min(8,n_k-b),stream=st),True)
    # Native successful sensory domain anchor; not a same-patch sensory control.
    x,y,m=TaskStream(92835039,'test').batch(32,'motion_direction');evaluate([dict(task='motion_direction',condition='native_anchor',x=x,y=y,metadata=m)],True)
    torch.mps.synchronize();immutable=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in initial.items());assert immutable
    assert sha(CKPT)==SHA and all(sha(p)==h for p,h in hashes.items())
    # Read persisted raw records rather than trusting the in-memory enumeration.
    saved=[json.loads(line) for line in (OUT/'model_trials.jsonl').read_text().splitlines()];summary=summarize(saved)
    counts={r['condition']:r['n'] for r in summary['conditions']};assert counts==dict(native_D0=n_d,native_D24=n_d,cue_retarget_D0=n_d,native_B20=n_k,cue_swap_B20=n_k,native_anchor=32)
    events=[m['event_type'] for r in saved if r['condition']=='native_B20' for m in r['metadata']];event_counts={k:events.count(k) for k in ('target','foil','catch')};assert event_counts==dict(target=n_k*57//100,foil=n_k*29//100,catch=n_k*14//100)
    summary.update(checkpoint=str(CKPT),checkpoint_sha256=SHA,step=cp['state']['step'],all_state_tensors_bitwise_unchanged=immutable,checkpoint_file_unchanged=True,source_hashes_unchanged=hashes,eval_mode=not model.training,all_requires_grad_false=all(not p.requires_grad for p in model.parameters()),training_updates=0,probe_fits=0,wrapper_parity=parity,event_counts=event_counts,elapsed_seconds=time.time()-budget['start_unix'],deadline_unix=deadline,completed_unix=time.time(),counts_verified=True)
    dump('model_summary.json',summary)
    lines=['# Frozen current-checkpoint motion diagnosis','',f"Checkpoint step {summary['step']}; SHA256 `{SHA}`. Frozen inference only, zero training/probe fits. Counts verified: {counts}. Elapsed {summary['elapsed_seconds']:.2f}s / 1200s nonrenewable cap.",'','|Task|Condition|N|Balanced accuracy|Macro AUC|Predicted class counts|','|---|---|---:|---:|---:|---|']
    for r in summary['conditions']:lines.append(f"|{r['task']}|{r['condition']}|{r['n']}|{r['balanced_accuracy']:.4f}|{r['auc_macro']:.4f}|{r['prediction_counts']}|")
    lines+=['','## Paired effects','```json',json.dumps(summary['paired'],indent=2),'```','','## Temporal readout','```json',json.dumps(summary['temporal'],indent=2),'```','','## Interpretation and scope','Only native final-report behavior is trained/deployed. Earlier-timestep scores apply the same unchanged readout to states without the native report phase: exploratory/OOD, not a fitted decoder or reaction time. D0/D24 have exactly identical nonblank evidence and labels. Stack3 means first two nominal blank updates retain motion images; subsequent D24 updates do not. Cue retargeting uses captured native rasters and preserves all dot pixels; Krauzlis edits only cue frames0/1 and leaves every movie/report pixel exact. Changed labels are recomputed from the new target; catches stay negative. Cue counterfactuals deliberately change label frequencies and are not classical cue-validity psychometrics. A native sensory anchor differs in aperture, density, speed and temporal demand: success is not proof of tiny-patch motion encoding. Single frozen checkpoint with synthetic independent episodes; intervals condition on this checkpoint. Failed behavior cannot uniquely distinguish inadequate local motion encoding, spatial selection, integration, retained state, or learned readout. No architecture rejection or exact causal mechanism is justified. No internal spatial inhibition/stimulation, attention maps, new decoder fits, or cloud changes performed.','', 'Source rules: `WorkingMemory/SpatialTaskBattery/stimuli.py:104-111` (duration), `:121-142` (Krauzlis), `:69-78` (sampling); `WorkingMemory/PlainBaseline/accum.py:54-68` (stack3/encoder); `SecondPass/SpatialReadout/model.py:38-51` (recurrence/final head); `ANALYSIS_SOP.md:98-104,156-165` (stack leakage, paired uncertainty, interpretation).','', 'Raw complete timestep logits, labels, seeds/trial IDs, schedules, event metadata and raster checks: `model_trials.jsonl`. Detailed metrics/relations: `model_summary.json`. Checkpoint/optimizer receipt: `model_checkpoint_inspection.json`, `model_optimizer_head_states.json`.']
    (OUT/'MODEL_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    assert time.time()<deadline;dump('model_completion.json',dict(status='complete',finished_unix=time.time(),elapsed_seconds=time.time()-budget['start_unix'],within_cap=True,all_weights_unchanged=immutable,counts=counts));signal.setitimer(signal.ITIMER_REAL,0)
    print('COMPLETE '+json.dumps(summary['conditions']),flush=True)

if __name__=='__main__':run()
