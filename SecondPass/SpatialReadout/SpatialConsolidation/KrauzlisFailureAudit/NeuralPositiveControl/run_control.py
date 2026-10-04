"""One predeclared architecture, finite nonrenewable 1800 s experiment."""
from control import *
import time,json,hashlib,signal,traceback,copy,collections
OUT=Path(__file__).resolve().parent
BUDGET=1800
START=None
DEVICE=None
records=[]
seen={}

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2))

def emit(kind,**kw):
    obj={'kind':kind,'elapsed':time.time()-START if START else None,**kw}
    with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(obj)+'\n')
    print(json.dumps(obj),flush=True)

def sync():
    if DEVICE=='mps': torch.mps.synchronize()

def remaining(): return START+BUDGET-time.time()

def register(x,y,meta,split):
    for i,m in enumerate(meta):
        assert int(y[i])==int(m['event_type']=='target')
        assert m['event_magnitude_degrees']==90
        assert x.shape[1]==m['baseline_transitions']+17
        h=hashlib.sha256(x[i,2:].numpy().tobytes()).hexdigest()
        if h in seen: assert seen[h]==split, 'Physical scene crossed a split'
        seen[h]=split
        m.update(physical_scene_sha256=h,experiment_split=split)
        records.append(m)
    return x,y,meta

def batch(stream,n,b,split): return register(*stream.batch(n,b),split)

def evaluate(model,bank,save=None):
    model.eval(); rows=[]
    with torch.no_grad():
        for x,y,meta in bank:
            for start in range(0,len(y),2):
                logits=model(x[start:start+2].to(DEVICE)).cpu()
                probs=logits.softmax(-1)[:,1].numpy()
                losses=F.cross_entropy(logits,y[start:start+2],reduction='none').numpy()
                for j,(p,l) in enumerate(zip(probs,losses)):
                    rows.append(dict(meta[start+j],label=int(y[start+j]),probability=float(p),loss=float(l),prediction=int(p>=.5)))
    if save: dump(save,rows)
    return rows

def basic(rows):
    y=np.array([r['label'] for r in rows]); p=np.array([r['probability'] for r in rows]); pred=p>=.5
    pos=y==1; neg=~pos
    auc=float(((p[pos,None]>p[None,neg])+.5*(p[pos,None]==p[None,neg])).mean()) if pos.any() and neg.any() else None
    return dict(n=len(rows),loss=float(np.mean([r['loss'] for r in rows])),accuracy=float((pred==y).mean()),balanced_accuracy=float(((pred[pos]).mean()+(~pred[neg]).mean())/2) if pos.any() and neg.any() else None,auc=auc,tp=int((pred&pos).sum()),tn=int((~pred&neg).sum()),fp=int((pred&neg).sum()),fn=int((~pred&pos).sum()),positive_rate=float(pred.mean()))

def metrics(rows):
    result=basic(rows)
    result['by_event']={event:basic([r for r in rows if r['event_type']==event]) for event in ('target','foil','catch')}
    result['by_B']={str(b):basic([r for r in rows if r['baseline_transitions']==b]) for b in (12,20,28)}
    rng=np.random.default_rng(197); boots=[]
    # Each row is a unique physical scene; stratify by B and label.
    strata=[[r for r in rows if r['baseline_transitions']==b and r['label']==y] for b in (12,20,28) for y in (0,1)]
    for _ in range(500):
        sample=[s[i] for s in strata if s for i in rng.integers(len(s),size=len(s))]
        m=basic(sample); boots.append([m['balanced_accuracy'],m['auc']])
    result['stratified_physical_scene_bootstrap_95ci']={key:np.quantile(np.array(boots)[:,i],[.025,.975]).tolist() for i,key in enumerate(('balanced_accuracy','auc'))}
    return result

def update(model,opt,batches):
    model.train(); opt.zero_grad(set_to_none=True); count=sum(len(y) for _,y,_ in batches); lossvalue=0.
    for x,y,_ in batches:
        x=x.to(DEVICE); y=y.to(DEVICE)
        loss=F.cross_entropy(model(x),y)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        (loss*len(y)/count).backward(); lossvalue+=float(loss.detach().cpu())*len(y)/count
    grad=math.sqrt(sum(float(p.grad.square().sum().detach().cpu()) for p in model.parameters() if p.grad is not None))
    if not math.isfinite(grad) or grad==0: raise RuntimeError('Invalid gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    opt.step()
    return lossvalue,grad

def build_bank(seed,split,n):
    return [batch(NativeStream(seed+i,split),n,b,split) for i,b in enumerate((12,20,28))]

def checkpoint(model,opt,name,step):
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'model_seed':72002,'pretrained':False},OUT/name)

def finish(result):
    result.update(elapsed_seconds=time.time()-START,budget_seconds=BUDGET,device=DEVICE,cpu_threads=torch.get_num_threads(),unique_physical_scenes=len(seen))
    dump('result.json',result); dump('scene_manifest.json',records)
    lines=['# Conventional neural positive control — actual execution','',f"Status: **{result.get('status','inconclusive')}**.",'',f"Elapsed {result['elapsed_seconds']:.1f} / {BUDGET} seconds; device {DEVICE}; one CPU thread.",'', '## Fixed-set diagnostic',json.dumps(result.get('overfit',{}),indent=2),'','## Fresh-stream generalization',json.dumps(result.get('generalization',{}),indent=2),'','## Integrity checks',json.dumps(result.get('checks',{}),indent=2),'','## Interpretation','A fixed-set fit is memorization/optimization evidence, not native-task acquisition. Only independently initialized fresh-stream held-out performance addresses generalization. No biological plausibility or universal neural solvability claim. This is one architecture, one seed, a limited wall budget, not an architecture sweep. Native timing, RGB, target/foil/catch meaning and ±90 event laws were not changed.','', 'Architecture, protocol, equations, seeds and limitations: see PROTOCOL.md. Full trial predictions and losses are saved, allowing independent metric replay.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'JOURNAL.md').write_text('# Execution journal\n\n'+f"Completed in {result['elapsed_seconds']:.1f}s. Outcome: {result.get('status')}.\n\n"+'See result.json, progress.jsonl, checks.json, optimizer checkpoints, prediction artifacts and REPORT.md. Original workers/renderers/checkpoints were not modified or loaded.\n')
    emit('complete',status=result.get('status'),elapsed_total=time.time()-START)

def main():
    global START,DEVICE
    if (OUT/'budget.json').exists(): raise RuntimeError('Nonrenewable budget already exists; no automatic restart')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
    # Persist before first MPS operation, including disposal/profile/diagnostic/eval/report.
    START=time.time(); dump('budget.json',dict(start_unix=START,deadline_unix=START+BUDGET,budget_seconds=BUDGET,renewable=False,pid=os.getpid(),device=DEVICE))
    def alarm(*_):
        dump('deadline_stop.json',dict(elapsed=time.time()-START,pid=os.getpid())); raise TimeoutError('Hard wall deadline')
    signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,BUDGET-3)
    result={}
    try:
        torch.manual_seed(72000); model=Model().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        # Fresh 32-example set: 16 target, 8 foil, 8 catch; all native B values.
        stream=NativeStream(910001,'train'); quota={'target':16,'foil':8,'catch':8}; accepted=[]; draws=0
        while sum(quota.values()):
            b=(12,20,28)[draws%3]; x,y,meta=stream.batch(1,b); draws+=1
            event=meta[0]['event_type']
            if quota[event]: quota[event]-=1; accepted.append(register(x,y,meta,'overfit'))
        dump('overfit_manifest.json',[m[0] for _,_,m in accepted])
        initial=evaluate(model,accepted); before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        st=time.time(); firstloss,grad=update(model,opt,accepted[:2]); sync(); profile=time.time()-st
        changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in model.named_parameters()}
        checks=dict(profile_update_seconds=profile,parameters=sum(p.numel() for p in model.parameters()),first_gradient_norm=grad,first_update_max_per_parameter=changes,all_parameter_tensors_updated=all(v>0 for v in changes.values()),initial_overfit=basic(initial),initial_optimizer_empty=True)
        # Identical grouping/single inference check and all-time input gradient on MPS.
        x,y,_=accepted[0]; x=x.to(DEVICE).requires_grad_(); model.zero_grad(set_to_none=True); F.cross_entropy(model(x),y.to(DEVICE)).backward(); g=x.grad.detach().abs().sum((0,2,3,4)).cpu().tolist()
        checks['input_gradient_by_frame']=g; assert min(g)>0
        model.eval()
        with torch.no_grad():
            other=next(a for a in accepted[1:] if a[0].shape[1]==accepted[0][0].shape[1])
            xx=torch.cat([accepted[0][0],other[0]]).to(DEVICE)
            e=(model(xx)-torch.cat([model(z[None]) for z in xx])).abs().max().item()
        checks['MPS_batch_inference_max_error']=e; assert e<1e-4
        dump('checks.json',checks); result['checks']=checks; emit('profile',**checks)
        rng=np.random.default_rng(501); history=[]; fit=False; steps=1; diagnostic_end=START+650
        # Full fixed-set epoch, exact per-example CE weighting, no validation tuning.
        while time.time()<diagnostic_end:
            order=rng.permutation(32)
            for start in range(0,32,8):
                chosen=[accepted[i] for i in order[start:start+8]]
                loss,grad=update(model,opt,chosen); steps+=1
            rows=evaluate(model,accepted); m=basic(rows); history.append(dict(step=steps,elapsed=time.time()-START,**m)); emit('overfit',step=steps,gradient_norm=grad,**m)
            if m['accuracy']>=.96875 and m['loss']<=.15: fit=True; break
        dump('overfit_curve.json',history); dump('overfit_predictions.json',rows)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':steps,'seed':72000},OUT/'overfit_only.pt')
        result['overfit']=dict(initial=basic(initial),final=basic(rows),fit_gate=fit,steps=steps,unique_trials=32,curation_native_draws=draws,trial_presentations=2+(steps-1)*8,gate='accuracy>=31/32 AND CE<=0.15',never_transferred_to_generalization=True)
        if not fit:
            # Cheap shuffled-label fit tests capacity, not an inference null test.
            del model,opt
            torch.manual_seed(72001); model=Model().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
            perm=np.random.default_rng(502).permutation(32); labels=[accepted[i][1].clone() for i in perm]
            shuffled=[(a[0],labels[i],a[2]) for i,a in enumerate(accepted)]
            pre=basic(evaluate(model,shuffled)); stop=min(time.time()+120,START+1470); n=0
            while time.time()<stop:
                order=rng.permutation(32)
                for i in range(0,32,8): update(model,opt,[shuffled[j] for j in order[i:i+8]]); n+=1
            post=basic(evaluate(model,shuffled)); result['permuted_label_diagnostic']=dict(initial=pre,final=post,steps=n,seconds_cap=120,permutation=perm.tolist())
            result['generalization']={'executed':False,'reason':'Prespecified fixed-set fit gate failed; no fresh generalization run justified'}
            result['status']='inconclusive: fixed-set fit gate failed within diagnostic allocation'; finish(result); return
        # Clean init, optimizer, independent training RNG and physical streams.
        del model,opt
        if DEVICE=='mps': torch.mps.empty_cache()
        torch.manual_seed(72002); model=Model().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
        assert len(opt.state)==0
        initial_state={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        torch.manual_seed(72002); reference=Model()
        assert all(torch.equal(initial_state[n],p.detach()) for n,p in reference.named_parameters()); del reference
        checks['generalization_direct_fresh_init_exact']=True; checks['generalization_initial_optimizer_empty']=True
        val=build_bank(930001,'val',100)
        pre=evaluate(model,val,'validation_initial_predictions.json'); best=basic(pre)['loss']; beststep=0
        checkpoint(model,opt,'selected.pt',0); valcurve=[dict(step=0,**basic(pre))]
        streams=[NativeStream(920001+i,'train') for i in range(3)]; counts=collections.Counter(); events=collections.Counter(); step=0; lastval=time.time()
        trainrng=np.random.default_rng(503); source_ids=[]
        while time.time()<START+1450:
            cell=step%3; b=(12,20,28)[cell]
            x,y,meta=batch(streams[cell],8,b,'train'); counts[b]+=8; events.update(m['event_type'] for m in meta)
            chunks=[(x[i:i+2],y[i:i+2],meta[i:i+2]) for i in range(0,8,2)]
            loss,grad=update(model,opt,chunks); step+=1
            if step%10==0: emit('train',step=step,episodes=step*8,loss=loss,gradient_norm=grad)
            if time.time()-lastval>=180 and time.time()<START+1370:
                v=evaluate(model,val); m=basic(v); valcurve.append(dict(step=step,**m)); emit('validation',step=step,**m)
                if m['loss']<best: best=m['loss']; beststep=step; checkpoint(model,opt,'selected.pt',step)
                lastval=time.time()
        # Fixed wall-time endpoint, validation selection only, no test feedback.
        v=evaluate(model,val,'validation_terminal_predictions.json'); vm=basic(v); valcurve.append(dict(step=step,**vm))
        if vm['loss']<best: best=vm['loss']; beststep=step; checkpoint(model,opt,'selected.pt',step)
        checkpoint(model,opt,'terminal.pt',step); dump('validation_curve.json',valcurve)
        dump('selection.json',dict(selected_step=beststep,terminal_step=step,criterion='minimum validation CE including fresh step0',selection_frozen_before_test=True,train_exposure=step*8,train_by_B=dict(counts),train_events=dict(events)))
        test=build_bank(940001,'test',200)
        terminal_rows=evaluate(model,test,'test_terminal_predictions.json'); terminal=metrics(terminal_rows)
        selected_state=torch.load(OUT/'selected.pt',map_location='cpu',weights_only=False); model.load_state_dict(selected_state['model']); del selected_state
        selected_rows=evaluate(model,test,'test_selected_predictions.json'); selected=metrics(selected_rows)
        changes={n:float((p.detach().cpu()-initial_state[n]).abs().max()) for n,p in model.named_parameters()}
        checks['selected_change_from_generalization_init']=changes; dump('checks.json',checks)
        result['generalization']=dict(executed=True,model_seed=72002,optimizer='AdamW lr0.0005 wd0.0001 clip5',train_episodes=step*8,updates=step,train_by_B=dict(counts),train_events=dict(events),initial_validation=basic(pre),terminal_validation=vm,selected_step=beststep,terminal_step=step,selected_test=selected,terminal_test=terminal)
        result['status']='held-out learning evidence' if selected['stratified_physical_scene_bootstrap_95ci']['balanced_accuracy'][0]>.5 and selected['stratified_physical_scene_bootstrap_95ci']['auc'][0]>.5 else 'inconclusive: no established held-out task acquisition in budget'
        finish(result)
    except Exception as exc:
        result.update(status='blocked or deadline-limited',error=repr(exc),traceback=traceback.format_exc()); finish(result); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__': main()
