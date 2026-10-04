"""One diagnosed optimizer correction; original absolute deadline, same model.
Not an architecture search, not a renewed budget, not an overfit warm start.
"""
import run_control as r
from control import *
import time,json,signal,collections
OUT=Path(__file__).resolve().parent
budget=json.loads((OUT/'budget.json').read_text()); deadline=budget['deadline_unix']
assert time.time()<deadline-300,'Insufficient original budget; no continuation'
assert not (OUT/'optimizer_correction_started.json').exists(),'One correction only'
torch.set_num_threads(1); torch.set_num_interop_threads(1)
r.START=budget['start_unix']; r.DEVICE=budget['device']
prior=json.loads((OUT/'result.json').read_text())
r.dump('initial_attempt_result.json',prior)
r.dump('optimizer_correction_started.json',dict(elapsed=time.time()-r.START,deadline_unix=deadline,architecture_changed=False,reason='Saved checkpoint shows 93.75% exactly-zero dense GELU activations and sharply reduced upstream gradients',changes='AdamW lr 0.001 -> 0.0001, effective batch32 instead of8 for fixed set; no initial2-example update',same_initial_model_seed=72000))
def alarm(*_): raise TimeoutError('Original hard deadline')
signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,max(1,deadline-time.time()-3))
result={'initial_attempt':prior,'checks':prior['checks']}
try:
    stream=NativeStream(910001,'train'); quota={'target':16,'foil':8,'catch':8}; accepted=[]; draws=0
    while sum(quota.values()):
        b=(12,20,28)[draws%3]; x,y,meta=stream.batch(1,b); draws+=1
        if quota[meta[0]['event_type']]: quota[meta[0]['event_type']]-=1; accepted.append(r.register(x,y,meta,'overfit'))
    torch.manual_seed(72000); model=Model().to(r.DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=0)
    initial=r.evaluate(model,accepted); curve=[]; step=0; fit=False
    while time.time()<r.START+1320:
        loss,grad=r.update(model,opt,accepted); step+=1
        if step%5==0:
            rows=r.evaluate(model,accepted); m=r.basic(rows); curve.append(dict(step=step,elapsed=time.time()-r.START,**m)); r.emit('corrected_overfit',step=step,gradient_norm=grad,**m)
            if m['accuracy']>=.96875 and m['loss']<=.15: fit=True; break
    rows=r.evaluate(model,accepted); r.dump('corrected_overfit_predictions.json',rows); r.dump('corrected_overfit_curve.json',curve)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'seed':72000},OUT/'corrected_overfit_only.pt')
    result['overfit']=dict(initial=r.basic(initial),final=r.basic(rows),fit_gate=fit,steps=step,unique_trials=32,trial_presentations=step*32,optimizer_correction=True,architecture_changes=0,never_transferred_to_generalization=True)
    if not fit:
        result['status']='inconclusive: same-architecture optimizer correction did not pass tiny-set gate'
        result['generalization']={'executed':False,'reason':'Native fixed-set gate still failed'}
    else:
        del model,opt
        if r.DEVICE=='mps': torch.mps.empty_cache()
        torch.manual_seed(72002); model=Model().to(r.DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
        assert not opt.state
        init={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        torch.manual_seed(72002); direct=Model()
        assert all(torch.equal(init[n],p.detach()) for n,p in direct.named_parameters()); del direct
        result['checks']['generalization_direct_fresh_init_exact']=True; result['checks']['generalization_initial_optimizer_empty']=True
        val=r.build_bank(930001,'val',100); initialval=r.evaluate(model,val,'validation_initial_predictions.json'); best=r.basic(initialval)['loss']; beststep=0; r.checkpoint(model,opt,'selected.pt',0)
        curve=[dict(step=0,**r.basic(initialval))]; streams=[NativeStream(920001+i,'train') for i in range(3)]; counts=collections.Counter(); events=collections.Counter(); step=0; lastval=time.time()
        while time.time()<r.START+1660:
            cell=step%3; b=(12,20,28)[cell]; x,y,meta=r.batch(streams[cell],8,b,'train'); counts[b]+=8; events.update(m['event_type'] for m in meta)
            loss,grad=r.update(model,opt,[(x[i:i+2],y[i:i+2],meta[i:i+2]) for i in range(0,8,2)]); step+=1
            if step%10==0: r.emit('fresh_train',step=step,episodes=step*8,loss=loss,gradient_norm=grad)
            if time.time()-lastval>120 and time.time()<r.START+1580:
                vr=r.evaluate(model,val); m=r.basic(vr); curve.append(dict(step=step,**m)); r.emit('fresh_validation',step=step,**m)
                if m['loss']<best: best=m['loss']; beststep=step; r.checkpoint(model,opt,'selected.pt',step)
                lastval=time.time()
        vr=r.evaluate(model,val,'validation_terminal_predictions.json'); vm=r.basic(vr); curve.append(dict(step=step,**vm))
        if vm['loss']<best: best=vm['loss']; beststep=step; r.checkpoint(model,opt,'selected.pt',step)
        r.checkpoint(model,opt,'terminal.pt',step); r.dump('validation_curve.json',curve)
        r.dump('selection.json',dict(selected_step=beststep,terminal_step=step,criterion='minimum validation CE including step0',selection_frozen_before_test=True,train_exposure=step*8,train_by_B=dict(counts),train_events=dict(events),optimizer='AdamW lr0.0001 wd0.0001 clip5',original_deadline_unix=deadline))
        test=r.build_bank(940001,'test',200)
        tr=r.evaluate(model,test,'test_terminal_predictions.json'); terminal=r.metrics(tr)
        selected_state=torch.load(OUT/'selected.pt',map_location='cpu',weights_only=False); model.load_state_dict(selected_state['model']); del selected_state
        sr=r.evaluate(model,test,'test_selected_predictions.json'); selected=r.metrics(sr)
        result['generalization']=dict(executed=True,model_seed=72002,optimizer='AdamW lr0.0001 wd0.0001 clip5',train_episodes=step*8,updates=step,train_by_B=dict(counts),train_events=dict(events),initial_validation=r.basic(initialval),terminal_validation=vm,selected_step=beststep,terminal_step=step,selected_test=selected,terminal_test=terminal)
        result['status']='held-out learning evidence' if selected['stratified_physical_scene_bootstrap_95ci']['balanced_accuracy'][0]>.5 and selected['stratified_physical_scene_bootstrap_95ci']['auc'][0]>.5 else 'inconclusive: no established held-out task acquisition in budget'
    r.dump('checks.json',result['checks']); r.finish(result)
except Exception as e:
    import traceback
    result.update(status='blocked or deadline-limited',error=repr(e),traceback=traceback.format_exc()); r.finish(result); raise
finally: signal.setitimer(signal.ITIMER_REAL,0)
