"""Fresh native single-head shared-attention training, one finite local allocation."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from SecondPass.PixelContrastiveVAE.train import GPULock,atomic_json,append_json,cpu_tree,replace_with_retry,configure,synchronize,metrics
from . import dataset
from .model import StabilizedAttentionKrauzlis
from .memory import GATE_NAMES

ROOT=Path(__file__).resolve().parents[2]
POOL_UPDATES=66
INIT_SEED,SHUFFLE_SEED=197211,197212


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in (
        'SecondPass/PixelContrastiveVAE/train.py',
        'SecondPass/PredictiveConsistencyVAE/queue.py','WorkingMemory/SpatialTaskBattery/stimuli.py',
        'WorkingMemory/stimuli.py','PreAttentiveVision/neuroscience_stimuli.py',
        'PreAttentiveVision/natural_stimuli.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def remaining(deadline,reserve=120):
    if time.time()>=deadline-reserve: raise TimeoutError('Finite local cap; no automatic extension')


def build(device):
    random.seed(INIT_SEED); np.random.seed(INIT_SEED); torch.manual_seed(INIT_SEED)
    if device=='cuda': torch.cuda.manual_seed_all(INIT_SEED)
    model=StabilizedAttentionKrauzlis().to(device)
    return model,torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-8,weight_decay=0.)


def batches(rows,rng):
    groups={cell:torch.randperm(len(items),generator=rng).split(32) for cell,items in rows.items()}
    result=[]
    for i in range(max(map(len,groups.values()))):
        order=torch.randperm(len(groups),generator=rng).tolist()
        for j in order:
            cell=list(groups)[j]
            if i<len(groups[cell]): result.append((cell,groups[cell][i].tolist()))
    return result


def update(model,opt,rows,indices,device,micro=4):
    model.train(); opt.zero_grad(set_to_none=True)
    totals=dict(loss=0.,decision_loss=0.,prediction_loss=0.,accuracy=0.,cls_std=0.,attention_entropy=0.,attention_peak=0.,attention_scale=0.,**{k:0. for k in GATE_NAMES})
    for start in range(0,len(indices),micro):
        chosen=[rows[i] for i in indices[start:start+micro]]
        x=torch.stack([r[0] for r in chosen]).to(device)
        y=torch.tensor([r[1] for r in chosen],device=device)
        logits,detail=model(x,return_details=True)
        ce=F.cross_entropy(logits,y); prediction=detail['prediction_loss']; loss=ce+.05*prediction
        weight=len(chosen)/len(indices); (weight*loss).backward()
        values=dict(loss=loss,decision_loss=ce,prediction_loss=prediction,
                    accuracy=(logits.argmax(-1)==y).float().mean(),
                    cls_std=detail['cls'].std(0,unbiased=False).mean())
        alpha=detail['attention'].flatten(2).detach()
        values.update(attention_entropy=-(alpha*alpha.clamp_min(1e-12).log()).sum(-1).mean(),
            attention_peak=alpha.max(-1).values.mean(),attention_scale=detail['attention_scale'])
        values.update(zip(GATE_NAMES,detail['gate_statistics']))
        for k,v in values.items(): totals[k]+=weight*float(v.detach())
    missing=[n for n,p in model.named_parameters() if p.grad is None]
    if missing or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Missing/nonfinite gradients: '+str(missing))
    norm=torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
    if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite gradient norm')
    torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True)
    clipped_norm=torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
    opt.step(); totals.update(batch_size=len(indices),gradient_norm=float(norm),clipped_gradient_norm=float(clipped_norm),gradient_clipped=bool(norm>100))
    return totals


def summary(scores,labels):
    scores=np.asarray(scores); labels=np.asarray(labels); choices=scores>0
    result=metrics(scores,labels,0.,1)
    result['balanced_accuracy']=.5*((~choices[labels==0]).mean()+choices[labels==1].mean())
    result.update(TP=int(((labels==1)&choices).sum()),TN=int(((labels==0)&~choices).sum()),
                  FP=int(((labels==0)&choices).sum()),FN=int(((labels==1)&~choices).sum()))
    return result


@torch.no_grad()
def evaluate(model,split,count,device,deadline):
    model.eval(); results=[]
    for cell in dataset.CELLS:
        remaining(deadline,30)
        rows=dataset.render(split,cell,count,block=0)
        scores=[]; labels=[]; metadata=[]; monitoring=[]; gate_rows=[]
        coord=torch.arange(10,device=device)*10+4.5
        yy,xx=torch.meshgrid(coord,coord,indexing='ij')
        masks=torch.stack([((xx-c)**2+(yy-50)**2<12**2).flatten() for c in (20,80)]).float()
        for i in range(0,count,4):
            remaining(deadline,30)
            selected=rows[i:i+4]
            logits,detail=model(torch.stack([r[0] for r in selected]).to(device),return_details=True)
            a=detail['attention'].flatten(2)
            post=torch.tensor([r[2]['first_postchange_frame']+7 for r in selected],device=device)
            late=a[torch.arange(len(selected),device=device),post]
            mass=late@masks.T
            side=torch.tensor([r[2]['target_location'] for r in selected],device=device)
            ix=torch.arange(len(selected),device=device)
            monitoring.extend(torch.stack((mass[ix,side],mass[ix,1-side],late.max(-1).values,
                -(late*late.clamp_min(1e-12).log()).sum(-1)),dim=-1).cpu().tolist())
            gate_rows.append(detail['gate_statistics'].cpu().tolist())
            scores.extend((logits[:,1]-logits[:,0]).cpu().tolist())
            labels.extend(r[1] for r in selected); metadata.extend(r[2] for r in selected)
        score=np.array(scores); label=np.array(labels)
        ce=float(F.binary_cross_entropy_with_logits(torch.tensor(score),torch.tensor(label,dtype=torch.float64)))
        events={}
        for event in ('target','foil','catch'):
            ix=[i for i,m in enumerate(metadata) if m['event_type']==event]
            events[event]=dict(n=len(ix),change_response_rate=float((score[ix]>0).mean()) if ix else None)
        results.append(dict(cell='B'+str(cell),n=count,loss=ce,**summary(score,label),events=events,
            attention=dict(zip(('post_target_mass','post_foil_mass','post_peak','post_entropy'),np.mean(monitoring,axis=0).tolist())),
            gates=dict(zip(GATE_NAMES,np.mean(gate_rows,axis=0).tolist())),
            predictions=[dict(score=float(s),label=int(y),event=m['event_type'],target_location=m['target_location'],trial_id=m['trial_id'])
                         for s,y,m in zip(scores,labels,metadata)]))
    return dict(n=3*count,loss=float(np.mean([r['loss'] for r in results])),
        balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in results])),
        auc=float(np.mean([r['auc'] for r in results])),cells=results,
        attention={k:float(np.mean([r['attention'][k] for r in results])) for k in results[0]['attention']},
        gates={k:float(np.mean([r['gates'][k] for r in results])) for k in GATE_NAMES},
        decision='terminal argmax, tie0; no test threshold fitting')


def save(run,model,opt,state,config,rng,device,best=False):
    payload=cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,config=config,
        shuffle_rng=rng.get_state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None))
    for name in ('latest','best') if best else ('latest',):
        temp=run/(name+'.pt.tmp'); torch.save(payload,temp);replace_with_retry(temp,run/(name+'.pt'))


def profile(args,run):
    if (run/'budget.json').exists(): raise RuntimeError('Existing allocation; no renewal')
    started=time.time(); deadline=started+14400
    atomic_json(run/'budget.json',dict(started=started,deadline=deadline,wall_seconds=14400,worker_pid=os.getpid()))
    configure(args.device); model,opt=build(args.device)
    del model,opt
    clock=time.time();rows=dataset.pool(0);generation=time.time()-clock
    model,opt=build(args.device);timings=[]
    for i in range(4):
        remaining(deadline);clock=time.time()
        update(model,opt,rows[28],list(range(32)),args.device,args.microbatch);synchronize(args.device)
        if i: timings.append(time.time()-clock)
    clock=time.time();evaluate(model,'val',100,args.device,deadline);synchronize(args.device)
    eval_cost=(time.time()-clock)/300
    cost=1.25*(float(np.median(timings))+generation/POOL_UPDATES)
    target=args.target_updates
    while target>=POOL_UPDATES:
        validation_count=len(set(range(50,target+1,50))|{33,target})
        reserve=600+1.25*eval_cost*(3*args.val_count*validation_count+3*args.test_count)
        if target*cost+reserve<=deadline-time.time(): break
        target-=POOL_UPDATES
    if target<POOL_UPDATES: raise RuntimeError('No complete pool fits budget')
    atomic_json(run/'profile.json',dict(started=started,deadline=deadline,target_updates=target,
        requested_target_updates=args.target_updates,update_seconds=float(np.median(timings)),
        pool_generation_seconds=generation,evaluation_seconds_per_movie=eval_cost,reserve_seconds=reserve,
        profile_weights_discarded=True,source_hashes=source_hashes(),
        device=args.device,microbatch=args.microbatch,val_count=args.val_count,test_count=args.test_count))


def production(args,run):
    if (run/'production_config.json').exists(): raise RuntimeError('Existing production; no automatic restart')
    pinned=json.loads((run/'profile.json').read_text())
    assert pinned['source_hashes']==source_hashes()
    for key in ('device','microbatch','val_count','test_count'): assert pinned[key]==getattr(args,key)
    target,deadline=pinned['target_updates'],pinned['deadline'];remaining(deadline)
    configure(args.device);model,opt=build(args.device);initial=cpu_tree(model.state_dict())
    rng=torch.Generator().manual_seed(SHUFFLE_SEED)
    config=dict(version='native_krauzlis_stabilized_attention_v1',fresh_whole_model=True,checkpoint_inputs=[],
        all_learned_parameters_trainable=True,heads=1,spatial_grid=[10,10],width=64,
        architecture='one shared attention; QK L2 normalization with sigmoid-bounded scale1..16; per-patch separate gate/candidate LayerNorm in spatialGRU and CLSGRU; alpha/(alpha+1/N) gates spatial update; original local comparisons',
        memory_normalization='gate/candidate preactivations across channels; stored state is not renormalized',
        objective='terminal CE +0.05 pixel-derived causal prediction error',metadata_inputs=False,
        simulator_regression_targets=False,phase_indices_inputs=False,angles=[26,28],speed=.375,
        source_hashes=source_hashes(),parameter_count=sum(p.numel() for p in model.parameters()),
        init_seed=INIT_SEED,shuffle_seed=SHUFFLE_SEED,dataset_namespaces=dataset.NAMESPACES,
        pool_size=1000,pool_epochs=2,pool_updates=66,batch_size=32,optimizer='Adam',lr=3e-4,
        precision='fp32',tf32=False,clipping=100.,gradient_renormalization=False,selection='minimum mean validation CE',
        analysis_after_learning=True,stimulation_during_training=False,
        reference_window=8,observation_window=4,intervening_frames=4,comparison_trace_decay=.8,
        explicit_comparison_channels=735,validation_every=50,first_validation=33,
        **{k:pinned[k] for k in ('started','deadline','device','microbatch','val_count','test_count','target_updates')})
    atomic_json(run/'production_config.json',config)
    state=dict(step=0,presentations=0,unique_movies=0,best_step=None,best_loss=None)
    selected_loaded=False
    def status(phase,**values):
        atomic_json(run/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=utc(),target_updates=target,
            **{k:v for k,v in state.items() if k!='order'},**values))
    def validate():
        status('validation');result=evaluate(model,'val',args.val_count,args.device,deadline)
        result['step']=state['step'];atomic_json(run/('validation_%06d.json'%state['step']),result)
        better=state['best_loss'] is None or result['loss']<state['best_loss']
        if better: state.update(best_step=state['step'],best_loss=result['loss'])
        save(run,model,opt,state,config,rng,args.device,best=better)
    try:
        checks=set(range(50,target+1,50))|{33,target}
        while state['step']<target:
            remaining(deadline);pool_id=state['step']//POOL_UPDATES
            status('generating_pool',next_pool=pool_id);rows=dataset.pool(pool_id)
            state.update(pool_index=pool_id,unique_movies=(pool_id+1)*1000)
            for epoch in range(2):
                order=batches(rows,rng);state.update(epoch=epoch,order=order,cursor=0)
                for cell,indices in order:
                    remaining(deadline);clock=time.time()
                    values=update(model,opt,rows[cell],indices,args.device,args.microbatch)
                    state['step']+=1;state['presentations']+=len(indices);state['cursor']+=1
                    append_json(run/'progress.jsonl',dict(utc=utc(),step=state['step'],cell=cell,
                        presentations=state['presentations'],update_seconds=time.time()-clock,**values))
                    status('training',cell=cell,**values)
                    if state['step']==1 or state['step']%50==0: save(run,model,opt,state,config,rng,args.device)
                    if state['step']==1:
                        saved=torch.load(run/'latest.pt',map_location='cpu');adam=saved['optimizer']['state']
                        assert len(adam)==len(list(model.parameters())) and all(float(s['step'])==1 for s in adam.values())
                        atomic_json(run/'production_verified.json',dict(cpu_reloaded=True,step=1,optimizer_states=len(adam),
                            changed_tensors=[k for k,v in saved['model'].items() if not torch.equal(v,initial[k])],
                            checkpoint_inputs=[],one_spatial_attention_head=True))
                        del saved,initial
                    if state['step'] in checks: validate()
            del rows
        selected=torch.load(run/'best.pt',map_location='cpu');model.load_state_dict(selected['model']);selected_loaded=True
        selected_step=selected['state']['step'];del selected
        status('final_test');final=evaluate(model,'test',args.test_count,args.device,deadline)
        final.update(selected_step=selected_step,test_thresholds_fitted=False)
        atomic_json(run/'final_test.json',final)
        (run/'FINAL_REPORT.md').write_text('# Shared-attention native Krauzlis\n\n'+
            f"Completed {state['step']} updates; selected {selected_step}. BA {final['balanced_accuracy']:.6f}, AUC {final['auc']:.6f}.\n\n"+
            'Native cue, competing patches, lifetime resets, .375px/frame,26/28degrees. One shared attention head.\n'+
            'No microstimulation/inhibition analysis performed; task learning does not establish attention causality.\n',encoding='utf-8')
        receipt=dict(status='complete',stop_reason='planned_complete',completed=utc(),updates=state['step'],
            presentations=state['presentations'],unique_movies=state['unique_movies'],selected_step=selected_step,
            elapsed_seconds=time.time()-pinned['started'],final_n=final['n'],no_cloud=True)
        atomic_json(run/'completion_verified.json',receipt);atomic_json(run/'live_status.json',dict(phase='complete',**receipt))
    except BaseException as exc:
        if state['step'] and not selected_loaded: save(run,model,opt,state,config,rng,args.device)
        atomic_json(run/'failure.json',dict(type=type(exc).__name__,message=str(exc),utc=utc(),step=state['step']))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--profile-only',action='store_true');p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--target-updates',type=int,default=2244);p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--val-count',type=int,default=100);p.add_argument('--test-count',type=int,default=1000)
    a=p.parse_args()
    if a.target_updates<66 or a.target_updates%66 or a.microbatch<1 or a.val_count<100 or a.test_count<100 or a.val_count%100 or a.test_count%100:
        p.error('Require complete66-update pools, positive microbatch and100-trial evaluation blocks')
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try: profile(a,run) if a.profile_only else production(a,run)
    finally: lock.close()


if __name__=='__main__': main()
