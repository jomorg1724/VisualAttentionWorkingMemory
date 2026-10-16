"""One fresh jointly trained model; unchanged task suite and stabilized backbone."""
import argparse,datetime,hashlib,json,os,random,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import StabilizedAttentionSuite
from .dataset import Stream,TASKS,CATALOG,NAMESPACES,all_cells
from SecondPass.JointTraining.core import BalancedScheduler,score_cell,summarize
from SecondPass.StabilizedAttentionKrauzlis.memory import GATE_NAMES
from SecondPass.PixelContrastiveVAE.train import GPULock,atomic_json,append_json,cpu_tree,replace_with_retry,configure,synchronize
from PreAttentiveVision.natural_stimuli import DATA_ROOT

ROOT=Path(__file__).resolve().parents[2]
SEED=199211

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def hashes():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/p for p in ('SecondPass/StabilizedAttentionKrauzlis/model.py',
        'SecondPass/StabilizedAttentionKrauzlis/memory.py','SecondPass/TaskSuite/suite.py','SecondPass/TaskSuite/catalog.json',
        'SecondPass/JointTraining/core.py','WorkingMemory/BatteryAudit/observers.py',
        'WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/PlainBaseline/variants.py','WorkingMemory/stimuli.py',
        'PreAttentiveVision/neuroscience_stimuli.py','PreAttentiveVision/natural_stimuli.py',
        'SecondPass/PixelContrastiveVAE/train.py','SecondPass/PredictiveConsistencyVAE/queue.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def build(device):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if device=='cuda':torch.cuda.manual_seed_all(SEED)
    m=StabilizedAttentionSuite().to(device)
    return m,torch.optim.Adam(m.parameters(),lr=3e-4,eps=1e-8)

def active_parameters(model,task):
    return [(n,p) for n,p in model.named_parameters() if not n.startswith('heads.') or n.startswith('heads.'+task+'.')]

def update(model,opt,x,y,task,device,micro=4):
    model.train();opt.zero_grad(set_to_none=True)
    totals=dict(loss=0.,decision_loss=0.,prediction_loss=0.,accuracy=0.,attention_entropy=0.,attention_peak=0.,**{k:0. for k in GATE_NAMES})
    for offset in range(0,len(x),micro):
        xx=x[offset:offset+micro].to(device);yy=y[offset:offset+micro].to(device)
        logits,details=model(xx,task,True)
        ce=F.cross_entropy(logits,yy);loss=ce+.05*details['prediction_loss']
        w=len(xx)/len(x);(w*loss).backward()
        a=details['attention'].flatten(2).detach()
        vals=dict(loss=loss,decision_loss=ce,prediction_loss=details['prediction_loss'],
            accuracy=(logits.argmax(-1)==yy).float().mean(),attention_entropy=-(a*a.clamp_min(1e-12).log()).sum(-1).mean(),
            attention_peak=a.max(-1).values.mean(),**dict(zip(GATE_NAMES,details['gate_statistics'])))
        for k,v in vals.items():totals[k]+=w*float(v.detach())
    active=active_parameters(model,task)
    missing=[n for n,p in active if p.grad is None]
    if missing or not all(bool(torch.isfinite(p.grad).all()) for _,p in active):raise FloatingPointError(str(missing))
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith('heads.') and not n.startswith('heads.'+task+'.'))
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in active],100.,error_if_nonfinite=True)
    after=torch.stack([p.grad.square().sum() for _,p in active]).sum().sqrt()
    opt.step()
    totals.update(gradient_norm=float(norm),clipped_gradient_norm=float(after),gradient_clipped=bool(norm>100),episodes=len(x),frames=len(x)*x.shape[1])
    return totals

def check_time(deadline,reserve=60):
    if time.time()>deadline-reserve:raise TimeoutError('Finite suite allocation exhausted')

@torch.no_grad()
def evaluate(model,split,n,krau_n,device,deadline,output,cells=None):
    stream=Stream(split);model.eval();rows=[];cells=all_cells() if cells is None else cells
    started=time.time()
    for task,cell in cells:
        check_time(deadline);count=krau_n if task=='krauzlis_cued_motion' else n
        yy=[];pp=[];mm=[];ce_sum=0.
        for i in range(0,count,4):
            check_time(deadline);x,y,meta=stream.batch(min(4,count-i),task,cell)
            logits=model(x.to(device),task)
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite evaluation logits')
            ce_sum+=float(F.cross_entropy(logits,y.to(device),reduction='sum'))
            yy.extend(y.tolist());pp.extend(logits.softmax(-1).cpu().tolist());mm.extend(meta)
        row=score_cell(task,cell,yy,pp,mm);row['loss']=ce_sum/count
        row['predictions']=[dict(label=y,probabilities=p,metadata=m) for y,p,m in zip(yy,pp,mm)]
        rows.append(row)
        atomic_json(output.with_suffix('.partial.json'),dict(split=split,completed_cells=len(rows),total_cells=len(cells)))
    result=dict(split=split,cells=rows,complete=True,seconds=time.time()-started,n=sum(r['n'] for r in rows))
    if len(cells)==35:
        summary=summarize(rows)
        mean_ba=float(np.mean([r['chance_normalized_ba'] for r in summary['tasks'].values()]))
        summary['historical_worst_task_selection_key']=summary.pop('selection_key')
        summary['selection_key']=[summary['equal_task_mean_auc'],mean_ba]
        result['summary']=summary
    atomic_json(output,result);return result

def save(run,m,opt,scheduler,stream,state,config,best=False):
    payload=cpu_tree(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        stream=stream.state_dict(),state=state,config=config,torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all()))
    for name in ('latest','best') if best else ('latest',):
        temp=run/(name+'.pt.tmp');torch.save(payload,temp);replace_with_retry(temp,run/(name+'.pt'))

def profile(run,args):
    assert not (run/'budget.json').exists()
    started=time.time();deadline=started+28800
    atomic_json(run/'budget.json',dict(started=started,deadline=deadline,wall_seconds=28800))
    frozen=hashes()
    for name in frozen:
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    atomic_json(run/'source_hashes.json',frozen)
    configure('cuda');m,opt=build('cuda');stream=Stream('train');timings={};eval_times={}
    for task,cell in all_cells():
        check_time(deadline)
        start=time.time();x,y,_=stream.batch(32,task,cell)
        values=update(m,opt,x,y,task,'cuda');synchronize('cuda');seconds=time.time()-start
        timings[task+'/'+cell]=seconds
        m.eval();start=time.time()
        with torch.no_grad():
            vx,vy,_=Stream('val').batch(8,task,cell)
            for i in range(0,8,4):m(vx[i:i+4].cuda(),task)
        synchronize('cuda');eval_times[task+'/'+cell]=(time.time()-start)/8
        atomic_json(run/'profile_progress.json',dict(task=task,cell=cell,completed_cells=len(timings),total_cells=35,seconds=seconds,**values))
    nval=64;ktval=100;ntest=256;kttest=1000
    fullval=sum(v*(ktval if k.startswith('krauzlis') else nval) for k,v in eval_times.items())
    final=sum(v*(kttest if k.startswith('krauzlis') else ntest) for k,v in eval_times.items())
    monitor=sum(v*100 for k,v in eval_times.items() if k.startswith('krauzlis'))
    per_task={t:float(np.mean([v for k,v in timings.items() if k.startswith(t+'/')])) for t in TASKS}
    cycle_cost=sum(per_task.values())
    target=args.target_updates
    while target>=130:
        full_checks=set(range(520,target+1,520))|{130,target}
        monitor_checks=set(range(130,target+1,130))-full_checks
        reserve=900+1.35*(len(full_checks)*fullval+len(monitor_checks)*monitor+final)
        if 1.25*(target/13)*cycle_cost+reserve<deadline-time.time():break
        target-=13
    if target<130:raise RuntimeError('No substantial full-task allocation fits')
    atomic_json(run/'profile.json',dict(started=started,deadline=deadline,target_updates=target,requested_updates=args.target_updates,
        per_cell_update_seconds=timings,per_cell_eval_seconds=eval_times,cycle_seconds=cycle_cost,reserve_seconds=reserve,
        estimated_full_validation_seconds=fullval,estimated_final_seconds=final,val_count=nval,krauzlis_val=ktval,
        test_count=ntest,krauzlis_test=kttest,source_hashes=frozen,profile_weights_discarded=True,
        dataset_manifest_sha256=sha(DATA_ROOT/'manifest.json')))

def production(run,args):
    assert not (run/'production_config.json').exists()
    pinned=json.loads((run/'profile.json').read_text());assert pinned['source_hashes']==hashes()
    assert pinned['dataset_manifest_sha256']==sha(DATA_ROOT/'manifest.json')
    configure('cuda');m,opt=build('cuda');stream=Stream('train');scheduler=BalancedScheduler(199212)
    target=pinned['target_updates'];deadline=pinned['deadline']
    config=dict(version='stabilized_attention_suite_v1',fresh_whole_model=True,checkpoint_inputs=[],
        architecture='identical stabilized Krauzlis backbone and shared64D FFN;13 task-selected linear output heads',
        parameter_count=sum(p.numel() for p in m.parameters()),parameter_tensors=len(list(m.parameters())),
        task_classes={t:s['classes'] for t,s in TASKS.items()},catalog=CATALOG,seed=SEED,stream_namespaces=NAMESPACES,
        optimizer='Adam',lr=3e-4,clipping=100.,batch_size=32,microbatch=4,precision='fp32',tf32=False,
        objective='terminal CE + .05 same pixel-prediction loss on every task',
        sampling='established equal-task BalancedScheduler; cells balanced within task; fresh32 episodes each update; no replay',
        metadata_inputs=False,external_task_identity='output head selection only',all_parameters_trainable=True,
        selection='equal-task mean AUC then mean chance-normalized BA;N0 excluded',
        **pinned)
    atomic_json(run/'production_config.json',config)
    state=dict(step=0,episodes=0,frames=0,task_updates={t:0 for t in TASKS},task_episodes={t:0 for t in TASKS},
        cell_updates={t+'/'+c:0 for t,c in all_cells()},best_step=None,best_key=None)
    full_checks=set(range(520,target+1,520))|{130,target}
    def status(phase,**kw):atomic_json(run/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=now(),target_updates=target,**state,**kw))
    selected_loaded=False
    try:
        while state['step']<target:
            check_time(deadline)
            if (run/'stop_requested.json').exists():
                save(run,m,opt,scheduler,stream,state,config);status('user_stopped');return
            task,cell=scheduler.next();status('rendering',task=task,cell=cell)
            start=time.time();x,y,_=stream.batch(32,task,cell)
            values=update(m,opt,x,y,task,'cuda');elapsed=time.time()-start
            state['step']+=1;state['episodes']+=32;state['frames']+=values['frames'];state['task_updates'][task]+=1;state['task_episodes'][task]+=32
            state['cell_updates'][task+'/'+cell]+=1
            append_json(run/'progress.jsonl',dict(step=state['step'],task=task,cell=cell,utc=now(),update_seconds=elapsed,**values))
            status('training',task=task,cell=cell,**{k:v for k,v in values.items() if k not in ('episodes','frames')})
            if state['step'] in (1,13) or state['step']%130==0:save(run,m,opt,scheduler,stream,state,config)
            if state['step']==13:
                saved=torch.load(run/'latest.pt',map_location='cpu');states=saved['optimizer']['state']
                assert len(states)==len(list(m.parameters()))
                assert all(torch.isfinite(v).all() for s in states.values() for v in s.values() if torch.is_tensor(v))
                atomic_json(run/'production_verified.json',dict(step=13,all13_tasks_updated=True,optimizer_states=len(states),cpu_reloaded=True))
            if state['step'] in full_checks:
                status('full_validation')
                r=evaluate(m,'val',64,100,'cuda',deadline,run/('validation_%06d.json'%state['step']))
                key=r['summary']['selection_key'];better=state['best_key'] is None or tuple(key)>tuple(state['best_key'])
                if better:state.update(best_key=key,best_step=state['step'])
                save(run,m,opt,scheduler,stream,state,config,best=better)
            elif state['step']%130==0:
                status('krauzlis_monitor')
                evaluate(m,'val',64,100,'cuda',deadline,run/('krauzlis_monitor_%06d.json'%state['step']),
                    cells=[(t,c) for t,c in all_cells() if t=='krauzlis_cued_motion'])
        saved=torch.load(run/'best.pt',map_location='cpu');m.load_state_dict(saved['model']);selected_loaded=True
        selected_step=saved['state']['step'];status('final_test',selected_step=selected_step)
        final=evaluate(m,'test',256,1000,'cuda',deadline,run/'final_test.json')
        atomic_json(run/'completion_verified.json',dict(status='complete',updates=state['step'],episodes=state['episodes'],
            selected_step=selected_step,final_n=final['n'],all35cells=True,completed=now()))
        status('complete',selected_step=selected_step)
    except BaseException as e:
        if state['step'] and not selected_loaded:save(run,m,opt,scheduler,stream,state,config)
        atomic_json(run/'failure.json',dict(error=repr(e),step=state['step'],utc=now()));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--profile-only',action='store_true')
    p.add_argument('--target-updates',type=int,default=17160);args=p.parse_args()
    assert args.target_updates%13==0
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:profile(run,args) if args.profile_only else production(run,args)
    finally:lock.close()

if __name__=='__main__':main()
