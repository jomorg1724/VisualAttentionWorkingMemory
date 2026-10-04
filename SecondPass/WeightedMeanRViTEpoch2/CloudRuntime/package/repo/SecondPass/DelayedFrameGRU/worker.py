"""Fresh, native Krauzlis-only delayed-frame CNN GRU experiment. No cloud API access."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from SecondPass.JointTraining.core import atomic_json, append_jsonl, cpu_tree, tree_equal, score_cell
from SecondPass.TaskSuite.suite import SuiteStream, TASKS as ALL_TASKS
from .model import DelayedFrameGRU

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.DelayedFrameGRU.worker'
TASK='krauzlis_cued_motion'
TASKS={TASK:ALL_TASKS[TASK]}
CELLS=[r['id'] for r in TASKS[TASK]['conditions']]
VERSION='independent_delayed_frame_stride1_residual_cnn_gru'
PROTOCOL=VERSION+'_wholemodel_fresh_native_v1'
INIT_SEED,CUDA_SEED,SCHEDULER_SEED=158192763,158292763,158392763
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=158492763,158592763,158692763
TARGET=4216
MIN_USEFUL_UPDATES=1000


@lru_cache(maxsize=1)
def parameter_count():
    """Measured from a fresh constructor, preserving live CPU/CUDA RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INIT_SEED)
        model=DelayedFrameGRU()
    return sum(p.numel() for p in model.parameters())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def cpu_setup():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(2)
    except RuntimeError: pass
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    elif str(device)=='mps': torch.mps.synchronize()

class Scheduler:
    def __init__(self,seed):
        self.rng=random.Random(seed); self.cells=[]; self.updates=0
    def next(self):
        if not self.cells: self.cells=list(CELLS); self.rng.shuffle(self.cells)
        self.updates+=1
        return TASK,self.cells.pop()
    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.getstate(),cells=self.cells,updates=self.updates))

class FreshStream(SuiteStream):
    def stream_seed(self,task,cell):
        if task!=TASK: raise ValueError('Native Krauzlis only')
        index,_=self._cell(task,cell)
        return dict(train=TRAIN_NAMESPACE,val=VAL_NAMESPACE,test=FINAL_NAMESPACE)[self.split]*100000+TASKS[task]['stream_id']*1000+index
    def batch(self,n,task,cell):
        if task!=TASK: raise ValueError('Native Krauzlis only')
        return super().batch(n,task,cell)

def provenance():
    return dict(initialization='whole model direct fresh constructor; no checkpoint inputs',
        architecture=VERSION,initialization_seed=INIT_SEED,cuda_seed=CUDA_SEED,scheduler_seed=SCHEDULER_SEED,
        mps_seed=INIT_SEED,
        train_namespace=TRAIN_NAMESPACE,validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,
        inherited_weights=False,inherited_optimizer=False,inherited_rng=False,inherited_stream=False,
        profile_state_inherited=False,inactive_heads=[],parameter_count=parameter_count(),
        independent_encoders=True,cnn_stride=1,frame_pairing="ordered current and previous; zero previous at firstframe",
        recurrent_core="standard torch.nn.GRU 512 to256",comparison_target_updates=TARGET,
        stimulus='unchanged native Krauzlis26/28degree B12/B20/B28; target57 foil29 catch14 per100')

def load_verified(receipt):
    path=Path(receipt['path'])
    if path.stat().st_size!=receipt['bytes'] or digest(path)!=receipt['sha256']: raise ValueError('Checkpoint integrity mismatch')
    return torch.load(path,map_location='cpu',weights_only=False)

class Session:
    def __init__(self,directory,config):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config; self.device=config['device']; self.latest=None
        random.seed(INIT_SEED); np.random.seed(INIT_SEED); torch.manual_seed(INIT_SEED)
        if str(self.device).startswith('cuda'): torch.cuda.manual_seed_all(CUDA_SEED)
        elif str(self.device)=='mps': torch.mps.manual_seed(INIT_SEED)
        self.model=DelayedFrameGRU().to(self.device)
        assert sum(p.numel() for p in self.model.parameters())==parameter_count()
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.scheduler=Scheduler(SCHEDULER_SEED); self.stream=FreshStream('train')
        self.state=dict(step=0,episodes=0,frames=0,optimizer_seconds=0.,best_step=None,best_key=None,
            best_checkpoint=None,selection_history=[],config=config,
            exposure={TASK:dict(updates=0,episodes=0,frames=0,cells={c:0 for c in CELLS})})

    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=utc(),
            architecture=VERSION,initialization='fresh',step=self.state['step'],episodes=self.state['episodes'],
            optimizer_seconds=self.state['optimizer_seconds'],deadline=self.config['deadline'],
            latest_checkpoint=self.latest,best_step=self.state['best_step'],**extra))

    def checkpoint(self,filename):
        path=self.directory/filename
        if path.exists(): raise FileExistsError('Immutable checkpoint: '+str(path))
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        if str(self.device).startswith('cuda'): rng['cuda']=torch.cuda.get_rng_state_all()
        elif str(self.device)=='mps': rng['mps']=torch.mps.get_rng_state()
        payload=cpu_tree(dict(schema=3,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            optimizer_names=[n for n,_ in self.model.named_parameters()],state=self.state,
            scheduler=self.scheduler.state_dict(),stream=self.stream.state_dict(),rng=rng,provenance=provenance()))
        temp=path.with_suffix('.pt.tmp')
        with temp.open('wb') as f: torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
        receipt=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path),verified=True,step=self.state['step'])
        assert tree_equal(payload,load_verified(receipt))
        self.latest=receipt; atomic_json(self.directory/'latest_checkpoint.json',receipt)
        if self.state['step']==0:
            atomic_json(self.directory/'initial_checkpoint.json',receipt)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(INIT_SEED)
                direct=DelayedFrameGRU()
            assert tree_equal(payload['model'],direct.state_dict())
            assert not payload['optimizer']['state'] and payload['stream']['streams']==[]
            atomic_json(self.directory/'constructor_equality.json',dict(verified=True,tensors=len(payload['model']),
                empty_adam=True,empty_stream=True,checkpoint=receipt,parameter_count=parameter_count()))
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=utc(),**receipt))
        if self.state['step'] in (1,2,3): verify_progress(self.directory)
        if self.state['step']==3 and not self.config.get('disposable_profile'):
            atomic_json(self.directory/'startup_ready.json',dict(verified=True,checkpoint=receipt,
                persisted_optimizer_evidence=verify_progress(self.directory)))
        return receipt

    def train_update(self,task,cell):
        effective,micro=self.config['effective_batch'],self.config['microbatch']
        assert effective%micro==0
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        sync(self.device); started=time.perf_counter(); loss_sum=0.; frames=0
        for _ in range(effective//micro):
            images,labels,_=self.stream.batch(micro,task,cell); frames+=images.shape[0]*images.shape[1]
            logits=self.model(images.to(self.device),task)
            loss=F.cross_entropy(logits,labels.to(self.device))
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
            (loss*(micro/effective)).backward(); loss_sum+=float(loss.detach())*micro/effective
            del images,labels,logits,loss
        norms={}
        for name,p in self.model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise FloatingPointError('Missing/nonfinite gradient: '+name)
            norms[name]=float(p.grad.square().sum())
        gradient_norm=math.sqrt(sum(norms.values()))
        if not math.isfinite(gradient_norm): raise FloatingPointError('Nonfinite gradient norm before Adam')
        self.optimizer.step(); sync(self.device)
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()): raise FloatingPointError('Nonfinite parameter')
        elapsed=time.perf_counter()-started; s=self.state
        s['step']+=1; s['episodes']+=effective; s['frames']+=frames; s['optimizer_seconds']+=elapsed
        exposure=s['exposure'][task]; exposure['updates']+=1; exposure['episodes']+=effective
        exposure['frames']+=frames; exposure['cells'][cell]+=effective
        row=dict(utc=utc(),step=s['step'],task=task,cell=cell,loss=loss_sum,seconds=elapsed,
            episodes=effective,frames=frames,grad_norm=gradient_norm,gradient_norms={k:math.sqrt(v) for k,v in norms.items()},
            cumulative_episodes=s['episodes'],cumulative_frames=s['frames'],optimizer_seconds=s['optimizer_seconds'],
            per_task_exposure=cpu_tree(s['exposure']),initialization='fresh',clipping=None)
        append_jsonl(self.directory/'progress.jsonl',row)
        print(json.dumps({k:row[k] for k in ('utc','step','task','cell','loss','seconds','cumulative_episodes')}),flush=True)
        if s['step']==3 and not self.config.get('disposable_profile') and (self.directory/'profile/profile.json').exists():
            prof=json.loads((self.directory/'profile/profile.json').read_text())
            costs={c:max(r['seconds'] for r in prof['rows'] if r['cell']==c) for c in CELLS}
            observed=[json.loads(line) for line in (self.directory/'progress.jsonl').read_text().splitlines()]
            ratio=sum(r['seconds'] for r in observed)/sum(costs[r['cell']] for r in observed)
            atomic_json(self.directory/'first_cycle_timing.json',dict(updates=3,matched_profile_ratio=ratio,
                material_slowdown=ratio>1.35,production_rows=observed))
        return row

def verify_progress(directory):
    d=Path(directory); initial=load_verified(json.loads((d/'initial_checkpoint.json').read_text()))
    receipt=json.loads((d/'latest_checkpoint.json').read_text()); saved=load_verified(receipt)
    assert initial['schema']==saved['schema']==3 and initial['provenance']==saved['provenance']==provenance()
    assert initial['state']['step']==initial['scheduler']['updates']==0 and not initial['optimizer']['state']
    assert initial['stream']['streams']==[] and initial['state']['best_checkpoint'] is None and initial['state']['selection_history']==[]
    cfg=initial['state']['config']
    if cfg['device']=='mps':
        for payload in (initial,saved):
            assert torch.is_tensor(payload['rng'].get('mps')) and payload['rng']['mps'].numel()>0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INIT_SEED); direct=DelayedFrameGRU()
    assert tree_equal(initial['model'],direct.state_dict())
    names=[n for n,_ in direct.named_parameters()]
    assert names==initial['optimizer_names']==saved['optimizer_names']
    assert sum(p.numel() for p in direct.parameters())==parameter_count()
    step=saved['state']['step']; assert step>0 and saved['scheduler']['updates']==receipt['step']==step
    assert saved['state']['episodes']==step*cfg['effective_batch'] and saved['state']['optimizer_seconds']>0
    assert set(saved['optimizer']['state'])==set(range(len(names)))
    assert len(saved['optimizer']['param_groups'])==1
    group=saved['optimizer']['param_groups'][0]
    assert group['lr']==1e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
    changed=[]
    for i,name in enumerate(names):
        opt=saved['optimizer']['state'][i]; assert float(opt['step'])==step,name
        for key in ('exp_avg','exp_avg_sq'):
            assert opt[key].shape==saved['model'][name].shape and torch.isfinite(opt[key]).all(),name
        assert torch.isfinite(saved['model'][name]).all(),name
        if not torch.equal(initial['model'][name],saved['model'][name]): changed.append(name)
    assert len(changed)==len(names),set(names)-set(changed)
    counts={r['cell']:r['state']['counts'][TASK] for r in saved['stream']['streams']}
    assert counts=={c:n for c,n in saved['state']['exposure'][TASK]['cells'].items() if n}
    assert sum(counts.values())==saved['state']['episodes']
    prefixes=('current_encoder.','previous_encoder.','gru.','classifier.')
    groups={p:[n for n in changed if n.startswith(p)] for p in prefixes}
    assert all(groups.values()),groups
    result=dict(verified=True,step=step,episodes=saved['state']['episodes'],checkpoint=receipt,parameter_count=parameter_count(),
        changed_parameters=len(changed),changed_by_module=groups,all_active_parameters_changed=True,stream_counts=counts,
        optimizer_seconds=saved['state']['optimizer_seconds'],fresh_initial_optimizer_empty=True,persisted_constructor_equality=True)
    atomic_json(d/'persisted_progress_verification.json',result); return result

def selection_key(result):
    rows=result['cells']
    if not result['complete'] or len(rows)!=3 or {r['cell'] for r in rows}!=set(CELLS): return None
    if any(r[k] is None for r in rows for k in ('auc','balanced_accuracy')): return None
    return [sum(r[k] for r in rows)/3 for k in ('auc','balanced_accuracy')]

@torch.no_grad()
def evaluate(model,split,count,microbatch,device,output,deadline):
    stream=FreshStream(split); model.eval(); rows=[]; started=time.perf_counter()
    complete=True
    for cell in CELLS:
        labels=[]; probabilities=[]; metadata=[]; before=time.perf_counter()
        for i in range(0,count,microbatch):
            if time.time()>=deadline: complete=False; break
            images,y,meta=stream.batch(min(microbatch,count-i),TASK,cell)
            logits=model(images.to(device),TASK)
            if not bool(torch.isfinite(logits).all()): raise FloatingPointError('Nonfinite evaluation logits')
            labels.extend(y.tolist()); probabilities.extend(logits.softmax(1).cpu().tolist()); metadata.extend(meta)
        if not complete: break
        sync(device); row=score_cell(TASK,cell,labels,probabilities,metadata); row['seconds']=time.perf_counter()-before
        for event,event_row in row['events'].items():
            ix=[j for j,m in enumerate(metadata) if m['event_type']==event]
            pairs=[(labels[j],int(np.argmax(probabilities[j]))) for j in ix]
            cm=[[sum(y==a and p==b for y,p in pairs) for b in range(2)] for a in range(2)]
            event_row.update(confusion=cm,hit_rate=cm[1][1]/sum(cm[1]) if sum(cm[1]) else None,
                specificity=cm[0][0]/sum(cm[0]) if sum(cm[0]) else None,
                false_positive_rate=cm[0][1]/sum(cm[0]) if sum(cm[0]) else None)
        row['trial_ids']=[m['suite_trial_id'] for m in metadata]
        row['labels']=labels; row['probabilities']=probabilities
        rows.append(row)
        atomic_json(str(output)+'.partial.json',dict(split=split,complete_cells=len(rows),expected_cells=3,cells=rows))
    result=dict(split=split,n=count,cells=rows,complete=complete and len(rows)==3,seconds=time.perf_counter()-started,
        namespace=VAL_NAMESPACE if split=='val' else FINAL_NAMESPACE,fixed_draws=True,
        selection_rule='three-condition mean AUC then BA; earlier ties; validation only')
    result['summary']=dict(selection_key=selection_key(result)); atomic_json(output,result); return result

def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    for key in ('cap_started','deadline','hard_deadline','retrieval_reserve_seconds','wall_cap_seconds'):
        if config.get(key)!=budget[key]: raise ValueError('Absolute cap changed')
    if budget['wall_cap_seconds']!=28800 or budget['hard_deadline']-budget['cap_started']!=28800: raise ValueError('Require unchanged eight-hour cap')
    if budget['retrieval_reserve_seconds']!=600 or budget['hard_deadline']-budget['deadline']!=600: raise ValueError('Require600s retrieval reserve')
    if not budget['cap_started']<=now<budget['deadline']: raise ValueError('Cap expired/not started')

def verify_sources(config):
    for path,sha in config['source_hashes'].items():
        if digest(path)!=sha: raise ValueError('Deployment source changed: '+path)

def config_for(budget):
    manifest=json.loads((ROOT.parent/'deployment_manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL: raise ValueError('Wrong deployment protocol')
    hashes={str(ROOT.parent/r['path']):r['sha256'] for r in manifest['files']}
    config=dict(**budget,protocol=PROTOCOL,device='cuda',
        effective_batch=32,microbatch=4,eval_microbatch=4,val_n=100,test_n=200,checkpoint_every=100,cpu_threads=2,
        source_hashes=hashes,runtime_root=str(ROOT),initialization=provenance(),all_trainable=True,
        lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clipping=None,precision='fp32',bptt='full',tf32=False)
    verify_sources(config); return config

def local_config_for(budget):
    manifest_path=ROOT/'runtime_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    hashes={str(Path(p) if Path(p).is_absolute() else ROOT/p):sha for p,sha in manifest['source_hashes'].items()}
    hashes[str(manifest_path)]=digest(manifest_path)
    cfg=dict(budget,protocol=PROTOCOL,device='mps',execution_placement='local',minimum_useful_updates=3,
        effective_batch=32,microbatch=1,eval_microbatch=1,val_n=100,test_n=200,checkpoint_every=100,cpu_threads=2,
        source_hashes=hashes,runtime_root=str(ROOT),initialization=provenance(),all_trainable=True,
        lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clipping=None,precision='fp32',bptt='full',tf32=False,
        hardware='Apple M4 Max 36GiB; MPSrecommendedmemory28.08GiB; micro1selectedbeforeprofile',
        platform_difference='Apple MPS versus prior CUDA; same effectivebatch32 and teaching, differentmicrobatch1')
    verify_sources(cfg); return cfg

def profile(directory):
    cfg=json.loads((directory/'profile_config.json').read_text()); verify_sources(cfg)
    session=Session(directory,cfg); session.checkpoint('initial.pt'); rows=[]; warmup_rows=[]
    if session.device=='cuda': torch.cuda.reset_peak_memory_stats()
    for i in range(9):
        if time.time()>=cfg['deadline']-60: raise RuntimeError('Bounded native profile exhausted')
        row=session.train_update(*session.scheduler.next())
        (warmup_rows if i<3 else rows).append(row)
        if session.state['step'] in (1,2,3): session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
    session.checkpoint('profile_terminal.pt')
    ev=evaluate(session.model,'val',20,cfg['eval_microbatch'],session.device,directory/'eval_timing.json',cfg['deadline']-30)
    if not ev['complete']: raise RuntimeError('Incomplete native three-condition evaluation profile')
    atomic_json(directory/'profile.json',dict(complete=True,architecture=VERSION,rows=rows,warmup_rows=warmup_rows,
        timing_policy='discard first native full three-condition cycle; use next two cycles only',evaluation_cells=ev['cells'],
        disposable=True,discard_all_state=True,verification=verify_progress(directory),torch_version=torch.__version__,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if session.device=='cuda' else None,
        mps_current_bytes=torch.mps.current_allocated_memory() if session.device=='mps' else None,
        mps_driver_bytes=torch.mps.driver_allocated_memory() if session.device=='mps' else None))

def measured_plan(directory,budget,now=None):
    now=time.time() if now is None else now; validate_budget(budget,budget,now)
    prof=json.loads((directory/'profile/profile.json').read_text())
    if not prof['complete'] or prof['architecture']!=VERSION or len(prof['rows'])!=6: raise ValueError('Require six native profiles')
    warmup=prof.get('warmup_rows',[])
    if len(warmup)!=3 or {r['cell'] for r in warmup}!=set(CELLS) or any(r['episodes']!=32 for r in warmup):
        raise ValueError('Require separate native three-condition warmup cycle')
    if len(prof['evaluation_cells'])!=3 or {r['cell'] for r in prof['evaluation_cells']}!=set(CELLS): raise ValueError('Missing native evaluation cells')
    costs={}
    for cell in CELLS:
        rows=[r for r in prof['rows'] if r['task']==TASK and r['cell']==cell]
        if len(rows)!=2 or any(r['episodes']!=32 for r in rows): raise ValueError('Require two batch32updates per cell')
        costs[cell]=max(r['seconds'] for r in rows)
    eval_costs=[r['seconds']/r['n'] for r in prof['evaluation_cells']]
    if not all(math.isfinite(v) and v>0 for v in [*costs.values(),*eval_costs]): raise ValueError('Invalid timings')
    val=1.35*sum(eval_costs)*100; test=2*val; overhead=2*val+2*test+900
    scheduler=Scheduler(SCHEDULER_SEED); schedule=[]; training=0.; best=None
    for _ in range(TARGET):
        _,cell=scheduler.next(); schedule.append(cell); training+=costs[cell]
        if 1.25*training+overhead+60<budget['deadline']-now: best=(len(schedule),training)
    minimum=3 if budget.get('execution_placement')=='local' else MIN_USEFUL_UPDATES
    if best is None or best[0]<minimum: raise RuntimeError(f'Fewer than{minimum}updates fit within unchanged cap')
    count,training=best
    return dict(max_steps=count,target_requested=TARGET,minimum_useful_updates=minimum,
        exposure_reduced=count<TARGET,total_episodes=count*32,validation_steps=[count//2,count],checkpoint_every=100,
        planned_exposure={c:dict(updates=schedule[:count].count(c),episodes=32*schedule[:count].count(c)) for c in CELLS},
        estimated_optimizer_seconds=training,estimated_total_remaining_seconds=1.25*training+overhead,
        slower_150pct_seconds=1.5*training+overhead,estimated_validation_seconds=val,estimated_one_test_seconds=test,
        final_reserve_seconds=val+2*test+180,update_estimate_seconds=1.25*max(costs.values()),
        method='discard3nativewarmupupdates; next6nativeeffectivebatch32updates at configuredmicrobatch two/cell; steadypercellmax; exactseededschedule; train1.25eval1.35; two val100+two test200/cell+900s; separate600sretrieval')

def prepare(directory,budget):
    validate_budget(budget,budget); directory.mkdir(parents=True,exist_ok=True)
    with (directory/'budget.json').open('x') as f: json.dump(budget,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.chmod(directory/'budget.json',0o444)
    cfg=config_for(budget); pd=directory/'profile'; pd.mkdir()
    atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],time.time()+1800)))

def pin(directory):
    budget=json.loads((directory/'budget.json').read_text()); cfg=config_for(budget); plan=measured_plan(directory,budget); cfg.update(plan)
    for name,value in [('allocation.json',plan),('config.json',cfg)]:
        with (directory/name).open('x') as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
        os.chmod(directory/name,0o444)
    return plan

def scientifically_finalized(*,failure,complete,actual,pinned,reason):
    """Completed evaluation may truthfully finalize a deadline-limited run."""
    return (failure is None and complete and 0<actual<=pinned
        and reason in ('planned_complete','wall_budget_reserve'))

def practical_success(result):
    """Provisional reporting criteria, independent of selection and stopping."""
    eligible=bool(result and result['complete'] and len(result['cells'])==3
        and {r['cell'] for r in result['cells']}==set(CELLS)
        and all(r['balanced_accuracy'] is not None for r in result['cells']))
    return dict(provisional=True,basis='selected-model fresh final tests only',metric='balanced_accuracy',
        acquisition_threshold_each_condition=.70,solved_target_threshold_each_condition=.90,
        acquisition_met=eligible and all(r['balanced_accuracy']>=.70 for r in result['cells']),
        solved_target_met=eligible and all(r['balanced_accuracy']>=.90 for r in result['cells']),
        affects_selection=False,affects_stopping=False)

def run(directory):
    config=json.loads((directory/'config.json').read_text()); validate_budget(json.loads((directory/'budget.json').read_text()),config); verify_sources(config)
    if (directory/'initial.pt').exists(): raise RuntimeError('Freshonly: no restart or overwrite')
    s=Session(directory,config); s.checkpoint('initial.pt')
    atomic_json(directory/'fresh_initialization.json',dict(verified=True,checkpoint=s.latest,**provenance()))
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete'; failure=None; terminal=None; terminal_test=None; selected_test=None
    def validate():
        s.status('validation')
        result=evaluate(s.model,'val',100,config['eval_microbatch'],s.device,directory/f"validation_{s.state['step']:06d}.json",config['deadline']-2*config['estimated_one_test_seconds']-120)
        key=selection_key(result); state=s.state; state['selection_history'].append(dict(step=state['step'],key=key,complete=result['complete']))
        name=f"validation_checkpoint_{state['step']:06d}.pt"
        if key is not None and (state['best_key'] is None or tuple(key)>tuple(state['best_key'])):
            state.update(best_key=key,best_step=state['step'],best_checkpoint=str(directory/name))
        s.checkpoint(name)
    try:
        while s.state['step']<config['max_steps']:
            if stopped[0]: reason='signal'; break
            if time.time()+config['final_reserve_seconds']+config['update_estimate_seconds']>=config['deadline']: reason='wall_budget_reserve'; break
            row=s.train_update(*s.scheduler.next()); step=s.state['step']
            if step in (1,2,3) or step%config['checkpoint_every']==0: s.checkpoint(f'checkpoint_{step:06d}.pt')
            s.status('training',last_update={k:row[k] for k in ('task','cell','loss','seconds')})
            if step in config['validation_steps'] and not stopped[0]: validate()
        terminal=s.checkpoint('terminal.pt')
        if s.state['step']>0 and not stopped[0]:
            history=s.state['selection_history']
            if (not history or history[-1]['step']!=s.state['step']) and len(history)<2 and time.time()+config['final_reserve_seconds']<config['deadline']:
                validate(); terminal=s.checkpoint('terminal_validated.pt')
            if time.time()<config['deadline']-60:
                different=s.state['best_step']!=s.state['step']; reserve=config['estimated_one_test_seconds'] if different else 0
                s.status('final_test_terminal')
                terminal_test=evaluate(s.model,'test',200,config['eval_microbatch'],s.device,directory/'test_terminal.json',config['deadline']-reserve-60)
                if not different and s.state['best_checkpoint']:
                    winner=torch.load(s.state['best_checkpoint'],map_location='cpu',weights_only=False)
                    assert tree_equal(winner['model'],cpu_tree(s.model.state_dict()))
                    selected_test=dict(reused_terminal=True,identical_model_verified=True,results=terminal_test); atomic_json(directory/'test_selected.json',selected_test)
                elif s.state['best_checkpoint'] and time.time()<config['deadline']-60:
                    winner=torch.load(s.state['best_checkpoint'],map_location='cpu',weights_only=False); s.model.load_state_dict(winner['model'])
                    s.status('final_test_selected'); selected_test=evaluate(s.model,'test',200,config['eval_microbatch'],s.device,directory/'test_selected.json',config['deadline']-45)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()); reason='worker_error'
        atomic_json(directory/'failure.json',failure)
        if terminal is None: terminal=s.latest
    selected=selected_test.get('results',selected_test) if selected_test else None
    complete=bool(terminal_test and terminal_test['complete'] and selected and selected['complete'])
    if complete:
        for a,b in zip(terminal_test['cells'],selected['cells']): assert a['trial_ids']==b['trial_ids'] and a['labels']==b['labels']
    saved=load_verified(terminal)
    report=dict(protocol=PROTOCOL,architecture=VERSION,initialization=provenance(),stop_reason=reason,failure=failure,
        terminal_checkpoint=terminal,terminal_step=saved['state']['step'],selected_step=saved['state']['best_step'],
        pinned_updates=config['max_steps'],exposure_completed=saved['state']['step']==config['max_steps'],
        cap_limited=reason=='wall_budget_reserve' and saved['state']['step']<config['max_steps'],
        selection_history=saved['state']['selection_history'],episodes=saved['state']['episodes'],exposure=saved['state']['exposure'],
        optimizer_seconds=saved['state']['optimizer_seconds'],config=config,terminal_test=terminal_test,selected_test=selected_test,
        final_coverage_complete=complete,finished_utc=utc(),limitations=['One fresh seed; no convergence or superiority claim.',
            'Unchanged native26/28degree task; all cells and target/foil/catch subgroups visible.',
            'Architecture package comparison; no isolated causal claim about prior failed transformers.'])
    report['practical_success_criterion']=practical_success(selected)
    atomic_json(directory/'report.json',report)
    lines=['# Fresh delayed-frame independent CNN GRU training',f"Updates {report['terminal_step']}; episodes {report['episodes']}; selected {report['selected_step']}; stop {reason}.",
        'Provisional reporting criteria: selected-model fresh final BA at least 0.70 in every condition for acquisition; 0.90 in every condition for the solved target.']
    for label,result in [('Terminal',terminal_test),('Selected',selected)]:
        lines += [f'\n## {label}\n','|Cell|n|BA|AUC|Target hit|Foil FPR|Catch FPR|','|---|---:|---:|---:|---:|---:|---:|']
        for row in result['cells'] if result else []:
            lines.append('|'+ '|'.join(str(row.get(k)) for k in ('cell','n','balanced_accuracy','auc','target_hit_rate','foil_false_alarm_rate','catch_false_positive_rate'))+'|')
            lines.append('\nEvents: `'+json.dumps(row['events'],sort_keys=True)+'`')
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n'); s.status('finished',stop_reason=reason,failure=failure,final_coverage_complete=complete)
    if report['terminal_step']>0: verify_progress(directory)
    return 0 if scientifically_finalized(failure=failure,complete=complete,actual=report['terminal_step'],
        pinned=config['max_steps'],reason=reason) else 2

def completion(directory,status,error=None):
    report=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {}
    latest=json.loads((directory/'latest_checkpoint.json').read_text()) if (directory/'latest_checkpoint.json').exists() else None
    terminal=report.get('terminal_checkpoint'); selected=None
    if terminal:
        saved=load_verified(terminal); source=saved['state']['best_checkpoint']
        if source: shutil.copy2(source,directory/'selected.pt'); selected=str(directory/'selected.pt')
    if latest: shutil.copy2(latest['path'],directory/'latest.pt')
    paths=[p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md') and p.name!='cloud_completion.json']
    paths += [p for p in (directory/'initial.pt',directory/'selected.pt',directory/'latest.pt',Path(terminal['path']) if terminal else directory/'terminal.pt') if p.is_file()]
    files=[dict(path=str(p),relative_path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]
    allocation=json.loads((directory/'allocation.json').read_text()) if (directory/'allocation.json').exists() else {}
    atomic_json(directory/'cloud_completion.json',dict(status=status,error=error,target_updates=TARGET,pinned_updates=allocation.get('max_steps'),
        actual_updates=latest['step'] if latest else 0,exposure_completed=report.get('exposure_completed',False),
        cap_limited=report.get('cap_limited',False),stop_reason=report.get('stop_reason'),
        selected_checkpoint=selected,terminal_checkpoint=terminal,latest=latest,
        artifact_manifest=files,finished_utc=utc()))

def supervise(directory,mode):
    budget=json.loads((directory/'budget.json').read_text()); validate_budget(budget,budget)
    target=directory/'profile' if mode=='profile' else directory
    cfg=json.loads((target/('profile_config.json' if mode=='profile' else 'config.json')).read_text())
    if mode=='run':
        validate_budget(budget,cfg)
        if time.time()+cfg['estimated_total_remaining_seconds']>=budget['deadline']: raise RuntimeError('Pinned allocation no longer fits original cap')
        with (directory/'production_claim.json').open('x') as f: json.dump(dict(pid=os.getpid(),observed=time.time()),f)
    deadline=min(budget['deadline'],cfg['deadline']); command=[sys.executable,'-u','-m',MODULE,mode,str(target)]
    child=subprocess.Popen(command,start_new_session=True)
    atomic_json(target/(mode+'_supervisor.json'),dict(pid=os.getpid(),worker_pid=child.pid,deadline=deadline,command=command))
    hard=False
    try: child.wait(timeout=max(0.,deadline-time.time()))
    except subprocess.TimeoutExpired:
        hard=True; os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=5)
    outcome=dict(returncode=child.returncode,status='complete' if child.returncode==0 and not hard else 'incomplete',
        hard_cap_triggered=hard,finished=time.time(),supervisor_pid=os.getpid(),worker_pid=child.pid,deadline=deadline)
    atomic_json(target/'supervisor_result.json',outcome)
    if mode=='run': completion(directory,'complete' if child.returncode==0 else 'incomplete')
    return outcome

def local_supervise(directory):
    """One fresh local attempt, pinning exposure before production; never resume."""
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required for authorized local attempt')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'budget.json').exists(): raise RuntimeError('No local restart or cap renewal')
    started=time.time()
    budget=dict(cap_started=started,deadline=started+28200,hard_deadline=started+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800,execution_placement='local',
        origin='First fresh accelerator profile; includes profile/training/evaluation/reporting; no renewal')
    with (directory/'budget.json').open('x') as f: json.dump(budget,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.chmod(directory/'budget.json',0o444)
    atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),**budget))
    cfg=local_config_for(budget); pd=directory/'profile'; pd.mkdir()
    atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],time.time()+1800)))
    os.chmod(pd/'profile_config.json',0o444)
    outcomes=[]
    try:
        outcome=supervise(directory,'profile'); outcomes.append(outcome)
        if outcome['returncode']!=0: raise RuntimeError('Bounded native local profile failed')
        plan=measured_plan(directory,budget); cfg.update(plan)
        for name,value in [('allocation.json',plan),('config.json',cfg)]:
            with (directory/name).open('x') as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
            os.chmod(directory/name,0o444)
        outcome=supervise(directory,'run'); outcomes.append(outcome)
        result=dict(status=outcome['status'],outcomes=outcomes,**budget)
    except Exception as exc:
        result=dict(status='error',error=repr(exc),traceback=traceback.format_exc(),outcomes=outcomes,**budget)
        atomic_json(directory/'failure.json',result)
        completion(directory,'incomplete',result['error'])
    atomic_json(directory/'local_supervisor_result.json',result)
    return result

def main():
    cpu_setup(); parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','profile','pin','run','verify','supervise-profile','supervise-run','local-supervise'])
    parser.add_argument('directory',type=Path); parser.add_argument('--budget',type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='prepare':
        if args.budget is None: parser.error('--budget required')
        return prepare(directory,json.loads(args.budget.read_text()))
    if args.mode=='pin': print(json.dumps(pin(directory))); return
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode=='local-supervise':
        result=local_supervise(directory); print(json.dumps(result),flush=True)
        raise SystemExit(0 if result['status']=='complete' else 2)
    if args.mode.startswith('supervise-'): raise SystemExit(supervise(directory,args.mode.split('-',1)[1])['returncode'])
    target=directory/'profile_config.json' if args.mode=='profile' else directory/'config.json'
    device=json.loads(target.read_text())['device']
    if device=='mps':
        if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
        lock_path=Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock')
    else:
        if device!='cuda' or not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one CUDA GPU required')
        lock_path=ROOT.parent/'delayed_frame_gru_worker.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))

if __name__=='__main__': main()
