"""Local fresh-only native Krauzlis adapter; reuses the existing training loop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types
import numpy as np
import torch
from SecondPass.SpatialReadout.FreshRun import worker as fresh
from SecondPass.SpatialReadout.SpatialConsolidation.model import SpatialConsolidation
from SecondPass.JointTraining import core
from SecondPass.TaskSuite.suite import TASKS as ALL_TASKS, task_classes
from SecondPass.SpatialReadout.protocol import supervise, digest, assert_no_worker

TASK='krauzlis_cued_motion'
TASKS={TASK:ALL_TASKS[TASK]}
CELLS=[c['id'] for c in TASKS[TASK]['conditions']]
ROOT=Path(__file__).resolve().parents[4]
VERSION='terminal_spatial_consolidation_krauzlis_only'
PROTOCOL=VERSION+'_wholemodel_fresh_local_v1'
INIT_SEED=118192763
SCHEDULER_SEED=118392763
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=118492763,118592763,118692763
clone=fresh.cloud.clone
atomic_json=core.atomic_json
load_verified=fresh.load_verified

class Scheduler(core.BalancedScheduler):
    __init__=clone(core.BalancedScheduler.__init__,TASKS=TASKS)
    next=clone(core.BalancedScheduler.next,TASKS=TASKS)

class FreshStream(fresh.FreshStream):
    stream_seed=clone(fresh.FreshStream.stream_seed,TRAIN_NAMESPACE=TRAIN_NAMESPACE,VAL_NAMESPACE=VAL_NAMESPACE,FINAL_NAMESPACE=FINAL_NAMESPACE)
    def batch(self,n,task,cell):
        if task!=TASK: raise ValueError('Krauzlis ONLY')
        return super().batch(n,task,cell)

def provenance():
    return dict(initialization='whole model direct fresh constructor; no checkpoint inputs',architecture=VERSION,
        initialization_seed=INIT_SEED,scheduler_seed=SCHEDULER_SEED,train_namespace=TRAIN_NAMESPACE,
        validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,inherited_weights=False,
        inherited_optimizer=False,inherited_rng=False,inherited_stream=False,profile_state_inherited=False,
        inactive_heads=[t for t in ALL_TASKS if t!=TASK])

class Session:
    __init__=clone(fresh.Session.__init__,INIT_SEED=INIT_SEED,SpatialReadout=SpatialConsolidation,
        BalancedScheduler=Scheduler,SCHEDULER_SEED=SCHEDULER_SEED,FreshStream=FreshStream,TASKS=TASKS)
    status=clone(fresh.Session.status,VERSION=VERSION)
    def train_update(self,task,cell):
        row=clone(fresh.Session.train_update)(self,task,cell)
        if self.state['step'] in (3,13) and not self.config.get('disposable_profile'):
            p=json.loads((self.directory/'profile/profile.json').read_text())
            costs={c:max(r['seconds'] for r in p['rows'] if r['cell']==c) for c in CELLS}
            rows=[json.loads(x) for x in (self.directory/'progress.jsonl').read_text().splitlines()]
            ratio=sum(r['seconds'] for r in rows)/sum(costs[r['cell']] for r in rows)
            atomic_json(self.directory/'first_cycle_timing.json',dict(updates=self.state['step'],matched_profile_ratio=ratio,material_slowdown=ratio>1.35,production_rows=rows))
        if self.state['step']==3 and not self.config.get('disposable_profile'):
            self.checkpoint('checkpoint_000003.pt')
        return row
    def checkpoint(self,filename):
        path=self.directory/filename
        if path.exists(): raise FileExistsError(path)
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
            mps=torch.mps.get_rng_state() if self.device=='mps' else None)
        payload=core.cpu_tree(dict(schema=3,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            optimizer_names=[n for n,_ in self.model.named_parameters()],state=self.state,
            scheduler=self.scheduler.state_dict(),stream=self.stream.state_dict(),rng=rng,provenance=provenance()))
        temp=path.with_suffix('.pt.tmp')
        with temp.open('wb') as f: torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
        receipt=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path),verified=True,step=self.state['step'])
        assert core.tree_equal(payload,load_verified(receipt))
        self.latest=receipt
        atomic_json(self.directory/'latest_checkpoint.json',receipt)
        if self.state['step']==0:
            atomic_json(self.directory/'initial_checkpoint.json',receipt)
            # Independent direct constructor, preserving the production RNG states.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(INIT_SEED)
                direct=SpatialConsolidation(task_classes()).state_dict()
                assert core.tree_equal(payload['model'],direct)
            assert not payload['optimizer']['state'] and payload['stream']['streams']==[]
            atomic_json(self.directory/'constructor_equality.json',dict(verified=True,tensors=len(direct),empty_adam=True,empty_stream=True,checkpoint=receipt))
        core.append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=fresh.cloud.original.utc(),**receipt))
        if self.state['step'] in (1,2,3): verify_progress(self.directory)
        return receipt

def verify_progress(directory):
    d=Path(directory); initial=load_verified(json.loads((d/'initial_checkpoint.json').read_text()))
    receipt=json.loads((d/'latest_checkpoint.json').read_text()); saved=load_verified(receipt)
    assert initial['state']['step']==0 and not initial['optimizer']['state'] and initial['stream']['streams']==[]
    step=saved['state']['step']; assert step>0 and saved['scheduler']['updates']==step
    assert saved['state']['episodes']==step*saved['state']['config']['effective_batch']
    changed=[]; inactive=[]
    for i,n in enumerate(saved['optimizer_names']):
        opt=saved['optimizer']['state'].get(i)
        if n.startswith('heads.') and not n.startswith('heads.'+TASK+'.'):
            assert opt is None and torch.equal(initial['model'][n],saved['model'][n]); inactive.append(n); continue
        assert opt is not None and float(opt['step'])==step,n
        assert all(torch.isfinite(v).all() for v in opt.values() if torch.is_tensor(v)),n
        if not torch.equal(initial['model'][n],saved['model'][n]): changed.append(n)
    groups={p:[n for n in changed if n.startswith(p)] for p in ('blocks.','acc.0.','acc.1.','acc.2.','spatial_input.','spatial_gru.','consolidation.','readout.','heads.'+TASK+'.')}
    assert all(groups.values()),groups
    result=dict(verified=True,step=step,episodes=saved['state']['episodes'],checkpoint=receipt,
        changed_parameters=len(changed),changed_by_module=groups,inactive_unchanged_heads=inactive,
        optimizer_seconds=saved['state']['optimizer_seconds'],fresh_initial_optimizer_empty=True)
    atomic_json(d/'persisted_progress_verification.json',result); return result

def score_cell(task,cell,labels,probabilities,metadata):
    result=core.score_cell(task,cell,labels,probabilities,metadata)
    for event,row in result['events'].items():
        pairs=[(int(y),int(np.argmax(p))) for y,p,m in zip(labels,probabilities,metadata) if m['event_type']==event]
        cm=[[sum(y==a and pred==b for y,pred in pairs) for b in range(2)] for a in range(2)]
        row.update(confusion=cm,hit_rate=cm[1][1]/sum(cm[1]) if sum(cm[1]) else None,
            specificity=cm[0][0]/sum(cm[0]) if sum(cm[0]) else None,
            false_positive_rate=cm[0][1]/sum(cm[0]) if sum(cm[0]) else None)
    return result

def selection_key(result):
    if not result['complete'] or len(result['cells'])!=3 or {r['cell'] for r in result['cells']}!=set(CELLS): return None
    if any(r[k] is None for r in result['cells'] for k in ('auc','balanced_accuracy')): return None
    return [sum(r[k] for r in result['cells'])/3 for k in ('auc','balanced_accuracy')]

def evaluate(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
    evaluator=clone(fresh.cloud.original.evaluate.__wrapped__,SuiteStream=FreshStream,score_cell=score_cell,
        summarize=clone(core.summarize,TASKS=TASKS))
    with torch.no_grad(): result=evaluator(model,split,n,krauzlis_n,microbatch,device,output,cells=[(TASK,c) for c in CELLS],deadline=deadline)
    result.update(namespace=VAL_NAMESPACE if split=='val' else FINAL_NAMESPACE,selection_rule='mean three-cell AUC then BA; earlier ties; validation only')
    result['summary']['selection_key']=selection_key(result); atomic_json(output,result); return result

def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    assert budget['deadline']-budget['cap_started']==28800
    assert all(config[k]==v for k,v in budget.items())
    assert budget['cap_started']<=now<budget['deadline']

def verify_sources(config):
    for p,h in config['source_hashes'].items():
        assert digest(p)==h,p

local=types.SimpleNamespace(**dict(vars(fresh.cloud),original=types.SimpleNamespace(**dict(vars(fresh.cloud.original),verify_sources=verify_sources))))
run=clone(fresh.run,Session=Session,evaluate=evaluate,selection_key=selection_key,validate_budget=validate_budget,
    cloud=local,PROTOCOL=PROTOCOL,VERSION=VERSION,provenance=provenance,verify_progress=verify_progress)

def profile(directory):
    cfg=json.loads((directory/'profile_config.json').read_text()); verify_sources(cfg)
    s=Session(directory,cfg); s.checkpoint('initial.pt'); rows=[]
    for _ in range(6):
        rows.append(s.train_update(*s.scheduler.next()))
    s.checkpoint('profile_terminal.pt')
    ev=evaluate(s.model,'val',20,20,4,s.device,directory/'eval_timing.json',cfg['deadline']-30)
    assert ev['complete']
    atomic_json(directory/'profile.json',dict(complete=True,rows=rows,evaluation_cells=ev['cells'],disposable=True,discard_all_state=True,verification=verify_progress(directory)))

def measured_plan(directory,budget):
    p=json.loads((directory/'profile/profile.json').read_text())
    assert p['complete'] and len(p['rows'])==6 and len(p['evaluation_cells'])==3
    costs={c:max(r['seconds'] for r in p['rows'] if r['cell']==c) for c in CELLS}
    val=1.35*sum(r['seconds']/r['n']*100 for r in p['evaluation_cells']); test=2*val
    overhead=2*val+2*test+900
    scheduler=Scheduler(SCHEDULER_SEED); schedule=[]; training=0.; best=None
    for i in range(10000):
        pair=scheduler.next(); schedule.append(pair); training+=costs[pair[1]]
        if 1.25*training+overhead<budget['deadline']-time.time(): best=(i+1,training)
    if best is None or best[0]<30: raise RuntimeError('Insufficient acquisition allowance')
    count,training=best
    return dict(max_steps=count,target_requested=10000,exposure_reduced=count<10000,total_episodes=32*count,
        planned_exposure={c:dict(updates=sum(cell==c for _,cell in schedule[:count]),episodes=32*sum(cell==c for _,cell in schedule[:count])) for c in CELLS},
        validation_steps=[count//2,count],estimated_optimizer_seconds=training,estimated_total_remaining_seconds=1.25*training+overhead,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values()),checkpoint_every=100,
        method='two complete batch32 micro4 native updates/cell, max per-cell; exact seeded schedule; train1.25/eval1.35; two val100 and two test200 per cell plus900s')

def supervisor(directory):
    directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'budget.json').exists(),'No restart or cap renewal'
    exclusive=assert_no_worker()
    budget=dict(cap_started=time.time()); budget.update(deadline=budget['cap_started']+28800.,wall_cap_seconds=28800)
    atomic_json(directory/'budget.json',budget)
    atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),exclusive=exclusive,**budget))
    hashes=json.loads((ROOT/'runtime_manifest.json').read_text())['source_hashes']
    cfg=dict(**budget,device='mps',effective_batch=32,microbatch=4,eval_microbatch=4,cpu_threads=2,source_hashes=hashes,
        initialization=provenance(),all_trainable=True,lr=1e-4,clipping=None,precision='fp32',bptt='full')
    pd=directory/'profile'; pd.mkdir(); atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],time.time()+1800)))
    cmd=[sys.executable,'-u','-m',__spec__.name]
    out=supervise(cmd+['profile',str(pd)],min(budget['deadline'],time.time()+1800),pd,'profile')
    if out['returncode']!=0: raise RuntimeError('Profile failed')
    plan=measured_plan(directory,budget); atomic_json(directory/'allocation.json',plan); cfg.update(plan)
    atomic_json(directory/'config.json',cfg)
    for name in ('budget.json','allocation.json','config.json'): os.chmod(directory/name,0o444)
    out=supervise(cmd+['run',str(directory)],budget['deadline'],directory,'production')
    atomic_json(directory/'supervisor_result.json',out)

def main():
    fresh.cloud.original.cpu_setup()
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['supervise','profile','run','probe','verify']); p.add_argument('directory',type=Path)
    a=p.parse_args(); d=a.directory.resolve()
    if a.mode=='probe':
        d.mkdir(parents=True,exist_ok=True)
        return supervise([sys.executable,'-c','import time; time.sleep(30)'],time.time()+2,d,'probe')
    if a.mode=='verify': print(json.dumps(verify_progress(d))); return
    if a.mode=='supervise': supervisor(d); return
    assert torch.backends.mps.is_available()
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.mps.manual_seed(INIT_SEED)
        if a.mode=='profile': profile(d)
        else: raise SystemExit(run(d))

if __name__=='__main__': main()
