"""Fresh-only v1 terminal consolidation adapter; no checkpoint/resume CLI.
Frozen FreshRun loop is reused with a private globals namespace, never mutated.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from SecondPass.SpatialReadout.FreshRun import worker as fresh
from SecondPass.SpatialReadout.SpatialConsolidation.model import SpatialConsolidation
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, tree_equal
from SecondPass.TaskSuite.suite import TASKS, task_classes

cloud=fresh.cloud
ROOT=Path(__file__).resolve().parents[3]
VERSION='terminal_spatial_consolidation'
PROTOCOL=VERSION+'_wholemodel_fresh_v1'
INIT_SEED,CUDA_SEED,SCHEDULER_SEED=108192763,108292763,108392763
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=108492763,108592763,108692763
TARGET=46800
digest=cloud.digest
load_verified=cloud.load_verified
# Clone function scopes only; imported modules and frozen files remain unchanged.
scope=dict(vars(fresh), **{k:v for k,v in globals().copy().items() if not k.startswith('__')})
scope['SpatialReadout']=SpatialConsolidation
FreshStream=type('FreshStream',(fresh.FreshStream,),{'stream_seed':cloud.clone(fresh.FreshStream.stream_seed,**scope)})
scope['FreshStream']=FreshStream
provenance=cloud.clone(fresh.provenance,**scope)
scope['provenance']=provenance
_native_verify=cloud.clone(fresh.verify_progress,**scope)

def verify_progress(directory):
    result=_native_verify(directory)
    directory=Path(directory)
    initial=load_verified(json.loads((directory/'initial_checkpoint.json').read_text()))
    saved=load_verified(json.loads((directory/'latest_checkpoint.json').read_text()))
    changed=[n for n in saved['model'] if n.startswith('consolidation.') and not torch.equal(saved['model'][n],initial['model'][n])]
    assert changed, 'Terminal transformer failed to update'
    result['consolidation_changed_tensors']=changed
    atomic_json(directory/'persisted_progress_verification.json',result)
    return result

scope['verify_progress']=verify_progress
Session=type('Session',(),{name:cloud.clone(fn,**scope) for name,fn in vars(fresh.Session).items() if callable(fn)})
selection_key=fresh.selection_key
evaluate=cloud.clone(fresh.evaluate,**scope)


def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    keys=('cap_started','deadline','hard_deadline','retrieval_reserve_seconds','wall_cap_seconds')
    if any(config.get(k)!=budget[k] for k in keys): raise ValueError('Absolute cap changed')
    if budget['wall_cap_seconds']!=43200 or budget['hard_deadline']-budget['cap_started']!=43200:
        raise ValueError('Require new twelve-hour pod-creation cap')
    if budget['retrieval_reserve_seconds']!=600 or budget['hard_deadline']-budget['deadline']!=600:
        raise ValueError('Require separate 600-second retrieval reserve')
    if not budget['cap_started']<=now<budget['deadline']: raise ValueError('Cap expired/not started')

config_for=cloud.clone(fresh.config_for,**scope)
scope.update(Session=Session,evaluate=evaluate,validate_budget=validate_budget)
run=cloud.clone(fresh.run,**scope)
completion=cloud.clone(fresh.completion,**scope)


def profile(directory):
    """Two complete per-cell passes through the exact production update path."""
    config=json.loads((directory/'profile_config.json').read_text())
    cloud.original.verify_sources(config)
    session=Session(directory,config)
    session.checkpoint('initial.pt')
    rows=[]
    # Canonical schedule cycle first permits verified step1/13 checkpoint counters.
    for _ in range(13):
        rows.append(session.train_update(*session.scheduler.next()))
        if session.state['step']==1: session.checkpoint('checkpoint_000001.pt')
    session.checkpoint('checkpoint_000013.pt')
    # Disposable only: profile each condition directly; do not checkpoint its
    # artificial scheduler cursor or transfer ANY of this state to production.
    timings=[]
    for repeat in range(2):
        for task,cell in cloud.original.all_cells():
            if time.time()>=config['deadline']-60: raise RuntimeError('Bounded profile expired')
            row=cloud.update(session.model,session.optimizer,session.stream,task,cell,32,4,session.device)
            timings.append(dict(row,repeat=repeat))
    ev=evaluate(session.model,'val',8,8,4,session.device,directory/'eval_timing.json',config['deadline']-30)
    if not ev['complete'] or len(ev['cells'])!=35: raise RuntimeError('Incomplete all-cell timing')
    atomic_json(directory/'profile.json',dict(complete=True,architecture=VERSION,rows=rows,training_cells=timings,
        evaluation_cells=ev['cells'],verification=verify_progress(directory),disposable=True,discard_all_state=True,
        initialization=provenance(),torch_version=torch.__version__,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


def measured_plan(directory,budget,now=None):
    now=time.time() if now is None else now
    validate_budget(budget,budget,now)
    p=json.loads((directory/'profile/profile.json').read_text())
    cells=set(cloud.original.all_cells())
    if not p['complete'] or p['architecture']!=VERSION: raise ValueError('Wrong profile')
    if len(p['evaluation_cells'])!=35 or {(r['task'],r['cell']) for r in p['evaluation_cells']}!=cells:
        raise ValueError('Require all35 evaluation cells')
    ec={(r['task'],r['cell']):r['seconds']/r['n'] for r in p['evaluation_cells']}
    tc={}
    for cell in cells:
        rows=[r for r in p['training_cells'] if (r['task'],r['cell'])==cell]
        if len(rows)!=2 or any(r['episodes']!=32 for r in rows): raise ValueError('Require two native batch32 updates per cell')
        tc[cell]=max(r['seconds'] for r in rows)
    if not all(math.isfinite(v) and v>0 for v in [*tc.values(),*ec.values()]): raise ValueError('Invalid timings')
    val=1.35*sum(ec[t,c]*(100 if t=='krauzlis_cued_motion' else 64) for t,c in cells)
    test=2*val
    scheduler=BalancedScheduler(SCHEDULER_SEED)
    schedule=[]; training=0.; best=None; overhead=2*val+2*test+900
    # Largest feasible whole13task exposure, not a promise that proposal fits.
    for i in range(TARGET):
        pair=scheduler.next(); schedule.append(pair); training+=tc[pair]
        total=1.25*training+overhead
        if (i+1)%13==0 and i+1>=26 and total<budget['deadline']-now:
            best=(i+1,training,total)
    if best is None: raise RuntimeError('No complete acquisition allocation fits')
    count,training,total=best
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()}
    for t,c in schedule[:count]:
        exposure[t]['updates']+=1; exposure[t]['episodes']+=32; exposure[t]['cells'][c]+=32
    return dict(max_steps=count,target_requested=TARGET,validation_steps=[count//26*13,count],total_episodes=count*32,
        exposure_reduced=count<TARGET,updates_per_task=count//13,planned_exposure=exposure,
        estimated_optimizer_seconds=training,estimated_total_remaining_seconds=total,slower_150pct_seconds=1.5*training+overhead,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(tc.values()),method='2 native batch32 updates per35cells, percell max; exact seeded schedule; train1.25/eval1.35; 2val+2test+900s; separate600s retrieval')


def prepare(directory,budget):
    validate_budget(budget,budget)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'budget.json').open('x') as f: json.dump(budget,f,indent=2)
    os.chmod(directory/'budget.json',0o444)
    config=config_for(budget)
    pd=directory/'profile'; pd.mkdir()
    atomic_json(pd/'profile_config.json',dict(config,disposable_profile=True,deadline=min(budget['deadline'],time.time()+1800)))


def pin(directory):
    budget=json.loads((directory/'budget.json').read_text())
    config=config_for(budget); plan=measured_plan(directory,budget)
    config.update(plan)
    # Checkpoint every1001 updates (~77/task), not13: bounded volume use.
    config['checkpoint_every']=1001
    for name,value in [('allocation.json',plan),('config.json',config)]:
        with (directory/name).open('x') as f: json.dump(value,f,indent=2)
        os.chmod(directory/name,0o444)
    return plan


def supervise(directory,mode):
    budget=json.loads((directory/'budget.json').read_text())
    validate_budget(budget,budget)
    target=directory/'profile' if mode=='profile' else directory
    config=json.loads((target/('profile_config.json' if mode=='profile' else 'config.json')).read_text())
    if mode=='run': validate_budget(budget,config)
    deadline=min(budget['deadline'],config['deadline'])
    command=[sys.executable,'-u','-m','SecondPass.SpatialReadout.SpatialConsolidation.worker',mode,str(target)]
    # Caller runs from packaged repo; parent independent billing guard is separate.
    return cloud.supervise(command,deadline,target,prefix=mode)


def main():
    cloud.original.cpu_setup()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','profile','pin','run','verify','supervise-profile','supervise-run'])
    p.add_argument('directory',type=Path)
    p.add_argument('--budget',type=Path)
    args=p.parse_args(); directory=args.directory.resolve()
    if args.mode=='prepare':
        if args.budget is None: p.error('--budget required')
        prepare(directory,json.loads(args.budget.read_text())); return
    if args.mode=='pin': print(json.dumps(pin(directory),indent=2)); return
    if args.mode=='verify': print(json.dumps(verify_progress(directory),indent=2)); return
    if args.mode.startswith('supervise-'):
        result=supervise(directory,args.mode.split('-',1)[1])
        print(json.dumps(result)); raise SystemExit(result['returncode'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Require exactly one CUDA GPU')
    with (ROOT.parent/'spatial_consolidation_fresh_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else:
            status=run(directory)
            completion(directory,'complete' if status==0 else 'incomplete')
            raise SystemExit(status)

if __name__=='__main__': main()
