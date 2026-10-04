"""Fresh local MPS replay adapter for the authors' analytic motion front end."""
import argparse, json, os, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
from SecondPass.TwoFrameRViTReplay import worker as replay
from SecondPass.TwoFrameRViT import worker as original
from SecondPass.DelayedFrameGRU import worker as local
from .model import StructuredMotionRViT

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.StructuredMotionRViT.worker'
VERSION='analytic_simoncelli_heeger_structured_motion_rvit'
PROTOCOL=VERSION+'_fresh_native_pool1000_epoch10_local_v1'
TASK,TASKS,CELLS=replay.TASK,replay.TASKS,replay.CELLS
INIT_SEED,CUDA_SEED,SCHEDULER_SEED=replay.INIT_SEED,replay.CUDA_SEED,replay.SCHEDULER_SEED
TARGET=4216
EXPECTED_PARAMETERS=7297650
atomic_json,append_jsonl,cpu_tree,tree_equal=replay.atomic_json,replay.append_jsonl,replay.cpu_tree,replay.tree_equal
digest,load_verified=replay.digest,replay.load_verified
clone=replay.clone
validate_budget,verify_sources=local.validate_budget,local.verify_sources

def sync(device):
    if str(device)=='mps': torch.mps.synchronize()
    elif str(device).startswith('cuda'): torch.cuda.synchronize(device)

def provenance():
    return dict(replay.provenance(),architecture=VERSION,parameter_count=EXPECTED_PARAMETERS,mps_seed=INIT_SEED,
        fixed_frontend='author analytic Simoncelli-Heeger 1998 coefficients stored as buffers; no pretrained learned weights',
        full_learned_initialization='fresh task CNN, spatial positions, recurrent visual block and classifier')

def validation_steps_for(count):
    return sorted(set(([100] if count>=100 else [])+([250] if count>=250 else [])+list(range(500,count+1,500))+[count]))

adapter=types.SimpleNamespace(**dict(vars(original),TwoFrameRViT=StructuredMotionRViT,sync=sync))
_init=clone(original.Session.__init__,TwoFrameRViT=StructuredMotionRViT,EXPECTED_PARAMETERS=EXPECTED_PARAMETERS)
_verify=clone(replay.verify_progress,base=adapter,EXPECTED_PARAMETERS=EXPECTED_PARAMETERS,provenance=provenance)
def verify_progress(directory):
    result=_verify(directory)
    d=Path(directory)
    first=load_verified(json.loads((d/'initial_checkpoint.json').read_text()))
    last=load_verified(result['checkpoint'])
    for payload in (first,last): assert torch.is_tensor(payload['rng'].get('mps')) and payload['rng']['mps'].numel()>0
    fixed=[k for k in first['model'] if k.startswith('motion_frontend.')]
    assert fixed and all(torch.equal(first['model'][k],last['model'][k]) for k in fixed)
    result['fixed_analytic_buffers_unchanged']=True
    atomic_json(d/'persisted_progress_verification.json',result); return result

class Session(replay.Session):
    def __init__(self,directory,config):
        if config['device']=='mps': torch.mps.manual_seed(INIT_SEED)
        _init(self,directory,config)
        self.stream=replay.ReplayStream(); self.scheduler=replay.ReplayScheduler(self.stream)
        self.state.update(unique_episodes_generated=0,unique_episodes_presented=0,replay_presentations=0)
    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=original.utc(),architecture=VERSION,
            initialization='fresh',step=self.state['step'],episodes=self.state['episodes'],
            optimizer_seconds=self.state['optimizer_seconds'],deadline=self.config['deadline'],latest_checkpoint=self.latest,
            best_step=self.state['best_step'],best_key=self.state['best_key'],
            latest_validation=self.state['selection_history'][-1] if self.state['selection_history'] else None,
            pool_index=self.stream.pool_index,epoch=self.stream.epoch,
            unique_episodes_generated=self.state['unique_episodes_generated'],
            unique_episodes_presented=self.state['unique_episodes_presented'],replay_presentations=self.state['episodes'],**extra))
    checkpoint=clone(local.Session.checkpoint,DelayedFrameGRU=StructuredMotionRViT,parameter_count=lambda:EXPECTED_PARAMETERS,
        provenance=provenance,verify_progress=verify_progress,VERSION=VERSION)
    train_update=clone(replay.Session.train_update,base=adapter)

_evaluate=clone(original.evaluate.__wrapped__,sync=sync)
evaluate=torch.no_grad()(_evaluate)
selection_key=original.selection_key
_run=clone(original.run,Session=Session,provenance=provenance,verify_progress=verify_progress,PROTOCOL=PROTOCOL,VERSION=VERSION,evaluate=evaluate)
def run(directory):
    code=_run(directory)
    report=json.loads((directory/'report.json').read_text()); saved=load_verified(report['terminal_checkpoint'])
    report.update(training_policy=provenance(),unique_episodes_generated=saved['state']['unique_episodes_generated'],
        unique_episodes_presented=saved['state']['unique_episodes_presented'],replay_presentations=saved['state']['episodes'])
    atomic_json(directory/'report.json',report)
    p=directory/'REPORT.md'; p.write_text(p.read_text().replace('; episodes ','; presentations ')+
        f"\nUnique movies generated {report['unique_episodes_generated']}; unique movies presented {report['unique_episodes_presented']}; replay presentations {report['replay_presentations']}.\n")
    return code
completion=replay.completion
supervise=clone(original.supervise,MODULE=MODULE,completion=completion)

def config_for(budget):
    manifest_path=ROOT/'runtime_manifest.json'; manifest=json.loads(manifest_path.read_text())
    hashes={str(Path(p) if Path(p).is_absolute() else ROOT/p):sha for p,sha in manifest['source_hashes'].items()}
    hashes[str(manifest_path)]=digest(manifest_path)
    cfg=dict(budget,protocol=PROTOCOL,device='mps',execution_placement='local',checkpoint_encoder=True,
        effective_batch=32,microbatch=1,eval_microbatch=1,val_n=100,test_n=200,checkpoint_every=100,cpu_threads=2,
        source_hashes=hashes,runtime_root=str(ROOT),initialization=provenance(),all_trainable=True,
        lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clipping=None,precision='fp32',bptt='full',tf32=False,
        training_policy='1000native movies reused10shuffled epochs then regenerate')
    verify_sources(cfg); return cfg

def profile(directory):
    cfg=json.loads((directory/'profile_config.json').read_text()); verify_sources(cfg)
    s=Session(directory,cfg); s.checkpoint('initial.pt'); warmup=[]; rows=[]
    for i in range(6):
        if time.time()>=cfg['deadline']-60: raise RuntimeError('Bounded native MPS profile expired')
        row=s.train_update(*s.scheduler.next()); (warmup if i<3 else rows).append(row)
        if s.state['step'] in (1,2,3): s.checkpoint(f"checkpoint_{s.state['step']:06d}.pt")
    s.checkpoint('profile_terminal.pt')
    ev=evaluate(s.model,'val',20,1,s.device,directory/'eval_timing.json',cfg['deadline']-30)
    assert ev['complete']
    atomic_json(directory/'profile.json',dict(complete=True,architecture=VERSION,warmup_rows=warmup,rows=rows,
        evaluation_cells=ev['cells'],disposable=True,discard_all_state=True,verification=verify_progress(directory),
        mps_current_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory()))

def measured_plan(directory,budget,now=None):
    now=time.time() if now is None else now; validate_budget(budget,budget,now)
    p=json.loads((directory/'profile/profile.json').read_text())
    assert p['complete'] and len(p['rows'])==3 and {r['cell'] for r in p['rows']}==set(CELLS)
    costs={r['cell']:r['seconds'] for r in p['rows']}
    assert all(r['episodes']==32 and r['seconds']>0 for r in p['rows'])
    val=1.35*sum(r['seconds']/r['n'] for r in p['evaluation_cells'])*100; test=2*val
    generation=sum(r.get('pool_generation_seconds',0.) for r in p['warmup_rows']); assert generation>0
    rng=random.Random(SCHEDULER_SEED+71); schedule=[]; training=0.; best=None
    for i in range(TARGET):
        if i%330==0: sizes={c:333+int(j==(i//330)%3) for j,c in enumerate(CELLS)}
        if i%33==0: schedule.extend(replay.epoch_batches(sizes,rng))
        cell,indices=schedule[i]; training+=costs[cell]
        gen=((i//330)+1)*generation
        total=1.25*(training+gen)+len(validation_steps_for(i+1))*val+2*test+900
        if total+60<budget['deadline']-now: best=(i+1,training,gen,total)
    if best is None or best[0]<330: raise RuntimeError('One full1000movie10epochpool does not fit finite local cap')
    count=best[0]//330*330
    training=sum(costs[c] for c,_ in schedule[:count]); gen=((count+329)//330)*generation
    total=1.25*(training+gen)+len(validation_steps_for(count))*val+2*test+900
    return dict(max_steps=count,target_requested=TARGET,exposure_reduced=count<TARGET,
        total_episodes=sum(len(ix) for _,ix in schedule[:count]),planned_presentations=sum(len(ix) for _,ix in schedule[:count]),
        planned_unique_movies=count//330*1000,validation_steps=validation_steps_for(count),checkpoint_every=100,
        estimated_optimizer_seconds=training,estimated_pool_generation_seconds=gen,estimated_total_remaining_seconds=total,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values())+generation,
        method='3nativewarmup+3timed MPS updates; full32cost for tails; poolgeneration each330updates; validation100/250/every500/terminal; complete replay pools prospectively within8h')

def local_supervise(directory,cap_started=None):
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'budget.json').exists(): raise RuntimeError('No restart or cap renewal')
    started=time.time() if cap_started is None else cap_started
    budget=dict(cap_started=started,deadline=started+28200,hard_deadline=started+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800,execution_placement='local',
        origin='First actual MPS compatibility/profile; fresh local cap, no renewal')
    validate_budget(budget,budget)
    with (directory/'budget.json').open('x') as f: json.dump(budget,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.chmod(directory/'budget.json',0o444)
    atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),**budget))
    cfg=config_for(budget); pd=directory/'profile'; pd.mkdir()
    atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],time.time()+3600)))
    outcomes=[]
    try:
        outcome=supervise(directory,'profile'); outcomes.append(outcome)
        if outcome['returncode']!=0: raise RuntimeError('Native MPS profile failed')
        plan=measured_plan(directory,budget); cfg.update(plan)
        for name,value in [('allocation.json',plan),('config.json',cfg)]:
            with (directory/name).open('x') as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
            os.chmod(directory/name,0o444)
        outcome=supervise(directory,'run'); outcomes.append(outcome)
        result=dict(status=outcome['status'],outcomes=outcomes,**budget)
    except Exception as exc:
        result=dict(status='error',error=repr(exc),outcomes=outcomes,**budget)
        atomic_json(directory/'failure.json',result); completion(directory,'incomplete',result['error'])
    atomic_json(directory/'local_supervisor_result.json',result); return result

def main():
    original.cpu_setup(); p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['local-supervise','profile','run','verify']); p.add_argument('directory',type=Path)
    p.add_argument('--cap-start',type=float)
    args=p.parse_args(); d=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(d))); return
    if args.mode=='local-supervise':
        result=local_supervise(d,args.cap_start); print(json.dumps(result),flush=True)
        raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    import fcntl
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(d)
        else: raise SystemExit(run(d))
if __name__=='__main__': main()
