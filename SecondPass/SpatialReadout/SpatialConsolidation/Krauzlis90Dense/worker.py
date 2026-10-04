"""Isolated Krauzlis +/-90 cloud arm; architecture reused, every weight fresh."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import torch
from SecondPass.SpatialReadout.FreshRun import worker as fresh
from SecondPass.SpatialReadout.SpatialConsolidation import worker as consolidation
from SecondPass.SpatialReadout.SpatialConsolidation.KrauzlisOnly import worker as single
from .model import DenseObserver
from SecondPass.TaskSuite.suite import task_classes
from .stimuli import SpatialBatteryStream

ROOT=Path(__file__).resolve().parents[4]
MODULE='SecondPass.SpatialReadout.SpatialConsolidation.Krauzlis90Dense.worker'
TASK=single.TASK
TASKS=single.TASKS
CELLS=single.CELLS
VERSION='global_dense_kda_gru_krauzlis90'
PROTOCOL=VERSION+'_wholemodel_fresh_cloud_v1'
INIT_SEED,CUDA_SEED,SCHEDULER_SEED=128192763,128292763,128392763
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=128492763,128592763,138692763
TARGET=7737
MIN_USEFUL_UPDATES=1000
# Private verifier excludes unrelated recognition assets; original modules untouched.
cloud=types.SimpleNamespace(**dict(vars(fresh.cloud),
    original=types.SimpleNamespace(**dict(vars(fresh.cloud.original),verify_sources=single.verify_sources))))
clone=cloud.clone
atomic_json=single.atomic_json
digest=single.digest
load_verified=fresh.load_verified
Scheduler=single.Scheduler
validate_budget=fresh.validate_budget
selection_key=single.selection_key
score_cell=single.score_cell

class FreshStream(fresh.FreshStream):
    stream_seed=clone(fresh.FreshStream.stream_seed,TRAIN_NAMESPACE=TRAIN_NAMESPACE,VAL_NAMESPACE=VAL_NAMESPACE,FINAL_NAMESPACE=FINAL_NAMESPACE,TASKS=TASKS)
    def _make_stream(self,task,cell):
        if task!=TASK: raise ValueError('Krauzlis90 only')
        return SpatialBatteryStream(self.stream_seed(task,cell),self.split)
    def batch(self,n,task,cell):
        if task!=TASK: raise ValueError('Krauzlis90 only')
        return super().batch(n,task,cell)

def provenance():
    return dict(initialization='whole model direct fresh constructor; no checkpoint inputs',
        architecture=VERSION,initialization_seed=INIT_SEED,cuda_seed=CUDA_SEED,
        scheduler_seed=SCHEDULER_SEED,train_namespace=TRAIN_NAMESPACE,
        validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,
        inherited_weights=False,inherited_optimizer=False,inherited_rng=False,
        inherited_stream=False,profile_state_inherited=False,inactive_heads=[],
        task_stream_policy='same fresh02 train/val seed namespaces, reset from zero; independent of model RNG; new final namespace',
        architecture_contract='RGB100 stack3 centered flatten90000 dense128 three global KDA H2 K8 V16 denseGRU128 densehead')


def verify_progress(directory):
    d=Path(directory)
    initial=load_verified(json.loads((d/'initial_checkpoint.json').read_text()))
    receipt=json.loads((d/'latest_checkpoint.json').read_text())
    saved=load_verified(receipt)
    assert initial['schema']==saved['schema']==3
    assert initial['provenance']==saved['provenance']==provenance()
    assert initial['state']['step']==initial['scheduler']['updates']==0
    assert not initial['optimizer']['state'] and initial['stream']['streams']==[]
    assert initial['state']['best_checkpoint'] is None and initial['state']['selection_history']==[]
    # Independent seeded constructor equality on persisted bytes, not a receipt flag.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INIT_SEED)
        direct=DenseObserver(task_classes())
    assert fresh.tree_equal(initial['model'],direct.state_dict())
    names=[n for n,_ in direct.named_parameters()]
    assert saved['optimizer_names']==initial['optimizer_names']==names
    assert set(saved['model'])==set(initial['model'])==set(direct.state_dict())
    step=saved['state']['step']
    assert step>0 and saved['scheduler']['updates']==receipt['step']==step
    assert saved['state']['episodes']==step*saved['state']['config']['effective_batch']
    assert saved['state']['optimizer_seconds']>0 and not fresh.tree_equal(initial['stream'],saved['stream'])
    changed=[]
    assert set(saved['optimizer']['state'])==set(range(len(names)))
    for i,n in enumerate(names):
        opt=saved['optimizer']['state'][i]
        assert float(opt['step'])==step,n
        for k in ('exp_avg','exp_avg_sq'):
            assert opt[k].shape==saved['model'][n].shape and torch.isfinite(opt[k]).all(),n
        assert torch.isfinite(saved['model'][n]).all(),n
        if not torch.equal(initial['model'][n],saved['model'][n]): changed.append(n)
    groups={p:[n for n in changed if n.startswith(p)] for p in
        ('encoder.','kda.0.','kda.1.','kda.2.','dense_gru.','readout.','heads.'+TASK+'.')}
    assert all(groups.values()),groups
    # At step1 alpha has no gradient on first frame, but later frames train it.
    assert len(changed)==len(names),(set(names)-set(changed))
    result=dict(verified=True,step=step,episodes=saved['state']['episodes'],checkpoint=receipt,
        changed_parameters=len(changed),changed_by_module=groups,all_active_parameters_changed=True,
        optimizer_seconds=saved['state']['optimizer_seconds'],fresh_initial_optimizer_empty=True,
        persisted_constructor_equality=True,inactive_unchanged_heads=[])
    atomic_json(d/'persisted_progress_verification.json',result)
    return result

class Session:
    __init__=clone(fresh.Session.__init__,INIT_SEED=INIT_SEED,CUDA_SEED=CUDA_SEED,
        SpatialReadout=DenseObserver,BalancedScheduler=Scheduler,SCHEDULER_SEED=SCHEDULER_SEED,FreshStream=FreshStream,TASKS=TASKS)
    status=clone(fresh.Session.status,VERSION=VERSION)
    train_update=single.Session.train_update
    _checkpoint=clone(fresh.Session.checkpoint,provenance=provenance,verify_progress=verify_progress)
    def checkpoint(self,filename):
        receipt=self._checkpoint(filename)
        # The single-task startup save is step3, not the generic step13 cycle.
        # Verify actual persisted full state before publishing launch readiness.
        if self.state['step']==3:
            verify_progress(self.directory)
        if self.state['step']==0:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if self.device=='cuda' else []):
                torch.manual_seed(INIT_SEED)
                expected=DenseObserver(task_classes()).state_dict()
            assert fresh.tree_equal(expected,fresh.cpu_tree(self.model.state_dict()))
            assert not self.optimizer.state and self.stream.state_dict()['streams']==[]
            atomic_json(self.directory/'constructor_equality.json',dict(verified=True,tensors=len(expected),empty_adam=True,empty_stream=True,checkpoint=receipt))
        return receipt

def evaluate(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
    evaluator=clone(cloud.original.evaluate.__wrapped__,SuiteStream=FreshStream,score_cell=score_cell,
        summarize=clone(single.core.summarize,TASKS=TASKS),sync=cloud.sync)
    with torch.no_grad():
        result=evaluator(model,split,n,krauzlis_n,microbatch,device,output,cells=[(TASK,c) for c in CELLS],deadline=deadline)
    result.update(namespace=VAL_NAMESPACE if split=='val' else FINAL_NAMESPACE,
        selection_rule='three-condition mean AUC then BA; earlier ties; validation only')
    result['summary']['selection_key']=selection_key(result)
    atomic_json(output,result)
    return result

_run=clone(fresh.run,**globals())
def run(directory):
    code=_run(directory)
    report=json.loads((directory/'report.json').read_text())
    report['limitations']=['One whole-model-fresh seed; no convergence or superiority claim.',
        'Only Krauzlis three conditions; +/-90 event magnitude is an explicit non-native adaptation.',
        'Target/foil/catch proportions and native timing retained; metadata never enters model.',
        'Package comparison changes capacity, global representation and recurrent state; not isolated convolution causality.',
        'Fresh train/val streams share fresh02 seed namespaces; final test namespace is new.']
    atomic_json(directory/'report.json',report)
    p=directory/'REPORT.md'
    p.write_text(p.read_text().replace('# Fresh final spatial ConvGRU training','# Fresh Krauzlis +/-90 all-dense KDA GRU training'))
    return code

completion=clone(fresh.completion,TARGET=TARGET)
profile=clone(single.profile,Session=Session,evaluate=evaluate,verify_progress=verify_progress,verify_sources=cloud.original.verify_sources)

def config_for(budget):
    manifest=json.loads((ROOT.parent/'deployment_manifest.json').read_text())
    for r in manifest['files']:
        if digest(ROOT.parent/r['path'])!=r['sha256']: raise ValueError('Deployment mismatch')
    hashes={str(ROOT.parent/r['path']):r['sha256'] for r in manifest['files']}
    return dict(**budget,protocol=PROTOCOL,device='cuda',effective_batch=32,microbatch=4,eval_microbatch=4,
        val_n=100,val_krauzlis_n=100,test_n=200,test_krauzlis_n=200,checkpoint_every=1000,cpu_threads=2,
        source_hashes=hashes,runtime_root=str(ROOT),initialization=provenance(),baseline='none',
        selection='mean three-cell AUC then BA',all_trainable=True,lr=1e-4,clipping=None,bptt='full',precision='fp32',tf32=False)

def measured_plan(directory,budget,now=None):
    now=time.time() if now is None else now
    validate_budget(budget,budget,now)
    p=json.loads((directory/'profile/profile.json').read_text())
    if not p['complete'] or len(p['rows'])!=6 or len(p['evaluation_cells'])!=3: raise ValueError('Incomplete profile')
    if {r['cell'] for r in p['evaluation_cells']}!=set(CELLS): raise ValueError('Missing evaluation conditions')
    costs={}
    for c in CELLS:
        rows=[r for r in p['rows'] if r['cell']==c and r['task']==TASK]
        if len(rows)!=2 or any(r['episodes']!=32 for r in rows): raise ValueError('Two batch32 updates per condition required')
        costs[c]=max(r['seconds'] for r in rows)
    ec=[r['seconds']/r['n'] for r in p['evaluation_cells']]
    if not all(math.isfinite(v) and v>0 for v in [*costs.values(),*ec]): raise ValueError('Invalid timings')
    val=1.35*sum(ec)*100; test=2*val; overhead=2*val+2*test+900
    scheduler=Scheduler(SCHEDULER_SEED); schedule=[]; training=0.; best=None
    for i in range(TARGET):
        pair=scheduler.next(); schedule.append(pair); training+=costs[pair[1]]
        # Reserve pin serialization and the next supervisor's process/import startup.
        # This is outside the run estimate, not an extension of either deadline.
        if 1.25*training+overhead+60<budget['deadline']-now: best=(i+1,training)
    if best is None or best[0]<MIN_USEFUL_UPDATES:
        raise RuntimeError('Fewer than1000 updates fit; stop, do not substitute feasibility pilot')
    count,training=best
    return dict(max_steps=count,target_requested=TARGET,minimum_useful_updates=MIN_USEFUL_UPDATES,
        exposure_reduced=count<TARGET,total_episodes=32*count,
        planned_exposure={c:dict(updates=sum(cell==c for _,cell in schedule[:count]),episodes=32*sum(cell==c for _,cell in schedule[:count])) for c in CELLS},
        validation_steps=[count//2,count],estimated_optimizer_seconds=training,
        estimated_total_remaining_seconds=1.25*training+overhead,slower_150pct_seconds=1.5*training+overhead,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values()),checkpoint_every=1000,
        method='six native batch32/micro4 updates, two per cell; per-cell max; seeded exact schedule; train1.25/eval1.35; two val100+two test200 per cell+900s; separate600s retrieval')

prepare=clone(consolidation.prepare,config_for=config_for,validate_budget=validate_budget)
def pin(directory):
    budget=json.loads((directory/'budget.json').read_text()); cfg=config_for(budget); plan=measured_plan(directory,budget); cfg.update(plan)
    for name,value in [('allocation.json',plan),('config.json',cfg)]:
        with (directory/name).open('x') as f: json.dump(value,f,indent=2)
        os.chmod(directory/name,0o444)
    return plan

def supervise(directory,mode):
    budget=json.loads((directory/'budget.json').read_text()); validate_budget(budget,budget)
    target=directory/'profile' if mode=='profile' else directory
    cfg=json.loads((target/('profile_config.json' if mode=='profile' else 'config.json')).read_text())
    if mode=='run':
        validate_budget(budget,cfg)
        if time.time()+cfg['estimated_total_remaining_seconds']>=budget['deadline']: raise RuntimeError('Pinned allocation no longer fits original cap')
        with (directory/'production_claim.json').open('x') as f: json.dump(dict(pid=os.getpid(),observed=time.time()),f)
    outcome=cloud.supervise([sys.executable,'-u','-m',MODULE,mode,str(target)],min(budget['deadline'],cfg['deadline']),target,prefix=mode)
    atomic_json(target/'supervisor_result.json',dict(outcome,status='complete' if outcome['returncode']==0 else 'failed'))
    return outcome

def main():
    cloud.original.cpu_setup()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','profile','pin','run','verify','supervise-profile','supervise-run']); p.add_argument('directory',type=Path); p.add_argument('--budget',type=Path)
    a=p.parse_args(); d=a.directory.resolve()
    if a.mode=='prepare':
        if a.budget is None: p.error('--budget required')
        return prepare(d,json.loads(a.budget.read_text()))
    if a.mode=='pin': print(json.dumps(pin(d))); return
    if a.mode=='verify': print(json.dumps(verify_progress(d))); return
    if a.mode.startswith('supervise-'):
        out=supervise(d,a.mode.split('-',1)[1]); raise SystemExit(out['returncode'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one CUDA GPU required')
    with (ROOT.parent/'krauzlis90_dense_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.mode=='profile': profile(d)
        else:
            code=run(d); completion(d,'complete' if code==0 else 'incomplete'); raise SystemExit(code)

if __name__=='__main__': main()
