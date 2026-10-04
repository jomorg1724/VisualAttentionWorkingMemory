"""One learned comparison delta; reuse frozen optimizer, streams and run loop."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import statistics
import subprocess
import sys
import time
import types

import numpy as np
import torch
from torch import nn
from SecondPass.JointTraining import worker as original
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, append_jsonl, cpu_tree, tree_equal, score_cell
from SecondPass.TaskSuite.suite import TASKS, SuiteStream, task_classes
from SecondPass.SpatialReadout import worker as inherited
from SecondPass.SpatialReadout.model import SpatialReadout
from SecondPass.SpatialReadout.state import map_adam, restore, load_verified
from SecondPass.SpatialReadout.protocol import digest, launchd_spec, supervise
from SecondPass.SpatialReadout.continuation_v2 import guard, verify_coverage

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_continuation_v2/terminal.pt')
SOURCE_SHA='1826a67acdebcf2f979f614c09131afefdf1920b5dff914784a34bf150c9a841'
PROTOCOL='learned_spatial_comparison_v1'
VAL_NAMESPACE=95192763
FINAL_NAMESPACE=95292763
INIT_SEED=95392763
CAP=86400.
TARGET=2600
ARMS=('control','candidate')
TARGET_TASKS=('orientation_cued','spatial_binding')
WORKER_LOCK=Path('/Users/jonathanmorgan/VAWMRuntime/.spatial_optimizer.lock')


class ComparisonReadout(SpatialReadout):
    def __init__(self, classes):
        super().__init__(classes)
        self.comparator=nn.Sequential(nn.Conv2d(128,64,1),nn.ReLU(),nn.Conv2d(64,64,1))
        nn.init.zeros_(self.comparator[-1].weight)
        nn.init.zeros_(self.comparator[-1].bias)

    def recurrent_states(self, images):
        images=self.frames(images); encoder_states=None; hidden=None; history=[]
        for t in range(images.shape[1]):
            field,encoder_states=self.encode_frame(images[:,t],encoder_states)
            sensory=self.spatial_input(field)
            previous=torch.zeros_like(sensory) if hidden is None else hidden
            hidden=self.spatial_gru(sensory,previous)
            comparison=self.comparator(torch.cat((previous,sensory),1))
            history.append(hidden+comparison)
        return history


def source_payload():
    if digest(SOURCE)!=SOURCE_SHA: raise ValueError('Not authorized terminal6760')
    source=torch.load(SOURCE,map_location='cpu')
    if source['state']['step']!=6760 or source['scheduler']['updates']!=10153 or source['scheduler']['tasks']:
        raise ValueError('Source lineage/scheduler mismatch')
    config=copy.deepcopy(source['state']['config'])
    old_root=Path(config['runtime_root'])
    config['source_hashes']={str(ROOT/Path(p).relative_to(old_root)):sha for p,sha in config['source_hashes'].items()}
    original.verify_sources(config)
    receipt=dict(path=str(SOURCE),sha256=SOURCE_SHA,bytes=SOURCE.stat().st_size,step=6760,verified=True)
    return source,receipt


def migrate(source,receipt,arm):
    assert arm in ARMS
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(INIT_SEED).get_state())
        model=(ComparisonReadout if arm=='candidate' else SpatialReadout)(task_classes())
    weights=cpu_tree(model.state_dict())
    for name,value in source['model'].items():
        if name not in weights or value.shape!=weights[name].shape:raise ValueError('Inherited name/shape changed: '+name)
        weights[name]=value.clone()
    names=[n for n,_ in model.named_parameters()]
    optimizer=map_adam(source['optimizer'],source['optimizer_names'],source['model'],names,weights)
    g=optimizer['param_groups'][0]
    assert (g['lr'],g['betas'],g['eps'],g['weight_decay'])==(1e-4,(.9,.999),1e-8,0)
    state=dict(step=0,episodes=0,frames=0,optimizer_seconds=0.,best_step=None,best_key=None,best_checkpoint=None,
        selection_history=[],parent=cpu_tree(source['state']),
        exposure={t:dict(updates=0,episodes=0,frames=0,cells={c['id']:0 for c in spec['conditions']}) for t,spec in TASKS.items()})
    return dict(schema=2,model=weights,optimizer=optimizer,optimizer_names=names,state=state,
        scheduler=cpu_tree(source['scheduler']),stream=cpu_tree(source['stream']),rng=cpu_tree(source['rng']),
        migration=dict(source=receipt,arm=arm,initialization_seed=INIT_SEED,selection_reset=True,profile_state_inherited=False,
            carried_names=list(source['model']),fresh_names=[n for n in names if n not in source['model']],
            inherited_optimizer_seconds=source['state']['optimizer_seconds'],inherited_global_updates=10153,
            inherited_convgru_updates=6760,architecture='h_next=original_gru(s,h); decision=h_next+MLP1x1([h,s]); recurrent carry=h_next'))


def selection_key(result):
    rows=[r for r in result['cells'] if r['task'] in TARGET_TASKS]
    suite=result['summary']['equal_task_mean_auc']
    if not result['complete'] or len(rows)!=8 or suite is None or any(r['auc'] is None for r in rows):return None
    return [statistics.mean(r['auc'] for r in rows),suite]


class EvaluationStream(SuiteStream):
    def __init__(self,split):
        if split not in ('val','test'):raise ValueError('Evaluation-only stream')
        super().__init__(split)
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return (VAL_NAMESPACE if self.split=='val' else FINAL_NAMESPACE)*100000+TASKS[task]['stream_id']*1000+index


def evaluate(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
    def scored(task,cell,labels,probabilities,metadata):
        row=score_cell(task,cell,labels,probabilities,metadata)
        if split=='test':
            row['paired_observations']=dict(labels=labels,probabilities=probabilities,
                metadata_sha256=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest())
        return row
    raw=original.evaluate.__wrapped__
    fn=types.FunctionType(raw.__code__,dict(raw.__globals__,SuiteStream=EvaluationStream,score_cell=scored),raw.__name__,raw.__defaults__,raw.__closure__)
    with torch.no_grad():result=fn(model,split,n,krauzlis_n,microbatch,device,output,cells=cells,deadline=deadline)
    result['summary']['selection_key']=selection_key(result)
    result['namespace']=VAL_NAMESPACE if split=='val' else FINAL_NAMESPACE
    result['selection_rule']='target8 equal-cell AUC then suite equal-task AUC; earlier ties; validation only'
    atomic_json(output,result)
    return result


class Session(inherited.TrainingSession):
    def __init__(self,directory,config,payload):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config;self.device=config['device']
        self.model=(ComparisonReadout if config['arm']=='candidate' else SpatialReadout)(task_classes()).to(self.device)
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.scheduler=BalancedScheduler(0);self.stream=SuiteStream('train')
        self.state=restore(payload,self.model,self.optimizer,self.scheduler,self.stream,self.device)
        self.state['config']=config;self.state['deadline']=config['deadline'];self.migration=payload['migration'];self.latest=None
        self.initial_new={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if n.startswith('comparator.')}
        if not config.get('disposable_profile'):
            baseline=json.loads((self.directory.parent/'baseline_validation.json').read_text())
            verify_coverage(baseline,64,100)
            key=selection_key(baseline)
            if key is None:raise ValueError('Baseline not eligible')
            self.state.update(best_key=key,best_step=0,best_checkpoint=str(self.directory/'migration.pt'),
                selection_history=[dict(step=0,key=key,complete=True,source='fresh_shared_baseline')])

    def checkpoint(self,filename):
        receipt=super().checkpoint(filename)
        if self.state['step'] in (2,13) and filename.startswith('checkpoint_'):
            # Automatic persisted-state proof also runs when the second arm starts.
            verify_progress(self.directory)
        return receipt

    def train_update(self,task,cell):
        row=super().train_update(task,cell)
        step=self.state['step']
        if self.config['arm']=='candidate' and step in (1,2,13):
            evidence={}
            for n,p in self.model.named_parameters():
                if n not in self.initial_new:continue
                norm=float(p.grad.detach().square().sum().cpu().sqrt()) if p.grad is not None else None
                changed=not torch.equal(p.detach().cpu(),self.initial_new[n])
                adam=self.optimizer.state[p]
                evidence[n]=dict(gradient_norm=norm,changed=changed,adam_step=float(adam['step']))
                if norm is None or not np.isfinite(norm) or float(adam['step'])!=step:raise ValueError('Comparator optimizer/gradient missing')
                if step==1 and n.startswith('comparator.0.') and (norm!=0 or changed):raise ValueError('First-layer initial zero gradient violated')
                if step>=2 and (norm<=0 or not changed):raise ValueError('Comparator failed to learn after initial update')
            append_jsonl(self.directory/'comparator_progress.jsonl',dict(step=step,parameters=evidence,verified=True))
        if step==13:
            rows=[json.loads(x) for x in (self.directory/'progress.jsonl').read_text().splitlines()]
            if not self.config.get('disposable_profile'):
                prof=json.loads((self.directory.parent/('profile_'+self.config['arm'])/'profile.json').read_text())
                expected={(r['task'],r['cell']):r['seconds'] for r in prof['rows']}
                ratio=sum(r['seconds'] for r in rows)/sum(expected[r['task'],r['cell']] for r in rows)
                atomic_json(self.directory/'first_cycle_timing.json',dict(matched_profile_ratio=ratio,updates=13,
                    material_slowdown=ratio>1.35,production_seconds=sum(r['seconds'] for r in rows)))
        return row


def verify_progress(directory):
    directory=Path(directory)
    initial=torch.load(directory/'migration.pt',map_location='cpu')
    receipt=json.loads((directory/'latest_checkpoint.json').read_text())
    saved=load_verified(receipt);step=saved['state']['step']
    assert step>=1 and saved['scheduler']['updates']==10153+step
    assert saved['state']['episodes']==step*32
    assert not tree_equal(initial['optimizer'],saved['optimizer'])
    assert not tree_equal(initial['stream'],saved['stream'])
    n='spatial_gru.gates.weight';i=saved['optimizer_names'].index(n)
    assert float(saved['optimizer']['state'][i]['step'])==float(initial['optimizer']['state'][i]['step'])+step
    assert not torch.equal(initial['model'][n],saved['model'][n])
    new=saved['migration']['fresh_names']
    changed=[n for n in new if not torch.equal(initial['model'][n],saved['model'][n])]
    if new and step>=2:assert changed==new
    for n in new:
        i=saved['optimizer_names'].index(n)
        assert i not in initial['optimizer']['state'] and float(saved['optimizer']['state'][i]['step'])==step
    result=dict(verified=True,additional_step=step,convgru_step=6760+step,global_optimizer_step=10153+step,
        episodes=step*32,checkpoint=receipt,optimizer_changed=True,stream_advanced=True,new_changed=changed,
        optimizer_seconds=saved['state']['optimizer_seconds'])
    atomic_json(directory/'persisted_progress_verification.json',result)
    return result


def validate_budget(budget,config):
    if budget['deadline']!=budget['cap_started']+CAP or any(config.get(k)!=v for k,v in budget.items()):raise ValueError('Cap changed')
    if not budget['cap_started']<=time.time()<budget['deadline']:raise ValueError('Cap expired')


def profile(directory,arm):
    config=json.loads((directory/'profile_config.json').read_text())
    source,receipt=source_payload();session=Session(directory,config,migrate(source,receipt,arm))
    session.checkpoint('migration.pt');rows=[]
    for _ in range(13):
        if time.time()>=min(config['deadline'],config['cap_started']+1800)-60:raise RuntimeError('Bounded profile exhausted')
        rows.append(session.train_update(*session.scheduler.next()))
        if session.state['step'] in (1,2):session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
    session.checkpoint('profile_terminal.pt')
    verified=verify_progress(directory)
    atomic_json(directory/'profile.json',dict(complete=True,rows=rows,verification=verified,disposable=True,discard_all_state=True))


def baseline(directory):
    config=json.loads((directory/'profile_control'/'profile_config.json').read_text())
    source,receipt=source_payload();payload=migrate(source,receipt,'control')
    session=Session(directory/'baseline',config,payload)
    result=evaluate(session.model,'val',64,100,4,'mps',directory/'baseline_validation.json',config['deadline']-60)
    verify_coverage(result,64,100)


def run_arm(directory,arm):
    config=json.loads((directory/'config.json').read_text());source,receipt=source_payload()
    payload=migrate(source,receipt,arm)
    scope=dict(inherited.run.__globals__,validate_budget=validate_budget,require_approval=lambda *_:None,
        verify_parent=lambda _:(payload,receipt),migrate=lambda p,_:p,TrainingSession=Session,
        evaluate=evaluate,selection_key=selection_key,PROTOCOL=PROTOCOL)
    fn=types.FunctionType(inherited.run.__code__,scope,inherited.run.__name__,inherited.run.__defaults__,inherited.run.__closure__)
    result=fn(directory)
    path=directory/'report.json';report=json.loads(path.read_text())
    report.update(parent_step=6760,parent_episodes=216320,inherited_global_updates=10153,
        cumulative_convgru_step=6760+report['terminal_step'],arm=arm,
        inherited_optimizer_seconds=source['state']['optimizer_seconds'],
        limitations=['Single-parent single-run architecture-package pilot; not comparison-versus-capacity isolation or biological attention evidence.',
            'All learned parameters trainable; native tasks, cues, objectives, scheduling and inherited Adam unchanged.',
            'Same fresh paired final draws across arms; photo identities retain official source splits.',
            'Eight target cells prioritized prospectively; all other task regressions and empty/Krauzlis strata reported.'])
    atomic_json(path,report)
    verify_progress(directory)
    return result


def measured_plan(directory,source):
    """All 5070 previous updates/all completed eval cells; matched 13-update profiles.

    Per-cell historical mean * max(1, profile/historical matched ratio) per arm.
    1.25 train and 1.35 eval margins; 5 full validations, worst-case 4 full final
    tests, 900 seconds I/O/report overhead. Reduce equal 13-step cycles only here.
    """
    previous=SOURCE.parent
    rows=[json.loads(x) for x in (previous/'progress.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1691,6761))
    costs={(t,c):statistics.mean(r['seconds'] for r in rows if (r['task'],r['cell'])==(t,c)) for t,c in original.all_cells()}
    evaluations=[]
    for name in ('validation_002951.json','validation_004225.json','validation_005486.json','validation_006760.json','test_terminal.json','test_selected.json'):
        result=json.loads((previous/name).read_text())
        if result.get('reused_terminal'):continue
        verify_coverage(result,128 if name.startswith('test') else 64,200 if name.startswith('test') else 100)
        evaluations.extend(result['cells'])
    evcost={(t,c):sum(r['seconds'] for r in evaluations if (r['task'],r['cell'])==(t,c))/sum(r['n'] for r in evaluations if (r['task'],r['cell'])==(t,c)) for t,c in costs}
    factors={}
    for arm in ARMS:
        p=json.loads((directory/('profile_'+arm)/'profile.json').read_text())
        assert p['complete'] and len(p['rows'])==13
        factors[arm]=max(1.,sum(r['seconds'] for r in p['rows'])/sum(costs[r['task'],r['cell']] for r in p['rows']))
    val=1.35*max(factors.values())*sum(evcost[t,c]*(100 if t=='krauzlis_cued_motion' else 64) for t,c in costs)
    test=2*val;overhead=5*val+4*test+900
    budget=json.loads((directory/'budget.json').read_text());remaining=budget['deadline']-time.time()
    scheduler=BalancedScheduler(0);scheduler.load_state_dict(source['scheduler'])
    schedule=[scheduler.next() for _ in range(TARGET)]
    count=TARGET
    while count>=26:
        training=sum(costs[x] for x in schedule[:count])*sum(factors.values())
        total=1.25*training+overhead
        if total<remaining:break
        count-=13
    if count<26:raise RuntimeError('Insufficient two-arm acquisition budget')
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()}
    for t,c in schedule[:count]:
        exposure[t]['updates']+=1;exposure[t]['episodes']+=32;exposure[t]['cells'][c]+=32
    return dict(max_steps=count,additional_target_requested=TARGET,validation_steps=[(count//26)*13,count],
        planned_exposure=exposure,combined_optimizer_seconds=training,estimated_total_remaining_seconds=total,
        slower_150pct_seconds=1.5*training+overhead,estimated_validation_seconds=val,estimated_one_test_seconds=test,
        final_reserve_seconds=val+2*test+180,update_estimate_seconds=max(costs.values())*max(factors.values())*1.25,
        timing_factors=factors,timing_method=measured_plan.__doc__,prior_updates=len(rows),prior_eval_cells=len(evaluations),
        total_episodes_per_arm=count*32,target_convgru_step=6760+count)


def common_config(directory,budget,source,arm):
    old=source['state']['config'];old_root=Path(old['runtime_root'])
    hashes={str(ROOT/Path(p).relative_to(old_root)):sha for p,sha in old['source_hashes'].items()}
    for p in Path(__file__).parent.glob('*'):
        if p.suffix in ('.py','.md'):hashes[str(p)]=digest(p)
    return dict(**budget,arm=arm,protocol=PROTOCOL,device='mps',effective_batch=32,microbatch=4,eval_microbatch=4,
        val_n=64,val_krauzlis_n=100,test_n=128,test_krauzlis_n=200,checkpoint_every=13,cpu_threads=2,
        worker_lock=str(WORKER_LOCK),source_hashes=hashes,catalog=old['catalog'],source_manifest_sha256=old['source_manifest_sha256'],
        parent_pointer=str(SOURCE),validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,
        selection='target8 mean AUC then suite equal-task AUC; earlier ties; baseline/midpoint/terminal only',
        profile_state_inherited=False,all_trainable=True,lr=1e-4,clipping=None,bptt='full',precision='fp32',
        runtime_root=str(ROOT),report=str(directory/'REPORT.md'))


def launch(directory):
    guard()
    if (directory/'launch_attempt.json').exists():raise RuntimeError('Refusing repeat launch')
    label='org.vawm.spatial-comparison-v1-'+digest(directory/'cpu_preflight.json')[:12]
    target=f'gui/{os.getuid()}/'+label
    spec=launchd_spec(label,directory,sys.executable,ROOT)
    spec.update(ProcessType='Interactive',ProgramArguments=[sys.executable,'-u','-m','SecondPass.SpatialComparisonReadout.experiment','supervise',str(directory)])
    path=directory/'experiment.launchd.plist'
    with path.open('xb') as f:plistlib.dump(spec,f)
    atomic_json(directory/'launch_attempt.json',dict(label=label,target=target,utc=original.utc()))
    subprocess.run(['/bin/launchctl','bootstrap',f'gui/{os.getuid()}',str(path)],check=True)
    observed=subprocess.check_output(['/bin/launchctl','print',target],text=True)
    result=dict(label=label,target=target,external_owner='launchd',KeepAlive=False,observed=observed)
    atomic_json(directory/'launch_receipt.json',result)
    return result


def production_supervise(directory):
    guard();source,_=source_payload()
    started=time.time();budget=dict(cap_started=started,deadline=started+CAP,wall_cap_seconds=CAP,
        origin='First actual-launcher disposable profile; both sequential arms and all evaluation/reporting; no renewal')
    with (directory/'budget.json').open('x') as f:json.dump(budget,f,indent=2);f.flush();os.fsync(f.fileno())
    os.chmod(directory/'budget.json',0o444)
    atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),parent_pid=os.getppid(),
        started_utc=original.utc(started),deadline_utc=original.utc(budget['deadline']),**budget))
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    outcomes=[]
    def child(mode,path,arm=None):
        command=[sys.executable,'-u','-m','SecondPass.SpatialComparisonReadout.experiment',mode,str(path)]
        if arm:command+=['--arm',arm]
        result=supervise(command,budget['deadline'],path,prefix=mode)
        outcomes.append(dict(mode=mode,arm=arm,**result))
        if result['returncode']!=0:raise RuntimeError(f'{mode}/{arm} exited {result}')
    try:
        for arm in ARMS:
            path=directory/('profile_'+arm);path.mkdir()
            config=common_config(directory,budget,source,arm);config['disposable_profile']=True
            atomic_json(path/'profile_config.json',config)
            child('profile',path,arm)
        plan=measured_plan(directory,source)
        atomic_json(directory/'allocation.json',plan);os.chmod(directory/'allocation.json',0o444)
        for arm in ARMS:
            path=directory/arm;path.mkdir()
            config=common_config(directory,budget,source,arm);config.update(plan)
            atomic_json(path/'config.json',config);atomic_json(path/'budget.json',budget)
            os.chmod(path/'config.json',0o444)
        child('baseline',directory)
        for arm in ARMS:child('run',directory/arm,arm)
        # Reporting is also a supervised subprocess inside the same absolute cap.
        child('report',directory)
        result=dict(complete=True,outcomes=outcomes,finished_utc=original.utc(),deadline=budget['deadline'])
    except Exception as exc:
        result=dict(complete=False,error=repr(exc),outcomes=outcomes,finished_utc=original.utc(),deadline=budget['deadline'])
        atomic_json(directory/'incomplete_report.json',result)
        (directory/'REPORT.md').write_text('# Spatial comparison incomplete\n\n'+repr(exc)+'\nSee per-arm progress/checkpoints and supervisor receipts. No superiority conclusion.\n')
    finally:
        awake.terminate();awake.wait(timeout=5)
    atomic_json(directory/'supervisor_result.json',result)
    return result


def main():
    original.cpu_setup()
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','supervise','profile','baseline','run','report','verify']);p.add_argument('directory');p.add_argument('--arm',choices=ARMS)
    args=p.parse_args();directory=Path(args.directory).resolve()
    if args.mode=='launch':print(json.dumps(launch(directory),indent=2));return
    if args.mode=='supervise':
        result=production_supervise(directory);print(json.dumps(result));raise SystemExit(0 if result['complete'] else 2)
    if args.mode=='verify':print(json.dumps(verify_progress(directory),indent=2));return
    if args.mode=='report':
        from .report import report
        report(directory);return
    guard()
    with WORKER_LOCK.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile':profile(directory,args.arm)
        elif args.mode=='baseline':baseline(directory)
        else:raise SystemExit(run_arm(directory,args.arm))


if __name__=='__main__':main()
