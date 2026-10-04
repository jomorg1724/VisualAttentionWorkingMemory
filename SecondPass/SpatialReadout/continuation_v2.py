"""Authorized unchanged ConvGRU continuation: 5070 added updates, one 24h cap.

Reuse the frozen optimizer/evaluator/supervisor; only version allocation, final
namespace, and initialization. No module globals or predecessor bytes change.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import plistlib
import statistics
import subprocess
import sys
import time
import types

import torch
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, tree_equal
from SecondPass.JointTraining.worker import cpu_setup, verify_sources, utc
from SecondPass.TaskSuite.suite import TASKS, SuiteStream
from . import worker
from .protocol import digest, assert_no_worker, launchd_spec, supervise
from .state import load_verified, map_adam

ROOT=Path(__file__).resolve().parents[2]
SOURCE_DIR=Path('/Users/jonathanmorgan/VAWMRuntime/final_convgru_01/run_qos_repair')
SOURCE_SHA='91bf029acafaf1a5e67fa3bf7ff659234deba39356c92f0cb36b0b05d91c02b5'
PROTOCOL='unchanged_final_convgru_continuation_v2_5070'
FINAL_NAMESPACE=94792763
START=1690
ADDITIONAL=5070
TARGET=6760
CAP=86400.
# Exhaustive policy migration allowlist. All other config fields remain exact.
ALLOWED_CONFIG=set('cap_started deadline wall_cap_seconds origin max_steps planned_cycles validation_steps estimated_validation_seconds estimated_one_test_seconds final_reserve_seconds update_estimate_seconds estimated_optimizer_seconds projected_optimizer_hours estimated_total_remaining_seconds slower_150pct_training_seconds planned_exposure timing_estimator cell_costs additional_target_requested protocol selection final_test_seed_namespace source_hashes budget_sha256 pinned_utc continuation'.split())


def new_budget(started):
    return dict(cap_started=started,deadline=started+CAP,wall_cap_seconds=CAP,
                origin='Authorized 5070-update continuation activation; includes migration, training, evaluation and reporting; never renew')


def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    if budget!=new_budget(budget['cap_started']) or any(config.get(k)!=v for k,v in budget.items()):
        raise ValueError('Changed/renewed continuation cap')
    if not budget['cap_started']<=now<budget['deadline']:raise ValueError('Expired/future continuation cap')


def activate_once(directory,started):
    directory=Path(directory)
    with (directory/'budget.json').open('x') as f:
        json.dump(new_budget(started),f,indent=2);f.flush();os.fsync(f.fileno())
    os.chmod(directory/'budget.json',0o444)


def guard():
    result=assert_no_worker()
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        argv=fields[1].split()
        if argv and Path(argv[0]).name.lower().startswith('python') and '-m' in argv:
            i=argv.index('-m')
            if len(argv)>i+2 and argv[i+1] in ('SecondPass.SpatialReadout.qos_repair','SecondPass.SpatialReadout.continuation_v2') and argv[i+2]=='run':
                raise RuntimeError('Other spatial worker: '+line)
    return result


def verify_coverage(result,n,krauzlis_n):
    expected={(t,c['id']) for t,s in TASKS.items() for c in s['conditions']}
    rows=result['cells']
    if not result['complete'] or len(rows)!=35 or {(r['task'],r['cell']) for r in rows}!=expected:
        raise ValueError('Incomplete predecessor coverage')
    for row in rows:
        if row['n']!=(krauzlis_n if row['task']=='krauzlis_cued_motion' else n):
            raise ValueError('Predecessor denominator mismatch')
        if sum(map(sum,row['confusion']))!=row['n']:raise ValueError('Invalid confusion denominator')


def verify_predecessor(directory):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    outcome=json.loads((directory/'production_supervisor_result.json').read_text())
    receipt=report['terminal_checkpoint']
    if outcome['returncode']!=0 or outcome['hard_cap_triggered'] or report['failure'] is not None or report['stop_reason']!='planned_complete_cycles':
        raise ValueError('Predecessor did not complete normally')
    for pid in (outcome['supervisor_pid'],outcome['worker_pid']):
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('Predecessor PID still present')
    if receipt['sha256']!=SOURCE_SHA or Path(receipt['path'])!=directory/'terminal.pt':
        raise ValueError('Not authorized exact terminal')
    source=load_verified(receipt);s=source['state']
    if source['schema']!=2 or s['step']!=START or s['episodes']!=START*32 or source['scheduler']['updates']!=3393+START or source['scheduler']['tasks']:
        raise ValueError('Source exposure/scheduler mismatch')
    if report['terminal_step']!=START or report['selected_step']!=START or not report['final_coverage_complete']:
        raise ValueError('Source report/selection mismatch')
    if source['rng']['mps'] is None:raise ValueError('Source MPS RNG missing')
    verify_sources(s['config'])
    SuiteStream('train').load_state_dict(source['stream'])
    from .model import SpatialReadout
    from SecondPass.TaskSuite.suite import task_classes
    with torch.random.fork_rng(devices=[]):model=SpatialReadout(task_classes())
    names=[n for n,_ in model.named_parameters()]
    if names!=source['optimizer_names']:raise ValueError('Adam names mismatch')
    model.load_state_dict(source['model'])
    remapped=map_adam(source['optimizer'],names,source['model'],names,model.state_dict())
    if not tree_equal(remapped,source['optimizer']):raise ValueError('Adam mapping mismatch')
    validation=json.loads((directory/'validation_001690.json').read_text())
    verify_coverage(validation,64,100)
    if s['best_step']!=START or s['best_key']!=worker.selection_key(validation) or len(s['selection_history'])!=2:
        raise ValueError('Baseline selection mismatch')
    winner=torch.load(s['best_checkpoint'],map_location='cpu')
    if not tree_equal(winner['model'],source['model']):raise ValueError('Baseline winner mismatch')
    terminal=json.loads((directory/'test_terminal.json').read_text())
    selected=json.loads((directory/'test_selected.json').read_text())
    verify_coverage(terminal,128,200)
    verify_coverage(selected.get('results',selected),128,200)
    return source,receipt


def measured_plan(directory,source,remaining=CAP):
    """Prespecified means of ALL repaired production rows, not recent fast rows.

    Per-cell evaluation seconds/episode pool both completed validation looks and
    the unique completed terminal test (selected was deduplicated). Use exact
    inherited future queues, 1.25x train, 1.35x eval, 900s startup/I/O/report.
    """
    directory=Path(directory)
    updates=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]
    if [r['step'] for r in updates]!=list(range(125,1691)):raise ValueError('Repaired progress not contiguous/full')
    evaluations=[]
    for name in ('validation_000845.json','validation_001690.json','test_terminal.json'):
        result=json.loads((directory/name).read_text());verify_coverage(result,128 if name.startswith('test') else 64,200 if name.startswith('test') else 100)
        evaluations.extend(result['cells'])
    costs=[]
    for task,spec in TASKS.items():
        for cell in spec['conditions']:
            key=(task,cell['id'])
            train=[r['seconds'] for r in updates if (r['task'],r['cell'])==key]
            ev=[r for r in evaluations if (r['task'],r['cell'])==key]
            if not train or len(ev)!=3:raise ValueError('Missing measured cell')
            cost=dict(task=task,cell=key[1],seconds=statistics.mean(train),measured_updates=len(train),
                      eval_seconds_per_episode=sum(r['seconds'] for r in ev)/sum(r['n'] for r in ev))
            if not all(math.isfinite(cost[k]) and cost[k]>0 for k in ('seconds','eval_seconds_per_episode')):raise ValueError('Invalid timing')
            costs.append(cost)
    lookup={(r['task'],r['cell']):r['seconds'] for r in costs}
    val=1.35*sum(r['eval_seconds_per_episode']*(100 if r['task']=='krauzlis_cued_motion' else 64) for r in costs)
    test=2*val
    scheduler=BalancedScheduler(0);scheduler.load_state_dict(source['scheduler'])
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()}
    training=0.
    for _ in range(ADDITIONAL):
        t,cell=scheduler.next();training+=lookup[t,cell]
        exposure[t]['updates']+=1;exposure[t]['episodes']+=32;exposure[t]['cells'][cell]+=32
    overhead=4*val+2*test+900.
    total=1.25*training+overhead
    if total>remaining:raise ValueError(f'Fixed 5070 updates do not fit: {total} > {remaining}; do not reduce or renew')
    return dict(max_steps=TARGET,additional_target_requested=ADDITIONAL,planned_cycles=390,
        validation_steps=[START+(390*i//4)*13 for i in range(1,5)],
        estimated_validation_seconds=val,estimated_one_test_seconds=test,
        final_reserve_seconds=val+2*test+180.,update_estimate_seconds=max(lookup.values())*1.25,
        estimated_optimizer_seconds=training,projected_optimizer_hours=training/3600.,
        estimated_total_remaining_seconds=total,slower_150pct_training_seconds=1.5*training+overhead,
        planned_exposure=exposure,cell_costs=costs,
        timing_estimator=dict(method=measured_plan.__doc__,train_margin=1.25,eval_margin=1.35,overhead_seconds=900.,
                              production_rows=len(updates),evaluation_unique_cells=len(evaluations)))


def build_config(source,plan,started,source_hashes=None):
    config=copy.deepcopy(source['state']['config'])
    config.update(plan);config.update(new_budget(started))
    config.update(protocol=PROTOCOL,final_test_seed_namespace=FINAL_NAMESPACE,
        selection='maximum lexicographic (equal-task mean AUC, mean chance-normalized BA), earlier ties; inherited validation1690 fallback; four scheduled complete looks only',
        pinned_utc=utc(started),continuation=dict(source_sha256=SOURCE_SHA,source_step=START,additional_updates=ADDITIONAL,
            target_cumulative_convgru_updates=TARGET,source_optimizer_seconds=source['state']['optimizer_seconds'],
            baseline_validation=str(SOURCE_DIR/'validation_001690.json'),
            early_cap_policy='No opportunistic extra look; preserve complete prior winner; scheduled looks only',
            old_config_is_historical=True))
    if source_hashes is not None:config['source_hashes']=source_hashes
    return config


def migrate_continuation(source,config):
    old=source['state']['config']
    changed={k for k in set(old)|set(config) if not tree_equal(old.get(k),config.get(k))}
    if changed-ALLOWED_CONFIG:raise ValueError('Unauthorized config migration: '+repr(changed-ALLOWED_CONFIG))
    if config['max_steps']!=TARGET or config['additional_target_requested']!=ADDITIONAL or config['final_test_seed_namespace']!=FINAL_NAMESPACE or config['validation_steps']!=[START+(390*i//4)*13 for i in range(1,5)]:
        raise ValueError('Changed immutable target/cadence/namespace')
    if any(config.get(k)!=v for k,v in new_budget(config['cap_started']).items()):raise ValueError('Invalid cap migration')
    result=copy.deepcopy(source)
    result['state']['config']=copy.deepcopy(config);result['state']['deadline']=config['deadline']
    return result


class FinalTestStream(SuiteStream):
    def __init__(self,split='test'):
        if split!='test':raise ValueError('Final-only stream')
        super().__init__(split)
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return FINAL_NAMESPACE*100000+TASKS[task]['stream_id']*1000+index


def evaluate(*args,**kwargs):
    scope=dict(worker.evaluate.__globals__,FinalTestStream=FinalTestStream,FINAL_TEST_NAMESPACE=FINAL_NAMESPACE)
    fn=types.FunctionType(worker.evaluate.__code__,scope,worker.evaluate.__name__,worker.evaluate.__defaults__,worker.evaluate.__closure__)
    return fn(*args,**kwargs)


def prepare(directory):
    directory=Path(directory).resolve()
    if directory.exists():raise FileExistsError('Continuation directory exists')
    guard();source,receipt=verify_predecessor(SOURCE_DIR)
    plan=measured_plan(SOURCE_DIR,source)
    hashes=dict(source['state']['config']['source_hashes'])
    for name in ('continuation_v2.py','test_continuation_v2.py','qos_repair.py','CONTINUATION_BRIEF.md'):
        path=Path(__file__).with_name(name);hashes[str(path)]=digest(path)
    if Path(source['state']['config']['runtime_root'])!=ROOT:raise ValueError('Prepare from verified runtime copy')
    directory.mkdir(parents=True)
    for path,sha in hashes.items():
        if digest(path)!=sha:raise ValueError('Frozen source changed')
        dst=directory/'locked_source'/Path(path).relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(path).read_bytes())
    timing_paths=[SOURCE_DIR/x for x in ('progress.jsonl','validation_000845.json','validation_001690.json','test_terminal.json','test_selected.json','production_supervisor_result.json','report.json')]
    result=dict(protocol=PROTOCOL,source=receipt,source_hashes=hashes,plan=plan,
        evidence_hashes={str(p):digest(p) for p in timing_paths},prepared_utc=utc(),runtime_root=str(ROOT),
        production_started=False,cap_not_started=True)
    atomic_json(directory/'preparation.json',result);os.chmod(directory/'preparation.json',0o444)
    return dict(directory=str(directory),optimizer_hours=plan['projected_optimizer_hours'],total_hours=plan['estimated_total_remaining_seconds']/3600,
                slower_150pct_hours=plan['slower_150pct_training_seconds']/3600,validation_steps=plan['validation_steps'])


def check_preparation(directory):
    prep=json.loads((Path(directory)/'preparation.json').read_text())
    for path,sha in {**prep['source_hashes'],**prep['evidence_hashes']}.items():
        if digest(path)!=sha:raise ValueError('Pinned source/evidence changed: '+path)
    source,receipt=verify_predecessor(SOURCE_DIR)
    if receipt!=prep['source'] or measured_plan(SOURCE_DIR,source)!=prep['plan']:raise ValueError('Source/budget changed')
    return prep,source


def checked_config(directory):
    directory=Path(directory);prep,source=check_preparation(directory)
    budget=json.loads((directory/'budget.json').read_text());config=json.loads((directory/'config.json').read_text())
    validate_budget(budget,config)
    expected=build_config(source,prep['plan'],budget['cap_started'],prep['source_hashes'])
    expected['budget_sha256']=digest(directory/'budget.json')
    if expected!=config:raise ValueError('Pinned config changed')
    verify_sources(config)
    if (directory/'migration.pt').exists():raise RuntimeError('No automatic restart')
    if time.time()+config['estimated_total_remaining_seconds']>config['deadline']:raise ValueError('Fixed exposure no longer fits cap')
    return config,source,prep['source']


def execute_worker(directory,config,payload,receipt,session_class=None,evaluator=None):
    """Reuse exact loop. Carried two-look history disables legacy extra early look.

    All four explicit continuation looks execute via validation_steps; history
    remains complete and baseline1690 remains eligible. No len(history) tricks.
    """
    class Session(worker.TrainingSession):
        def train_update(self,task,cell):
            row=super().train_update(task,cell)
            if self.state['step']==START+1:self.checkpoint('first_resumed_update.pt')
            return row
    scope=dict(worker.run.__globals__,validate_budget=validate_budget,
        require_approval=lambda directory,config:None,
        verify_parent=lambda pointer:(payload,receipt),migrate=lambda source,parent_receipt:source,
        TrainingSession=session_class or Session,evaluate=evaluator or evaluate,PROTOCOL=PROTOCOL)
    fn=types.FunctionType(worker.run.__code__,scope,worker.run.__name__,worker.run.__defaults__,worker.run.__closure__)
    return fn(directory)


def run(directory):
    config,source,receipt=checked_config(directory);guard()
    with Path(config['worker_lock']).open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        payload=migrate_continuation(source,config)
        result=execute_worker(directory,config,payload,receipt)
    # Historical architecture-origin fields stay in checkpoints; report this run.
    path=Path(directory)/'report.json';report=json.loads(path.read_text())
    report.update(continuation=config['continuation'],additional_completed_updates=report['terminal_step']-START,
                  cumulative_convgru_updates=report['terminal_step'],
                  limitations=['Unchanged additional acquisition, not architectural superiority or convergence evidence.',
                    'Validation draws reused for selection; fresh final seeds retain official source identities.',
                    'All task/cell curves retained; N0 specificity and Krauzlis event denominators separate.'])
    atomic_json(path,report)
    return result


def continuation_spec(label,directory,python,repo):
    spec=launchd_spec(label,directory,python,repo)
    spec.update(ProcessType='Interactive',ProgramArguments=[str(python),'-u','-m','SecondPass.SpatialReadout.continuation_v2','supervise',str(directory)])
    return spec


def launch(directory):
    directory=Path(directory).resolve();prep,source=check_preparation(directory);guard()
    if (directory/'budget.json').exists():raise RuntimeError('Activation already occurred')
    label='org.vawm.spatialreadout.continuation-v2-'+digest(directory/'preparation.json')[:12]
    target=f'gui/{os.getuid()}/'+label
    spec=continuation_spec(label,directory,sys.executable,ROOT)
    with (directory/'continuation.launchd.plist').open('xb') as f:plistlib.dump(spec,f);f.flush();os.fsync(f.fileno())
    with (directory/'launch_attempt.json').open('x') as f:
        json.dump(dict(label=label,job_handle=target,epoch=time.time(),preparation_sha256=digest(directory/'preparation.json')),f);f.flush();os.fsync(f.fileno())
    subprocess.run(['/bin/launchctl','bootstrap',f'gui/{os.getuid()}',str(directory/'continuation.launchd.plist')],check=True)
    observed=subprocess.run(['/bin/launchctl','print',target],capture_output=True,text=True,check=True)
    receipt=dict(job_handle=target,launchctl=observed.stdout,external_owner='launchd',KeepAlive=False,restarts_authorized=False)
    atomic_json(directory/'launch_receipt.json',receipt)
    return receipt


def production_supervise(directory):
    directory=Path(directory);prep,source=check_preparation(directory);guard()
    attempt=json.loads((directory/'launch_attempt.json').read_text())
    if attempt['preparation_sha256']!=digest(directory/'preparation.json'):raise ValueError('Launch preparation changed')
    # CPU verification precedes activation; durable deadline precedes all MPS.
    started=time.time();activate_once(directory,started)
    config=build_config(source,prep['plan'],started,prep['source_hashes']);config['budget_sha256']=digest(directory/'budget.json')
    atomic_json(directory/'config.json',config);os.chmod(directory/'config.json',0o444)
    atomic_json(directory/'production_activated.json',dict(supervisor_pid=os.getpid(),parent_pid=os.getppid(),
        started_utc=utc(started),deadline_utc=utc(config['deadline']),deadline=config['deadline'],config_sha256=digest(directory/'config.json')))
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:result=supervise([sys.executable,'-u','-m','SecondPass.SpatialReadout.continuation_v2','run',str(directory)],config['deadline'],directory)
    finally:awake.terminate();awake.wait(timeout=5)
    if not (directory/'report.json').exists():
        atomic_json(directory/'report.json',dict(protocol=PROTOCOL,stop_reason='worker_exit_without_report',supervisor=result,final_coverage_complete=False))
    return result


def main():
    cpu_setup()
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','supervise','run']);parser.add_argument('directory')
    args=parser.parse_args();directory=Path(args.directory).resolve()
    if args.mode=='run':raise SystemExit(run(directory))
    result={'prepare':prepare,'launch':launch,'supervise':production_supervise}[args.mode](directory)
    print(json.dumps(result,indent=2),flush=True)
    if args.mode=='supervise':raise SystemExit(result['returncode'])


if __name__=='__main__':main()
