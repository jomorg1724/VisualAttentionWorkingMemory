"""Pinned protocol, conservative all-cell planner, launchd and wall guards."""
import hashlib
import os
from pathlib import Path

import signal
import subprocess
import time
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json
from SecondPass.JointTraining.continuation_v3 import selection_key
from SecondPass.TaskSuite.suite import TASKS, SuiteStream

PROTOCOL='kda_final_convgru_warm_start_v1'
FINAL_TEST_NAMESPACE=94692763
TRAIN_MARGIN=1.25
EVAL_MARGIN=1.35
OVERHEAD_SECONDS=900.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_budget(started):
    return dict(cap_started=started,deadline=started+28800.,wall_cap_seconds=28800,
        origin='First disposable MPS profile; includes profiling/review/production/evaluation/reporting; never renew')


def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    if budget!=new_budget(budget['cap_started']) or any(config.get(k)!=v for k,v in budget.items()):
        raise ValueError('Changed or renewed wall budget')
    if not budget['cap_started']<=now<budget['deadline']: raise ValueError('Expired/future wall budget')


def plan(rows,scheduler_state,remaining):
    expected={(t,c['id']) for t,s in TASKS.items() for c in s['conditions']}
    if len(rows)!=35 or {(r['task'],r['cell']) for r in rows}!=expected:
        raise ValueError('Require exactly all 35 distinct measured cells')
    costs={(r['task'],r['cell']):r['seconds'] for r in rows}
    val=EVAL_MARGIN*sum(r['eval_seconds']/r['eval_n']*(100 if r['task']=='krauzlis_cued_motion' else 64) for r in rows)
    test=2*val
    overhead=2*val+2*test+OVERHEAD_SECONDS
    scheduler=BalancedScheduler(0); scheduler.load_state_dict(scheduler_state)
    if scheduler.tasks: raise ValueError('Requires carried complete-cycle boundary')
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()}
    training=0.; cycles=0
    for _ in range(165):
        future=[scheduler.next() for _ in range(13)]
        candidate=training+sum(costs[x] for x in future)
        if candidate*TRAIN_MARGIN+overhead>remaining: break
        training=candidate; cycles+=1
        for t,c in future:
            exposure[t]['updates']+=1; exposure[t]['episodes']+=32; exposure[t]['cells'][c]+=32
    if cycles<4: raise ValueError('Insufficient useful acquisition with required evaluations')
    return dict(max_steps=cycles*13,planned_cycles=cycles,effective_batch=32,microbatch=4,eval_microbatch=4,
        validation_steps=[(cycles//2)*13,cycles*13],val_n=64,val_krauzlis_n=100,test_n=128,test_krauzlis_n=200,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,
        final_reserve_seconds=val+2*test+180,update_estimate_seconds=max(costs.values())*TRAIN_MARGIN,
        estimated_optimizer_seconds=training,projected_optimizer_hours=training/3600,
        estimated_total_remaining_seconds=TRAIN_MARGIN*training+overhead,
        slower_150pct_training_seconds=1.5*training+overhead,planned_exposure=exposure,checkpoint_every=13,
        timing_estimator=dict(train_margin=TRAIN_MARGIN,eval_margin=EVAL_MARGIN,overhead_seconds=OVERHEAD_SECONDS,
            method='One effective batch32 update and 8-episode eval per cell, exact inherited future scheduler; 2 complete validations + worst-case 2 final tests',
            includes_optional_baseline=False),cell_costs=rows)


class FinalTestStream(SuiteStream):
    def __init__(self,split='test'):
        if split!='test': raise ValueError('Final-only namespace')
        super().__init__('test')

    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return FINAL_TEST_NAMESPACE*100000+TASKS[task]['stream_id']*1000+index


def is_worker(command):
    # ps emits argv display text, not shell syntax: embedded unbalanced quotes
    # in unrelated program arguments must not crash the worker guard.
    tokens=command.split()
    if not tokens or not Path(tokens[0]).name.lower().startswith('python'): return False
    # Refuse all repository compute-worker modules (profile included), not the
    # caffeinate/shell wrappers which merely repeat their child's argv.
    if '-m' in tokens:
        i=tokens.index('-m'); module=tokens[i+1] if len(tokens)>i+1 else ''
        role=tokens[i+2] if len(tokens)>i+2 else ''
        if module=='SecondPass.SpatialReadout.worker': return True
        if module.startswith('SecondPass.JointTraining.') and role in ('run','worker','profile'): return True
        if module.startswith(('WorkingMemory.','PreAttentiveVision.')): return True
    return any('VisualAttentionWorkingMemory/' in token and token.endswith('.py') and
               any(x in token.lower() for x in ('train','worker','profile')) for token in tokens[1:])


def assert_no_worker(exclude=None):
    exclude=set(exclude or ())|{os.getpid()}
    observed=[]
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,lstart=,command='],text=True).splitlines():
        fields=line.strip().split(None,7)
        if len(fields)==8 and int(fields[0]) not in exclude and is_worker(fields[7]): observed.append(line.strip())
    if observed: raise RuntimeError('Other compute worker: '+repr(observed))
    return dict(checked_epoch=time.time(),conflicting_workers=[],method='Executable-aware ps scan plus shared advisory worker lock')


def identity(pid):
    return subprocess.check_output(['ps','-p',str(pid),'-o','pid=,ppid=,lstart=,command='],text=True).strip()


def supervise(command,deadline,directory,prefix='production'):
    directory=Path(directory)
    if time.time()>=deadline: raise ValueError('Expired supervisor deadline')
    child=subprocess.Popen(command,start_new_session=True)
    atomic_json(directory/(prefix+'_supervisor.json'),dict(pid=os.getpid(),worker_pid=child.pid,
        supervisor_identity=identity(os.getpid()),worker_identity=identity(child.pid),deadline=deadline,command=command))
    hard=False
    try:
        child.wait(timeout=max(0.,deadline-time.time()))
    except subprocess.TimeoutExpired:
        hard=True; os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=5)
    result=dict(returncode=child.returncode,hard_cap_triggered=hard,finished=time.time(),
        supervisor_pid=os.getpid(),worker_pid=child.pid,deadline=deadline)
    atomic_json(directory/(prefix+'_supervisor_result.json'),result)
    return result


def launchd_spec(label,directory,python,repo):
    directory=Path(directory)
    return dict(Label=label,ProgramArguments=[str(python),'-u','-m','SecondPass.SpatialReadout.launch','supervise',str(directory)],
        WorkingDirectory=str(repo),RunAtLoad=True,KeepAlive=False,ProcessType='Background',
        EnvironmentVariables={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','PYTHONUNBUFFERED':'1'},
        StandardOutPath=str(directory/'launchd.stdout.log'),StandardErrorPath=str(directory/'launchd.stderr.log'))


def require_approval(directory,config):
    import json
    directory=Path(directory)
    approval=json.loads((directory/'REVIEW_APPROVED.json').read_text())
    if approval.get('approved') is not True or not approval.get('reviewer') or approval.get('config_sha256')!=digest(directory/'config.json') or approval.get('source_hashes')!=config['source_hashes']:
        raise ValueError('Independent review does not approve exact config/sources')
    return approval
