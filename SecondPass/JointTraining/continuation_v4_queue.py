"""Read-only predecessor gate and finite CPU queue; no accelerator work while waiting."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .core import atomic_json, tree_equal, BalancedScheduler
from SecondPass.TaskSuite.suite import TASKS

CLEANUP_GRACE = 300.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def identity(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True,check=False)
    if result.returncode not in (0,1):raise ValueError('OS process lookup failed')
    return result.stdout.strip() or None


def validate_identity(value,role,source):
    if value is None:return
    # ps lstart contributes exactly five fields; inspect executable + interpreter argv.
    tokens=value.split()[5:]
    if not tokens or not Path(tokens[0]).name.lower().startswith('python'):
        raise ValueError('Predecessor executable identity mismatch')
    i=1
    while i<len(tokens) and tokens[i] in {'-u','-B','-O','-OO','-I','-E','-s','-S','-b','-bb','-q'}:i+=1
    if tokens[i:i+3]!=['-m','SecondPass.JointTraining.continuation_v3',role] or len(tokens)!=i+4 or Path(tokens[-1]).resolve()!=Path(source).resolve():
        raise ValueError('Predecessor module/role/path identity mismatch')


def verify_hashes(hashes):
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('Frozen source digest changed: '+path)


def verify_cells(data,n,k,namespace=None):
    expected={(t,c['id']) for t,s in TASKS.items() for c in s['conditions']}
    cells=data.get('cells',[])
    if not data.get('complete') or data.get('complete_cells')!=35 or len(cells)!=35 or {(r['task'],r['cell']) for r in cells}!=expected:
        raise ValueError('Incomplete or duplicate 35-cell coverage')
    if namespace is not None and data.get('seed_namespace')!=namespace:raise ValueError('Wrong final namespace')
    for r in cells:
        if r['n']!=(k if r['task']=='krauzlis_cued_motion' else n) or sum(sum(row) for row in r['confusion'])!=r['n']:
            raise ValueError('Evaluation precision/confusion denominator mismatch')


def verify_predecessor(directory):
    """Full normal completion, exact terminal lineage and OS exit, never EOF."""
    import torch
    from . import continuation_v3 as v3
    directory=Path(directory).resolve()
    sup=load(directory/'supervisor.json')
    if any(identity(sup[k]) is not None for k in ('pid','worker_pid')):
        raise ValueError('Predecessor supervisor or worker still alive')
    result=load(directory/'supervisor_result.json'); cfg=load(directory/'config.json'); budget=load(directory/'budget.json')
    if result.get('returncode')!=0 or result.get('hard_cap_triggered') is not False:
        raise ValueError('Predecessor did not exit normally')
    if result['supervisor_pid']!=sup['pid'] or result['worker_pid']!=sup['worker_pid'] or not (result['deadline']==sup['deadline']==budget['deadline']==cfg['deadline']):
        raise ValueError('Predecessor process/deadline receipt mismatch')
    report=load(directory/'report.json')
    if report.get('protocol')!=v3.PROTOCOL or report.get('stop_reason')!='planned_complete_cycles' or report.get('failure') is not None or report.get('terminal_step')!=2860 or report.get('episodes')!=91520 or not report.get('final_coverage_complete'):
        raise ValueError('Incomplete or failed predecessor report')
    if any((directory/f).exists() for f in ('failure.json','finalization_failure.json','guardian_hard_stop.json')):
        raise ValueError('Predecessor failure receipt present')
    verify_hashes({**cfg['source_hashes'],**cfg.get('continuation_source_hashes',{})})
    source=report['terminal_checkpoint']
    if Path(source['path']).resolve()!=directory/'terminal.pt' or source['step']!=2860 or not source['verified'] or digest(directory/'terminal.pt')!=source['sha256'] or (directory/'terminal.pt').stat().st_size!=source['bytes']:
        raise ValueError('Terminal checkpoint receipt/digest mismatch')
    saved=torch.load(directory/'terminal.pt',map_location='cpu'); state=saved['state']
    if saved['schema']!=1 or state['step']!=2860 or state['episodes']!=91520 or saved['scheduler']['updates']!=2860 or state['config']!=cfg:
        raise ValueError('Terminal checkpoint state/config mismatch')
    for t,e in state['exposure'].items():
        if e['updates']!=220 or e['episodes']!=7040 or sum(e['cells'].values())!=7040:raise ValueError('Source task exposure mismatch')
    baseline=load(directory/'validation_002860.json'); verify_cells(baseline,64,100)
    if v3.selection_key(baseline) is None:raise ValueError('Baseline metrics incomplete')
    validated=torch.load(directory/'validation_checkpoint_002860.pt',map_location='cpu')
    if validated['state']['step']!=2860 or not all(tree_equal(saved[key],validated[key]) for key in ('model','optimizer','scheduler','stream','rng')):
        raise ValueError('Terminal differs from completed validation checkpoint')
    if not any(r['step']==2860 and r['complete'] for r in validated['state']['selection_history']):raise ValueError('Terminal validation selection record missing')
    terminal=load(directory/'test_terminal.json'); selected=load(directory/'test_selected.json')
    verify_cells(terminal,128,200,v3.FINAL_TEST_NAMESPACE)
    if selected.get('reused_terminal'):
        if not selected.get('identical_model_verified') or selected.get('selected_step')!=2860 or report['selected_step']!=2860 or selected.get('results')!=terminal:
            raise ValueError('Unverified selected/terminal deduplication')
        selected=selected['results']
    verify_cells(selected,128,200,v3.FINAL_TEST_NAMESPACE)
    progress=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]
    if [r['step'] for r in progress]!=list(range(716,2861)) or any(r['cumulative_episodes']!=r['step']*32 for r in progress):
        raise ValueError('Predecessor progress not contiguous 716..2860')
    paths=['config.json','budget.json','supervisor.json','supervisor_result.json','report.json','validation_002860.json','validation_checkpoint_002860.pt','test_terminal.json','test_selected.json','progress.jsonl']
    return dict(source_checkpoint=source,verified_epoch=time.time(),normal_completion=True,os_exit_verified=True,
        prior_deadline=cfg['deadline'],artifact_sha256={name:digest(directory/name) for name in paths},
        prior_completed_selection_history=validated['state']['selection_history'],prior_completed_best_step=validated['state']['best_step'])


def queue_decision(*,now,expiry,alive,result):
    if now>=expiry:raise ValueError('Bounded predecessor queue expired')
    if result is not None and (result.get('returncode')!=0 or result.get('hard_cap_triggered') is not False):raise ValueError('Predecessor failed; no activation')
    if alive:return 'waiting'
    if result is None:raise ValueError('Predecessor exited without normal-completion receipt')
    return 'verify'


def no_other_worker():
    from .continuation_v4 import is_joint_worker
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)==2 and int(fields[0])!=os.getpid() and is_joint_worker(fields[1]):
            raise ValueError('Another local joint worker exists: '+fields[0])


def namespace_unused(source):
    from .continuation_v4 import FINAL_TEST_NAMESPACE
    checked=[]
    for directory in sorted(Path(source).parent.glob('fresh_kda_joint_01*')):
        if directory.name=='fresh_kda_joint_01_continuation_v4_8h':continue
        for path in sorted(directory.glob('*.json')):
            # No metric is used: search only namespace-bearing fields recursively.
            data=load(path)
            def used(obj):
                if isinstance(obj,dict):return any((k in ('seed_namespace','final_test_seed_namespace') and v==FINAL_TEST_NAMESPACE) or used(v) for k,v in obj.items())
                if isinstance(obj,list):return any(used(v) for v in obj)
                return False
            if used(data):raise ValueError('Final namespace already used: '+str(path))
            checked.append(str(path))
    return dict(namespace=FINAL_TEST_NAMESPACE,unused=True,checked_paths=checked)


def timing_plan(source,saved,project=False):
    from .continuation_v4 import plan_continuation
    source=Path(source); state=copy.deepcopy(saved['state']); scheduler=BalancedScheduler(0);scheduler.load_state_dict(saved['scheduler'])
    if project:
        while state['step']<2860:
            t,c=scheduler.next();state['step']+=1;state['episodes']+=32
            state['exposure'][t]['updates']+=1;state['exposure'][t]['episodes']+=32;state['exposure'][t]['cells'][c]+=32
    progress=[json.loads(line) for line in (source/'progress.jsonl').read_text().splitlines()]
    paths=[p for p in sorted(source.glob('validation_[0-9]*.json')) if not p.name.endswith('.partial.json')]
    validations=[load(p) for p in paths]
    for data in validations:verify_cells(data,64,100)
    if not validations:raise ValueError('No complete v3 validation timing')
    evaluation_paths=list(paths)
    for name in ('test_terminal.json','test_selected.json'):
        path=source/name
        if path.exists():
            data=load(path)
            if data.get('reused_terminal'):continue  # one timing measurement, not two
            verify_cells(data,128,200)
            validations.append(data);evaluation_paths.append(path)
    # Match v3's empirical-mean estimator, pooling every complete look per cell.
    # The unchanged 15% optimizer margin and 600s overhead remain explicit.
    import statistics
    basis=copy.deepcopy(load(paths[-1]))
    maximum=copy.deepcopy(basis)
    for rows,reducer in ((basis['cells'],statistics.mean),(maximum['cells'],max)):
        for row in rows:
            row['seconds']=reducer(r['seconds']/r['n']*row['n'] for v in validations for r in v['cells'] if (r['task'],r['cell'])==(row['task'],row['cell']))
    plan=plan_continuation(state['config'],state,scheduler.state_dict(),progress,basis)
    return plan,dict(progress_rows=len(progress),progress_path=str(source/'progress.jsonl'),validation_paths=[str(p) for p in paths],
        evaluation_timing_paths=[str(p) for p in evaluation_paths],
        evaluation_estimator='mean observed seconds/episode per cell over all completed v3 validation and final evaluations',optimizer_estimator='mean per cell over every available v3 update',
        sensitivity_max_cell_evaluation_seconds=8*sum(r['seconds'] for r in maximum['cells']),
        sensitivity_max_cell_total_seconds=plan['allocation']['optimizer_with_15_percent_margin']+600+8*sum(r['seconds'] for r in maximum['cells']),
        projected_terminal_scheduler=project,allocation=plan['allocation'],estimated_total_seconds=plan['estimated_total_seconds'])


def prepare_queue(directory,source):
    import torch
    from . import worker as w
    from .continuation_v4 import AUTHORIZATION
    directory=Path(directory).resolve();source=Path(source).resolve()
    if directory.exists():raise ValueError('Refusing existing destination or queue restart')
    cfg=load(source/'config.json');w.verify_sources(cfg)
    verify_hashes(cfg.get('continuation_source_hashes',{}))
    hashes={**cfg['source_hashes'],**cfg['continuation_source_hashes']}
    for name in ('continuation_v3.py','continuation_v4.py','continuation_v4_queue.py'):
        path=Path(__file__).with_name(name).resolve();hashes[str(path)]=digest(path)
    verify_hashes(hashes)
    if cfg['max_steps']!=2860 or cfg['device']!='mps' or cfg['effective_batch']!=32 or cfg['microbatch']!=4 or cfg['val_n']!=64 or cfg['test_n']!=128:raise ValueError('Unexpected predecessor recipe')
    sup=load(source/'supervisor.json'); identities={str(sup[k]):identity(sup[k]) for k in ('pid','worker_pid')}
    for key,role in (('pid','run'),('worker_pid','worker')):validate_identity(identities[str(sup[key])],role,source)
    pointer=load(source/'latest_checkpoint.json')
    if digest(pointer['path'])!=pointer['sha256']:raise ValueError('Checkpoint pointer digest mismatch')
    saved=torch.load(pointer['path'],map_location='cpu')
    plan,basis=timing_plan(source,saved,project=saved['state']['step']<2860)
    namespace=namespace_unused(source)
    manifest=dict(authorization=AUTHORIZATION,created_epoch=time.time(),source=str(source),predecessor=sup,predecessor_identities=identities,
        queue_expiry=cfg['deadline']+CLEANUP_GRACE,cleanup_grace_seconds=CLEANUP_GRACE,source_hashes=hashes,
        source_config_sha256=digest(source/'config.json'),source_budget_sha256=digest(source/'budget.json'),
        additional_updates=2145,target_step=5005,activation_budget_seconds=28800,no_accelerator_work=True,
        activation_conditions='normal exit receipts; both OS PIDs gone; terminal2860 full-state digest; complete baseline and both 35-cell finals; frozen sources; exact exposure fit; exclusive worker lock')
    directory.mkdir(parents=True)
    for name,obj in [('queue_manifest',manifest),('plan_preview',plan),('timing_basis_preview',basis),('namespace_check',namespace)]:atomic_json(directory/(name+'.json'),obj)
    for path in hashes:
        dest=directory/'locked_source'/Path(path).relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes())
    print(json.dumps(dict(prepared=str(directory),expiry=manifest['queue_expiry'],estimated_total_seconds=plan['estimated_total_seconds'],allocation=plan['allocation']),default=str),flush=True)


def activate(directory,manifest):
    import torch
    from . import continuation_v4 as v
    from .launch import supervise
    source=Path(manifest['source']);directory=Path(directory)
    no_other_worker();verify_hashes(manifest['source_hashes'])
    if digest(source/'config.json')!=manifest['source_config_sha256'] or digest(source/'budget.json')!=manifest['source_budget_sha256']:raise ValueError('Predecessor budget/config changed')
    completion=verify_predecessor(source);namespace_unused(source)
    saved=torch.load(completion['source_checkpoint']['path'],map_location='cpu')
    plan,basis=timing_plan(source,saved)
    plan['baseline_validation']=str(source/'validation_002860.json')
    plan['historical_validation_paths']=list(saved['state']['config'].get('historical_validation_paths',[]))+[saved['state']['config']['baseline_validation']]+[str(p) for p in sorted(source.glob('validation_[0-9]*.json')) if not p.name.endswith('.partial.json')]
    plan['continuation_source_hashes']={p:h for p,h in manifest['source_hashes'].items() if p not in plan['source_hashes']}
    completion.update(authorization=v.AUTHORIZATION,baseline_sha256=digest(source/'validation_002860.json'),baseline_model_identical=True,no_accelerator_work_in_preparation=True)
    for name,obj in [('plan',plan),('timing_basis',basis),('migration',completion)]:atomic_json(directory/(name+'.json'),obj)
    if time.time()>=manifest['queue_expiry']:raise ValueError('Queue expired during verification')
    if (directory/'budget.json').exists():raise ValueError('Refusing automatic restart or renewed cap')
    # Activation origin is after CPU feasibility, before migration or any new MPS work.
    budget=v.new_budget(time.time(),completion['source_checkpoint']);atomic_json(directory/'budget.json',budget)
    config=dict(plan,**budget);atomic_json(directory/'config.json',config)
    os.chmod(directory/'budget.json',0o444);os.chmod(directory/'config.json',0o444)
    atomic_json(directory/'activation.json',dict(queue_pid=os.getpid(),activation_epoch=budget['cap_started'],deadline=budget['deadline'],normal_predecessor=completion,config_sha256=digest(directory/'config.json')))
    atomic_json(directory/'queue_status.json',dict(phase='activated_not_yet_verified_training',queue_pid=os.getpid(),budget=budget,epoch=time.time()))
    result=supervise(['/usr/bin/caffeinate','-i',sys.executable,'-u','-m','SecondPass.JointTraining.continuation_v4','worker',str(directory)],budget['deadline'],directory)
    if not (directory/'report.json').exists():
        atomic_json(directory/'report.json',dict(protocol=v.PROTOCOL,stop_reason='supervisor_worker_exit_without_report',supervisor=result,final_coverage_complete=False))
        (directory/'REPORT.md').write_text('# v4 interrupted\nNo complete final report. See supervisor and persisted checkpoints. No automatic renewal.\n')
    atomic_json(directory/'queue_result.json',dict(phase='finished',supervisor=result,epoch=time.time()))
    return result['returncode']


def watch(directory):
    import fcntl
    import traceback
    directory=Path(directory).resolve()
    with (directory/'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'queue_handoff.json').exists() or (directory/'budget.json').exists():raise ValueError('Refusing queue restart')
        manifest=load(directory/'queue_manifest.json');source=Path(manifest['source'])
        verify_hashes(manifest['source_hashes'])
        atomic_json(directory/'queue_handoff.json',dict(pid=os.getpid(),identity=identity(os.getpid()),queue_expiry=manifest['queue_expiry'],manifest_sha256=digest(directory/'queue_manifest.json'),started_epoch=time.time(),stdout_stderr_attached=True))
        last_print=0.
        try:
            while True:
                alive=False
                for pid,expected in manifest['predecessor_identities'].items():
                    current=identity(pid)
                    if current is not None:
                        if current!=expected:raise ValueError('Predecessor PID identity changed; fail closed')
                        alive=True
                result=load(source/'supervisor_result.json') if (source/'supervisor_result.json').exists() else None
                phase=queue_decision(now=time.time(),expiry=manifest['queue_expiry'],alive=alive,result=result)
                old=load(source/'live_status.json')
                status=dict(phase='queued_waiting_predecessor' if phase=='waiting' else 'verifying_predecessor',queue_pid=os.getpid(),epoch=time.time(),queue_expiry=manifest['queue_expiry'],predecessor_phase=old['phase'],predecessor_step=old['step'],new_worker_started=False,cap_started=False)
                atomic_json(directory/'queue_status.json',status)
                if time.time()-last_print>=60:print(json.dumps(status),flush=True);last_print=time.time()
                if phase=='verify':return activate(directory,manifest)
                time.sleep(min(5.,max(.05,manifest['queue_expiry']-time.time())))
        except Exception as exc:
            failure=dict(phase='blocked',error=str(exc),traceback=traceback.format_exc(),epoch=time.time(),new_worker_started=(directory/'supervisor.json').exists())
            atomic_json(directory/'queue_result.json',failure);atomic_json(directory/'queue_status.json',failure)
            print(json.dumps(failure),flush=True);return 2


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','queue']);parser.add_argument('directory');parser.add_argument('--source')
    args=parser.parse_args()
    if args.mode=='prepare':
        if not args.source:parser.error('--source required')
        prepare_queue(args.directory,args.source)
    else:raise SystemExit(watch(args.directory))


if __name__=='__main__':main()
