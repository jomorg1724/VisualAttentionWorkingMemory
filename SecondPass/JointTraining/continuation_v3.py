"""Explicitly authorized eight-hour continuation; frozen v1/v2 remain immutable."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

from .core import BalancedScheduler, atomic_json, restore_checkpoint, summarize
from SecondPass.TaskSuite.suite import SuiteStream, TASKS

PROTOCOL = 'continuation_v3_eight_hour'
AUTHORIZATION = 'go ahead and set up a 8 hour training run for more trsining and more updates'
SELECTION = 'maximum lexicographic (equal-task mean validation AUC, mean chance-normalized task BA); equal eligible-cell means; earlier ties'
FINAL_TEST_NAMESPACE = 94392763
SOURCE_SHA256 = 'f8f308e64c97c99a3ce7c7b1b68ed5386bbc91c35df2197aef1a02cbdbf00048'


def new_budget(started, source):
    return dict(cap_started=started, deadline=started+28800., wall_cap_seconds=28800,
                authorization=AUTHORIZATION, budget_source_sha256=source['sha256'],
                note='New explicit user authorization; receipt before first MPS work, including migration. No renewal.')


def selection_key(validation):
    if not validation['complete']:
        return None
    tasks = validation['summary']['tasks']
    if set(tasks) != set(TASKS):
        return None
    auc = [tasks[t]['auc'] for t in TASKS]
    ba = [tasks[t]['chance_normalized_ba'] for t in TASKS]
    if None in auc or None in ba:
        return None
    return [sum(auc)/len(auc), sum(ba)/len(ba)]


def restore_for_continuation(receipt, model, optimizer, scheduler, stream, config, baseline, now=None):
    import torch
    now = time.time() if now is None else now
    path = Path(receipt['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Source checkpoint digest mismatch')
    saved = torch.load(path, map_location='cpu')
    prior = saved['state']['config']
    mutable = {'cap_started','deadline','wall_cap_seconds','authorization','budget_source_sha256','note',
               'max_steps','planned_cycles','planned_exposure','validation_steps','final_reserve_seconds',
               'estimated_validation_seconds','estimated_one_test_seconds','estimated_cycle_seconds',
               'estimated_total_seconds','update_estimate_seconds','protocol_version','amendment',
               'final_test_seed_namespace','continuation_source_hashes','selection','allocation',
               'baseline_validation','historical_validation_paths'}
    for key in prior.keys() | config.keys():
        if key not in mutable and prior.get(key) != config.get(key):
            raise ValueError('Unauthorized protocol change: '+key)
    expected = new_budget(config['cap_started'], receipt)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Unauthorized budget field: '+key)
    if config['selection'] != SELECTION or config['protocol_version'] != PROTOCOL:
        raise ValueError('Selection/protocol mismatch')
    if now >= config['deadline']:
        raise ValueError('New wall cap expired')
    if now < config['cap_started'] or config['cap_started'] <= prior['deadline']:
        raise ValueError('New authorization must follow completed prior allowance')
    if saved['state']['step'] >= config['max_steps']:
        raise ValueError('Continuation horizon already complete')
    key = selection_key(baseline)
    if key is None:
        raise ValueError('Complete baseline validation required')
    state = copy.deepcopy(restore_checkpoint(path, model, optimizer, scheduler, stream, config['device']))
    state['historical_selection'] = dict(rule=prior.get('selection'), history=copy.deepcopy(state['selection_history']),
        best_step=state['best_step'], best_key=state['best_key'], best_checkpoint=state['best_checkpoint'],
        note='Historical winner retained; incompatible keys never compared with v3 keys.')
    state['previous_elapsed_cap_seconds'] = state.get('elapsed_cap_seconds',0.)
    state['elapsed_cap_seconds'] = now-config['cap_started']
    state['deadline'] = config['deadline']; state['config'] = copy.deepcopy(config)
    state['continuation'] = dict(protocol=PROTOCOL, source=receipt, carried_step=state['step'],
        carried_episodes=state['episodes'], carried_optimizer_seconds=state['optimizer_seconds'], previous_deadline=prior['deadline'])
    state['selection_history'] = [dict(step=state['step'],key=key,complete=True,protocol=PROTOCOL,
        baseline_reused=True, validation=config.get('baseline_validation'))]
    state['best_step'] = state['step']; state['best_key'] = key; state['best_checkpoint'] = receipt['path']
    return state


def plan_continuation(old, state, scheduler_state, progress, validation):
    """Maximize complete cycles using exact future queues and observed cell costs."""
    import statistics
    expected = {(t,c['id']) for t,s in TASKS.items() for c in s['conditions']}
    per_cell = {key:[] for key in expected}
    for row in progress:
        per_cell[(row['task'],row['cell'])].append(row['seconds'])
    if any(not values for values in per_cell.values()):
        raise ValueError('Every cell needs prior observed optimizer timing')
    costs = {key:statistics.mean(values) for key,values in per_cell.items()}
    if {(r['task'],r['cell']) for r in validation['cells']} != expected:
        raise ValueError('Every validation cell is required for timing')
    def eval_cost(n,k):
        return sum(r['seconds']/r['n']*(k if r['task']=='krauzlis_cued_motion' else n) for r in validation['cells'])
    val = eval_cost(old['val_n'],old['val_krauzlis_n']); test = eval_cost(128,200)
    overhead = 4*val+2*test+600.
    scheduler=BalancedScheduler(0); scheduler.load_state_dict(scheduler_state)
    exposure=copy.deepcopy(state['exposure']); added=0; training=0.
    while True:
        trial=[scheduler.next() for _ in range(13)]
        candidate=training+sum(costs[x] for x in trial)
        next_total=candidate*1.15+overhead
        if next_total > 28800.:
            break
        training=candidate; added+=13
        for t,c in trial:
            exposure[t]['updates']+=1; exposure[t]['episodes']+=old['effective_batch']
            exposure[t]['cells'][c]+=old['effective_batch']
    if added < 52:
        raise ValueError('Budget does not fit meaningful acquisition and four looks')
    cycles=added//13; steps=state['step']+added
    return dict(old, max_steps=steps, planned_cycles=steps//13,
        planned_exposure={t:dict(updates=e['updates'],episodes=e['episodes'],cell_episodes=e['cells']) for t,e in exposure.items()},
        validation_steps=[state['step']+(cycles*k//4)*13 for k in (1,2,3,4)],
        final_reserve_seconds=val+2*test+300., estimated_validation_seconds=val,
        estimated_one_test_seconds=test,estimated_cycle_seconds=training/cycles,
        update_estimate_seconds=max(r['seconds'] for r in progress)*1.25,
        estimated_total_seconds=training*1.15+overhead,
        protocol_version=PROTOCOL,selection=SELECTION,final_test_seed_namespace=FINAL_TEST_NAMESPACE,
        allocation=dict(additional_updates=added,additional_episodes=added*old['effective_batch'],
            additional_updates_per_task=cycles,additional_episodes_per_task=cycles*old['effective_batch'],
            cumulative_updates=steps,cumulative_episodes=state['episodes']+added*old['effective_batch'],
            expected_optimizer_seconds=training,optimizer_with_15_percent_margin=training*1.15,
            four_validation_seconds=4*val,two_final_test_seconds=2*test,save_startup_report_grace_seconds=600.,
            next_cycle_projected_seconds=next_total,
            measured_equal_task_seconds=statistics.mean([statistics.mean([costs[(t,c['id'])] for c in s['conditions']]) for t,s in TASKS.items()]),
            cell_seconds={t:{c['id']:costs[(t,c['id'])] for c in s['conditions']} for t,s in TASKS.items()},
            timing_only=True,prior_test_scores_used=False,prior_tests_seen=True,
            source_identity_reuse=True,exploratory=True))


class FinalTestStream(SuiteStream):
    def __init__(self, split='test'):
        if split != 'test':
            raise ValueError('Final namespace is test-only')
        super().__init__('test')

    def stream_seed(self, task, cell):
        index,_=self._cell(task,cell)
        return FINAL_TEST_NAMESPACE*100000+TASKS[task]['stream_id']*1000+index


def is_joint_worker(command):
    from .continuation_v2 import is_joint_worker as old_guard
    tokens=command.split()
    if old_guard(command): return True
    if not tokens or not Path(tokens[0]).name.lower().startswith('python'): return False
    i=1
    while i<len(tokens) and tokens[i] in {'-u','-B','-O','-OO','-I','-E','-s','-S','-b','-bb','-q'}: i+=1
    return tokens[i:i+3]==['-m','SecondPass.JointTraining.continuation_v3','worker']


def evaluate_final(model, config, device, output, deadline):
    # Versioned wrapper reuses the exact v2 scoring loop without changing its globals.
    import types
    from .continuation_v2 import evaluate_final as original
    scope=dict(original.__globals__,FinalTestStream=FinalTestStream,FINAL_TEST_NAMESPACE=FINAL_TEST_NAMESPACE)
    evaluator=types.FunctionType(original.__code__,scope,original.__name__,original.__defaults__,original.__closure__)
    return evaluator(model,config,device,output,deadline)


def publish_curves(directory, config):
    """Cumulative validation-only curves; each look retains all cells/strata."""
    paths=[Path(p) for p in config.get('historical_validation_paths',[])]
    paths += [Path(config['baseline_validation'])]
    paths += sorted(Path(directory).glob('validation_[0-9]*.json'))
    looks=[]; seen=set()
    for path in paths:
        if not path.exists() or path.name.endswith('.partial.json'): continue
        step=int(path.stem.rsplit('_',1)[-1]) if path.stem.rsplit('_',1)[-1].isdigit() else 1
        if step in seen: continue
        seen.add(step); data=json.loads(path.read_text())
        looks.append(dict(step=step,cumulative_episodes=step*config['effective_batch'],path=str(path),
            summary=data['summary'],cells=data['cells'],complete=data['complete'],
            v3_selection_eligible=step>=config.get('allocation',{}).get('cumulative_updates',config['max_steps'])-config.get('allocation',{}).get('additional_updates',config['max_steps']-1)))
    looks.sort(key=lambda r:r['step'])
    atomic_json(Path(directory)/'validation_curves.json',dict(looks=looks,selection=SELECTION,
        note='Same fixed validation draws; prior looks historical only. All cells, strata and task curves retained.'))
    return looks


def run(directory, config, receipt):
    import os
    import signal
    import traceback
    import torch
    from . import worker as w
    from .core import append_jsonl, save_checkpoint, tree_equal, may_update
    directory=Path(directory)
    if (directory/'migration.pt').exists(): raise RuntimeError('Refusing automatic restart')
    w.verify_sources(config)
    for path,digest in config.get('continuation_source_hashes',{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise ValueError('Frozen continuation source changed: '+path)
    baseline=json.loads(Path(config['baseline_validation']).read_text())
    device=config['device']
    model=w.AccumulatorBaseline(w.task_classes(),stack=3,center=True,accumulator='kda').to(device)
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    scheduler=BalancedScheduler(config['scheduler_seed']); stream=SuiteStream('train')
    state=restore_for_continuation(receipt,model,optimizer,scheduler,stream,config,baseline)
    if (directory/'migration.json').exists():
        provenance=json.loads((directory/'migration.json').read_text())
        state['historical_selection']['completed_history']=provenance['prior_completed_selection_history']
        state['historical_selection']['completed_best_step']=provenance['prior_completed_best_step']
    carried=state['step']; carried_episodes=state['episodes']; carried_seconds=state['optimizer_seconds']
    latest=None; stopped=[False]; handlers={}
    def status(phase,**extra):
        atomic_json(directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=w.utc(),step=state['step'],
            episodes=state['episodes'],frames=state['frames'],optimizer_seconds=state['optimizer_seconds'],
            additional_updates=state['step']-carried,additional_episodes=state['episodes']-carried_episodes,
            additional_optimizer_seconds=state['optimizer_seconds']-carried_seconds,
            cap_started=config['cap_started'],deadline_epoch=config['deadline'],deadline=w.utc(config['deadline']),
            remaining_seconds=max(0.,config['deadline']-time.time()),latest_checkpoint=latest,best_step=state['best_step'],
            carried_step=carried,pinned_target=config['max_steps'],**extra))
    def checkpoint(filename):
        nonlocal latest
        state['elapsed_cap_seconds']=time.time()-config['cap_started']
        latest=save_checkpoint(directory/filename,model,optimizer,scheduler,stream,state,device)
        atomic_json(directory/'latest_checkpoint.json',latest)
        append_jsonl(directory/'checkpoints.jsonl',dict(utc=w.utc(),**latest))
        return latest
    checkpoint('migration.pt')
    original=torch.load(receipt['path'],map_location='cpu'); migrated=torch.load(directory/'migration.pt',map_location='cpu')
    checks={k:tree_equal(original[k],migrated[k]) for k in ('model','optimizer','scheduler','stream','rng')}
    if not all(checks.values()): raise RuntimeError('Migration changed acquired state: '+str(checks))
    atomic_json(directory/'resume_integrity.json',dict(exact_nonallocation_state=True,checks=checks,source=receipt,migration=latest,
        carried_step=carried,carried_episodes=carried_episodes,carried_optimizer_seconds=carried_seconds,
        historical_selection=state['historical_selection'],baseline_selection=state['selection_history'],verified_utc=w.utc()))
    del original,migrated
    publish_curves(directory,config)
    for sig in (signal.SIGTERM,signal.SIGINT): handlers[sig]=signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def validate():
        status('validation'); w.verify_sources(config)
        validation=w.evaluate(model,'val',config['val_n'],config['val_krauzlis_n'],config['eval_microbatch'],device,
            directory/f"validation_{state['step']:06d}.json",deadline=config['deadline']-2*config['estimated_one_test_seconds']-120)
        key=selection_key(validation)
        # Override the obsolete worst-task key in this new artifact only.
        validation['summary']['historical_rule_key']=validation['summary']['selection_key']
        validation['summary']['selection_key']=key; validation['selection_rule']=SELECTION
        atomic_json(directory/f"validation_{state['step']:06d}.json",validation)
        state['selection_history'].append(dict(step=state['step'],key=key,complete=validation['complete'],protocol=PROTOCOL))
        filename=f"validation_checkpoint_{state['step']:06d}.pt"
        if key is not None and tuple(key)>tuple(state['best_key']):
            state['best_key']=key; state['best_step']=state['step']; state['best_checkpoint']=str(directory/filename)
        checkpoint(filename); publish_curves(directory,config)
    status('training',initialization='exact_terminal715_full_state_continuation')
    failure=None; reason='planned_complete_cycles'; terminal_test=None; selected_test=None
    try:
        while state['step']<config['max_steps']:
            if not may_update(now=time.time(),deadline=config['deadline'],reserve=config['final_reserve_seconds'],
                estimate=config['update_estimate_seconds'],step=state['step'],max_steps=config['max_steps'],stopped=stopped[0]):
                reason='signal' if stopped[0] else 'wall_budget_reserve'; break
            task,cell=scheduler.next()
            row=w.update(model,optimizer,stream,task,cell,config['effective_batch'],config['microbatch'],device)
            state['step']+=1; state['episodes']+=row['episodes']; state['frames']+=row['frames']; state['optimizer_seconds']+=row['seconds']
            e=state['exposure'][task]; e['updates']+=1; e['episodes']+=row['episodes']; e['frames']+=row['frames']; e['cells'][cell]+=row['episodes']
            row.update(utc=w.utc(),step=state['step'],cumulative_episodes=state['episodes'],cumulative_frames=state['frames'],
                optimizer_seconds=state['optimizer_seconds'],additional_optimizer_seconds=state['optimizer_seconds']-carried_seconds,
                additional_updates=state['step']-carried,additional_episodes=state['episodes']-carried_episodes,
                elapsed_cap_seconds=time.time()-config['cap_started'],previous_elapsed_cap_seconds=state['previous_elapsed_cap_seconds'],
                per_task_exposure=state['exposure'],clipping=None,protocol=PROTOCOL,carried_step=carried,latest_validation=state['selection_history'][-1])
            append_jsonl(directory/'progress.jsonl',row)
            print(json.dumps({k:row[k] for k in ('utc','step','task','cell','loss','seconds','cumulative_episodes')}),flush=True)
            if state['step']==carried+1 or state['step']%config['checkpoint_every']==0: checkpoint(f"checkpoint_{state['step']:06d}.pt")
            status('training',last_update={k:row[k] for k in ('task','cell','loss','grad_norm','seconds')})
            if state['step'] in config['validation_steps'] and not stopped[0]: validate()
        terminal=checkpoint('terminal.pt')
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()); atomic_json(directory/'failure.json',failure)
        reason='nonfinite_safe_stop' if isinstance(exc,FloatingPointError) else 'worker_error'
        state=restore_checkpoint(latest['path'],model,optimizer,scheduler,stream,device)
        terminal=checkpoint('terminal_recovered.pt')
    try:
        if failure is None and state['selection_history'][-1]['step']!=state['step'] and time.time()+config['final_reserve_seconds']<config['deadline']:
            validate(); terminal=checkpoint('terminal.pt')
        if time.time()<config['deadline']-60:
            status('final_test_terminal')
            selected_reserve=config['estimated_one_test_seconds'] if state['best_step']!=state['step'] else 0
            terminal_test=evaluate_final(model,config,device,directory/'test_terminal.json',config['deadline']-selected_reserve-60)
            if state['best_step']==state['step']:
                saved=torch.load(state['best_checkpoint'],map_location='cpu')
                if not tree_equal(saved['model'],{k:v.detach().cpu() for k,v in model.state_dict().items()}):
                    raise RuntimeError('Same-step checkpoint is not identical; cannot deduplicate')
                selected_test=dict(reused_terminal=True,identical_model_verified=True,selected_step=state['best_step'],results=terminal_test)
                atomic_json(directory/'test_selected.json',selected_test)
            elif time.time()<config['deadline']-60:
                saved=torch.load(state['best_checkpoint'],map_location='cpu'); model.load_state_dict(saved['model']); del saved
                status('final_test_selected')
                selected_test=evaluate_final(model,config,device,directory/'test_selected.json',config['deadline']-45)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()); atomic_json(directory/'finalization_failure.json',failure)
    selected=selected_test.get('results',selected_test) if selected_test else None
    complete=bool(terminal_test and terminal_test['complete'] and selected and selected['complete'])
    looks=publish_curves(directory,config)
    report=dict(protocol=PROTOCOL,stop_reason=reason,failure=failure,config=config,carried_step=carried,
        terminal_step=state['step'],selected_step=state['best_step'],selected_checkpoint=state['best_checkpoint'],
        historical_selection=state['historical_selection'],selection_history=state['selection_history'],terminal_checkpoint=terminal,
        episodes=state['episodes'],additional_episodes=state['episodes']-carried_episodes,frames=state['frames'],
        optimizer_seconds=state['optimizer_seconds'],additional_optimizer_seconds=state['optimizer_seconds']-carried_seconds,
        exposure=state['exposure'],terminal_test=terminal_test,selected_test=selected_test,final_coverage_complete=complete,
        finished_utc=w.utc(),elapsed_cap_seconds=time.time()-config['cap_started'],validation_curves=str(directory/'validation_curves.json'),
        limitations=['Exploratory continuation: prior test results were seen; this is not a wholly untouched test population.',
            'New final-only draws reuse official test source identities; draws are not new source identities.',
            'Checkpoint selection uses validation only, equal-task mean AUC then mean chance-normalized BA; earlier ties.',
            'Historical worst-task-rule winner remains historical and was not silently reselected.',
            'Single seed; no convergence claim, source-independent uncertainty, paired-delay or novel-location claim.',
            'N0 specificity/FPR and Krauzlis target/foil/catch rates and side/event denominators remain separate.'])
    atomic_json(directory/'report.json',report)
    lines=['# Eight-hour joint KDA continuation v3','',f"Terminal {state['step']}; selected {state['best_step']}; cumulative episodes {state['episodes']}; additional {state['episodes']-carried_episodes}.",
        f"Stop: {reason}. Complete terminal + selected 35-cell tests: {complete}.",'','## Comparable fixed-draw validation: baseline versus latest','',
        '| Task | Baseline AUC | Latest AUC | Baseline BA | Latest BA |','|---|---:|---:|---:|---:|']
    last=looks[-1]['summary']['tasks'] if looks else baseline['summary']['tasks']
    for task in TASKS:
        a=baseline['summary']['tasks'][task]; b=last[task]
        lines.append(f"| {task} | {a['auc']} | {b['auc']} | {a.get('balanced_accuracy')} | {b.get('balanced_accuracy')} |")
    lines += ['','## Full cell and strata reporting','',
        '`test_terminal.json` and `test_selected.json` contain all cell confusion/BA/AUC, strata, N0 specificity/FPR and Krauzlis denominators; `validation_curves.json` retains cumulative per-task/cell curves.',
        'Old versus new test draws are not directly paired comparisons. No old test score was used to allocate or select this continuation.','']
    lines += ['- '+s for s in report['limitations']]
    for label,result in (('Terminal',terminal_test),('Selected',selected)):
        lines += ['',f'## {label}: every evaluated cell','',
            '| Task | Cell | n | BA | AUC | Specificity | FPR |','|---|---|---:|---:|---:|---:|---:|']
        for r in result['cells'] if result else []:
            lines.append(f"| {r['task']} | {r['cell']} | {r['n']} | {r['balanced_accuracy']} | {r['auc']} | {r.get('specificity')} | {r.get('false_positive_rate')} |")
            if r['task']=='krauzlis_cued_motion':
                lines.append(f"\n{r['cell']}: target hit {r['target_hit_rate']}; foil FPR {r['foil_false_alarm_rate']}; catch FPR {r['catch_false_positive_rate']}. Event/side counts: `{json.dumps(r['events'],sort_keys=True)}`.\n")
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    status('finished',report=str(directory/'report.json'),stop_reason=reason,failure=failure,final_coverage_complete=complete)
    for sig,handler in handlers.items(): signal.signal(sig,handler)
    return report


def validate_launch(config, budget, receipt, now=None):
    now=time.time() if now is None else now
    expected=new_budget(budget['cap_started'],receipt)
    if budget!=expected or any(config.get(k)!=v for k,v in budget.items()):
        raise ValueError('Refusing changed or renewed budget')
    if now>=budget['deadline'] or now<budget['cap_started']:
        raise ValueError('Refusing expired or future budget')


def prepare(directory, source_directory):
    """CPU-only preparation; does not start the new accelerator allowance."""
    import torch
    from . import worker as w
    from .core import tree_equal
    directory=Path(directory); source_directory=Path(source_directory).resolve()
    if directory.exists(): raise RuntimeError('Refusing existing destination')
    source=source_directory/'terminal.pt'; digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=SOURCE_SHA256: raise ValueError('Explicit terminal715 digest mismatch')
    saved=torch.load(source,map_location='cpu'); state=saved['state']
    if state['step']!=715 or state['episodes']!=22880: raise ValueError('Expected terminal715/22880')
    w.verify_sources(state['config'])
    completed=torch.load(source_directory/'validation_checkpoint_000715.pt',map_location='cpu')
    if not tree_equal(saved['model'],completed['model']): raise ValueError('Baseline validation model differs from terminal715')
    validation_path=source_directory/'validation_000715.json'; baseline=json.loads(validation_path.read_text())
    if not baseline['complete'] or baseline['complete_cells']!=35: raise ValueError('Incomplete baseline')
    progress=[json.loads(line) for line in (source_directory/'progress.jsonl').read_text().splitlines()]
    if [r['step'] for r in progress]!=list(range(1,716)): raise ValueError('Prior progress is not contiguous 1..715')
    plan=plan_continuation(state['config'],state,saved['scheduler'],progress,baseline)
    plan['baseline_validation']=str(validation_path)
    original=source_directory.parent/'fresh_kda_joint_01'
    plan['historical_validation_paths']=[str(p) for p in sorted(original.glob('validation_[0-9]*.json')) if not p.name.endswith('.partial.json')]
    plan['continuation_source_hashes']=dict(state['config'].get('continuation_source_hashes',{}))
    plan['continuation_source_hashes'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt=dict(path=str(source),sha256=digest,step=715,verified=True,bytes=source.stat().st_size)
    directory.mkdir(parents=True)
    atomic_json(directory/'plan.json',plan)
    atomic_json(directory/'migration.json',dict(source_checkpoint=receipt,authorization=AUTHORIZATION,
        prior_deadline=state['deadline'],prior_completed_selection_history=completed['state']['selection_history'],
        prior_completed_best_step=completed['state']['best_step'],baseline_model_identical=True,
        baseline_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        no_accelerator_work_in_preparation=True))
    for path,digest in {**plan['source_hashes'],**plan['continuation_source_hashes']}.items():
        destination=directory/'locked_source'/Path(path).relative_to(Path.cwd())
        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(Path(path).read_bytes())
    atomic_json(directory/'timing_basis.json',dict(prior_progress_path=str(source_directory/'progress.jsonl'),
        prior_rows=len(progress),validation_path=str(validation_path),allocation=plan['allocation']))
    print(json.dumps(dict(directory=str(directory),max_steps=plan['max_steps'],validation_steps=plan['validation_steps'],
        allocation={k:v for k,v in plan['allocation'].items() if k!='cell_seconds'},estimated_total_seconds=plan['estimated_total_seconds']),indent=2))


def main():
    import argparse
    import fcntl
    import os
    import subprocess
    import sys
    from . import worker as w
    from .launch import supervise
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','run','worker']); parser.add_argument('directory')
    parser.add_argument('--source'); args=parser.parse_args(); directory=Path(args.directory).resolve()
    if args.mode=='prepare':
        if not args.source: parser.error('--source required')
        prepare(directory,args.source); return
    if args.mode=='run' and (directory/'budget.json').exists():
        raise RuntimeError('Refusing automatic restart or budget renewal')
    if (directory/'migration.pt').exists(): raise RuntimeError('Refusing already started migration')
    migration=json.loads((directory/'migration.json').read_text()); receipt=migration['source_checkpoint']
    if receipt['sha256']!=SOURCE_SHA256: raise ValueError('Wrong explicitly authorized source')
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        tokens=line.strip().split(None,1)
        if len(tokens)==2 and int(tokens[0])!=os.getpid() and is_joint_worker(tokens[1]):
            raise RuntimeError('Another local joint worker exists: '+tokens[0])
    if args.mode=='run':
        plan=json.loads((directory/'plan.json').read_text())
        w.verify_sources(plan)
        for path,digest in plan['continuation_source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise ValueError('Continuation source changed')
        # Last CPU setup is complete. This durable receipt precedes all new MPS work.
        budget=new_budget(time.time(),receipt); atomic_json(directory/'budget.json',budget)
        config=dict(plan,**budget); atomic_json(directory/'config.json',config)
        os.chmod(directory/'budget.json',0o444); os.chmod(directory/'config.json',0o444)
        result=supervise(['/usr/bin/caffeinate','-i',sys.executable,'-u','-m',
            'SecondPass.JointTraining.continuation_v3','worker',str(directory)],budget['deadline'],directory)
        if not (directory/'report.json').exists():
            latest=json.loads((directory/'latest_checkpoint.json').read_text()) if (directory/'latest_checkpoint.json').exists() else None
            atomic_json(directory/'report.json',dict(protocol=PROTOCOL,stop_reason='supervisor_worker_exit_without_report',
                supervisor=result,latest_checkpoint=latest,final_coverage_complete=False,
                note='No success or convergence inferred; inspect persisted progress and partial final cell files.'))
            (directory/'REPORT.md').write_text('# Continuation interrupted\n\nFinal coverage incomplete; see report.json and partial cell files. No automatic renewal.\n')
        print(json.dumps(result),flush=True); raise SystemExit(result['returncode'])
    config=json.loads((directory/'config.json').read_text()); budget=json.loads((directory/'budget.json').read_text())
    validate_launch(config,budget,receipt)
    if hashlib.sha256(Path(config['baseline_validation']).read_bytes()).hexdigest()!=migration['baseline_sha256']:
        raise ValueError('Baseline validation artifact changed')
    with (directory.parent/'.joint_training_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        w.cpu_setup()
        report=run(directory,config,receipt)
        raise SystemExit(0 if report['failure'] is None and report['final_coverage_complete'] else 2)


if __name__=='__main__':
    main()
