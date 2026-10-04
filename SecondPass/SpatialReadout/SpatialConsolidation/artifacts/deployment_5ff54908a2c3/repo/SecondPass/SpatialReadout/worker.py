"""One optimizer worker; reusable exact CPU/MPS path, never auto-resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
import types
import torch
from SecondPass.JointTraining import worker as original
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, append_jsonl, tree_equal
from SecondPass.TaskSuite.suite import SuiteStream, TASKS, task_classes
from .model import SpatialReadout
from .state import verify_parent, migrate, snapshot, restore, save_payload, load_verified
from .protocol import (selection_key, FinalTestStream, FINAL_TEST_NAMESPACE, validate_budget,
                       assert_no_worker, require_approval, PROTOCOL)


class TrainingSession:
    def __init__(self,directory,config,payload):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config; self.device=config['device']
        self.model=SpatialReadout(task_classes()).to(self.device)
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.scheduler=BalancedScheduler(0); self.stream=SuiteStream('train')
        self.state=restore(payload,self.model,self.optimizer,self.scheduler,self.stream,self.device)
        self.state['config']=config; self.state['deadline']=config['deadline']; self.migration=payload['migration']
        self.latest=None

    def checkpoint(self,filename):
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        payload=snapshot(self.model,self.optimizer,self.scheduler,self.stream,self.state,self.migration,self.device)
        self.latest=save_payload(self.directory/filename,payload)
        atomic_json(self.directory/'latest_checkpoint.json',self.latest)
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=original.utc(),**self.latest))
        return self.latest

    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=original.utc(),
            branch_step=self.state['step'],branch_episodes=self.state['episodes'],parent_step=self.state['parent']['step'],
            optimizer_seconds=self.state['optimizer_seconds'],deadline=self.config['deadline'],latest_checkpoint=self.latest,
            best_step=self.state['best_step'],**extra))

    def train_update(self,task,cell):
        row=original.update(self.model,self.optimizer,self.stream,task,cell,
            self.config['effective_batch'],self.config['microbatch'],self.device)
        s=self.state
        s['step']+=1; s['episodes']+=row['episodes']; s['frames']+=row['frames']; s['optimizer_seconds']+=row['seconds']
        e=s['exposure'][task]; e['updates']+=1; e['episodes']+=row['episodes']; e['frames']+=row['frames']; e['cells'][cell]+=row['episodes']
        row.update(utc=original.utc(),step=s['step'],cumulative_episodes=s['episodes'],cumulative_frames=s['frames'],
            optimizer_seconds=s['optimizer_seconds'],parent_step=s['parent']['step'],parent_episodes=s['parent']['episodes'],
            cumulative_lineage_steps=s['parent']['step']+s['step'],cumulative_lineage_episodes=s['parent']['episodes']+s['episodes'],
            per_task_exposure=s['exposure'],elapsed_cap_seconds=time.time()-self.config['cap_started'],clipping=None,
            latest_validation=s['selection_history'][-1] if s['selection_history'] else None)
        append_jsonl(self.directory/'progress.jsonl',row)
        print(json.dumps({k:row[k] for k in ('utc','step','task','cell','loss','seconds','cumulative_episodes')}),flush=True)
        return row


def evaluate(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
    # Reuse frozen suite's complete metrics/strata/partial-coverage loop. A copied
    # function scope changes ONLY final draw namespace, never the global harness.

    # PyTorch no_grad wraps the original function; use its wrapped function and
    # explicitly enter no_grad here. This avoids modifying original module globals.
    raw=original.evaluate.__wrapped__
    scope=dict(raw.__globals__,SuiteStream=FinalTestStream if split=='test' else SuiteStream)
    evaluator=types.FunctionType(raw.__code__,scope,raw.__name__,raw.__defaults__,raw.__closure__)
    with torch.no_grad():
        result=evaluator(model,split,n,krauzlis_n,microbatch,device,output,cells=cells,deadline=deadline)
    result['summary']['selection_key']=selection_key(result)
    result['selection_rule']='equal-task mean AUC then mean chance-normalized BA; earlier ties; validation only'
    if split=='test': result['final_test_namespace']=FINAL_TEST_NAMESPACE
    atomic_json(output,result)
    return result


def profile(directory):
    directory=Path(directory); config=json.loads((directory/'profile_config.json').read_text())
    budget=json.loads((directory/'budget.json').read_text()); validate_budget(budget,config)
    original.verify_sources(config)
    parent,receipt=verify_parent(config['parent_pointer'])
    # MPS initialized only after the durable cap receipt and exclusive lock.
    session=TrainingSession(directory/'disposable',config,migrate(parent,receipt))
    initial=session.checkpoint('migration.pt')
    rows=[]
    for task,cell in original.all_cells():
        if time.time()>min(config['deadline'],config['cap_started']+1800)-60:
            raise RuntimeError('Disposable profile exceeded its bounded 30-minute allocation')
        row=session.train_update(task,cell)
        measured=evaluate(session.model,'val',8,8,4,session.device,
            directory/'disposable'/f'eval_{task}_{cell}.json',config['deadline']-60,cells=[(task,cell)])
        if not measured['complete']: raise RuntimeError('Profile timing incomplete')
        row=dict(task=task,cell=cell,seconds=row['seconds'],loss=row['loss'],eval_seconds=measured['seconds'],eval_n=8,
            effective_batch=32,microbatch=4,disposable=True,
            allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
        rows.append(row); append_jsonl(directory/'profile_progress.jsonl',row)
        if len(rows)==1: session.checkpoint('checkpoint_first_update.pt')
        atomic_json(directory/'profile.json',dict(complete=False,rows=rows,**budget))
    receipt=session.checkpoint('checkpoint_disposable_final.pt')
    before=torch.rand(16,device='mps').cpu()
    restore(load_verified(receipt),session.model,session.optimizer,session.scheduler,session.stream,'mps')
    after=torch.rand(16,device='mps').cpu()
    if not torch.equal(before,after): raise RuntimeError('MPS RNG roundtrip failed')
    original_payload=load_verified(initial); saved=load_verified(receipt)
    changed=[n for n in saved['model'] if n.startswith('spatial_gru.') and not torch.equal(saved['model'][n],original_payload['model'][n])]
    if len(changed)!=4: raise RuntimeError('ConvGRU parameters did not all change')
    result=dict(complete=True,rows=rows,mps_rng_replay=True,first_migration=initial,checkpoint=receipt,
        changed_convgru_tensors=changed,production_updates=0,discard_all_profile_state=True,torch_version=torch.__version__,
        seconds=time.time()-config['cap_started'],**budget)
    atomic_json(directory/'profile.json',result)
    print(json.dumps(dict(profile_complete=True,seconds=result['seconds'],checkpoint=receipt)),flush=True)


def run(directory):
    directory=Path(directory); config=json.loads((directory/'config.json').read_text())
    validate_budget(json.loads((directory/'budget.json').read_text()),config)
    require_approval(directory,config); original.verify_sources(config)
    if (directory/'migration.pt').exists(): raise RuntimeError('Refusing production restart')
    parent,receipt=verify_parent(config['parent_pointer']); payload=migrate(parent,receipt)
    session=TrainingSession(directory,config,payload)
    migration=session.checkpoint('migration.pt'); saved=load_verified(migration)
    checks={k:tree_equal(saved[k],payload[k]) for k in ('model','optimizer','optimizer_names','scheduler','stream','rng')}
    if not all(checks.values()): raise RuntimeError('Migration full-state preservation failed: '+str(checks))
    atomic_json(directory/'migration_integrity.json',dict(checks=checks,source=receipt,migration=migration,verified_utc=original.utc()))
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete_cycles'; failure=None; terminal=None; terminal_test=None; selected_test=None

    def validate():
        session.status('validation')
        result=evaluate(session.model,'val',config['val_n'],config['val_krauzlis_n'],config['eval_microbatch'],session.device,
            directory/f"validation_{session.state['step']:06d}.json",config['deadline']-2*config['estimated_one_test_seconds']-120)
        key=selection_key(result); s=session.state
        s['selection_history'].append(dict(step=s['step'],key=key,complete=result['complete']))
        filename=f"validation_checkpoint_{s['step']:06d}.pt"
        if key is not None and (s['best_key'] is None or tuple(key)>tuple(s['best_key'])):
            s['best_key']=key; s['best_step']=s['step']; s['best_checkpoint']=str(directory/filename)
        session.checkpoint(filename)

    try:
        while session.state['step']<config['max_steps']:
            if stopped[0]: reason='signal'; break
            if time.time()+config['final_reserve_seconds']+config['update_estimate_seconds']>=config['deadline']:
                reason='wall_budget_reserve'; break
            task,cell=session.scheduler.next()
            row=session.train_update(task,cell)
            step=session.state['step']
            if step in (1,2) or step%config['checkpoint_every']==0: session.checkpoint(f'checkpoint_{step:06d}.pt')
            session.status('training',last_update={k:row[k] for k in ('task','cell','loss','seconds')})
            if step in config['validation_steps'] and not stopped[0]: validate()
        terminal=session.checkpoint('terminal.pt')
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        atomic_json(directory/'failure.json',failure); reason='worker_error'
        # Do not publish advanced sampler/partial or contaminated optimizer state.
        session.state=restore(load_verified(session.latest),session.model,session.optimizer,session.scheduler,session.stream,session.device)
        terminal=session.checkpoint('terminal_recovered.pt')
    try:
        if failure is None and session.state['step']>0 and not stopped[0]:
            # If the cap stops training early, the second look is at that actual
            # terminal endpoint, never an extra opportunistic selection look.
            history=session.state['selection_history']
            if (not history or history[-1]['step']!=session.state['step']) and len(history)<2:
                if time.time()+config['final_reserve_seconds']<config['deadline']:
                    validate(); terminal=session.checkpoint('terminal_validated.pt')
            if time.time()<config['deadline']-60:
                session.status('final_test_terminal')
                different=session.state['best_step']!=session.state['step']
                reserve=config['estimated_one_test_seconds'] if different else 0
                terminal_test=evaluate(session.model,'test',128,200,4,session.device,directory/'test_terminal.json',config['deadline']-reserve-60)
                if not different:
                    winner=torch.load(session.state['best_checkpoint'],map_location='cpu')
                    if not tree_equal(winner['model'],{n:v.detach().cpu() for n,v in session.model.state_dict().items()}):
                        raise RuntimeError('Same-step final deduplication model mismatch')
                    selected_test=dict(reused_terminal=True,identical_model_verified=True,results=terminal_test)
                    atomic_json(directory/'test_selected.json',selected_test)
                elif session.state['best_checkpoint'] is not None and time.time()<config['deadline']-60:
                    session.model.load_state_dict(torch.load(session.state['best_checkpoint'],map_location='cpu')['model'])
                    session.status('final_test_selected')
                    selected_test=evaluate(session.model,'test',128,200,4,session.device,directory/'test_selected.json',config['deadline']-45)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        atomic_json(directory/'finalization_failure.json',failure)
    selected=selected_test.get('results',selected_test) if selected_test else None
    complete=bool(terminal_test and terminal_test['complete'] and selected and selected['complete'])
    report=dict(protocol=PROTOCOL,stop_reason=reason,failure=failure,terminal_checkpoint=terminal,
        terminal_step=session.state['step'],selected_step=session.state['best_step'],selection_history=session.state['selection_history'],
        parent_step=3393,parent_episodes=108576,branch_episodes=session.state['episodes'],branch_exposure=session.state['exposure'],
        optimizer_seconds=session.state['optimizer_seconds'],config=config,terminal_test=terminal_test,selected_test=selected_test,
        final_coverage_complete=complete,finished_utc=original.utc(),
        limitations=['Targeted warm-start architecture experiment, not matched causal control or superiority evidence.',
            'Inherited CNN/KDA/heads and Adam have unequal prior experience to fresh recurrence/readout.',
            'All profile state discarded; no optional baseline; historical scores contextual only.',
            'Official source-identity splits unchanged; fresh final draws are not fresh image identities.',
            'N0 specificity/FPR separate; all Krauzlis target/foil/catch denominators retained.',
            'Single seed, no convergence, novel-location or paired-delay claim.'])
    atomic_json(directory/'report.json',report)
    lines=['# KDA with final spatial ConvGRU',f"\nStop: {reason}; branch updates {session.state['step']}; selected {session.state['best_step']}; parent 3393.",
        f'\nComplete selected/terminal final coverage: {complete}.', '\nFull task/cell confusion, BA/AUC, strata and event denominators are in report.json and test_*.json.']
    for label,result in (('Terminal',terminal_test),('Selected',selected)):
        lines += ['',f'## {label} all completed cells','| Task | Cell | n | BA | AUC | Specificity | FPR |','|---|---|---:|---:|---:|---:|---:|']
        for r in result['cells'] if result else []:
            lines.append(f"| {r['task']} | {r['cell']} | {r['n']} | {r['balanced_accuracy']} | {r['auc']} | {r.get('specificity')} | {r.get('false_positive_rate')} |")
    lines+=['']+['- '+x for x in report['limitations']]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    session.status('finished',stop_reason=reason,failure=failure,final_coverage_complete=complete)
    return 0 if failure is None and complete and session.state['step']==config['max_steps'] else 2


def main():
    import fcntl
    original.cpu_setup()
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['profile','run']); parser.add_argument('directory')
    args=parser.parse_args(); directory=Path(args.directory).resolve()
    assert_no_worker()
    config_path=directory/('profile_config.json' if args.mode=='profile' else 'config.json')
    lock_path=Path(json.loads(config_path.read_text())['worker_lock'])
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))


if __name__=='__main__': main()
