"""Fresh-only no-CLS learner. Never reads a predecessor or profile checkpoint."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import torch
from SecondPass.SpatialComparisonReadout.CloudRun import worker as cloud
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, append_jsonl, cpu_tree, tree_equal
from SecondPass.TaskSuite.suite import CATALOG, TASKS, SuiteStream, task_classes
from .model import SpatialRecurrentConvDecoder, VERSION

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = VERSION + '_fresh_v1'
INIT_SEED, CUDA_SEED, SCHEDULER_SEED = 97592763, 97692763, 97792763
TRAIN_NAMESPACE, VAL_NAMESPACE, FINAL_NAMESPACE = 97892763, 97992763, 98092763
TARGET = 5200
digest = cloud.digest
load_verified = cloud.load_verified


class FreshStream(SuiteStream):
    def stream_seed(self, task, cell):
        index, _ = self._cell(task, cell)
        namespace = dict(train=TRAIN_NAMESPACE, val=VAL_NAMESPACE, test=FINAL_NAMESPACE)[self.split]
        return namespace*100000 + TASKS[task]['stream_id']*1000 + index


def provenance():
    return dict(initialization='all learned tensors newly constructed; no checkpoint input',
                architecture=VERSION, initialization_seed=INIT_SEED, cuda_seed=CUDA_SEED,
                scheduler_seed=SCHEDULER_SEED, train_namespace=TRAIN_NAMESPACE,
                validation_namespace=VAL_NAMESPACE, final_namespace=FINAL_NAMESPACE,
                inherited_weights=False, inherited_optimizer=False, inherited_rng=False,
                inherited_stream=False, profile_state_inherited=False)


class Session:
    def __init__(self, directory, config):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config; self.device = config['device']; self.latest = None
        random.seed(INIT_SEED); np.random.seed(INIT_SEED); torch.manual_seed(INIT_SEED)
        self.model = SpatialRecurrentConvDecoder(task_classes()).to(self.device)
        if str(self.device).startswith('cuda'): torch.cuda.manual_seed_all(CUDA_SEED)
        assert all(p.requires_grad and p.dtype == torch.float32 for p in self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)
        self.scheduler = BalancedScheduler(SCHEDULER_SEED); self.stream = FreshStream('train')
        self.state = dict(step=0, episodes=0, frames=0, optimizer_seconds=0., best_step=None,
                          best_key=None, best_checkpoint=None, selection_history=[], config=config,
                          exposure={t:dict(updates=0,episodes=0,frames=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()})

    def checkpoint(self, filename):
        path = self.directory/filename
        if path.exists(): raise FileExistsError('Immutable checkpoint: '+str(path))
        self.state['elapsed_cap_seconds'] = time.time()-self.config['cap_started']
        rng = dict(cpu=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
        if str(self.device).startswith('cuda'): rng['cuda'] = torch.cuda.get_rng_state_all()
        payload = cpu_tree(dict(schema=3, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                               optimizer_names=[n for n,_ in self.model.named_parameters()], state=self.state,
                               scheduler=self.scheduler.state_dict(), stream=self.stream.state_dict(), rng=rng,
                               provenance=provenance()))
        temporary = path.with_suffix('.pt.tmp')
        with temporary.open('wb') as f:
            torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(temporary,path)
        receipt = dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path), verified=True, step=self.state['step'])
        if not tree_equal(payload,load_verified(receipt)): raise RuntimeError('Checkpoint readback mismatch')
        self.latest = receipt
        atomic_json(self.directory/'latest_checkpoint.json',receipt)
        if self.state['step']==0: atomic_json(self.directory/'initial_checkpoint.json',receipt)
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=cloud.original.utc(),**receipt))
        if self.state['step'] in (1,13): verify_progress(self.directory)
        return receipt

    def status(self, phase, **extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=cloud.original.utc(),
            architecture=VERSION, initialization='fresh', step=self.state['step'], episodes=self.state['episodes'],
            optimizer_seconds=self.state['optimizer_seconds'], deadline=self.config['deadline'],
            latest_checkpoint=self.latest,best_step=self.state['best_step'],**extra))

    def train_update(self, task, cell):
        row = cloud.update(self.model,self.optimizer,self.stream,task,cell,
                           self.config['effective_batch'],self.config['microbatch'],self.device)
        s = self.state
        s['step']+=1; s['episodes']+=row['episodes']; s['frames']+=row['frames']; s['optimizer_seconds']+=row['seconds']
        e=s['exposure'][task]; e['updates']+=1; e['episodes']+=row['episodes']; e['frames']+=row['frames']; e['cells'][cell]+=row['episodes']
        row.update(utc=cloud.original.utc(),step=s['step'],cumulative_episodes=s['episodes'],cumulative_frames=s['frames'],
                   optimizer_seconds=s['optimizer_seconds'],per_task_exposure=cpu_tree(s['exposure']),
                   latest_validation=s['selection_history'][-1] if s['selection_history'] else None,
                   elapsed_cap_seconds=time.time()-self.config['cap_started'],clipping=None,initialization='fresh')
        append_jsonl(self.directory/'progress.jsonl',row)
        print(json.dumps({k:row[k] for k in ('utc','step','task','cell','loss','seconds','cumulative_episodes')}),flush=True)
        if s['step']==13 and not self.config.get('disposable_profile') and (self.directory/'profile/profile.json').exists():
            prof=json.loads((self.directory/'profile/profile.json').read_text())
            ratio=s['optimizer_seconds']/sum(r['seconds'] for r in prof['rows'])
            atomic_json(self.directory/'first_cycle_timing.json',dict(matched_profile_ratio=ratio,material_slowdown=ratio>1.35,updates=13))
        return row


def verify_progress(directory):
    directory=Path(directory)
    initial=load_verified(json.loads((directory/'initial_checkpoint.json').read_text()))
    receipt=json.loads((directory/'latest_checkpoint.json').read_text()); saved=load_verified(receipt)
    assert initial['schema']==saved['schema']==3
    assert initial['state']['step']==initial['scheduler']['updates']==0 and not initial['optimizer']['state']
    assert initial['stream']['streams']==[] and initial['state']['best_checkpoint'] is None
    assert initial['provenance']==saved['provenance']==provenance()
    step=saved['state']['step']; assert step>0 and saved['scheduler']['updates']==step
    assert saved['state']['episodes']==step*saved['state']['config']['effective_batch']
    assert not tree_equal(initial['stream'],saved['stream'])
    changed=[]
    for i,name in enumerate(saved['optimizer_names']):
        expected = saved['state']['exposure'][name.split('.')[1]]['updates'] if name.startswith('heads.') else step
        opt=saved['optimizer']['state'].get(i)
        if expected==0: assert opt is None; continue
        assert opt is not None and float(opt['step'])==expected, name
        assert all(torch.isfinite(v).all() for v in opt.values() if torch.is_tensor(v)), name
        if not torch.equal(initial['model'][name],saved['model'][name]): changed.append(name)
    for prefix in ('blocks.','acc.','spatial_input.','memory.','readout.','heads.'):
        assert any(n.startswith(prefix) for n in changed), prefix
    result=dict(verified=True,step=step,episodes=saved['state']['episodes'],changed_parameters=len(changed),
                checkpoint=receipt,optimizer_seconds=saved['state']['optimizer_seconds'],fresh_initial_optimizer_empty=True)
    atomic_json(directory/'persisted_progress_verification.json',result)
    return result


def selection_key(result):
    if not result['complete'] or len(result['cells'])!=35: return None
    auc=result['summary']['equal_task_mean_auc']
    ba=[r['chance_normalized_ba'] for r in result['summary']['tasks'].values()]
    return [auc,sum(ba)/len(ba)] if auc is not None and None not in ba else None


def evaluate(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
    evaluator=cloud.clone(cloud.original.evaluate.__wrapped__,SuiteStream=FreshStream,sync=cloud.sync)
    with torch.no_grad(): result=evaluator(model,split,n,krauzlis_n,microbatch,device,output,cells=cells,deadline=deadline)
    result.update(namespace=VAL_NAMESPACE if split=='val' else FINAL_NAMESPACE,
                  selection_rule='equal-task mean AUC then equal-task chance-normalized BA; earlier ties; validation only')
    result['summary']['selection_key']=selection_key(result)
    atomic_json(output,result); return result


def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    keys=('cap_started','deadline','hard_deadline','retrieval_reserve_seconds','wall_cap_seconds')
    if any(config.get(k)!=budget[k] for k in keys): raise ValueError('Absolute cap changed')
    if budget['wall_cap_seconds']!=14400 or budget['hard_deadline']-budget['cap_started']!=14400:
        raise ValueError('Require new four-hour pod-creation cap')
    if budget['retrieval_reserve_seconds']!=600 or budget['hard_deadline']-budget['deadline']!=600:
        raise ValueError('Require separate 600-second retrieval reserve')
    if not budget['cap_started']<=now<budget['deadline']: raise ValueError('Cap expired/not started')


def config_for(budget):
    manifest=json.loads((ROOT.parent/'deployment_manifest.json').read_text())
    for row in manifest['files']:
        if digest(ROOT.parent/row['path'])!=row['sha256']: raise ValueError('Deployment digest mismatch: '+row['path'])
    hashes={str(ROOT.parent/r['path']):r['sha256'] for r in manifest['files'] if r['path'].startswith('repo/') and r['path'].endswith(('.py','.md','.json'))}
    config=dict(**budget,protocol=PROTOCOL,device='cuda',effective_batch=32,microbatch=4,eval_microbatch=4,
                val_n=64,val_krauzlis_n=100,test_n=128,test_krauzlis_n=200,checkpoint_every=13,cpu_threads=2,
                source_hashes=hashes,catalog=CATALOG,source_manifest_sha256=digest(ROOT/'PreAttentiveVision/data/bsds500/manifest.json'),
                runtime_root=str(ROOT),initialization=provenance(),baseline='none',selection='equal-task AUC then chance-normalized BA',
                all_trainable=True,lr=1e-4,clipping=None,bptt='full',precision='fp32',tf32=False)
    cloud.original.verify_sources(config); return config


def profile(directory):
    config=json.loads((directory/'profile_config.json').read_text()); cloud.original.verify_sources(config)
    session=Session(directory,config); session.checkpoint('initial.pt'); rows=[]
    for _ in range(13):
        if time.time()>=config['deadline']-60: raise RuntimeError('Bounded profile exhausted')
        rows.append(session.train_update(*session.scheduler.next()))
        if session.state['step'] in (1,2): session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
    session.checkpoint('profile_terminal.pt')
    ev=evaluate(session.model,'val',8,8,4,session.device,directory/'eval_timing.json',config['deadline']-30)
    if not ev['complete'] or len(ev['cells'])!=35: raise RuntimeError('Incomplete 35-cell profile')
    atomic_json(directory/'profile.json',dict(complete=True,architecture=VERSION,rows=rows,evaluation_cells=ev['cells'],
        verification=verify_progress(directory),disposable=True,discard_all_state=True,initialization=provenance(),
        torch_version=torch.__version__,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


def measured_plan(directory,budget,now=None):
    now=time.time() if now is None else now
    prof=json.loads((directory/'profile/profile.json').read_text()); cells=set(cloud.original.all_cells())
    if not prof['complete'] or prof.get('architecture')!=VERSION or len(prof['rows'])!=13 or {r['task'] for r in prof['rows']}!=set(TASKS):
        raise ValueError('Require fresh 13-task profile')
    if len(prof['evaluation_cells'])!=35 or {(r['task'],r['cell']) for r in prof['evaluation_cells']}!=cells:
        raise ValueError('Require full35 evaluation timings')
    ec={(r['task'],r['cell']):r['seconds']/r['n'] for r in prof['evaluation_cells']}
    if not all(math.isfinite(v) and v>0 for v in ec.values()): raise ValueError('Invalid timings')
    tc={r['task']:r['seconds']*max(1.,max(ec[t,c] for t,c in cells if t==r['task'])/ec[r['task'],r['cell']]) for r in prof['rows']}
    val=1.35*sum(ec[t,c]*(100 if t=='krauzlis_cued_motion' else 64) for t,c in cells); test=2*val
    scheduler=BalancedScheduler(SCHEDULER_SEED); schedule=[scheduler.next() for _ in range(TARGET)]
    count=TARGET; overhead=2*val+2*test+900
    while count>=26:
        training=sum(tc[t] for t,c in schedule[:count]); total=1.25*training+overhead
        if total<budget['deadline']-now: break
        count-=13
    if count<26: raise RuntimeError('No complete acquisition allocation fits four-hour cap')
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()}
    for t,c in schedule[:count]: exposure[t]['updates']+=1; exposure[t]['episodes']+=32; exposure[t]['cells'][c]+=32
    return dict(max_steps=count,target_requested=TARGET,validation_steps=[count//26*13,count],total_episodes=count*32,
                exposure_reduced=count<TARGET,updates_per_task=count//13,planned_exposure=exposure,
                estimated_optimizer_seconds=training,estimated_total_remaining_seconds=total,
                slower_150pct_seconds=1.5*training+overhead,estimated_validation_seconds=val,estimated_one_test_seconds=test,
                final_reserve_seconds=val+2*test+180,update_estimate_seconds=1.25*max(tc.values()),
                method='Fresh13 updates + all35 eval8 cells; slowest-cell task ratio; train1.25/eval1.35; 2val+2test+900s; separate600s retrieval; reduce only before production by13')


def run(directory):
    config=json.loads((directory/'config.json').read_text())
    validate_budget(json.loads((directory/'budget.json').read_text()),config); cloud.original.verify_sources(config)
    if (directory/'initial.pt').exists(): raise RuntimeError('Fresh-only: no restart/overwrite')
    session=Session(directory,config); session.checkpoint('initial.pt')
    atomic_json(directory/'fresh_initialization.json',dict(verified=True,checkpoint=session.latest,**provenance()))
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete_cycles'; failure=None; terminal=None; terminal_test=None; selected_test=None
    def validate():
        session.status('validation')
        result=evaluate(session.model,'val',64,100,4,session.device,directory/f"validation_{session.state['step']:06d}.json",
                        config['deadline']-2*config['estimated_one_test_seconds']-120)
        key=selection_key(result); s=session.state
        s['selection_history'].append(dict(step=s['step'],key=key,complete=result['complete']))
        name=f"validation_checkpoint_{s['step']:06d}.pt"
        if key is not None and (s['best_key'] is None or tuple(key)>tuple(s['best_key'])):
            s.update(best_key=key,best_step=s['step'],best_checkpoint=str(directory/name))
        session.checkpoint(name)
    try:
        while session.state['step']<config['max_steps']:
            if stopped[0]: reason='signal'; break
            if time.time()+config['final_reserve_seconds']+config['update_estimate_seconds']>=config['deadline']:
                reason='wall_budget_reserve'; break
            row=session.train_update(*session.scheduler.next()); step=session.state['step']
            if step in (1,2) or step%config['checkpoint_every']==0: session.checkpoint(f'checkpoint_{step:06d}.pt')
            session.status('training',last_update={k:row[k] for k in ('task','cell','loss','seconds')})
            if step in config['validation_steps'] and not stopped[0]: validate()
        terminal=session.checkpoint('terminal.pt')
        if session.state['step']>0 and not stopped[0]:
            history=session.state['selection_history']
            if (not history or history[-1]['step']!=session.state['step']) and len(history)<2 and time.time()+config['final_reserve_seconds']<config['deadline']:
                validate(); terminal=session.checkpoint('terminal_validated.pt')
            if time.time()<config['deadline']-60:
                session.status('final_test_terminal'); different=session.state['best_step']!=session.state['step']
                reserve=config['estimated_one_test_seconds'] if different else 0
                terminal_test=evaluate(session.model,'test',128,200,4,session.device,directory/'test_terminal.json',config['deadline']-reserve-60)
                if not different:
                    winner=cloud.trusted_load(session.state['best_checkpoint'])
                    if not tree_equal(winner['model'],cpu_tree(session.model.state_dict())): raise RuntimeError('Same-step model mismatch')
                    selected_test=dict(reused_terminal=True,identical_model_verified=True,results=terminal_test)
                    atomic_json(directory/'test_selected.json',selected_test)
                elif session.state['best_checkpoint'] is not None and time.time()<config['deadline']-60:
                    session.model.load_state_dict(cloud.trusted_load(session.state['best_checkpoint'])['model'])
                    session.status('final_test_selected')
                    selected_test=evaluate(session.model,'test',128,200,4,session.device,directory/'test_selected.json',config['deadline']-45)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()); reason='worker_error'
        atomic_json(directory/'failure.json',failure)
        # Preserve only the last fully verified checkpoint, never partial updates.
        if terminal is None: terminal=session.latest
    selected=selected_test.get('results',selected_test) if selected_test else None
    complete=bool(terminal_test and terminal_test['complete'] and selected and selected['complete'])
    persisted=load_verified(terminal)
    report=dict(protocol=PROTOCOL,architecture=VERSION,initialization=provenance(),stop_reason=reason,failure=failure,
                terminal_checkpoint=terminal,terminal_step=persisted['state']['step'],selected_step=persisted['state']['best_step'],
                selection_history=persisted['state']['selection_history'],episodes=persisted['state']['episodes'],
                exposure=persisted['state']['exposure'],optimizer_seconds=persisted['state']['optimizer_seconds'],
                config=config,terminal_test=terminal_test,selected_test=selected_test,final_coverage_complete=complete,
                finished_utc=cloud.original.utc(),limitations=['One fresh seed; no convergence or superiority claim.',
                'Original BSDS identity splits; new draws are not new image identities.',
                'Empty recognition specificity/FPR separate; all Krauzlis event subgroups retained.'])
    atomic_json(directory/'report.json',report)
    lines=['# Fresh no-CLS convolutional-decoder training',f"Updates {report['terminal_step']}; episodes {report['episodes']}; selected {report['selected_step']}; stop {reason}."]
    for label,result in [('Terminal',terminal_test),('Selected',selected)]:
        lines += [f'\n## {label}','|Task|Cell|n|BA|AUC|Specificity|FPR|','|---|---|---:|---:|---:|---:|---:|']
        for r in result['cells'] if result else []:
            lines.append('|'+ '|'.join(str(r.get(k)) for k in ('task','cell','n','balanced_accuracy','auc','specificity','false_positive_rate'))+'|')
            if r['task']=='krauzlis_cued_motion': lines.append('\nEvents: `'+json.dumps(r['events'],sort_keys=True)+'`')
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    session.status('finished',stop_reason=reason,failure=failure,final_coverage_complete=complete)
    if report['terminal_step']>0: verify_progress(directory)
    return 0 if failure is None and complete and report['terminal_step']==config['max_steps'] else 2


def completion(directory,status,error=None):
    report=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {}
    latest=json.loads((directory/'latest_checkpoint.json').read_text()) if (directory/'latest_checkpoint.json').exists() else None
    terminal=report.get('terminal_checkpoint'); selected=None
    if terminal:
        saved=load_verified(terminal); source=saved['state']['best_checkpoint']
        if source: shutil.copy2(source,directory/'selected.pt'); selected=str(directory/'selected.pt')
    if latest: shutil.copy2(latest['path'],directory/'latest.pt')
    paths=[p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md') and p.name!='cloud_completion.json']
    paths += [p for p in (directory/'selected.pt',directory/'latest.pt',Path(terminal['path']) if terminal else directory/'terminal.pt') if p.is_file()]
    files=[dict(path=str(p),relative_path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]
    allocation=json.loads((directory/'allocation.json').read_text()) if (directory/'allocation.json').exists() else {}
    atomic_json(directory/'cloud_completion.json',dict(status=status,error=error,target_updates=TARGET,
                pinned_updates=allocation.get('max_steps'),actual_updates=latest['step'] if latest else 0,
                selected_checkpoint=selected,terminal_checkpoint=terminal,latest=latest,artifact_manifest=files,finished_utc=cloud.original.utc()))


def supervisor(directory,budget):
    validate_budget(budget,budget); directory.mkdir(parents=True,exist_ok=True)
    with (directory/'activation.json').open('x') as f:
        json.dump(dict(supervisor_pid=os.getpid(),**budget),f,indent=2); f.flush(); os.fsync(f.fileno())
    atomic_json(directory/'budget.json',budget); outcomes=[]
    try:
        config=config_for(budget); pd=directory/'profile'; pd.mkdir()
        pc=dict(config,disposable_profile=True,deadline=min(budget['deadline'],time.time()+1800))
        atomic_json(pd/'profile_config.json',pc); os.chdir(ROOT)
        command=[sys.executable,'-u','-m','SecondPass.SpatialRecurrentConvDecoder.worker']
        outcome=cloud.supervise(command+['profile',str(pd)],pc['deadline'],pd,prefix='profile'); outcomes.append(outcome)
        if outcome['returncode']!=0: raise RuntimeError('Disposable profile failed')
        plan=measured_plan(directory,budget); atomic_json(directory/'allocation.json',plan)
        config.update(plan); atomic_json(directory/'config.json',config)
        os.chmod(directory/'allocation.json',0o444); os.chmod(directory/'config.json',0o444)
        outcome=cloud.supervise(command+['run',str(directory)],budget['deadline'],directory,prefix='production'); outcomes.append(outcome)
        result=dict(status='complete' if outcome['returncode']==0 else 'incomplete',outcomes=outcomes,**budget)
    except Exception as exc:
        result=dict(status='error',error=repr(exc),traceback=traceback.format_exc(),outcomes=outcomes,**budget)
    atomic_json(directory/'supervisor_result.json',result); completion(directory,result['status'],result.get('error'))
    return result


def main():
    cloud.original.cpu_setup()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['supervise','profile','run','verify']); p.add_argument('directory',type=Path)
    p.add_argument('--budget',type=Path,help='Parent-supplied immutable four-hour pod-creation budget')
    args=p.parse_args(); directory=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(directory),indent=2)); return
    if args.mode=='supervise':
        if args.budget is None: p.error('--budget required')
        result=supervisor(directory,json.loads(args.budget.read_text())); print(json.dumps(result))
        raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Require exactly one CUDA GPU')
    with (ROOT.parent/'convdecoder_fresh_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))


if __name__=='__main__': main()
