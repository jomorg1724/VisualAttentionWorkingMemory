"""Versioned CUDA adapters; frozen native sources remain byte-identical."""
from pathlib import Path
import types
import torch

PROTOCOL = 'spatial_comparison_cloud_candidate_v1'


def trusted_load(path, **kwargs):
    """Only locally packaged/checksummed experiment checkpoints, not public files."""
    kwargs.setdefault('map_location', 'cpu')
    return torch.load(path, weights_only=False, **kwargs)


class TorchCompat:
    load = staticmethod(trusted_load)

    def __getattr__(self, name):
        return getattr(torch, name)


def clone(fn, **scope):
    result = types.FunctionType(fn.__code__, dict(fn.__globals__, **scope), fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import numpy as np
from SecondPass.SpatialComparisonReadout import experiment as exp
from SecondPass.SpatialReadout import state as native_state
from SecondPass.SpatialReadout import worker as inherited
from SecondPass.JointTraining import worker as original
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, append_jsonl, cpu_tree, tree_equal, score_cell
from SecondPass.TaskSuite.suite import TASKS, SuiteStream, task_classes
from SecondPass.SpatialReadout.continuation_v2 import verify_coverage
from SecondPass.SpatialReadout.protocol import digest, supervise

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(os.environ.get('VAWM_CLOUD_ASSETS', str(ROOT.parent/'assets')))
CUDA_SEED = 95492763
TORCH = TorchCompat()
load_verified = clone(native_state.load_verified, torch=TORCH)
save_payload = clone(native_state.save_payload, load_verified=load_verified)
verify_progress = clone(exp.verify_progress, torch=TORCH, load_verified=load_verified)


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    elif str(device) == 'mps': torch.mps.synchronize()


update = clone(original.update, sync=sync)
ORIGINAL = types.SimpleNamespace(**dict(vars(original), update=update))
native_train_update = clone(inherited.TrainingSession.train_update, original=ORIGINAL)


def source_payload():
    path = ASSETS/'terminal6760.pt'
    if not path.exists(): path = exp.SOURCE
    if digest(path) != exp.SOURCE_SHA: raise ValueError('Unauthorized source checkpoint')
    source = trusted_load(path)
    if source['state']['step'] != 6760 or source['scheduler']['updates'] != 10153 or source['scheduler']['tasks']:
        raise ValueError('Wrong source exposure/queue')
    config = copy.deepcopy(source['state']['config'])
    old = Path(config['runtime_root'])
    config['source_hashes'] = {str(ROOT/Path(p).relative_to(old)): h for p,h in config['source_hashes'].items()}
    original.verify_sources(config)
    return source, dict(path=str(path), sha256=exp.SOURCE_SHA, bytes=path.stat().st_size, step=6760, verified=True)


def portable_adam_schema(saved):
    """Make Torch 2.8's implicit Adam default explicit, preserving all state."""
    result = copy.deepcopy(saved)
    if 'decoupled_weight_decay' in inspect.signature(torch.optim.Adam).parameters:
        for group in result['param_groups']:
            group.setdefault('decoupled_weight_decay', False)
    return result


def migrate(source, receipt, device):
    payload = exp.migrate(source, receipt, 'candidate')
    payload['optimizer'] = portable_adam_schema(payload['optimizer'])
    payload['migration']['adam_schema_adapter'] = 'v1: explicit legacy decoupled_weight_decay=False when supported; moments/steps/options unchanged'
    payload['migration'].update(cloud_protocol=PROTOCOL, cuda_seed=CUDA_SEED,
        cuda_rng_policy='fresh seeded CUDA generator on MPS-to-CUDA transition; restore saved CUDA state on cloud resume',
        inherited_mps_rng_retained_as_provenance=True)
    if str(device).startswith('cuda'):
        payload['rng']['cuda'] = [torch.Generator(device=device).manual_seed(CUDA_SEED).get_state()]
    return payload


def restore(payload, model, optimizer, scheduler, stream, device):
    state = native_state.restore(payload, model, optimizer, scheduler, stream, device)
    if str(device).startswith('cuda'):
        if len(payload['rng'].get('cuda', [])) != torch.cuda.device_count(): raise ValueError('CUDA RNG topology/state missing')
        torch.cuda.set_rng_state_all(payload['rng']['cuda'])
    return state


class Session(inherited.TrainingSession):
    def __init__(self, directory, config, payload):
        self.directory = Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config; self.device=config['device']
        self.model=exp.ComparisonReadout(task_classes()).to(self.device)
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.scheduler=BalancedScheduler(0); self.stream=SuiteStream('train')
        self.state=restore(payload,self.model,self.optimizer,self.scheduler,self.stream,self.device)
        self.state['config']=config; self.state['deadline']=config['deadline']
        self.migration=payload['migration']; self.latest=None
        self.inherited_mps_rng=payload['rng']['mps']
        self.initial_new={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if n.startswith('comparator.')}
        if not config.get('disposable_profile') and self.state['step']==0:
            baseline=json.loads((ASSETS/'baseline_validation.json').read_text())
            verify_coverage(baseline,64,100)
            if baseline.get('namespace') != exp.VAL_NAMESPACE: raise ValueError('Baseline namespace mismatch')
            key=exp.selection_key(baseline)
            if key is None: raise ValueError('Incomplete baseline')
            self.state.update(best_key=key,best_step=0,best_checkpoint=str(self.directory/'migration.pt'),
                selection_history=[dict(step=0,key=key,complete=True,source='reused_baseline',
                    sha256=digest(ASSETS/'baseline_validation.json'),new_peek=False)])

    def checkpoint(self, filename):
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        payload=native_state.snapshot(self.model,self.optimizer,self.scheduler,self.stream,self.state,self.migration,self.device)
        payload['rng']['mps']=self.inherited_mps_rng
        if str(self.device).startswith('cuda'): payload['rng']['cuda']=torch.cuda.get_rng_state_all()
        self.latest=save_payload(self.directory/filename,payload)
        atomic_json(self.directory/'latest_checkpoint.json',self.latest)
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=original.utc(),**self.latest))
        if self.state['step'] in (2,13) and filename.startswith('checkpoint_'):
            verify_progress(self.directory)
        return self.latest

    def train_update(self, task, cell):
        row=native_train_update(self,task,cell)
        step=self.state['step']
        if step in (1,2,13):
            evidence={}
            for n,p in self.model.named_parameters():
                if n not in self.initial_new: continue
                norm=float(p.grad.detach().square().sum().cpu().sqrt()) if p.grad is not None else None
                changed=not torch.equal(p.detach().cpu(),self.initial_new[n])
                adam=float(self.optimizer.state[p]['step'])
                if norm is None or not np.isfinite(norm) or adam!=step: raise ValueError('Comparator gradient/Adam invalid')
                if step==1 and n.startswith('comparator.0.') and (norm!=0 or changed): raise ValueError('Initial residual gradient violated')
                if step>=2 and (norm<=0 or not changed): raise ValueError('Comparator did not learn')
                evidence[n]=dict(gradient_norm=norm,changed=changed,adam_step=adam)
            append_jsonl(self.directory/'comparator_progress.jsonl',dict(step=step,parameters=evidence,verified=True))
        if step==13 and not self.config.get('disposable_profile'):
            prof=json.loads((self.directory/'profile'/'profile.json').read_text())
            rows=[json.loads(x) for x in (self.directory/'progress.jsonl').read_text().splitlines()]
            ratio=sum(r['seconds'] for r in rows)/sum(r['seconds'] for r in prof['rows'])
            atomic_json(self.directory/'first_cycle_timing.json',dict(matched_profile_ratio=ratio,material_slowdown=ratio>1.35,updates=13))
        return row


def evaluate(model, split, n, krauzlis_n, microbatch, device, output, deadline, cells=None):
    evaluator=clone(original.evaluate.__wrapped__,SuiteStream=exp.EvaluationStream,sync=sync)
    with torch.no_grad(): result=evaluator(model,split,n,krauzlis_n,microbatch,device,output,cells=cells,deadline=deadline)
    result.update(namespace=exp.VAL_NAMESPACE if split=='val' else exp.FINAL_NAMESPACE,
                  selection_rule='target8 mean AUC then suite equal-task AUC; earlier ties; validation only')
    result['summary']['selection_key']=exp.selection_key(result)
    atomic_json(output,result)
    return result


def budget_from_creation(created):
    return dict(cap_started=created, hard_deadline=created+28800., deadline=created+28200., wall_cap_seconds=28800.,
                retrieval_reserve_seconds=600., origin='Pod creation; setup/profile/train/evaluation/retrieval; no renewal')


def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    if budget!=budget_from_creation(budget['cap_started']) or any(config.get(k)!=v for k,v in budget.items()):
        raise ValueError('Cloud cap changed')
    if not budget['cap_started']<=now<budget['deadline']: raise ValueError('Cloud compute cap expired')


def config_for(source,budget):
    config=exp.common_config(Path('.'),budget,source,'candidate')
    config.update(protocol=PROTOCOL,device='cuda',runtime_root=str(ROOT),worker_lock=str(ROOT.parent/'run'/'worker.lock'),
                  cpu_threads=2,selection='target8 then suite AUC; reused baseline plus scheduled midpoint/terminal only',
                  precision='fp32',tf32=False,all_trainable=True,report=str(ROOT.parent/'run'/'REPORT.md'))
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):config['source_hashes'][str(p)]=digest(p)
    return config


def profile(directory):
    config=json.loads((directory/'profile_config.json').read_text())
    source,receipt=source_payload()
    session=Session(directory,config,migrate(source,receipt,'cuda'))
    initial=session.checkpoint('migration.pt'); rows=[]
    for _ in range(13):
        if time.time()>=config['deadline']-60:raise RuntimeError('Disposable profile deadline exhausted')
        rows.append(session.train_update(*session.scheduler.next()))
        if session.state['step'] in (1,2):session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
    final=session.checkpoint('profile_terminal.pt')
    before=torch.rand(16,device='cuda').cpu()
    restore(load_verified(final),session.model,session.optimizer,session.scheduler,session.stream,'cuda')
    assert torch.equal(before,torch.rand(16,device='cuda').cpu()), 'CUDA RNG failed exact roundtrip'
    ev=evaluate(session.model,'val',8,8,4,'cuda',directory/'eval_timing.json',config['deadline']-30)
    if not ev['complete']:raise RuntimeError('Incomplete profile timing')
    result=dict(complete=True,rows=rows,evaluation_cells=ev['cells'],cuda_rng_replay=True,verification=verify_progress(directory),
                disposable=True,discard_all_state=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__)
    atomic_json(directory/'profile.json',result)


def measured_plan(directory,source,budget):
    prof=json.loads((directory/'profile'/'profile.json').read_text())
    if not prof['complete'] or len(prof['rows'])!=13:raise ValueError('Incomplete profile')
    historical=json.loads((ASSETS/'historical_costs.json').read_text())
    old={(r['task'],r['cell']):r['seconds'] for r in historical['cells']}
    factors={r['task']:r['seconds']/old[r['task'],r['cell']] for r in prof['rows']}
    costs={k:v*factors[k[0]] for k,v in old.items()}
    evalcost={(r['task'],r['cell']):r['seconds']/r['n'] for r in prof['evaluation_cells']}
    val=1.35*sum(evalcost[t,c]*(100 if t=='krauzlis_cued_motion' else 64) for t,c in costs)
    test=2*val
    scheduler=BalancedScheduler(0);scheduler.load_state_dict(source['scheduler'])
    schedule=[scheduler.next() for _ in range(2600)]
    count=2600;overhead=2*val+2*test+900
    while count>=26:
        training=sum(costs[x] for x in schedule[:count]);total=1.25*training+overhead
        if total<budget['deadline']-time.time():break
        count-=13
    if count<26:raise RuntimeError('Insufficient candidate acquisition budget')
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in s['conditions']}) for t,s in TASKS.items()}
    for t,c in schedule[:count]:
        exposure[t]['updates']+=1;exposure[t]['episodes']+=32;exposure[t]['cells'][c]+=32
    return dict(max_steps=count,additional_target_requested=2600,validation_steps=[count//26*13,count],
        total_episodes=count*32,planned_exposure=exposure,estimated_optimizer_seconds=training,
        estimated_total_remaining_seconds=total,slower_150pct_seconds=1.5*training+overhead,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values()),timing_factors=factors,
        method='All prior 5070 per-cell train timings scaled by same-task matched CUDA cycle; all35 CUDA validation8 timings; 1.25 train/1.35 eval margins; 900s overhead; 600s separate retrieval reserve')


def run(directory):
    source,receipt=source_payload();payload=migrate(source,receipt,'cuda')
    fn=clone(inherited.run,validate_budget=validate_budget,require_approval=lambda *_:None,
        verify_parent=lambda _:(payload,receipt),migrate=lambda p,_:p,TrainingSession=Session,
        evaluate=evaluate,selection_key=exp.selection_key,PROTOCOL=PROTOCOL,torch=TORCH,
        load_verified=load_verified,restore=restore)
    code=fn(directory)
    report=json.loads((directory/'report.json').read_text())
    report.update(parent_step=6760,parent_episodes=216320,cumulative_convgru_step=6760+report['terminal_step'],
        inherited_global_updates=10153,arm='candidate',baseline='reused_baseline; no new baseline peek',
        limitations=['Single warm-start candidate; no concurrent control, causal attribution or superiority claim.',
            'All learned parameters trainable; unchanged native stimuli/cues/labels, Adam1e-4, fp32 full BPTT.',
            'MPS-to-CUDA transition is not bitwise-equivalent training; CPU/NumPy/Python/stream inherited, CUDA RNG fresh then checkpointed.',
            'Fresh final draws relative to completed ConvGRU; fixed existing comparison namespace, unchanged source identity splits.',
            'All35 conditions reported; N0 specificity separate, Krauzlis target/foil/catch retained; selection validation-only.'])
    atomic_json(directory/'report.json',report)
    selected=report['selected_test'];selected=selected.get('results',selected) if selected else None
    lines=['# Cloud candidate: learned spatial comparison',f"Additional updates: {report['terminal_step']}; selected: {report['selected_step']}; stop: {report['stop_reason']}.",
        'Single-arm warm-start acquisition, not a controlled comparison. Baseline validation reused without a new peek.']
    for label,result in [('Terminal',report['terminal_test']),('Selected',selected)]:
        lines += [f'\n## {label}', '|Task|Cell|n|BA|AUC|Specificity|FPR|','|---|---|---:|---:|---:|---:|---:|']
        for r in result['cells'] if result else []:
            lines.append('|'+ '|'.join(str(r.get(k)) for k in ('task','cell','n','balanced_accuracy','auc','specificity','false_positive_rate'))+'|')
        for r in result['cells'] if result else []:
            if r['task']=='krauzlis_cued_motion':lines += [f"\nKrauzlis {r['cell']} event strata: `"+json.dumps(r.get('events',{}),sort_keys=True)+'`']
    lines+=['\n## Limitations']+['- '+x for x in report['limitations']]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    verify_progress(directory)
    return code


def completion(directory, status, error=None):
    report_path=directory/'report.json';report=json.loads(report_path.read_text()) if report_path.exists() else {}
    latest_path=directory/'latest_checkpoint.json';latest=json.loads(latest_path.read_text()) if latest_path.exists() else None
    selected_path=None;terminal_path=None
    if report.get('terminal_checkpoint'):
        terminal_path=report['terminal_checkpoint']['path']
        saved=trusted_load(terminal_path)
        selected_path=saved['state']['best_checkpoint']
        if selected_path:shutil.copy2(selected_path,directory/'selected.pt');selected_path=str(directory/'selected.pt')
    if latest:shutil.copy2(latest['path'],directory/'latest.pt')
    paths=[p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md') and p.name!='cloud_completion.json']
    paths += [p for p in (directory/'selected.pt',directory/'latest.pt',Path(terminal_path) if terminal_path else directory/'terminal.pt') if p.is_file()]
    manifest=[dict(path=str(p),relative_path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]
    atomic_json(directory/'cloud_completion.json',dict(status=status,error=error,target_updates=2600,
        pinned_updates=json.loads((directory/'allocation.json').read_text()).get('max_steps') if (directory/'allocation.json').exists() else None,
        actual_updates=latest['step'] if latest else 0,candidate_selected=selected_path,candidate_terminal=terminal_path,
        latest=latest,artifact_manifest=manifest,finished_utc=original.utc()))


def supervisor(directory,created):
    directory.mkdir(parents=True,exist_ok=True)
    budget=budget_from_creation(created);validate_budget(budget,budget)
    with (directory/'activation.json').open('x') as f:
        json.dump(dict(supervisor_pid=os.getpid(),parent_pid=os.getppid(),**budget),f,indent=2);f.flush();os.fsync(f.fileno())
    atomic_json(directory/'budget.json',budget)
    outcomes=[]
    try:
        source,receipt=source_payload();atomic_json(directory/'source_receipt.json',receipt)
        atomic_json(directory/'deployment_manifest.json',json.loads((ROOT.parent/'deployment_manifest.json').read_text()))
        config=config_for(source,budget)
        pd=directory/'profile';pd.mkdir()
        pc=dict(config,disposable_profile=True);pc['deadline']=min(budget['deadline'],time.time()+1800)
        atomic_json(pd/'profile_config.json',pc)
        command=[sys.executable,'-u','-m','SecondPass.SpatialComparisonReadout.CloudRun.worker','profile',str(pd)]
        outcome=supervise(command,pc['deadline'],pd,prefix='profile');outcomes.append(outcome)
        if outcome['returncode']!=0:raise RuntimeError('Candidate profile failed')
        plan=measured_plan(directory,source,budget);atomic_json(directory/'allocation.json',plan)
        config.update(plan);atomic_json(directory/'config.json',config)
        os.chmod(directory/'config.json',0o444);os.chmod(directory/'allocation.json',0o444)
        command=[sys.executable,'-u','-m','SecondPass.SpatialComparisonReadout.CloudRun.worker','run',str(directory)]
        outcome=supervise(command,budget['deadline'],directory,prefix='production');outcomes.append(outcome)
        status='complete' if outcome['returncode']==0 else 'incomplete'
        result=dict(status=status,outcomes=outcomes,**budget)
    except Exception as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc())
        atomic_json(directory/'supervisor_error.json',error)
        result=dict(status='error',outcomes=outcomes,**error,**budget)
    atomic_json(directory/'supervisor_result.json',result)
    completion(directory,result['status'],result.get('error'))
    return result


def main():
    original.cpu_setup()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['supervise','profile','run','verify']);p.add_argument('directory');p.add_argument('--created',type=float)
    args=p.parse_args();directory=Path(args.directory).resolve()
    if args.mode=='supervise':
        result=supervisor(directory,args.created);print(json.dumps(result));return
    if args.mode=='verify':print(json.dumps(verify_progress(directory),indent=2));return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Require exactly one CUDA GPU')
    directory.mkdir(parents=True,exist_ok=True)
    with (ROOT.parent/'run'/'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile':profile(directory)
        else:raise SystemExit(run(directory))


if __name__=='__main__': main()
