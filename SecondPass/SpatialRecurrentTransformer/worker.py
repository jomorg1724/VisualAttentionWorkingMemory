"""Single-arm adapter of proven CloudRun; no infrastructure or credential code."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import torch
from SecondPass.SpatialComparisonReadout.CloudRun import worker as cloud
from SecondPass.SpatialReadout import state as native
from SecondPass.SpatialReadout import worker as inherited
from SecondPass.SpatialComparisonReadout.experiment import selection_key
from SecondPass.JointTraining.core import BalancedScheduler, atomic_json, append_jsonl, cpu_tree, tree_equal
from SecondPass.TaskSuite.suite import TASKS, SuiteStream, task_classes
from .model import SpatialRecurrentTransformer, VERSION

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = VERSION
INIT_SEED = 96592763
CUDA_SEED = 96692763
VAL_NAMESPACE = 96792763
FINAL_NAMESPACE = 96892763
TARGET = 2600
digest = cloud.digest
load_verified = cloud.load_verified
save_payload = cloud.save_payload
restore = cloud.restore


def source_payload(pointer, allow_fixture=False):
    pointer = Path(pointer)
    receipt = json.loads(pointer.read_text())
    if receipt.get('schema') != 1: raise ValueError('Source pointer schema must be 1')
    fixture = receipt.get('fixture_only', False)
    if fixture and not allow_fixture: raise ValueError('Fixture forbidden in production')
    if not fixture and (receipt.get('step'), receipt.get('cumulative_step'), receipt.get('scheduler_updates')) != (2600, 9360, 12753):
        raise ValueError('Require completed comparison terminal: added2600/cumulative9360/global12753; never early117')
    path = Path(receipt['path'])
    if not path.is_absolute(): path = pointer.parent/path
    receipt['path'] = str(path.resolve())
    source = load_verified(receipt)
    if source['schema'] != 2 or source['state']['step'] != receipt['step'] or source['scheduler']['updates'] != receipt['scheduler_updates'] or source['scheduler']['tasks']:
        raise ValueError('Source schema/exposure/task-cycle mismatch')
    if source['state']['episodes'] != receipt['step']*32: raise ValueError('Source episode mismatch')
    if not fixture:
        if receipt['sha256']!='e47031f3c66fefdb174384ad442688554966f3ce2cc86a9562e2ec99640ae72c':
            raise ValueError('Unauthorized production parent SHA256')
        if source['state']['parent']['step'] != 6760 or not any(n.startswith('comparator.') for n in source['model']):
            raise ValueError('Not comparison terminal parent6760')
        if source['state']['step']+source['state']['parent']['step']!=receipt['cumulative_step']:
            raise ValueError('Cumulative exposure does not match actual checkpoint')
    required = ('model','optimizer','optimizer_names','scheduler','stream','rng','state')
    if any(k not in source for k in required): raise ValueError('Incomplete source')
    if any(k not in source['rng'] for k in ('cpu','numpy','python','mps')): raise ValueError('Missing inherited RNG')
    receipt['verified'] = True
    return source, receipt


def migrate(source, receipt, device):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(INIT_SEED).get_state())
        model = SpatialRecurrentTransformer(task_classes())
    weights = cpu_tree(model.state_dict()); carried = []
    for name in weights:
        if name.startswith(('blocks.','proj.','acc.','heads.','spatial_input.')):
            if name not in source['model'] or source['model'][name].shape != weights[name].shape:
                raise ValueError('Inherited name/shape mismatch: '+name)
            weights[name] = source['model'][name].clone(); carried.append(name)
    names = [n for n,_ in model.named_parameters()]
    opt = native.map_adam(source['optimizer'], source['optimizer_names'], source['model'], names, weights)
    opt = cloud.portable_adam_schema(opt)
    g = opt['param_groups'][0]
    if (g['lr'], g['betas'], g['eps'], g['weight_decay']) != (1e-4,(.9,.999),1e-8,0) or g.get('decoupled_weight_decay',False):
        raise ValueError('One unchanged Adam1e-4 recipe required')
    state = dict(step=0, episodes=0, frames=0, optimizer_seconds=0., best_step=None, best_key=None, best_checkpoint=None,
        selection_history=[], parent=cpu_tree(source['state']),
        exposure={t:dict(updates=0,episodes=0,frames=0,cells={c['id']:0 for c in spec['conditions']}) for t,spec in TASKS.items()})
    rng = cpu_tree(source['rng'])
    transition = dict(policy='fresh CUDA generator for new architecture; inherited CUDA/MPS states are provenance, not replay equivalence',seed=CUDA_SEED)
    if str(device).startswith('cuda'):
        if torch.cuda.device_count()!=1: raise ValueError('Require one CUDA device')
        if 'cuda' in rng: transition['parent_cuda_rng'] = cpu_tree(rng['cuda'])
        rng['cuda'] = [torch.Generator(device=device).manual_seed(CUDA_SEED).get_state()]
    migration = dict(source=receipt, architecture=VERSION, initialization_seed=INIT_SEED, carried_names=carried,
        fresh_names=[n for n in names if n not in carried], removed_names=[n for n in source['model'] if n not in weights],
        inherited_scheduler_updates=source['scheduler']['updates'], inherited_cumulative_step=receipt['cumulative_step'],
        selection_reset=True, profile_state_inherited=False, functionally_identical=False, cuda_transition=transition,
        adam_schema_adapter='CloudRun explicit decoupled_weight_decay=False; exact compatible name/shape moments')
    return dict(schema=2,model=weights,optimizer=opt,optimizer_names=names,state=state,
        scheduler=cpu_tree(source['scheduler']),stream=cpu_tree(source['stream']),rng=rng,migration=migration)


class Session(inherited.TrainingSession):
    def __init__(self,directory,config,payload):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config;self.device=config['device']
        self.model=SpatialRecurrentTransformer(task_classes()).to(self.device)
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.scheduler=BalancedScheduler(0);self.stream=SuiteStream('train')
        self.state=restore(payload,self.model,self.optimizer,self.scheduler,self.stream,self.device)
        self.state.update(config=config,deadline=config['deadline']);self.migration=payload['migration'];self.latest=None
        self.provenance_rng={k:cpu_tree(v) for k,v in payload['rng'].items() if k not in ('cpu','numpy','python','cuda')}
        if not str(self.device).startswith('cuda') and 'cuda' in payload['rng']:self.provenance_rng['cuda']=cpu_tree(payload['rng']['cuda'])
        self.initial_new={n:payload['model'][n].clone() for n in self.migration['fresh_names']}

    def checkpoint(self,filename):
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        payload=native.snapshot(self.model,self.optimizer,self.scheduler,self.stream,self.state,self.migration,self.device)
        payload['rng'].update(self.provenance_rng)
        if str(self.device).startswith('cuda'):payload['rng']['cuda']=torch.cuda.get_rng_state_all()
        self.latest=save_payload(self.directory/filename,payload)
        atomic_json(self.directory/'latest_checkpoint.json',self.latest)
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=cloud.original.utc(),**self.latest))
        if self.state['step'] in (2,13) and filename.startswith('checkpoint_'):verify_progress(self.directory)
        return self.latest

    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=cloud.original.utc(),
            architecture=VERSION,branch_step=self.state['step'],branch_episodes=self.state['episodes'],
            parent_step=self.migration['inherited_cumulative_step'],source_branch_step=self.state['parent']['step'],
            cumulative_lineage_step=self.migration['inherited_cumulative_step']+self.state['step'],
            global_optimizer_step=self.migration['inherited_scheduler_updates']+self.state['step'],
            optimizer_seconds=self.state['optimizer_seconds'],deadline=self.config['deadline'],latest_checkpoint=self.latest,
            best_step=self.state['best_step'],**extra))

    def train_update(self,task,cell):
        def write_progress(path,row):
            parent=self.migration['inherited_cumulative_step']
            row.update(source_branch_parent_step=self.state['parent']['step'],parent_step=parent,parent_episodes=parent*32,
                cumulative_lineage_steps=parent+row['step'],cumulative_lineage_episodes=parent*32+row['cumulative_episodes'],
                global_optimizer_step=self.migration['inherited_scheduler_updates']+row['step'],architecture=VERSION)
            append_jsonl(path,row)
        row=cloud.clone(cloud.native_train_update,append_jsonl=write_progress)(self,task,cell)
        step=self.state['step']
        if step in (1,2,13):
            evidence={}
            for n,p in self.model.named_parameters():
                if n not in self.initial_new:continue
                norm=float(p.grad.detach().norm().cpu()) if p.grad is not None else None
                changed=not torch.equal(p.detach().cpu(),self.initial_new[n])
                adam=float(self.optimizer.state[p]['step'])
                if norm is None or not (0<norm<float('inf')) or not changed or adam!=step:raise ValueError('New core/CLS gradient or update missing: '+n)
                evidence[n]=dict(gradient_norm=norm,changed=changed,adam_step=adam)
            append_jsonl(self.directory/'new_core_progress.jsonl',dict(step=step,parameters=evidence,verified=True))
        if step==13 and not self.config.get('disposable_profile'):
            prof=json.loads((self.directory/'profile'/'profile.json').read_text())
            rows=[json.loads(x) for x in (self.directory/'progress.jsonl').read_text().splitlines()]
            ratio=sum(r['seconds'] for r in rows)/sum(r['seconds'] for r in prof['rows'])
            atomic_json(self.directory/'first_cycle_timing.json',dict(matched_profile_ratio=ratio,material_slowdown=ratio>1.35,updates=13))
        return row


def verify_progress(directory):
    directory=Path(directory)
    initial=cloud.trusted_load(directory/'migration.pt')
    receipt=json.loads((directory/'latest_checkpoint.json').read_text())
    saved=load_verified(receipt);step=saved['state']['step'];names=saved['optimizer_names']
    assert step>0 and saved['scheduler']['updates']==initial['scheduler']['updates']+step
    assert saved['state']['episodes']==step*saved['state']['config']['effective_batch']
    assert not tree_equal(initial['stream'],saved['stream'])
    changed=[]
    for n in initial['migration']['fresh_names']:
        i=names.index(n)
        assert i not in initial['optimizer']['state'] and float(saved['optimizer']['state'][i]['step'])==step
        assert not torch.equal(initial['model'][n],saved['model'][n]),n
        changed.append(n)
    n='spatial_input.weight';i=names.index(n)
    assert float(saved['optimizer']['state'][i]['step'])==float(initial['optimizer']['state'][i]['step'])+step
    assert not torch.equal(initial['model'][n],saved['model'][n])
    result=dict(verified=True,additional_step=step,cumulative_lineage_step=initial['migration']['inherited_cumulative_step']+step,
        new_changed=changed,checkpoint=receipt,optimizer_seconds=saved['state']['optimizer_seconds'])
    atomic_json(directory/'persisted_progress_verification.json',result)
    return result


class EvaluationStream(SuiteStream):
    def __init__(self,split):
        if split not in ('val','test'):raise ValueError('Evaluation only')
        super().__init__(split)
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return (VAL_NAMESPACE if self.split=='val' else FINAL_NAMESPACE)*100000+TASKS[task]['stream_id']*1000+index


def evaluate(model,split,n,krauzlis_n,microbatch,device,output,deadline,cells=None):
    fn=cloud.clone(cloud.original.evaluate.__wrapped__,SuiteStream=EvaluationStream,sync=cloud.sync)
    with torch.no_grad():result=fn(model,split,n,krauzlis_n,microbatch,device,output,cells=cells,deadline=deadline)
    result.update(namespace=VAL_NAMESPACE if split=='val' else FINAL_NAMESPACE,selection_rule='target8 mean AUC then equal-task mean AUC; earlier ties; validation only')
    result['summary']['selection_key']=selection_key(result)
    atomic_json(output,result)
    return result


def validate_budget(budget,config,now=None):
    now=time.time() if now is None else now
    keys=('cap_started','deadline','hard_deadline','retrieval_reserve_seconds')
    if any(config.get(k)!=budget[k] for k in keys):raise ValueError('Absolute cap changed')
    if budget['hard_deadline']-budget['deadline']!=budget['retrieval_reserve_seconds'] or budget['retrieval_reserve_seconds']<0:
        raise ValueError('Invalid retrieval reserve')
    if not budget['cap_started']<=now<budget['deadline']:raise ValueError('Cap expired/not started')


def config_for(source,receipt,pointer,budget):
    # Dependency-closed package manifest is verified before any accelerator use.
    manifest=json.loads((ROOT.parent/'deployment_manifest.json').read_text())
    hashes={str(ROOT.parent/r['path']):r['sha256'] for r in manifest['files'] if r['path'].startswith('repo/') and r['path'].endswith(('.py','.json','.md'))}
    config=dict(**budget,protocol=PROTOCOL,device='cuda',effective_batch=32,microbatch=4,eval_microbatch=4,
        val_n=64,val_krauzlis_n=100,test_n=128,test_krauzlis_n=200,checkpoint_every=13,cpu_threads=2,
        source_hashes=hashes,catalog=source['state']['config']['catalog'],source_manifest_sha256=source['state']['config']['source_manifest_sha256'],
        parent_pointer=str(Path(pointer).resolve()),parent_receipt=receipt,runtime_root=str(ROOT),
        validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,baseline='none; changed architecture',
        selection='target8 mean AUC then suite equal-task AUC; scheduled midpoint/terminal only; earlier ties',
        profile_state_inherited=False,all_trainable=True,lr=1e-4,clipping=None,bptt='full',precision='fp32',tf32=False)
    cloud.original.verify_sources(config)
    return config


def profile(directory):
    config=json.loads((directory/'profile_config.json').read_text())
    cloud.original.verify_sources(config)
    source,receipt=source_payload(config['parent_pointer'])
    session=Session(directory,config,migrate(source,receipt,config['device']))
    initial=load_verified(session.checkpoint('migration.pt'));rows=[]
    for _ in range(13):
        if time.time()>=config['deadline']-60:raise RuntimeError('Bounded profile exhausted')
        rows.append(session.train_update(*session.scheduler.next()))
        if session.state['step'] in (1,2):session.checkpoint(f"checkpoint_{session.state['step']:06d}.pt")
    final=session.checkpoint('profile_terminal.pt')
    before=torch.rand(16,device=config['device']).cpu()
    restore(load_verified(final),session.model,session.optimizer,session.scheduler,session.stream,config['device'])
    assert torch.equal(before,torch.rand(16,device=config['device']).cpu()),'Device RNG replay failed'
    ev=evaluate(session.model,'val',8,8,4,config['device'],directory/'eval_timing.json',config['deadline']-30)
    if not ev['complete'] or len(ev['cells'])!=35:raise RuntimeError('Incomplete 35-cell timings')
    atomic_json(directory/'profile.json',dict(complete=True,architecture=VERSION,rows=rows,evaluation_cells=ev['cells'],
        verification=verify_progress(directory),disposable=True,discard_all_state=True,torch_version=torch.__version__,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(config['device']).startswith('cuda') else None))


def measured_plan(directory,source,budget,now=None):
    now=time.time() if now is None else now
    prof=json.loads((directory/'profile'/'profile.json').read_text())
    cells=set(cloud.original.all_cells())
    if not prof['complete'] or prof.get('architecture')!=VERSION or len(prof['rows'])!=13 or {r['task'] for r in prof['rows']}!=set(TASKS):
        raise ValueError('Require this architecture fresh 13-task profile')
    if len(prof['evaluation_cells'])!=35 or {(r['task'],r['cell']) for r in prof['evaluation_cells']}!=cells:
        raise ValueError('Require this architecture 35-cell eval timings')
    ec={(r['task'],r['cell']):r['seconds']/r['n'] for r in prof['evaluation_cells']}
    # No old-model timings: conservatively assign each task its observed update
    # time inflated by that task's slowest measured evaluation-cell ratio.
    tc={r['task']:r['seconds']*max(1.,max(ec[t,c] for t,c in cells if t==r['task'])/ec[r['task'],r['cell']]) for r in prof['rows']}
    val=1.35*sum(ec[t,c]*(100 if t=='krauzlis_cued_motion' else 64) for t,c in cells)
    test=2*val
    scheduler=BalancedScheduler(0);scheduler.load_state_dict(source['scheduler'])
    schedule=[scheduler.next() for _ in range(TARGET)]
    training=sum(tc[t] for t,c in schedule);overhead=2*val+2*test+900
    total=1.25*training+overhead
    if total>=budget['deadline']-now:raise RuntimeError('Fixed2600 exposure does not fit unchanged deadline; no automatic reduction/extension')
    exposure={t:dict(updates=0,episodes=0,cells={c['id']:0 for c in spec['conditions']}) for t,spec in TASKS.items()}
    for t,c in schedule:exposure[t]['updates']+=1;exposure[t]['episodes']+=32;exposure[t]['cells'][c]+=32
    return dict(max_steps=TARGET,additional_target_requested=TARGET,validation_steps=[1300,2600],total_episodes=TARGET*32,
        planned_exposure=exposure,estimated_optimizer_seconds=training,estimated_total_remaining_seconds=total,
        slower_150pct_seconds=1.5*training+overhead,estimated_validation_seconds=val,estimated_one_test_seconds=test,
        final_reserve_seconds=val+2*test+180,update_estimate_seconds=1.25*max(tc.values()),
        method='New-model13 native updates; each task inflated by slowest/observed35-cell eval8 per-episode ratio; train1.25/eval1.35 margins; 2val+2test+900s; fixed2600, never shrink or extend')


def run(directory):
    config=json.loads((directory/'config.json').read_text())
    source,receipt=source_payload(config['parent_pointer']);payload=migrate(source,receipt,config['device'])
    fn=cloud.clone(inherited.run,validate_budget=validate_budget,require_approval=lambda *_:None,
        verify_parent=lambda _:(payload,receipt),migrate=lambda p,_:p,TrainingSession=Session,
        evaluate=evaluate,selection_key=selection_key,PROTOCOL=PROTOCOL,torch=cloud.TORCH,load_verified=load_verified,restore=restore)
    code=fn(directory)
    report=json.loads((directory/'report.json').read_text())
    report.update(parent_step=receipt['cumulative_step'],source_branch_step=receipt['step'],parent_episodes=receipt['cumulative_step']*32,
        cumulative_lineage_step=receipt['cumulative_step']+report['terminal_step'],architecture=VERSION,baseline='none; changed architecture',
        limitations=['One warm-start architecture, not a causal superiority test or convergence claim.',
            'Inherited CNN/KDA/spatial_input/heads and compatible named Adam; fresh transformer, persistent CLS, CLS-only readout.',
            'No same-model parent baseline or stale profile. Profile weights/Adam/streams/RNG discarded before production.',
            'All weights trainable, one Adam1e-4, fp32 full BPTT, no clipping, unchanged native tasks/official BSDS identity splits.',
            'Fresh validation/test namespaces; target8 validation selection then equal-task mean AUC. All35 cells, N0 specificity and Krauzlis events retained.'])
    atomic_json(directory/'report.json',report)
    selected=report['selected_test'];selected=selected.get('results',selected) if selected else None
    lines=['# Recurrent convolutional transformer with recurrent CLS',f"Additional updates {report['terminal_step']}; selected {report['selected_step']}; stop {report['stop_reason']}."]
    for label,result in [('Terminal',report['terminal_test']),('Selected',selected)]:
        lines += [f'\n## {label}','|Task|Cell|n|BA|AUC|Specificity|FPR|','|---|---|---:|---:|---:|---:|---:|']
        for r in result['cells'] if result else []:
            lines.append('|'+ '|'.join(str(r.get(k)) for k in ('task','cell','n','balanced_accuracy','auc','specificity','false_positive_rate'))+'|')
            if r['task']=='krauzlis_cued_motion':lines.append('\nEvent strata: `'+json.dumps(r.get('events',{}),sort_keys=True)+'`\n')
    lines+=['\n## Limitations']+['- '+x for x in report['limitations']]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if report['terminal_step']>0:verify_progress(directory)
    return code


def supervisor(directory,pointer,budget):
    validate_budget(budget,budget)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'activation.json').open('x') as f:
        json.dump(dict(supervisor_pid=os.getpid(),**budget),f,indent=2);f.flush();os.fsync(f.fileno())
    atomic_json(directory/'budget.json',budget)
    outcomes=[]
    try:
        source,receipt=source_payload(pointer)
        atomic_json(directory/'source_receipt.json',receipt)
        config=config_for(source,receipt,pointer,budget)
        pd=directory/'profile';pd.mkdir()
        pc=dict(config,disposable_profile=True,deadline=min(budget['deadline'],time.time()+1800))
        atomic_json(pd/'profile_config.json',pc)
        # Explicit cwd fixes child module lookup; no inherited shell path guess.
        os.chdir(ROOT)
        command=[sys.executable,'-u','-m','SecondPass.SpatialRecurrentTransformer.worker','profile',str(pd)]
        outcome=cloud.supervise(command,pc['deadline'],pd,prefix='profile');outcomes.append(outcome)
        if outcome['returncode']!=0:raise RuntimeError('New-model bounded profile failed')
        plan=measured_plan(directory,source,budget)
        atomic_json(directory/'allocation.json',plan);config.update(plan);atomic_json(directory/'config.json',config)
        os.chmod(directory/'allocation.json',0o444);os.chmod(directory/'config.json',0o444)
        command=[sys.executable,'-u','-m','SecondPass.SpatialRecurrentTransformer.worker','run',str(directory)]
        outcome=cloud.supervise(command,budget['deadline'],directory,prefix='production');outcomes.append(outcome)
        result=dict(status='complete' if outcome['returncode']==0 else 'incomplete',outcomes=outcomes,**budget)
    except Exception as exc:
        import traceback
        result=dict(status='error',error=repr(exc),traceback=traceback.format_exc(),outcomes=outcomes,**budget)
    atomic_json(directory/'supervisor_result.json',result)
    cloud.completion(directory,result['status'],result.get('error'))
    return result


def main():
    cloud.original.cpu_setup()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['supervise','profile','run','verify','verify-source'])
    p.add_argument('directory',type=Path)
    p.add_argument('--source-pointer',type=Path)
    p.add_argument('--budget',type=Path,help='Immutable parent-owned absolute budget JSON; no default or renewal')
    args=p.parse_args();directory=args.directory.resolve()
    if args.mode=='verify-source':
        _,receipt=source_payload(args.source_pointer);print(json.dumps(receipt,indent=2));return
    if args.mode=='verify':print(json.dumps(verify_progress(directory),indent=2));return
    if args.mode=='supervise':
        if args.source_pointer is None or args.budget is None:p.error('--source-pointer and --budget required')
        result=supervisor(directory,args.source_pointer.resolve(),json.loads(args.budget.read_text()))
        print(json.dumps(result));raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Require exactly one CUDA GPU')
    lock=ROOT.parent/'recurrent_transformer_worker.lock'
    with lock.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile':profile(directory)
        else:raise SystemExit(run(directory))


if __name__=='__main__':main()
