"""Local pretrained-VAE sensory encoder plus fresh full-BPTT RViT classifier."""
import argparse, copy, json, math, os, random, signal, time, types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from SecondPass.SingleStimulusRViT import worker as replay
from SecondPass.SingleStimulusRViT.stimuli import SingleStimulusStream
from SecondPass.DelayedFrameGRU import worker as local
from SecondPass.ThreeFrameConvVAE.resume_worker import tensor_digest
from .model import VAERViT

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.VAERViTInput10.worker'
VERSION='vae9900_mu_sliding_triplet_rvit_input10'
PROTOCOL=VERSION+'_new_classification_pool1000_epoch10_mps_v1'
TARGET=990
TASK,TASKS,CELLS=replay.TASK,replay.TASKS,replay.CELLS
INIT_SEED,SCHEDULER_SEED=replay.INIT_SEED,replay.SCHEDULER_SEED
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=183101,183102,183103
atomic_json,append_jsonl,cpu_tree,tree_equal,digest,load_verified=replay.atomic_json,replay.append_jsonl,replay.cpu_tree,replay.tree_equal,replay.digest,replay.load_verified
verify_sources,validate_budget=local.verify_sources,local.validate_budget
validation_steps_for=replay.validation_steps_for

class FreshStream(SingleStimulusStream):
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return dict(train=TRAIN_NAMESPACE,val=VAL_NAMESPACE,test=FINAL_NAMESPACE)[self.split]*100000+TASKS[task]['stream_id']*1000+index

class ReplayStream(replay.ReplayStream):
    stream_seed=FreshStream.stream_seed
    _new_pool=replay.clone(replay.ReplayStream._new_pool,base=types.SimpleNamespace(FreshStream=FreshStream))
    load_state_dict=replay.clone(replay.ReplayStream.load_state_dict,base=types.SimpleNamespace(FreshStream=FreshStream))

evaluate=torch.no_grad()(replay.clone(replay.original.evaluate.__wrapped__,FreshStream=FreshStream,score_cell=replay.score_cell,
    sync=local.sync,VAL_NAMESPACE=VAL_NAMESPACE,FINAL_NAMESPACE=FINAL_NAMESPACE))
selection_key=replay.selection_key

def provenance(config):
    return dict(architecture=VERSION,input_scale=10.0,input_scale_location='after spatial positions and token_norm, before recurrent block',initialization='VAE9900 encoder and mu head transferred; whole recurrence and decoder fresh',
        encoder_checkpoint=config['encoder_checkpoint'],encoder_sha256=config['encoder_sha256'],encoder_source_step=9900,
        inherited_encoder=True,inherited_recurrent=False,inherited_classifier=False,inherited_optimizer=False,
        inherited_rng=False,inherited_stream=False,profile_state_inherited=False,all_trainable=True,
        initialization_seed=INIT_SEED,train_namespace=TRAIN_NAMESPACE,validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,
        training='1000 fresh movies, ten shuffled epochs, then replace; final binary crossentropy only',
        stimulus='existing no-cue single-stimulus29/37/45frames, native26/28changes; change57/nochange43',
        checkpoint_policy='only best.pt and latest.pt')

class Session:
    def __init__(self,directory,config):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True); self.config=config; self.device=config['device']; self.latest=None
        torch.manual_seed(INIT_SEED); random.seed(INIT_SEED); np.random.seed(INIT_SEED)
        if self.device=='mps': torch.mps.manual_seed(INIT_SEED)
        self.model=VAERViT(checkpoint_encoder=True,checkpoint_recurrent=True)
        transfer=self.model.load_pretrained_encoder(config['encoder_checkpoint'],expected_sha256=config['encoder_sha256'])
        # One direct constructor comparison binds every fresh core parameter; no initial checkpoint file.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(INIT_SEED)
            direct=VAERViT(checkpoint_encoder=True,checkpoint_recurrent=True)
            direct.load_pretrained_encoder(config['encoder_checkpoint'],expected_sha256=config['encoder_sha256'])
        assert tree_equal(self.model.state_dict(),direct.state_dict())
        names=[n for n,_ in self.model.named_parameters()]
        initial_hashes={n:tensor_digest(p) for n,p in self.model.named_parameters()}
        del direct
        self.model=self.model.to(self.device)
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.stream=ReplayStream(); self.scheduler=replay.ReplayScheduler(self.stream)
        self.state=dict(step=0,episodes=0,frames=0,optimizer_seconds=0.,best_step=None,best_key=None,best_checkpoint=None,
            selection_history=[],config=config,unique_episodes_generated=0,unique_episodes_presented=0,replay_presentations=0,
            exposure={TASK:dict(updates=0,episodes=0,frames=0,cells={c:0 for c in CELLS})})
        assert not self.optimizer.state and not self.stream.state_dict()['streams']
        atomic_json(self.directory/'initialization.json',dict(verified=True,provenance=provenance(config),transfer=transfer,
            parameter_count=sum(p.numel() for p in self.model.parameters()),optimizer_names=names,initial_parameter_hashes=initial_hashes,
            direct_constructor_equality=True,empty_adam=True,zero_classification_counters=True,empty_streams=True))

    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=local.utc(),architecture=VERSION,
            step=self.state['step'],episodes=self.state['episodes'],optimizer_seconds=self.state['optimizer_seconds'],
            deadline=self.config['deadline'],latest_checkpoint=self.latest,best_step=self.state['best_step'],best_key=self.state['best_key'],
            pool_index=self.stream.pool_index,epoch=self.stream.epoch,unique_episodes_generated=self.state['unique_episodes_generated'],
            latest_validation=self.state['selection_history'][-1] if self.state['selection_history'] else None,**extra))

    def checkpoint(self):
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        if self.device=='mps': rng['mps']=torch.mps.get_rng_state()
        payload=cpu_tree(dict(schema=3,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),state=self.state,
            optimizer_names=[n for n,_ in self.model.named_parameters()],scheduler=self.scheduler.state_dict(),stream=self.stream.state_dict(),
            rng=rng,provenance=provenance(self.config)))
        path=self.directory/'latest.pt'; temporary=self.directory/'latest.pt.tmp'
        with temporary.open('wb') as stream: torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        receipt=dict(path=str(path),step=self.state['step'],bytes=path.stat().st_size,sha256=digest(path),verified=True)
        assert tree_equal(payload,load_verified(receipt)); self.latest=receipt; atomic_json(self.directory/'latest_checkpoint.json',receipt)
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=local.utc(),**receipt))
        if self.state['step'] in (1,2,3):
            proof=verify_progress(self.directory); atomic_json(self.directory/f"optimizer_proof_{self.state['step']:06d}.json",proof)
            if self.state['step']==3: atomic_json(self.directory/'startup_ready.json',dict(verified=True,checkpoint=receipt,persisted_optimizer_evidence=proof))
        return receipt

    def promote_best(self):
        temporary=self.directory/'best.pt.tmp'; os.link(self.directory/'latest.pt',temporary); os.replace(temporary,self.directory/'best.pt')
        receipt=dict(self.latest,path=str(self.directory/'best.pt')); assert digest(receipt['path'])==receipt['sha256']
        atomic_json(self.directory/'best_checkpoint.json',receipt)

    def train_update(self,task,cell):
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        images,labels,actual_cell,indices=self.stream.current_batch(); assert task==TASK and cell==actual_cell
        actual=len(labels); local.sync(self.device); started=time.perf_counter(); loss_sum=0.
        for i in range(0,actual,self.config['microbatch']):
            x,y=images[i:i+self.config['microbatch']].to(self.device),labels[i:i+self.config['microbatch']].to(self.device)
            loss=F.cross_entropy(self.model(x,task),y)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite loss')
            weight=len(y)/actual; (loss*weight).backward(); loss_sum+=float(loss.detach())*weight
        norms=[]
        for name,p in self.model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise FloatingPointError('invalid gradient '+name)
            norms.append(float(p.grad.square().sum()))
        norm=math.sqrt(sum(norms)); assert math.isfinite(norm)
        self.optimizer.step(); local.sync(self.device)
        assert all(torch.isfinite(p).all() for p in self.model.parameters())
        self.stream.commit(cell,indices); elapsed=time.perf_counter()-started; s=self.state; frames=actual*images.shape[1]
        s['step']+=1; s['episodes']+=actual; s['frames']+=frames; s['optimizer_seconds']+=elapsed
        s.update(unique_episodes_generated=(self.stream.pool_index+1)*1000,unique_episodes_presented=self.stream.pool_index*1000+len(self.stream.used),replay_presentations=s['episodes'])
        e=s['exposure'][TASK]; e['updates']+=1; e['episodes']+=actual; e['frames']+=frames; e['cells'][cell]+=actual
        row=dict(utc=local.utc(),step=s['step'],task=task,cell=cell,loss=loss_sum,seconds=elapsed,episodes=actual,frames=frames,
            grad_norm=norm,cumulative_episodes=s['episodes'],pool_index=self.stream.pool_index,epoch=self.stream.epoch,
            unique_episodes_generated=s['unique_episodes_generated'],unique_episodes_presented=s['unique_episodes_presented'],
            pool_generation_seconds=self.stream.generation_seconds,partial_batch=actual<32)
        append_jsonl(self.directory/'progress.jsonl',row); print(json.dumps(row),flush=True); return row


def verify_progress(directory):
    directory=Path(directory); initial=json.loads((directory/'initialization.json').read_text())
    receipt=json.loads((directory/'latest_checkpoint.json').read_text()); saved=load_verified(receipt); step=saved['state']['step']
    names=saved['optimizer_names']; assert names==initial['optimizer_names'] and saved['provenance']==initial['provenance']
    assert step>0 and step==receipt['step']==saved['scheduler']['updates']
    assert set(saved['optimizer']['state'])==set(range(len(names))); changed=[]
    for i,name in enumerate(names):
        value=saved['model'][name]; opt=saved['optimizer']['state'][i]
        assert torch.isfinite(value).all() and float(opt['step'])==step
        assert all(torch.isfinite(opt[k]).all() and opt[k].shape==value.shape for k in ('exp_avg','exp_avg_sq'))
        if tensor_digest(value)!=initial['initial_parameter_hashes'][name]: changed.append(name)
    assert len(changed)==len(names)
    r=saved['stream']['replay']; assert r['presentations']==saved['state']['episodes']==saved['state']['replay_presentations']
    assert r['presentation_counts']==saved['state']['exposure'][TASK]['cells']
    assert sum(m['state']['counts'][TASK] for m in saved['stream']['streams'])==saved['state']['unique_episodes_generated']
    assert saved['rng']['mps'].numel()>0
    result=dict(verified=True,step=step,checkpoint=receipt,episodes=saved['state']['episodes'],parameter_count=initial['parameter_count'],
        changed_parameter_tensors=len(changed),all_trainable_parameters_changed=True,all_named_adam_steps_verified=True,
        direct_constructor_and_encoder_transfer_verified=initial['direct_constructor_equality'])
    atomic_json(directory/'persisted_progress_verification.json',result); return result


def profile(directory):
    cfg=json.loads((directory/'profile_config.json').read_text()); verify_sources(cfg); session=Session(directory,cfg); warm=[]; rows=[]
    for i in range(6):
        if time.time()>=cfg['deadline']-60: raise RuntimeError('Profile deadline')
        row=session.train_update(*session.scheduler.next()); (warm if i<3 else rows).append(row)
    ev=evaluate(session.model,'val',20,1,session.device,directory/'eval_timing.json',cfg['deadline']-30); assert ev['complete']
    atomic_json(directory/'profile.json',dict(complete=True,rows=rows,warmup_rows=warm,evaluation_cells=ev['cells'],
        disposable=True,discard_all_state=True,mps_driver_bytes=torch.mps.driver_allocated_memory(),
        optimizer_tensors=len(session.optimizer.state),optimizer_updates=session.state['step'],checkpoints_written=False))


def measured_plan(directory,budget):
    local.validate_budget(budget,budget); p=json.loads((directory/'profile/profile.json').read_text()); assert p['complete'] and len(p['rows'])==3
    costs={r['cell']:r['seconds'] for r in p['rows']}; assert set(costs)==set(CELLS)
    val=1.35*sum(r['seconds']/r['n'] for r in p['evaluation_cells'])*100; test=2*val
    generation=sum(r['pool_generation_seconds'] for r in p['warmup_rows']); rng=random.Random(SCHEDULER_SEED+71)
    schedule=[]; train=0.; best=None; now=time.time()
    for i in range(TARGET):
        if i%330==0: sizes={c:333+int(j==(i//330)%3) for j,c in enumerate(CELLS)}
        if i%33==0: schedule.extend(replay.epoch_batches(sizes,rng))
        train+=costs[schedule[i][0]]; gen=(i//330+1)*generation
        total=1.25*(train+gen)+len(validation_steps_for(i+1))*val+2*test+900
        if total+60<budget['deadline']-now: best=i+1
    if best is None or best<330: raise RuntimeError('One complete replay pool cannot fit authorized cap')
    count=best//330*330; train=sum(costs[c] for c,_ in schedule[:count]); gen=count//330*generation
    return dict(max_steps=count,target_requested=TARGET,exposure_reduced=count<TARGET,
        total_episodes=sum(len(ix) for _,ix in schedule[:count]),planned_unique_movies=count//330*1000,
        validation_steps=validation_steps_for(count),estimated_total_remaining_seconds=1.25*(train+gen)+len(validation_steps_for(count))*val+2*test+900,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values())+generation)


def run(directory):
    cfg=json.loads((directory/'config.json').read_text()); validate_budget(json.loads((directory/'budget.json').read_text()),cfg); verify_sources(cfg)
    if (directory/'initialization.json').exists(): raise RuntimeError('No restart')
    session=Session(directory,cfg); stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete'; failure=None; final=None; selected=None
    def validate():
        session.status('validation'); result=evaluate(session.model,'val',100,1,session.device,directory/f"validation_{session.state['step']:06d}.json",cfg['deadline']-2*cfg['estimated_one_test_seconds']-120)
        key=selection_key(result); state=session.state
        state['selection_history'].append(dict(step=state['step'],key=key,cells=result['cells'],complete=result['complete']))
        better=bool(result['complete'] and (state['best_key'] is None or tuple(key)>tuple(state['best_key'])))
        if better: state.update(best_step=state['step'],best_key=key,best_checkpoint=str(directory/'best.pt'))
        session.checkpoint()
        if better: session.promote_best()
        session.status('validation_complete')
    try:
        while session.state['step']<cfg['max_steps']:
            if stopped[0]: reason='signal'; break
            if time.time()+cfg['final_reserve_seconds']+cfg['update_estimate_seconds']>=cfg['deadline']: reason='wall_budget_reserve'; break
            row=session.train_update(*session.scheduler.next()); step=session.state['step']
            if step in (1,2,3) or step%100==0: session.checkpoint()
            session.status('training',last_loss=row['loss'])
            if step in cfg['validation_steps'] and not stopped[0]: validate()
        if session.state['step']>0 and not stopped[0] and (not session.state['selection_history'] or session.state['selection_history'][-1]['step']!=session.state['step']): validate()
        session.checkpoint()
        if session.state['step']>0 and not stopped[0]:
            session.status('final_test_terminal'); final=evaluate(session.model,'test',200,1,session.device,directory/'test_terminal.json',cfg['deadline']-cfg['estimated_one_test_seconds']-60)
            if session.state['best_step']==session.state['step']:
                selected=dict(final,reused_terminal=True); atomic_json(directory/'test_selected.json',selected)
            elif session.state['best_checkpoint']:
                saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False); session.model.load_state_dict(saved['model'])
                selected=evaluate(session.model,'test',200,1,session.device,directory/'test_selected.json',cfg['deadline']-60)
    except Exception as exc:
        import traceback
        failure=dict(error=repr(exc),traceback=traceback.format_exc()); atomic_json(directory/'failure.json',failure); reason='worker_error'
    complete=bool(final and selected and final['complete'] and selected['complete'])
    if complete: assert all(a['trial_ids']==b['trial_ids'] for a,b in zip(final['cells'],selected['cells']))
    actual=session.state['step']; report=dict(protocol=PROTOCOL,architecture=VERSION,initialization=provenance(cfg),stop_reason=reason,failure=failure,
        terminal_checkpoint=session.latest,terminal_step=actual,selected_step=session.state['best_step'],pinned_updates=cfg['max_steps'],
        exposure_completed=actual==cfg['max_steps'],cap_limited=reason=='wall_budget_reserve',episodes=session.state['episodes'],
        unique_movies_generated=session.state['unique_episodes_generated'],selection_history=session.state['selection_history'],
        terminal_test=final,selected_test=selected,final_coverage_complete=complete,config=cfg)
    atomic_json(directory/'report.json',report); session.status('finished',stop_reason=reason,final_coverage_complete=complete)
    return 0 if not failure and complete and reason in ('planned_complete','wall_budget_reserve') else 2


def completion(directory,status,error=None):
    report=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {}
    latest=json.loads((directory/'latest_checkpoint.json').read_text()) if (directory/'latest_checkpoint.json').exists() else None
    best=json.loads((directory/'best_checkpoint.json').read_text()) if (directory/'best_checkpoint.json').exists() else None
    paths=[p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md','.pt') and p.name!='cloud_completion.json']
    files=[dict(path=str(p),relative_path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)]
    atomic_json(directory/'cloud_completion.json',dict(status=status,error=error,target_updates=TARGET,pinned_updates=report.get('pinned_updates'),
        actual_updates=latest['step'] if latest else 0,exposure_completed=report.get('exposure_completed',False),cap_limited=report.get('cap_limited',False),
        selected_checkpoint=best['path'] if best else None,terminal_checkpoint=latest,latest=latest,artifact_manifest=files,finished_utc=local.utc()))

supervise=replay.clone(local.supervise,MODULE=MODULE,completion=completion)

def local_supervise(directory):
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'budget.json').exists(): raise RuntimeError('No cap renewal')
    start=time.time(); budget=dict(cap_started=start,deadline=start+28200,hard_deadline=start+28800,retrieval_reserve_seconds=600,wall_cap_seconds=28800,execution_placement='local')
    with (directory/'budget.json').open('x') as stream: json.dump(budget,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
    (directory/'budget.json').chmod(0o444); atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),**budget))
    manifest=json.loads((ROOT/'runtime_manifest.json').read_text())
    cfg=dict(budget,protocol=PROTOCOL,device='mps',source_hashes=manifest['source_hashes'],encoder_checkpoint=manifest['encoder_checkpoint'],encoder_sha256=manifest['encoder_sha256'],
        effective_batch=32,microbatch=1,eval_microbatch=1,cpu_threads=2,lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,precision='fp32',clipping=None,bptt='full',checkpoint_every=100)
    cfg['source_hashes'][cfg['encoder_checkpoint']]=cfg['encoder_sha256']; cfg['source_hashes'][str(ROOT/'runtime_manifest.json')]=digest(ROOT/'runtime_manifest.json')
    verify_sources(cfg); profile_directory=directory/'profile'; profile_directory.mkdir()
    atomic_json(profile_directory/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],start+3600)))
    outcomes=[]
    try:
        outcome=supervise(directory,'profile'); outcomes.append(outcome)
        if outcome['returncode']!=0: raise RuntimeError('Native profile failed')
        plan=measured_plan(directory,budget); cfg.update(plan)
        for name,value in [('allocation.json',plan),('config.json',cfg)]:
            with (directory/name).open('x') as stream: json.dump(value,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
            (directory/name).chmod(0o444)
        outcome=supervise(directory,'run'); outcomes.append(outcome); result=dict(status=outcome['status'],outcomes=outcomes,**budget)
    except Exception as exc:
        result=dict(status='error',error=repr(exc),outcomes=outcomes,**budget); atomic_json(directory/'failure.json',result); completion(directory,'incomplete',result['error'])
    atomic_json(directory/'local_supervisor_result.json',result); return result


def main():
    local.cpu_setup(); parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['local-supervise','profile','run','verify']); parser.add_argument('directory',type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode=='local-supervise':
        result=local_supervise(directory); print(json.dumps(result),flush=True); raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    import fcntl
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))
if __name__=='__main__': main()
