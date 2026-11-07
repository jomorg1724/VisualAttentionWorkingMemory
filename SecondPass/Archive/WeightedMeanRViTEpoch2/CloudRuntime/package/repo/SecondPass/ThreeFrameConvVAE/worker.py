"""Bounded fresh MPS VAE training on unlabeled native single-stimulus triplets."""
import argparse, copy, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from SecondPass.SingleStimulusRViT import worker as replay
from SecondPass.SingleStimulusRViT.stimuli import SingleStimulusStream
from SecondPass.DelayedFrameGRU import worker as local
from .model import ThreeFrameConvVAE, losses

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.ThreeFrameConvVAE.worker'
VERSION='three_frame_conv_vae_spatial256x13x13'
PROTOCOL=VERSION+'_fresh_single_stimulus_pool1000_epoch10_local_v1'
TASK,TASKS,CELLS=replay.TASK,replay.TASKS,replay.CELLS
INIT_SEED,CUDA_SEED,SCHEDULER_SEED=replay.INIT_SEED,replay.CUDA_SEED,replay.SCHEDULER_SEED
TARGET=10000
EXPECTED_PARAMETERS=9507913
BETA=1e-4
atomic_json,append_jsonl,cpu_tree,tree_equal=replay.atomic_json,replay.append_jsonl,replay.cpu_tree,replay.tree_equal
digest,load_verified=replay.digest,replay.load_verified
validate_budget,verify_sources=local.validate_budget,local.verify_sources
clone=replay.clone


def provenance():
    return dict(architecture=VERSION,parameter_count=EXPECTED_PARAMETERS,initialization='whole VAE direct fresh constructor',
        initialization_seed=INIT_SEED,mps_seed=INIT_SEED,inherited_weights=False,inherited_optimizer=False,
        inherited_rng=False,inherited_stream=False,profile_state_inherited=False,
        pool_size=1000,replay_epochs=10,updates_per_pool=330,latent_shape=[256,13,13],beta=BETA,
        objective='three-frame reconstruction plus mean Gaussian KL; no response labels',
        sampling='each movie presentation uniformly draws a fresh ordered three-frame window wholly within active frames7..T-2')


def validation_steps_for(count):
    return sorted(set(([100] if count>=100 else [])+([250] if count>=250 else [])+list(range(500,count+1,500))+[count]))

class TripletStream(replay.ReplayStream):
    def __init__(self):
        super().__init__(); self.window_rng=random.Random(SCHEDULER_SEED+811); self.window_starts=[]
    def current_triplets(self):
        cell,indices=self.current; data=[]; self.window_starts=[]
        for index in indices:
            movie=self.data[cell][index][0]
            start=self.window_rng.randrange(7,len(movie)-3)
            data.append(movie[start:start+3]); self.window_starts.append(start)
        return torch.stack(data),cell,indices
    def state_dict(self):
        result=super().state_dict(); result['replay']['window_rng']=self.window_rng.getstate()
        return result
    def load_state_dict(self,state):
        super().load_state_dict(state); self.window_rng.setstate(state['replay']['window_rng'])

class Session:
    def __init__(self,directory,config):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config; self.device=config['device']; self.latest=None
        torch.manual_seed(INIT_SEED); np.random.seed(INIT_SEED); random.seed(INIT_SEED)
        if self.device=='mps': torch.mps.manual_seed(INIT_SEED)
        self.model=ThreeFrameConvVAE().to(self.device)
        assert sum(p.numel() for p in self.model.parameters())==EXPECTED_PARAMETERS
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.stream=TripletStream(); self.scheduler=replay.ReplayScheduler(self.stream)
        self.state=dict(step=0,episodes=0,frames=0,optimizer_seconds=0.,best_step=None,best_key=None,best_checkpoint=None,
            selection_history=[],config=config,unique_episodes_generated=0,unique_episodes_presented=0,replay_presentations=0,
            exposure={TASK:dict(updates=0,episodes=0,frames=0,cells={c:0 for c in CELLS})})
    checkpoint=clone(local.Session.checkpoint,DelayedFrameGRU=ThreeFrameConvVAE,parameter_count=lambda:EXPECTED_PARAMETERS,
        provenance=provenance,verify_progress=lambda d:verify_progress(d),VERSION=VERSION)
    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=local.utc(),architecture=VERSION,
            initialization='fresh',step=self.state['step'],episodes=self.state['episodes'],triplet_presentations=self.state['episodes'],
            optimizer_seconds=self.state['optimizer_seconds'],deadline=self.config['deadline'],latest_checkpoint=self.latest,
            best_step=self.state['best_step'],best_key=self.state['best_key'],
            latest_validation=self.state['selection_history'][-1] if self.state['selection_history'] else None,
            pool_index=self.stream.pool_index,epoch=self.stream.epoch,
            unique_episodes_generated=self.state['unique_episodes_generated'],unique_episodes_presented=self.state['unique_episodes_presented'],**extra))
    def train_update(self,task,cell):
        x,actual_cell,indices=self.stream.current_triplets(); assert cell==actual_cell and task==TASK
        actual=len(x); micro=self.config['microbatch']; self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        local.sync(self.device); started=time.perf_counter(); averages={}
        for i in range(0,actual,micro):
            target=x[i:i+micro].to(self.device); output=self.model(target)
            metrics=losses(output['reconstruction'],target,output['mu'],output['logvar'],beta=BETA)
            if any(not bool(torch.isfinite(v)) for v in metrics.values()): raise FloatingPointError('nonfinite VAE loss')
            weight=len(target)/actual; (metrics['loss']*weight).backward()
            for k,v in metrics.items(): averages[k]=averages.get(k,0.)+float(v.detach())*weight
        norms={}
        for name,p in self.model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise FloatingPointError('missing/nonfinite gradient '+name)
            norms[name]=float(p.grad.square().sum())
        norm=math.sqrt(sum(norms.values()))
        if not math.isfinite(norm): raise FloatingPointError('nonfinite gradient norm')
        self.optimizer.step(); local.sync(self.device)
        assert all(torch.isfinite(p).all() for p in self.model.parameters())
        self.stream.commit(cell,indices); elapsed=time.perf_counter()-started; s=self.state
        s['step']+=1; s['episodes']+=actual; s['frames']+=3*actual; s['optimizer_seconds']+=elapsed
        s.update(unique_episodes_generated=(self.stream.pool_index+1)*1000,
            unique_episodes_presented=self.stream.pool_index*1000+len(self.stream.used),replay_presentations=s['episodes'])
        e=s['exposure'][TASK]; e['updates']+=1; e['episodes']+=actual; e['frames']+=3*actual; e['cells'][cell]+=actual
        row=dict(utc=local.utc(),step=s['step'],task=TASK,cell=cell,episodes=actual,frames=actual*3,seconds=elapsed,
            grad_norm=norm,**averages,triplet_window_starts=self.stream.window_starts,pool_index=self.stream.pool_index,
            epoch=self.stream.epoch,unique_episodes_generated=s['unique_episodes_generated'],
            unique_episodes_presented=s['unique_episodes_presented'],triplet_presentations=s['episodes'],
            pool_generation_seconds=self.stream.generation_seconds,partial_batch=actual<32,beta=BETA)
        append_jsonl(self.directory/'progress.jsonl',row); print(json.dumps(row),flush=True); return row

def verify_progress(directory):
    d=Path(directory); first=load_verified(json.loads((d/'initial_checkpoint.json').read_text()))
    receipt=json.loads((d/'latest_checkpoint.json').read_text()); last=load_verified(receipt)
    assert first['state']['step']==first['scheduler']['updates']==0 and not first['optimizer']['state'] and first['stream']['streams']==[]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INIT_SEED); direct=ThreeFrameConvVAE()
    assert tree_equal(first['model'],direct.state_dict()) and first['provenance']==last['provenance']==provenance()
    names=[n for n,_ in direct.named_parameters()]; assert len(names)==126 and first['optimizer_names']==last['optimizer_names']==names
    step=last['state']['step']; assert step>0 and step==last['scheduler']['updates']==receipt['step']
    r=last['stream']['replay']; assert last['state']['episodes']==r['presentations']
    assert sum(r['presentation_counts'].values())==r['presentations'] and r['window_rng']
    assert sum(m['state']['counts'][TASK] for m in last['stream']['streams'])==last['state']['unique_episodes_generated']
    assert set(last['optimizer']['state'])==set(range(126))
    changed=[]
    for i,name in enumerate(names):
        opt=last['optimizer']['state'][i]; assert float(opt['step'])==step
        assert torch.isfinite(last['model'][name]).all()
        assert all(torch.isfinite(opt[k]).all() and opt[k].shape==last['model'][name].shape for k in ('exp_avg','exp_avg_sq'))
        if not torch.equal(first['model'][name],last['model'][name]): changed.append(name)
    assert len(changed)==126
    groups={p:[n for n in changed if n.startswith(p)] for p in ('encoder.','mu_head.','logvar_head.','decoder.')}
    assert all(groups.values())
    if last['state']['config']['device']=='mps':
        assert first['rng']['mps'].numel()>0 and last['rng']['mps'].numel()>0
    result=dict(verified=True,step=step,episodes=last['state']['episodes'],checkpoint=receipt,parameter_count=EXPECTED_PARAMETERS,
        changed_parameters=126,changed_by_module=groups,all_active_parameters_changed=True,persisted_constructor_equality=True,
        unique_episodes_generated=last['state']['unique_episodes_generated'],unique_episodes_presented=last['state']['unique_episodes_presented'])
    atomic_json(d/'persisted_progress_verification.json',result); return result

def reconstruction_metrics(output,target):
    metrics=losses(output,target,target.new_zeros((len(target),1)),target.new_zeros((len(target),1)),beta=0.)
    metrics.pop('loss'); metrics.pop('kl')
    metrics['temporal_difference_mse']=((output[:,1:]-output[:,:-1])-(target[:,1:]-target[:,:-1])).square().mean()
    return metrics

@torch.no_grad()
def evaluate(model,split,movies,windows,device,output,deadline,save_images=True):
    stream=SingleStimulusStream(split); rng=random.Random(182701 if split=='val' else 182702)
    rows=[]; model.eval(); begin=time.perf_counter()
    for cell in CELLS:
        totals={}; start_time=time.perf_counter(); count=0; identities=[]; triplet_starts=[]
        for movie_index in range(movies):
            if time.time()>=deadline: atomic_json(output,dict(complete=False,cells=rows,split=split)); return dict(complete=False,cells=rows)
            image,_,meta=stream.batch(1,TASK,cell)
            starts=rng.sample(list(range(7,image.shape[1]-3)),windows)
            x=torch.stack([image[0,s:s+3] for s in starts]).to(device); predicted=model(x)
            metrics=losses(predicted['reconstruction'],x,predicted['mu'],predicted['logvar'],beta=BETA)
            metrics['temporal_difference_mse']=((predicted['reconstruction'][:,1:]-predicted['reconstruction'][:,:-1])-(x[:,1:]-x[:,:-1])).square().mean()
            for baseline_name,baseline in [('gray',torch.full_like(x,.5)),('copy_middle',x[:,1:2].expand_as(x))]:
                for k,v in reconstruction_metrics(baseline,x).items(): metrics[baseline_name+'_'+k]=v
            for k,v in metrics.items(): totals[k]=totals.get(k,0.)+float(v)*windows
            count+=windows; identities.append(meta[0]['suite_trial_id']); triplet_starts.append(starts)
            if save_images and movie_index==0:
                pair=torch.cat((x[0],predicted['reconstruction'][0]),dim=0).clamp(0,1).cpu()
                array=(pair.permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
                canvas=np.concatenate((np.concatenate(array[:3],axis=1),np.concatenate(array[3:],axis=1)),axis=0)
                Image.fromarray(canvas).save(str(output)+'.'+cell+'.png')
        local.sync(device)
        rows.append(dict(cell=cell,n=movies,triplets=count,metrics={k:v/count for k,v in totals.items()},
            seconds=time.perf_counter()-start_time,trial_ids=identities,window_starts=triplet_starts))
        atomic_json(str(output)+'.partial.json',dict(split=split,cells=rows,complete=False))
    result=dict(split=split,complete=True,cells=rows,seconds=time.perf_counter()-begin,
        selection_reconstruction=sum(r['metrics']['recon'] for r in rows)/3,
        grouped_movies=True,windows_per_movie=windows,evaluation='deterministic posterior mean reconstruction')
    atomic_json(output,result); return result

def profile(directory):
    cfg=json.loads((directory/'profile_config.json').read_text()); verify_sources(cfg)
    s=Session(directory,cfg); s.checkpoint('initial.pt'); warmup=[]; rows=[]
    for i in range(6):
        if time.time()>=cfg['deadline']-60: raise RuntimeError('Profile cap exhausted')
        row=s.train_update(*s.scheduler.next()); (warmup if i<3 else rows).append(row)
        if s.state['step'] in (1,2,3): s.checkpoint(f"checkpoint_{s.state['step']:06d}.pt")
    s.checkpoint('profile_terminal.pt')
    ev=evaluate(s.model,'val',8,3,s.device,directory/'eval_timing.json',cfg['deadline']-30)
    assert ev['complete']
    atomic_json(directory/'profile.json',dict(complete=True,warmup_rows=warmup,rows=rows,evaluation_cells=ev['cells'],
        disposable=True,discard_all_state=True,verification=verify_progress(directory)))

def config_for(budget):
    manifest_path=ROOT/'runtime_manifest.json'; manifest=json.loads(manifest_path.read_text())
    hashes={str(Path(p) if Path(p).is_absolute() else ROOT/p):sha for p,sha in manifest['source_hashes'].items()}
    hashes[str(manifest_path)]=digest(manifest_path)
    cfg=dict(budget,protocol=PROTOCOL,device='mps',effective_batch=32,microbatch=4,eval_microbatch=3,cpu_threads=2,
        source_hashes=hashes,runtime_root=str(ROOT),initialization=provenance(),all_trainable=True,lr=1e-4,
        betas=[.9,.999],eps=1e-8,weight_decay=0,clipping=None,precision='fp32',beta=BETA,
        val_movies=64,test_movies=128,eval_windows=3,checkpoint_every=100)
    verify_sources(cfg); return cfg

def measured_plan(directory,budget):
    validate_budget(budget,budget); p=json.loads((directory/'profile/profile.json').read_text())
    assert p['complete'] and len(p['rows'])==3
    costs={r['cell']:r['seconds'] for r in p['rows']}; assert set(costs)==set(CELLS)
    val=1.35*sum(r['seconds']/r['n'] for r in p['evaluation_cells'])*64; test=2*val
    generation=sum(r['pool_generation_seconds'] for r in p['warmup_rows']); assert generation>0
    rng=random.Random(SCHEDULER_SEED+71); schedule=[]; train=0.; best=None; now=time.time()
    for i in range(TARGET):
        if i%330==0: sizes={c:333+int(j==(i//330)%3) for j,c in enumerate(CELLS)}
        if i%33==0: schedule.extend(replay.epoch_batches(sizes,rng))
        train+=costs[schedule[i][0]]
        gen=((i//330)+1)*generation
        total=1.25*(train+gen)+len(validation_steps_for(i+1))*val+2*test+900
        if total+60<budget['deadline']-now: best=i+1
    if best is None or best<330: raise RuntimeError('One complete replay pool cannot fit new local cap')
    count=best//330*330; train=sum(costs[c] for c,_ in schedule[:count]); gen=count//330*generation
    total=1.25*(train+gen)+len(validation_steps_for(count))*val+2*test+900
    return dict(max_steps=count,target_requested=TARGET,total_episodes=sum(len(ix) for _,ix in schedule[:count]),
        planned_triplet_presentations=sum(len(ix) for _,ix in schedule[:count]),planned_unique_movies=count//330*1000,
        exposure_reduced=count<TARGET,validation_steps=validation_steps_for(count),estimated_total_remaining_seconds=total,
        estimated_one_test_seconds=test,estimated_validation_seconds=val,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values())+generation,checkpoint_every=100)

def run(directory):
    cfg=json.loads((directory/'config.json').read_text()); validate_budget(json.loads((directory/'budget.json').read_text()),cfg); verify_sources(cfg)
    if (directory/'initial.pt').exists(): raise RuntimeError('Fresh-only; no restart')
    s=Session(directory,cfg); s.checkpoint('initial.pt'); stopped=[False]
    import signal
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete'; failure=None; final=None; selected=None
    def validate():
        s.status('validation'); output=directory/f"validation_{s.state['step']:06d}.json"
        result=evaluate(s.model,'val',64,3,s.device,output,cfg['deadline']-2*cfg['estimated_one_test_seconds']-120)
        key=result.get('selection_reconstruction') if result['complete'] else None
        state=s.state; state['selection_history'].append(dict(step=state['step'],reconstruction=key,complete=result['complete']))
        name=f"validation_checkpoint_{state['step']:06d}.pt"
        if key is not None and (state['best_key'] is None or key<state['best_key']):
            state.update(best_key=key,best_step=state['step'],best_checkpoint=str(directory/name))
        s.checkpoint(name); s.status('validation_complete')
    try:
        while s.state['step']<cfg['max_steps']:
            if stopped[0]: reason='signal'; break
            if time.time()+cfg['final_reserve_seconds']+cfg['update_estimate_seconds']>=cfg['deadline']: reason='wall_budget_reserve'; break
            row=s.train_update(*s.scheduler.next()); step=s.state['step']
            if step in (1,2,3) or step%100==0: s.checkpoint(f'checkpoint_{step:06d}.pt')
            s.status('training',last_loss=row['loss'])
            if step in cfg['validation_steps'] and not stopped[0]: validate()
        if s.state['step']>0 and not stopped[0] and (not s.state['selection_history'] or s.state['selection_history'][-1]['step']!=s.state['step']): validate()
        terminal=s.checkpoint('terminal.pt')
        if s.state['step']>0 and not stopped[0]:
            s.status('final_test_terminal'); final=evaluate(s.model,'test',128,3,s.device,directory/'test_terminal.json',cfg['deadline']-cfg['estimated_one_test_seconds']-60)
            if s.state['best_step']==s.state['step']:
                selected=dict(final,reused_terminal=True); atomic_json(directory/'test_selected.json',selected)
            elif s.state['best_checkpoint']:
                saved=torch.load(s.state['best_checkpoint'],map_location='cpu',weights_only=False); s.model.load_state_dict(saved['model'])
                selected=evaluate(s.model,'test',128,3,s.device,directory/'test_selected.json',cfg['deadline']-60)
    except Exception as exc:
        import traceback
        failure=dict(error=repr(exc),traceback=traceback.format_exc()); atomic_json(directory/'failure.json',failure)
        terminal=s.latest; reason='worker_error'
    complete=bool(final and selected and final['complete'] and selected['complete'])
    if complete:
        assert all(a['trial_ids']==b['trial_ids'] and a['window_starts']==b['window_starts'] for a,b in zip(final['cells'],selected['cells']))
    saved=load_verified(terminal)
    report=dict(protocol=PROTOCOL,architecture=VERSION,initialization=provenance(),stop_reason=reason,failure=failure,
        terminal_checkpoint=terminal,terminal_step=saved['state']['step'],selected_step=saved['state']['best_step'],
        pinned_updates=cfg['max_steps'],exposure_completed=saved['state']['step']==cfg['max_steps'],
        cap_limited=reason=='wall_budget_reserve',triplet_presentations=saved['state']['episodes'],
        unique_movies_generated=saved['state']['unique_episodes_generated'],unique_movies_presented=saved['state']['unique_episodes_presented'],
        selection_history=saved['state']['selection_history'],terminal_test=final,selected_test=selected,final_coverage_complete=complete,config=cfg)
    atomic_json(directory/'report.json',report); s.status('finished',stop_reason=reason,final_coverage_complete=complete)
    return 0 if not failure and complete and reason in ('planned_complete','wall_budget_reserve') else 2

completion=clone(local.completion,TARGET=TARGET)
supervise=clone(local.supervise,MODULE=MODULE,completion=completion)
def local_supervise(directory):
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'budget.json').exists(): raise RuntimeError('No restart/cap renewal')
    start=time.time(); budget=dict(cap_started=start,deadline=start+28200,hard_deadline=start+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800,execution_placement='local')
    with (directory/'budget.json').open('x') as f: json.dump(budget,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.chmod(directory/'budget.json',0o444); atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),**budget))
    cfg=config_for(budget); pd=directory/'profile'; pd.mkdir()
    atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],time.time()+3600)))
    outcomes=[]
    try:
        outcome=supervise(directory,'profile'); outcomes.append(outcome)
        if outcome['returncode']!=0: raise RuntimeError('VAE native profile failed')
        plan=measured_plan(directory,budget); cfg.update(plan)
        for name,value in [('allocation.json',plan),('config.json',cfg)]:
            with (directory/name).open('x') as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
            os.chmod(directory/name,0o444)
        outcome=supervise(directory,'run'); outcomes.append(outcome); result=dict(status=outcome['status'],outcomes=outcomes,**budget)
    except Exception as exc:
        result=dict(status='error',error=repr(exc),outcomes=outcomes,**budget); atomic_json(directory/'failure.json',result)
        completion(directory,'incomplete',result['error'])
    atomic_json(directory/'local_supervisor_result.json',result); return result

def main():
    local.cpu_setup(); p=argparse.ArgumentParser(); p.add_argument('mode',choices=['local-supervise','profile','run','verify']); p.add_argument('directory',type=Path)
    args=p.parse_args(); d=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(d))); return
    if args.mode=='local-supervise':
        result=local_supervise(d); print(json.dumps(result),flush=True); raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    import fcntl
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(d)
        else: raise SystemExit(run(d))
if __name__=='__main__': main()
