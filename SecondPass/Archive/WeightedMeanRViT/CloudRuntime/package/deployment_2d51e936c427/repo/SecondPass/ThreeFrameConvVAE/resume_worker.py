"""Exact VAE recovery with only latest.pt and best.pt; original cap unchanged."""
import argparse, copy, hashlib, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
from . import worker as base
MODULE='SecondPass.ThreeFrameConvVAE.resume_worker'


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def restore_rng(rng,device):
    torch.set_rng_state(rng['cpu']); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
    if device=='mps': torch.mps.set_rng_state(rng['mps'])
    elif str(device).startswith('cuda'): torch.cuda.set_rng_state_all(rng['cuda'])


def verify_progress(directory):
    receipt=json.loads((directory/'latest_checkpoint.json').read_text()); saved=base.load_verified(receipt)
    recovery=json.loads((directory/'recovery.json').read_text()); step=saved['state']['step']
    assert saved['schema']==3 and saved['provenance']==base.provenance()
    assert step==saved['scheduler']['updates']==receipt['step'] and step>=recovery['restored_step']
    names=saved['optimizer_names']; assert len(names)==126 and names==recovery['optimizer_names']
    assert set(saved['optimizer']['state'])==set(range(126))
    optgroup=saved['optimizer']['param_groups']; assert len(optgroup)==1
    assert optgroup[0]['lr']==1e-4 and optgroup[0]['betas']==(.9,.999) and optgroup[0]['eps']==1e-8 and optgroup[0]['weight_decay']==0
    changed=[]
    for i,name in enumerate(names):
        value=saved['model'][name]; opt=saved['optimizer']['state'][i]
        assert float(opt['step'])==step and torch.isfinite(value).all()
        assert all(torch.isfinite(opt[key]).all() and opt[key].shape==value.shape for key in ('exp_avg','exp_avg_sq'))
        if tensor_digest(value)!=recovery['restored_parameter_hashes'][name]: changed.append(name)
    if step>recovery['restored_step']: assert len(changed)==126
    r=saved['stream']['replay']; assert r['presentations']==saved['state']['episodes']
    assert sum(r['presentation_counts'].values())==r['presentations'] and r['window_rng']
    assert r['presentation_counts']==saved['state']['exposure'][base.TASK]['cells']
    assert sum(m['state']['counts'][base.TASK] for m in saved['stream']['streams'])==saved['state']['unique_episodes_generated']
    if saved['state']['config']['device']=='mps': assert saved['rng']['mps'].numel()>0
    result=dict(verified=True,step=step,episodes=saved['state']['episodes'],checkpoint=receipt,
        additional_updates=step-recovery['restored_step'],parameter_count=base.EXPECTED_PARAMETERS,
        changed_parameters_since_restoration=len(changed),all126_adam_steps_verified=True,
        provenance='historical fresh-constructor receipt retained as metadata; exact checkpoint5600 continuation')
    base.atomic_json(directory/'persisted_progress_verification.json',result); return result


class ResumedSession(base.Session):
    def __init__(self,directory,config):
        super().__init__(directory,config)
        recovery=json.loads((self.directory/'recovery.json').read_text())
        receipt=recovery['restored_checkpoint']; saved=base.load_verified(receipt)
        assert saved['provenance']==base.provenance()
        assert saved['state']['step']==recovery['restored_step']==saved['scheduler']['updates']
        self.model.load_state_dict(saved['model']); self.optimizer.load_state_dict(saved['optimizer'])
        self.stream.load_state_dict(saved['stream']); self.scheduler.load_state_dict(saved['scheduler'])
        self.state=copy.deepcopy(saved['state']); self.state['config']=config; self.latest=receipt
        self.restored_step=saved['state']['step']
        assert base.tree_equal(base.cpu_tree(self.optimizer.state_dict()),saved['optimizer'])
        assert base.tree_equal(self.stream.state_dict(),saved['stream'])
        # The earlier best weights were deleted; their validation history remains descriptive only.
        self.state.update(best_step=None,best_key=None,best_checkpoint=None)
        restore_rng(saved['rng'],self.device)
        assert torch.equal(torch.get_rng_state(),saved['rng']['cpu'])
        if self.device=='mps': assert torch.equal(torch.mps.get_rng_state(),saved['rng']['mps'])
        base.atomic_json(self.directory/'restoration_verified.json',dict(verified=True,restored_step=self.restored_step,
            restored_presentations=self.state['episodes'],full_optimizer_stream_scheduler_rng_restored=True,
            checkpoint=receipt,learned_parameter_tensors=126,architecture_unchanged=True,previous_best_weights_available=False))

    def checkpoint(self,filename='latest.pt'):
        assert filename=='latest.pt'
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        if self.device=='mps': rng['mps']=torch.mps.get_rng_state()
        payload=base.cpu_tree(dict(schema=3,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            optimizer_names=[n for n,_ in self.model.named_parameters()],state=self.state,
            scheduler=self.scheduler.state_dict(),stream=self.stream.state_dict(),rng=rng,provenance=base.provenance()))
        path=self.directory/'latest.pt'; temporary=self.directory/'latest.pt.tmp'
        with temporary.open('wb') as stream: torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        receipt=dict(path=str(path),bytes=path.stat().st_size,sha256=base.digest(path),verified=True,step=self.state['step'])
        assert base.tree_equal(payload,base.load_verified(receipt))
        self.latest=receipt; base.atomic_json(self.directory/'latest_checkpoint.json',receipt)
        base.append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=base.local.utc(),**receipt))
        verification=verify_progress(self.directory)
        if self.restored_step<self.state['step']<=self.restored_step+3:
            base.atomic_json(self.directory/f"recovery_proof_{self.state['step']:06d}.json",verification)
            if self.state['step']==self.restored_step+3:
                base.atomic_json(self.directory/'startup_ready.json',dict(verified=True,checkpoint=receipt,persisted_optimizer_evidence=verification,recovery=True))
        return receipt

    def promote_best(self):
        source=self.directory/'latest.pt'; temporary=self.directory/'best.pt.tmp'; destination=self.directory/'best.pt'
        os.link(source,temporary); os.replace(temporary,destination)
        receipt=dict(self.latest,path=str(destination)); assert base.digest(destination)==receipt['sha256']
        base.atomic_json(self.directory/'best_checkpoint.json',receipt)


def run(directory):
    cfg=json.loads((directory/'config.json').read_text()); base.validate_budget(json.loads((directory/'budget.json').read_text()),cfg); base.verify_sources(cfg)
    session=ResumedSession(directory,cfg); session.checkpoint(); stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete'; failure=None; final=None; selected=None
    def validate():
        session.status('validation'); output=directory/f"validation_{session.state['step']:06d}.json"
        result=base.evaluate(session.model,'val',64,3,session.device,output,cfg['deadline']-2*cfg['estimated_one_test_seconds']-120)
        key=result.get('selection_reconstruction') if result['complete'] else None
        state=session.state; state['selection_history'].append(dict(step=state['step'],reconstruction=key,complete=result['complete'],available_recovery_candidate=True))
        better=key is not None and (state['best_key'] is None or key<state['best_key'])
        if better: state.update(best_key=key,best_step=state['step'],best_checkpoint=str(directory/'best.pt'))
        session.checkpoint()
        if better: session.promote_best()
        session.status('validation_complete')
    try:
        # Seed the best from available weights; preserve the training RNG across this deterministic evaluation.
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),mps=torch.mps.get_rng_state())
        validate(); restore_rng(rng,session.device); session.checkpoint()
        while session.state['step']<cfg['max_steps']:
            if stopped[0]: reason='signal'; break
            if time.time()+cfg['final_reserve_seconds']+cfg['update_estimate_seconds']>=cfg['deadline']: reason='wall_budget_reserve'; break
            row=session.train_update(*session.scheduler.next()); step=session.state['step']
            if step<=session.restored_step+3 or step%100==0: session.checkpoint()
            session.status('training',last_loss=row['loss'])
            if step in cfg['validation_steps'] and not stopped[0]: validate()
        if session.state['step']>session.restored_step and not stopped[0] and session.state['selection_history'][-1]['step']!=session.state['step']: validate()
        session.checkpoint()
        if not stopped[0]:
            session.status('final_test_terminal'); final=base.evaluate(session.model,'test',128,3,session.device,directory/'test_terminal.json',cfg['deadline']-cfg['estimated_one_test_seconds']-60)
            if session.state['best_step']==session.state['step']:
                selected=dict(final,reused_terminal=True); base.atomic_json(directory/'test_selected.json',selected)
            elif session.state['best_checkpoint']:
                saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False); session.model.load_state_dict(saved['model'])
                selected=base.evaluate(session.model,'test',128,3,session.device,directory/'test_selected.json',cfg['deadline']-60)
    except Exception as exc:
        import traceback
        failure=dict(error=repr(exc),traceback=traceback.format_exc()); base.atomic_json(directory/'failure.json',failure); reason='worker_error'
    complete=bool(final and selected and final['complete'] and selected['complete'])
    if complete: assert all(a['trial_ids']==b['trial_ids'] and a['window_starts']==b['window_starts'] for a,b in zip(final['cells'],selected['cells']))
    saved=base.load_verified(session.latest); recovery=json.loads((directory/'recovery.json').read_text())
    report=dict(protocol=base.PROTOCOL,architecture=base.VERSION,initialization=base.provenance(),stop_reason=reason,failure=failure,
        terminal_checkpoint=session.latest,terminal_step=saved['state']['step'],selected_step=saved['state']['best_step'],pinned_updates=cfg['max_steps'],
        exposure_completed=saved['state']['step']==cfg['max_steps'],cap_limited=reason=='wall_budget_reserve',triplet_presentations=saved['state']['episodes'],
        unique_movies_generated=saved['state']['unique_episodes_generated'],unique_movies_presented=saved['state']['unique_episodes_presented'],
        selection_history=saved['state']['selection_history'],terminal_test=final,selected_test=selected,final_coverage_complete=complete,config=cfg,recovery=recovery,
        additional_updates_after_recovery=saved['state']['step']-session.restored_step,
        discarded_unsaved_attempt_updates=recovery['discarded_unsaved_attempt_updates'],discarded_unsaved_attempt_presentations=recovery['discarded_unsaved_attempt_presentations'],
        selection_limitation='Historical best5500 weights unavailable; selected among resumed5600 and later validations only',checkpoint_policy='only latest.pt and best.pt')
    base.atomic_json(directory/'report.json',report); session.status('finished',stop_reason=reason,final_coverage_complete=complete)
    return 0 if not failure and complete and reason in ('planned_complete','wall_budget_reserve') else 2


def completion(directory,status,error=None):
    report=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {}
    latest=json.loads((directory/'latest_checkpoint.json').read_text()) if (directory/'latest_checkpoint.json').exists() else None
    best=json.loads((directory/'best_checkpoint.json').read_text()) if (directory/'best_checkpoint.json').exists() else None
    paths=[p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md','.png') and p.name!='cloud_completion.json']
    paths += [p for p in (directory/'latest.pt',directory/'best.pt') if p.exists()]
    files=[dict(path=str(p),relative_path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=base.digest(p)) for p in sorted(set(paths))]
    base.atomic_json(directory/'cloud_completion.json',dict(status=status,error=error,target_updates=base.TARGET,pinned_updates=9900,
        actual_updates=latest['step'] if latest else 0,exposure_completed=report.get('exposure_completed',False),cap_limited=report.get('cap_limited',False),
        stop_reason=report.get('stop_reason'),selected_checkpoint=best['path'] if best else None,terminal_checkpoint=latest,latest=latest,
        artifact_manifest=files,finished_utc=base.local.utc(),checkpoint_policy='latest.pt and best.pt only'))

supervise=base.clone(base.supervise,MODULE=MODULE,completion=completion)

def local_supervise(directory):
    budget=json.loads((directory/'budget.json').read_text()); cfg=json.loads((directory/'config.json').read_text())
    base.validate_budget(budget,cfg); base.verify_sources(cfg)
    base.atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),**budget))
    try:
        outcome=supervise(directory,'run'); result=dict(status=outcome['status'],outcomes=[outcome],**budget)
    except Exception as exc:
        import traceback
        result=dict(status='error',error=repr(exc),traceback=traceback.format_exc(),**budget)
        base.atomic_json(directory/'failure.json',result); completion(directory,'incomplete',result['error'])
    base.atomic_json(directory/'local_supervisor_result.json',result); return result


def main():
    base.local.cpu_setup(); parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['local-supervise','run','verify'])
    parser.add_argument('directory',type=Path); args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode=='local-supervise':
        result=local_supervise(directory); print(json.dumps(result),flush=True); raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    import fcntl
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); raise SystemExit(run(directory))
if __name__=='__main__': main()
