"""Explicit continuation from saved model, Adam, scheduler, streams and RNGs."""
import argparse, copy, hashlib, json, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from . import train as base
from .model import ExpandedSuite, PARENT

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_sources(saved):
    for name, expected in saved['config']['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Original source changed: {name}')
    assert digest(PARENT) == saved['config']['parent_sha256']

def restore(saved, device='cpu'):
    model = ExpandedSuite().to(device)
    model.load_state_dict(saved['model'], strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    scheduler = base.Scheduler()
    scheduler.rng.setstate(saved['scheduler']['rng'])
    scheduler.tasks = copy.deepcopy(saved['scheduler']['tasks'])
    scheduler.cells = copy.deepcopy(saved['scheduler']['cells'])
    stream = base.Stream('train'); stream.load_state_dict(saved['stream'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    return model, opt, scheduler, stream

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--wall-seconds',type=int,required=True)
    p.add_argument('--target-steps',type=int,help='Explicitly authorized cumulative target override')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if not 180<a.wall_seconds<=57600:raise ValueError('Finite allocation <=16 hours required')
    saved=torch.load(a.source/'latest.pt',map_location='cpu',weights_only=False)
    target=saved['config']['steps'] if a.target_steps is None else a.target_steps
    if target<=saved['step'] or target%base.CYCLE:raise ValueError('Target must advance the checkpoint in complete task cycles')
    verify_sources(saved)
    originals={n:digest(a.source/(n+'.pt')) for n in ('best','latest')}
    receipt=a.source/'continuation_ready.json'
    if not receipt.exists():receipt=a.source/'night_stop_verified.json'
    pause=json.loads(receipt.read_text())
    assert all(originals[n]==pause[n]['sha256'] for n in originals)
    a.run.mkdir(parents=True,exist_ok=False)
    started=time.time();deadline=started+a.wall_seconds
    lock=base.GPULock(ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:
        base.configure(a.device)
        model,opt,scheduler,stream=restore(saved,a.device)
        step=start_step=saved['step'];best=saved['best']
        config=copy.deepcopy(saved['config'])
        config.update(started=started,deadline=deadline,resumed_step=step,steps=target,
            resumed_from=str(a.source.resolve()),original_checkpoint_sha256=originals,
            original_config=copy.deepcopy(saved['config']),resume_source_sha256=digest(__file__),
            allocation='Explicit user-requested continuation under a finite cap',
            target_override_authorized=a.target_steps is not None)
        base.atomic_json(a.run/'config.json',config)
        shutil.copy2(a.source/'best.pt',a.run/'best.pt')
        def save(name='latest'):
            payload=base.cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),stream=stream.state_dict(),step=step,best=best,config=config,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
            temp=a.run/(name+'.pt.tmp');torch.save(payload,temp);base.replace_with_retry(temp,a.run/(name+'.pt'))
        def status(phase):
            base.atomic_json(a.run/'status.json',dict(phase=phase,step=step,target=target,
                resumed_step=start_step,new_updates=step-start_step,deadline=deadline,elapsed=time.time()-started))
        assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
        if a.device=='cuda':assert all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
        save();save('initial')
        base.atomic_json(a.run/'restoration_receipt.json',dict(step=step,optimizer_states=len(opt.state),
            rngs_restored=True,original_hashes=originals,first_next_update=step+1))
        reason='planned_complete'
        try:
            while step<target:
                if (a.run/'stop_requested.json').exists():reason='user_stop';break
                if time.time()>deadline-180:reason='time_cap';break
                task,cell=scheduler.next();x,y,meta=stream.batch(32,task,cell)
                metrics=base.update(model,opt,x,y,meta,task,a.device);step+=1
                base.append_json(a.run/'progress.jsonl',dict(step=step,task=task,cell=cell,time=time.time(),**metrics));status('training')
                if step%base.CYCLE==0:save()
                if step%(20*base.CYCLE)==0:
                    status('validating')
                    result=base.evaluate(model,a.device,'val',32,deadline-180,base.monitoring_cells(),a.run)
                    base.atomic_json(a.run/f'validation_{step:06d}.json',result)
                    if result['equal_task_mean_auc']>best:best=result['equal_task_mean_auc'];save('best')
            save()
            if reason=='planned_complete':
                status('final_validation')
                result=base.evaluate(model,a.device,'val',32,deadline-180,base.monitoring_cells(),a.run)
                base.atomic_json(a.run/'validation_final.json',result)
                if result['equal_task_mean_auc']>best:best=result['equal_task_mean_auc'];save('best')
                selected=copy.deepcopy(model)
                selected.load_state_dict(torch.load(a.run/'best.pt',map_location=a.device,weights_only=False)['model'])
                status('final_evaluation')
                result=base.evaluate(selected,a.device,'val',64,deadline-90,run_dir=a.run)
                base.atomic_json(a.run/'validation_full_selected.json',result)
                result=base.evaluate(selected,a.device,'test',256,deadline-90,run_dir=a.run)
                base.atomic_json(a.run/'final_test.json',result)
        except (TimeoutError,InterruptedError) as exc:
            reason='time_cap' if isinstance(exc,TimeoutError) else 'user_stop';save()
        except BaseException as exc:
            save();base.atomic_json(a.run/'failure.json',dict(error=repr(exc),step=step));raise
        assert all(digest(a.source/(n+'.pt'))==sha for n,sha in originals.items())
        base.atomic_json(a.run/'exit.json',dict(reason=reason,step=step,training_complete=step==target,
            evaluation_complete=(a.run/'final_test.json').exists(),original_checkpoints_unchanged=True))
        status('complete' if reason=='planned_complete' else 'stopped')
    finally:lock.close()

if __name__=='__main__':main()
