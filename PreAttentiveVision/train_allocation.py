"""Task-allocation sibling worker from trained late-SE parent. No implicit launch or budget renewal.

Execute a fully specified JSON job only after task/budget authorization. A job
contains config, kind, deadline (absolute Unix), source_hashes, out and result.
The coordinator must launch only one GPU worker and enforce the same deadline.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys
import time
import json
import csv
import random
import threading
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from PreAttentiveVision.train import write,read,sha


def worker(job):
    deadline=float(job['deadline'])
    if time.time()>=deadline:raise TimeoutError('Authorized deadline already passed')
    timer=threading.Timer(deadline-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PreAttentiveVision.models import build_encoder
    from PreAttentiveVision.hybrid_models import build_hybrid
    from PreAttentiveVision.allocation_sampler import TaskLocalStreams,SCHEDULES,TRAIN_BASE_SEED,exposure_counts
    from PreAttentiveVision.decoder_multitask import MultitaskPairClassifier
    from PreAttentiveVision.neuroscience_stimuli import TaskStream,TASK_CLASSES
    from PreAttentiveVision.evaluate_multitask import evaluate_tasks
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for name,digest in job['source_hashes'].items():
        if sha(HERE/name)!=digest:raise RuntimeError('Source identity changed: '+name)
    cfg=job['config'];tasks=cfg['task_classes']
    if not tasks or any(TASK_CLASSES.get(k)!=v for k,v in tasks.items()):
        raise ValueError('Requested task heads must match defined stimulus tasks')
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    encoder=build_hybrid('convnext_se_residual')
    torch.manual_seed(seed+500000);torch.cuda.manual_seed_all(seed+500000)
    model=MultitaskPairClassifier(encoder,tasks).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    stream=TaskLocalStreams(TRAIN_BASE_SEED,split='train');step=0
    schedule=SCHEDULES[cfg['model']]
    if job.get('checkpoint'):
        cp=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
        if cp['version']!='pav_task_allocation_v1' or cp['config']!=cfg or cp['source_hashes']!=job['source_hashes']:
            raise RuntimeError('Checkpoint incompatible with ordered-task run')
        model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer'])
        stream.load_state_dict(cp['stream']);step=cp['step']
        random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy'])
        torch.set_rng_state(cp['rng']['torch']);torch.cuda.set_rng_state(cp['rng']['cuda'])
    else:
        if sha(job['parent_checkpoint'])!=job['parent_sha256']:raise RuntimeError('Parent bytes differ from pinned identity')
        parent=torch.load(job['parent_checkpoint'],map_location='cpu',weights_only=False)
        if parent['version']!='pav_hybrid_continuation_v1' or parent['config']['model']!='convnext_se_residual' or parent['step']!=1512:raise ValueError('Expected trained late-SE parent1512')
        model.load_state_dict(parent['model'],strict=True);optimizer.load_state_dict(parent['optimizer']);step=parent['step']
        transfer=dict(transfer='Exact model and Adam restore; NEW task-local sampler protocol, not old stream continuation',inherited_adam_states=len(optimizer.state),new_sampler=stream.state_dict())
        random.setstate(parent['rng']['python']);np.random.set_state(parent['rng']['numpy'])
        torch.set_rng_state(parent['rng']['torch']);torch.cuda.set_rng_state(parent['rng']['cuda'])
        write(out/'transfer.json',dict(transfer,parent_checkpoint=job['parent_checkpoint'],parent_sha256=job['parent_sha256']))

    def save():
        path=out/f'checkpoint_{step:06d}.pt'
        if not path.exists():
            state=dict(version='pav_task_allocation_v1',config=cfg,source_hashes=job['source_hashes'],step=step,
                lineage=dict(parent_checkpoint=job['parent_checkpoint'],parent_sha256=job['parent_sha256'],parent_step=1512),
                model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
            with (out/'checkpoint_index.jsonl').open('a') as f:
                f.write(json.dumps(dict(file=path.name,sha256=sha(path),step=step,training_pairs=step*cfg['batch_size']))+'\n')
        return str(path)

    started=time.monotonic();kind=job['kind'];timings=[];task_names=list(tasks)
    if kind in ('train','profile'):
        model.train();torch.cuda.reset_peak_memory_stats()
        if not job.get('checkpoint'):save()
        fields=['step','task','fresh_pairs','loss','grad_norm','accuracy','step_seconds','wall_utc']
        with (out/'metrics.csv').open('a',newline='') as f:
            log=csv.DictWriter(f,fields)
            if f.tell()==0:log.writeheader()
            while step<job['target']:
                if time.time()>deadline-5:break
                # Deterministic round robin yields equal per-task exposure at multiples of task count.
                task=schedule[(step-1512)%len(schedule)];tick=time.monotonic()
                x,y,_=stream.batch(cfg['batch_size'],task=task);x=x.cuda();y=y.cuda()
                optimizer.zero_grad(set_to_none=True);logits=model(x,task);loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();torch.cuda.synchronize();step+=1;elapsed=time.monotonic()-tick;timings.append(elapsed)
                log.writerow(dict(step=step,task=task,fresh_pairs=step*cfg['batch_size'],loss=float(loss),
                    grad_norm=float(norm),accuracy=float((logits.argmax(1)==y).float().mean()),
                    step_seconds=elapsed,wall_utc=datetime.now(timezone.utc).isoformat()));f.flush()
                if step%128==0:print(json.dumps(dict(step=step,task=task,loss=float(loss))),flush=True)
                if step%256==0:save()
        result=dict(status='completed' if step==job['target'] else 'budget_stopped',step=step,
            checkpoint=save(),fresh_pairs=step*cfg['batch_size'],additional_pairs=(step-1512)*cfg['batch_size'],additional_updates=step-1512,worker_seconds=time.monotonic()-started,
            per_task_additional_updates=exposure_counts(step-1512,cfg['model']))
        if kind=='profile':
            model.eval();inference=[]
            for task in task_names:
                tick=time.monotonic();x,_,_=stream.batch(cfg['batch_size'],task=task)
                with torch.no_grad():model(x.cuda(),task)
                torch.cuda.synchronize();inference.append(time.monotonic()-tick)
            result.update(train_step_mean=float(np.mean(timings[len(schedule):] or timings)),train_step_median=float(np.median(timings[len(schedule):] or timings)),
                train_step_p90=float(np.quantile(timings[8:] or timings,.9)),
                eval_batch_mean=float(np.mean(inference)),eval_task_batch_seconds=dict(zip(task_names,inference)),
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                encoder_params=sum(p.numel() for p in model.encoder.parameters()),
                decoder_params=sum(p.numel() for p in model.decoder.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    elif kind=='eval':
        if not job.get('checkpoint') and not job.get('allow_parent_eval'):raise ValueError('Evaluation requires saved checkpoint or explicit parent reference')
        model.eval();evaluation=TaskLocalStreams(job['eval_seed'],split=job['split']);rows=[]
        for task in task_names:
            for offset in range(0,job['n_per_task'],cfg['batch_size']):
                if time.time()>deadline-8:raise TimeoutError('Evaluation deadline reached')
                x,y,metadata=evaluation.batch(min(cfg['batch_size'],job['n_per_task']-offset),task=task)
                with torch.no_grad():probabilities=model(x.cuda(),task).softmax(1).cpu().tolist()
                for p,label,meta in zip(probabilities,y.tolist(),metadata):
                    rows.append(dict(model=cfg['model'],seed=seed,step=step,task=task,label=label,
                        probabilities=p,difficulty=meta.get('difficulty','unspecified'),
                        base_id=meta.get('base_id'),trial_id=meta['trial_id'],metadata=meta))
        # Retain raw scores even if a later summary operation reaches the cap.
        with (out/'predictions.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        result=evaluate_tasks(rows,tasks,resamples=job.get('resamples',0))
        result.update(status='completed',split=job['split'],step=step,evaluated_pairs=len(rows),
            worker_seconds=time.monotonic()-started,predictions=str(out/'predictions.jsonl'),checkpoint=job.get('checkpoint') or job['parent_checkpoint'])
        write(out/'summary.json',result)
    else:raise ValueError('Unknown worker kind')
    write(job['result'],result);timer.cancel()


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: train_allocation.py AUTHORIZED_JOB.json')
    worker(read(sys.argv[1]))
