"""Causal temporal learner with a frozen successful spatial encoder. No implicit launch or budget renewal.

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
PAV=HERE.parent
sys.path.insert(0,str(PAV.parent))
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
    from PreAttentiveVision.allocation_sampler import TaskLocalStreams,SCHEDULES
    from PreAttentiveVision.TemporalIntegration.accumulators import StreamingPAVClassifier
    from PreAttentiveVision.TemporalIntegration.eval_temporal import summarize_modes, swapped_labels
    from PreAttentiveVision.decoder_multitask import MultitaskPairClassifier
    from PreAttentiveVision.neuroscience_stimuli import TaskStream,TASK_CLASSES
    from PreAttentiveVision.evaluate_multitask import evaluate_tasks
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for name,digest in job['source_hashes'].items():
        if sha(PAV/name)!=digest:raise RuntimeError('Source identity changed: '+name)
    cfg=job['config'];tasks=cfg['task_classes']
    if not tasks or any(TASK_CLASSES.get(k)!=v for k,v in tasks.items()):
        raise ValueError('Requested task heads must match defined stimulus tasks')
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if sha(job['parent_checkpoint'])!=job['parent_sha256']:raise RuntimeError('Frozen-parent identity mismatch')
    parent=torch.load(job['parent_checkpoint'],map_location='cpu',weights_only=False)
    if parent['version']!='pav_task_allocation_v1' or parent['step']!=2268:raise ValueError('Expected successful frozen-encoder parent2268')
    encoder=build_hybrid('convnext_se_residual')
    encoder.load_state_dict({k[len('encoder.'):]:v for k,v in parent['model'].items() if k.startswith('encoder.')},strict=True)
    for parameter in encoder.parameters():parameter.requires_grad_(False)
    encoder.eval()
    if job['kind']=='reference':
        model=MultitaskPairClassifier(encoder,tasks);model.load_state_dict(parent['model'],strict=True);model=model.cuda().eval()
        optimizer=None
    else:
        model=StreamingPAVClassifier(encoder,tasks,cfg['model'],common_seed=cfg['common_seed'],core_seed=cfg['core_seed']).cuda()
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    stream=TaskLocalStreams(cfg['train_base_seed'],split='train');step=0
    schedule=SCHEDULES['contour_focus']
    if job.get('checkpoint'):
        cp=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
        if cp['version']!='pav_temporal_v1' or cp['config']!=cfg or cp['source_hashes']!=job['source_hashes']:raise RuntimeError('Temporal checkpoint incompatible')
        model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer']);stream.load_state_dict(cp['stream']);step=cp['step']
        random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy'])
        torch.set_rng_state(cp['rng']['torch']);torch.cuda.set_rng_state(cp['rng']['cuda'])

    def save():
        path=out/f'checkpoint_{step:06d}.pt'
        if not path.exists():
            state=dict(version='pav_temporal_v1',config=cfg,source_hashes=job['source_hashes'],step=step,
                lineage=dict(parent_checkpoint=job['parent_checkpoint'],parent_sha256=job['parent_sha256'],parent_step=2268),
                model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
            with (out/'checkpoint_index.jsonl').open('a') as f:
                f.write(json.dumps(dict(file=path.name,sha256=sha(path),step=step,training_pairs=step*cfg['batch_size']))+'\n')
        return str(path)

    started=time.monotonic();kind=job['kind'];timings=[];task_names=list(tasks)
    if kind in ('train','profile'):
        model.train();model.encoder.eval();torch.cuda.reset_peak_memory_stats()
        if not job.get('checkpoint'):save()
        fields=['step','task','fresh_pairs','loss','grad_norm','accuracy','step_seconds','wall_utc']
        with (out/'metrics.csv').open('a',newline='') as f:
            log=csv.DictWriter(f,fields)
            if f.tell()==0:log.writeheader()
            while step<job['target']:
                if time.time()>deadline-5:break
                # Deterministic round robin yields equal per-task exposure at multiples of task count.
                task=schedule[step%len(schedule)];tick=time.monotonic()
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
            checkpoint=save(),fresh_pairs=step*cfg['batch_size'],worker_seconds=time.monotonic()-started,
            per_task_updates={t:sum(schedule[i%len(schedule)]==t for i in range(step)) for t in task_names})
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
                projection_params=sum(p.numel() for p in model.projections.parameters()),
                core_params=sum(p.numel() for p in model.accumulators.parameters()),
                readout_params=sum(p.numel() for p in model.readout.parameters()),
                state_elements_per_example=model.state_elements_per_example,
                state_bytes_per_example=model.state_elements_per_example*4,
                state_bytes_per_batch=model.state_elements_per_example*4*cfg['batch_size'],
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    elif kind in ('eval','reference'):
        if kind=='eval' and not job.get('checkpoint'):raise ValueError('Temporal evaluation requires saved checkpoint')
        model.eval();evaluation=TaskLocalStreams(job['eval_seed'],split=job['split']);rows=[]
        for task in task_names:
            for offset in range(0,job['n_per_task'],cfg['batch_size']):
                if time.time()>deadline-8:raise TimeoutError('Evaluation deadline reached')
                x,y,metadata=evaluation.batch(min(cfg['batch_size'],job['n_per_task']-offset),task=task)
                modes=['normal'] if job['split']=='val' or kind=='reference' else ['normal','reset_before_second','frame_swap']
                for mode in modes:
                    transformed=swapped_labels(y,task) if mode=='frame_swap' else y
                    with torch.no_grad():
                        gpu_x=x.cuda()
                        logits=model(gpu_x.flip(1),task) if mode=='frame_swap' else model(gpu_x,task,reset_before_second=True) if mode=='reset_before_second' else model(gpu_x,task)
                        probabilities=logits.softmax(1).cpu().tolist()
                    for probability,label,meta in zip(probabilities,transformed.tolist(),metadata):
                        rows.append(dict(model=cfg['model'],seed=seed,step=step,task=task,mode=mode,label=label,
                            probabilities=probability,difficulty=meta.get('difficulty','unspecified'),
                            base_id=meta.get('base_id'),trial_id=meta['trial_id'],metadata=meta))
        # Retain raw scores even if a later summary operation reaches the cap.
        with (out/'predictions.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        result=summarize_modes(rows,tasks,resamples=job.get('resamples',0))
        result.update(status='completed',split=job['split'],step=step,evaluated_pairs=job['n_per_task']*len(tasks),scored_presentations=len(rows),
            worker_seconds=time.monotonic()-started,predictions=str(out/'predictions.jsonl'),checkpoint=job.get('checkpoint') or job['parent_checkpoint'])
        write(out/'summary.json',result)
    else:raise ValueError('Unknown worker kind')
    write(job['result'],result);timer.cancel()


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: train_temporal.py AUTHORIZED_JOB.json')
    worker(read(sys.argv[1]))
