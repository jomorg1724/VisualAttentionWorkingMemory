"""Single WM GPU worker. Only explicit jobs with a fixed absolute deadline run."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,csv,random,threading
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha

def schedule(step,cfg):
    cycle=cfg['cycle']
    name=cycle[step%len(cycle)]
    return cfg['cells'][name]['family'],name,'focused'


def worker(job):
    deadline=float(job['deadline'])
    if time.time()>=deadline:raise TimeoutError('Experiment deadline passed')
    timer=threading.Timer(deadline-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from WorkingMemory.RecurrentComparison.model import load_parent,VERSION,optimizer_groups
    from WorkingMemory.stimuli import SequenceStream
    from WorkingMemory.RecurrentComparison.evaluate import evaluate_rows
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    for name,digest in job['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Pinned source changed: '+name)
    if sha(job['parent_checkpoint'])!=job['parent_sha256']:raise RuntimeError('Parent identity changed')
    cfg=job['config'];out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    parent=torch.load(job['parent_checkpoint'],map_location='cpu',weights_only=False)
    model=load_parent(parent,cfg['arm'],cfg['common_seed'],cfg['core_seed'],cfg['activation_checkpoint']).cuda()
    optimizer=torch.optim.Adam(optimizer_groups(model,cfg['new_lr'],cfg['parent_lr'],cfg['weight_decay']),eps=cfg['adam_eps'])
    stream=SequenceStream(cfg['train_seed'],'train');step=0;counts=Counter();cell_counts=Counter();metadata_counts=Counter()
    if job.get('checkpoint'):
        cp_path=Path(job['checkpoint']);index=[json.loads(s) for s in (cp_path.parent/'checkpoint_index.jsonl').read_text().splitlines()]
        if not any(v['file']==cp_path.name and v['sha256']==sha(cp_path) for v in index):raise RuntimeError('Checkpoint index identity failed')
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['version']!=VERSION or cp['config']!=cfg or cp['source_hashes']!=job['source_hashes']:raise RuntimeError('WM checkpoint incompatible')
        model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer']);stream.load_state_dict(cp['stream'])
        step=cp['step'];counts.update(cp['counts']);cell_counts.update(cp['cell_counts']);metadata_counts.update(cp['metadata_counts'])
        random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy']);torch.set_rng_state(cp['rng']['torch']);torch.cuda.set_rng_state(cp['rng']['cuda'])

    def save():
        path=out/f'checkpoint_{step:06d}.pt'
        if not path.exists():
            state=dict(version=VERSION,config=cfg,source_hashes=job['source_hashes'],step=step,
                lineage=dict(warm_start_version=VERSION,parent_checkpoint=job['parent_checkpoint'],parent_sha256=job['parent_sha256'],parent_step=6860,new_optimizer=True),
                model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),
                counts=dict(counts),cell_counts=dict(cell_counts),metadata_counts=dict(metadata_counts),
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
            with (out/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(file=path.name,sha256=sha(path),step=step,episodes=counts['episodes']))+'\n')
        return str(path)

    def train_batch(family,condition,n):
        tick=time.monotonic();x,y,metadata=stream.batch(n,family,condition)
        record=(step%64==0 or job['kind']=='profile')
        optimizer.zero_grad(set_to_none=True)
        output=model(x.cuda(),family,diagnostic=record)
        logits,states,zs=output if record else (output,None,None)
        loss=F.cross_entropy(logits,y.cuda())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        detail={}
        if record:
            detail['state_mean']={k:sum(float(s[k]) for s in states)/len(states) for k in states[0]}
            detail['state_peak']={k:max(float(s[k]) for s in states) for k in states[0] if k.endswith('_max')}
            detail['early_representation_gradient_norm']=float(zs[0].grad.norm())
            detail['late_representation_gradient_norm']=float(zs[-1].grad.norm())
            detail['core_gradient_norm']=float(sum(p.grad.square().sum() for p in model.memory.parameters() if p.grad is not None).sqrt())
            recurrent=model.memory.raw_recurrent if model.arm=='ei_adaptive' else model.memory.recurrent.weight
            gradient=recurrent.grad.detach()
            detail['recurrent_gradient_rms']=float(gradient.square().mean().sqrt())
            detail['recurrent_gradient_median_abs']=float(gradient.abs().median())
            detail['current_gradient_fraction_below_adam_eps']=float((gradient.abs()<cfg['adam_eps']).float().mean())
            effective_before=(model.memory.recurrent_weight() if model.arm=='ei_adaptive' else recurrent).detach().clone()
            if model.arm=='ei_adaptive':
                raw=model.memory.raw_recurrent;before=raw.detach().clone()
                detail['raw_recurrent_gradient_norm']=float(raw.grad.norm())
                detail['raw_recurrent_relative_gradient_norm']=float(raw.grad.norm()/raw.norm())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();torch.cuda.synchronize()
        if record:
            effective_after=(model.memory.recurrent_weight() if model.arm=='ei_adaptive' else recurrent).detach()
            detail['effective_recurrent_relative_update']=float((effective_after-effective_before).norm()/effective_before.norm())
        if record and model.arm=='ei_adaptive':detail['raw_recurrent_relative_update']=float((raw.detach()-before).norm()/before.norm())
        return dict(loss=float(loss),grad_norm=float(norm),clipped=int(float(norm)>cfg['grad_clip']),
                    accuracy=float((logits.argmax(1).cpu()==y).float().mean()),diagnostics=json.dumps(detail),
                    step_seconds=time.monotonic()-tick,frames=int(x.shape[1]),metadata=metadata)

    started=time.monotonic();kind=job['kind'];model.train()
    if kind=='profile':
        torch.cuda.reset_peak_memory_stats();measurements=[]
        for family,condition in job['profile_cells']:
            if time.time()>=deadline-10:raise TimeoutError('Profile reached cap')
            result=train_batch(family,condition,cfg['batch_size']);meta=result.pop('metadata')
            tick=time.monotonic();x,_,_=stream.batch(cfg['batch_size'],family,condition)
            model.eval()
            with torch.no_grad():model(x.cuda(),family)
            torch.cuda.synchronize();result['eval_seconds']=time.monotonic()-tick;model.train()
            measurements.append(dict(family=family,condition=condition,**result))
        step=len(measurements);counts['episodes']=step*cfg['batch_size']
        result=dict(status='completed',measurements=measurements,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),trainable_parameters=sum(p.numel() for p in model.parameters()),
            batch_size=cfg['batch_size'],checkpoint=save(),worker_seconds=time.monotonic()-started)
    elif kind=='train':
        if not job.get('checkpoint'):save()
        fields=['step','stage','family','condition','episodes','visual_updates','loss','grad_norm','clipped','accuracy','diagnostics','frames','step_seconds','wall_utc']
        with (out/'metrics.csv').open('a',newline='') as f:
            log=csv.DictWriter(f,fields)
            if f.tell()==0:log.writeheader()
            while step<job['target']:
                if time.time()>=deadline-10:break
                family,name,stage=schedule(step,cfg)
                condition=cfg['cells'][name]['condition']
                result=train_batch(family,condition,cfg['batch_size']);n=cfg['batch_size']
                counts['episodes']+=n;counts['visual_updates']+=n*result['frames'];counts[stage+'_episodes']+=n
                cell_counts[stage+'/'+family+'/'+name]+=n
                for meta in result.pop('metadata'):
                    for key,value in meta.get('counts',{}).items():
                        if isinstance(value,(int,float)):metadata_counts[key]+=value
                    for key in ('frame_count','encoder_updates','evidence_events','sample_presentations','cue_presentations','probe_presentations','report_presentations','blank_presentations','distractor_presentations'):
                        if isinstance(meta.get(key),(int,float)):metadata_counts[key]+=meta[key]
                step+=1
                log.writerow(dict(step=step,stage=stage,family=family,condition=name,episodes=counts['episodes'],visual_updates=counts['visual_updates'],wall_utc=datetime.now(timezone.utc).isoformat(),**result));f.flush()
                if step%64==0:print(json.dumps(dict(step=step,stage=stage,family=family,condition=name,episodes=counts['episodes'],visual_updates=counts['visual_updates'],loss=result['loss'])),flush=True)
                if step%256==0:save()
        result=dict(status='completed' if step==job['target'] else 'budget_stopped',step=step,checkpoint=save(),
            counts=dict(counts),cell_counts=dict(cell_counts),metadata_counts=dict(metadata_counts),worker_seconds=time.monotonic()-started)
    elif kind=='eval':
        if not job.get('checkpoint'):raise ValueError('Evaluation requires a trained checkpoint')
        model.eval();evaluation=SequenceStream(job['eval_seed'],job['split']);rows=[]
        with (out/'predictions.jsonl').open('w') as f:
            for name,cell in cfg['cells'].items():
                condition=cell['condition']
                for family in [cell['family']]:
                    for offset in range(0,job['n_per_cell'],cfg['batch_size']):
                        if time.time()>=deadline-10:raise TimeoutError('Evaluation reached cap; raw partial predictions preserved')
                        n=min(cfg['batch_size'],job['n_per_cell']-offset)
                        x,y,metadata=evaluation.batch(n,family,condition)
                        with torch.no_grad():probabilities=model(x.cuda(),family,reset_memory_each_frame=job.get('reset_memory',False)).softmax(1).cpu().tolist()
                        for p,label,meta in zip(probabilities,y.tolist(),metadata):
                            row=dict(task=family,condition=name,protocol=condition['protocol'],label=label,probabilities=p,
                                base_id=meta.get('base_id'),trial_id=meta['trial_id'],metadata=meta,difficulty=meta.get('difficulty','unspecified'))
                            rows.append(row);f.write(json.dumps(row)+'\n')
                        f.flush()
                    print(json.dumps(dict(evaluated=family+'/'+name,episodes=job['n_per_cell'])),flush=True)
        result=evaluate_rows(rows,resamples=job.get('resamples',0))
        result.update(status='completed',step=step,split=job['split'],predictions=str(out/'predictions.jsonl'),checkpoint=job['checkpoint'],worker_seconds=time.monotonic()-started)
        write(out/'summary.json',result)
    else:raise ValueError('Unknown worker kind')
    write(job['result'],result);timer.cancel()

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: train.py EXPLICIT_JOB.json')
    worker(read(sys.argv[1]))
