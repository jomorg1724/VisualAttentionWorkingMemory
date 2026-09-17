"""Single WM GPU worker. Only explicit jobs with a fixed absolute deadline run."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,csv,random,threading
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha

JOINT=['anchor']+['integration','recall']*4+['integration','anchor','recall']+['integration','recall']*4
assert len(JOINT)==20 and [JOINT.count(k) for k in ('anchor','integration','recall')]==[2,9,9]


def schedule(step,cfg):
    families=list(cfg['task_classes'])
    if step<cfg['bridge_steps']:
        conditions=list(cfg['bridge_conditions'])
        return families[step%7],conditions[(step//7)%len(conditions)],'bridge'
    j=step-cfg['bridge_steps'];group=JOINT[j%20]
    index=(j//20)*JOINT.count(group)+JOINT[:j%20].count(group)
    conditions=[k for k,v in cfg['train_conditions'].items() if v['protocol']==group]
    if not conditions:raise ValueError('Missing training conditions for '+group)
    return families[index%7],conditions[(index//7)%len(conditions)],'joint'


def worker(job):
    deadline=float(job['deadline'])
    if time.time()>=deadline:raise TimeoutError('Experiment deadline passed')
    timer=threading.Timer(deadline-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from WorkingMemory.model import load_parent,VERSION
    from WorkingMemory.stimuli import SequenceStream
    from WorkingMemory.evaluate import evaluate_rows
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    for name,digest in job['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Pinned source changed: '+name)
    if sha(job['parent_checkpoint'])!=job['parent_sha256']:raise RuntimeError('Parent identity changed')
    cfg=job['config'];out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    parent=torch.load(job['parent_checkpoint'],map_location='cpu',weights_only=False)
    model=load_parent(parent,cfg['task_classes'],cfg['activation_checkpoint']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
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
                lineage=dict(parent_checkpoint=job['parent_checkpoint'],parent_sha256=job['parent_sha256'],parent_step=4032,new_optimizer=True),
                model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),
                counts=dict(counts),cell_counts=dict(cell_counts),metadata_counts=dict(metadata_counts),
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
            with (out/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(file=path.name,sha256=sha(path),step=step,episodes=counts['episodes']))+'\n')
        return str(path)

    def train_batch(family,condition,n):
        tick=time.monotonic();x,y,metadata=stream.batch(n,family,condition)
        optimizer.zero_grad(set_to_none=True);logits=model(x.cuda(),family);loss=F.cross_entropy(logits,y.cuda())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();torch.cuda.synchronize()
        return dict(loss=float(loss),grad_norm=float(norm),accuracy=float((logits.argmax(1).cpu()==y).float().mean()),
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
        fields=['step','stage','family','condition','episodes','visual_updates','loss','grad_norm','accuracy','frames','step_seconds','wall_utc']
        with (out/'metrics.csv').open('a',newline='') as f:
            log=csv.DictWriter(f,fields)
            if f.tell()==0:log.writeheader()
            while step<job['target']:
                if time.time()>=deadline-10:break
                family,name,stage=schedule(step,cfg)
                condition=(cfg['bridge_conditions'] if stage=='bridge' else cfg['train_conditions'])[name]
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
            for name,condition in job['conditions'].items():
                for family in cfg['task_classes']:
                    for offset in range(0,job['n_per_cell'],cfg['batch_size']):
                        if time.time()>=deadline-10:raise TimeoutError('Evaluation reached cap; raw partial predictions preserved')
                        n=min(cfg['batch_size'],job['n_per_cell']-offset)
                        x,y,metadata=evaluation.batch(n,family,condition)
                        with torch.no_grad():probabilities=model(x.cuda(),family).softmax(1).cpu().tolist()
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
