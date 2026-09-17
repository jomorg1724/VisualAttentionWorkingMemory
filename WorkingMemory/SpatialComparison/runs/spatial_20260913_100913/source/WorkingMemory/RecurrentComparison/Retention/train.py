"""Single WM GPU worker. Only explicit jobs with a fixed absolute deadline run."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,csv,random,threading,copy
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha

def schedule(step,cfg):
    from WorkingMemory.RecurrentComparison.Retention.protocol import scheduled_name
    name=scheduled_name(step,cfg)
    return cfg['cells'][name]['family'],name,'retention'


def worker(job):
    deadline=float(job['deadline'])
    if time.time()>=deadline:raise TimeoutError('Experiment deadline passed')
    timer=threading.Timer(deadline-time.time(),lambda:os._exit(124));timer.daemon=True;timer.start()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from WorkingMemory.RecurrentComparison.model import RecurrentOpponent,optimizer_groups
    from WorkingMemory.stimuli import TASK_CLASSES
    VERSION="ei_retention_core_readout_v1"
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
    if parent['version']!='ei_readout_only_continuation_v1' or parent['step']!=9840:raise RuntimeError('Expected strict refitted EI9840 policy migration')
    model=RecurrentOpponent(TASK_CLASSES,'ei_adaptive',cfg['common_seed'],cfg['core_seed'],False)
    model.load_state_dict(parent['model'],strict=True);model.cuda()
    trainable_prefixes=('memory.','memory_output.','readout.heads.motion_direction.','readout.heads.orientation.')
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(trainable_prefixes))
    model.eval()
    optimizer=torch.optim.Adam(optimizer_groups(model,cfg['new_lr'],cfg['parent_lr'],cfg['weight_decay']),eps=cfg['adam_eps'])
    optimizer.load_state_dict(parent['optimizer'])
    # Same architecture and exact parameter ordering/group construction; verify every stored moment shape.
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state=optimizer.state.get(parameter,{})
            if 'exp_avg' in state and state['exp_avg'].shape!=parameter.shape:raise RuntimeError('Adam moment shape mismatch')
    stream=SequenceStream(cfg['train_seed'],'train');stream.load_state_dict(parent['stream'])
    step=parent['step'];counts=Counter(parent['counts']);cell_counts=Counter(parent['cell_counts']);metadata_counts=Counter(parent['metadata_counts'])
    random.setstate(parent['rng']['python']);np.random.set_state(parent['rng']['numpy']);torch.set_rng_state(parent['rng']['torch']);torch.cuda.set_rng_state(parent['rng']['cuda'])
    initial_frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    write(out/'initialization.json',dict(parent_step=9840,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],adam_state_transferred=True,sampler_state_transferred=True,all_upstream_eval=True))
    if job.get('checkpoint'):
        cp_path=Path(job['checkpoint']);index=[json.loads(s) for s in (cp_path.parent/'checkpoint_index.jsonl').read_text().splitlines()]
        if not any(v['file']==cp_path.name and v['sha256']==sha(cp_path) for v in index):raise RuntimeError('Checkpoint index identity failed')
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['version']!=VERSION or cp['config']!=cfg or cp['source_hashes']!=job['source_hashes']:raise RuntimeError('WM checkpoint incompatible')
        model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer']);stream.load_state_dict(cp['stream'])
        step=cp['step'];counts=Counter(cp['counts']);cell_counts=Counter(cp['cell_counts']);metadata_counts=Counter(cp['metadata_counts'])
        random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy']);torch.set_rng_state(cp['rng']['torch']);torch.cuda.set_rng_state(cp['rng']['cuda'])

    def save():
        path=out/f'checkpoint_{step:06d}.pt'
        if not path.exists():
            state=dict(version=VERSION,config=cfg,source_hashes=job['source_hashes'],step=step,
                lineage=dict(warm_start_version=VERSION,parent_checkpoint=job['parent_checkpoint'],parent_sha256=job['parent_sha256'],parent_step=9840,new_optimizer=False,expanded_protocol_only=True),
                model=model.state_dict(),optimizer=optimizer.state_dict(),stream=stream.state_dict(),scheduler=dict(version='seeded_permuted80_v1',seed=cfg['scheduler_seed'],offset=step-9840),
                counts=dict(counts),cell_counts=dict(cell_counts),metadata_counts=dict(metadata_counts),
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
            with (out/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(file=path.name,sha256=sha(path),step=step,episodes=counts['episodes']))+'\n')
        return str(path)

    def features(images,record=False):
        traces=();memory=None;states=[]
        for t in range(images.shape[1]):
            with torch.no_grad():
                result=model._sensory(images[:,t],*traces);field,traces=result[0],result[1:]
                z=model.memory_input(field)
            r,memory,stats=model.memory(z,memory,diagnostic=record)
            if record:
                for v in memory:
                    if v.requires_grad:v.retain_grad()
                states.append(memory)
        with torch.no_grad():sensory=model.readout.trunk(torch.cat((field.mean((2,3)),field.amax((2,3))),1))
        return sensory,r,states

    def train_batch(family,condition,n):
        tick=time.monotonic();x,y,metadata=stream.batch(n,family,condition);record=(step%64==0 or job['kind']=='profile')
        optimizer.zero_grad(set_to_none=True);sensory,r,states=features(x.cuda(),record)
        logits=model.classify(sensory+model.memory_output(r),family)
        if job['kind']=='profile' and not getattr(train_batch,'checked',False):
            with torch.no_grad():direct=model(x.cuda(),family)
            if not torch.equal(logits.detach(),direct):raise RuntimeError('Frozen extractor/output mismatch')
            train_batch.checked=True
        loss=F.cross_entropy(logits,y.cuda().long())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad} if record else {}
        loss.backward();detail={}
        if record:
            detail['projection_gradient_norm']=float(model.memory_output.weight.grad.norm())
            detail['head_gradient_norm']=float(model.readout.heads[family].weight.grad.norm())
            detail['core_gradient_norm']=float(sum(p.grad.square().sum() for p in model.memory.parameters() if p.grad is not None).sqrt())
            detail['raw_recurrent_gradient_rms']=float(model.memory.raw_recurrent.grad.square().mean().sqrt())
            for where,state in [('early',states[0]),('late',states[-1])]:
                for index,tensor in enumerate(state):
                    detail[where+('_r' if index==0 else '_a')+'_gradient_norm']=None if tensor.grad is None else float(tensor.grad.norm())
                    detail[where+('_r' if index==0 else '_a')+'_rms']=float(tensor.detach().square().mean().sqrt())
            if job['kind']=='profile' and not (detail['early_r_gradient_norm'] is not None and detail['early_r_gradient_norm']>0):raise RuntimeError('Early recurrent gradient missing')
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['grad_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();torch.cuda.synchronize()
        if record:
            detail['relative_parameter_updates']={name:float((p.detach()-before[name]).norm()/before[name].norm().clamp_min(1e-12)) for name,p in model.named_parameters() if name in before}
        if job['kind']=='profile':
            for name,p in model.named_parameters():
                if name in initial_frozen and not torch.equal(p.detach().cpu(),initial_frozen[name]):raise RuntimeError('Frozen parameter changed '+name)
            detail['all_frozen_parameters_exact']=True
        return dict(loss=float(loss),grad_norm=float(norm),clipped=int(float(norm)>cfg['grad_clip']),accuracy=float((logits.argmax(1).cpu()==y).float().mean()),diagnostics=json.dumps(detail),step_seconds=time.monotonic()-tick,frames=int(x.shape[1]),metadata=metadata)

    started=time.monotonic();kind=job['kind'];model.eval()
    if kind=='profile':
        torch.cuda.reset_peak_memory_stats();measurements=[]
        for family,condition in job['profile_cells']:
            if time.time()>=deadline-10:raise TimeoutError('Profile reached cap')
            result=train_batch(family,condition,cfg['batch_size']);meta=result.pop('metadata')
            tick=time.monotonic();x,_,_=stream.batch(cfg['batch_size'],family,condition)
            model.eval()
            with torch.no_grad():model(x.cuda(),family)
            torch.cuda.synchronize();result['eval_seconds']=time.monotonic()-tick;model.eval()
            measurements.append(dict(family=family,condition=condition,**result))
        step=9840+len(measurements);counts['episodes']=78720+len(measurements)*cfg['batch_size']
        result=dict(status='completed',measurements=measurements,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
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
        for name,p in model.named_parameters():
            if name in initial_frozen and not torch.equal(p.detach().cpu(),initial_frozen[name]):raise RuntimeError('Frozen parameter changed '+name)
        result=dict(status='completed' if step==job['target'] else 'budget_stopped',step=step,checkpoint=save(),
            counts=dict(counts),cell_counts=dict(cell_counts),metadata_counts=dict(metadata_counts),worker_seconds=time.monotonic()-started)
    elif kind=='eval':
        # Without a new-protocol checkpoint, evaluate the untouched trained EI9840 parent.
        model.eval();evaluation=SequenceStream(job['eval_seed'],job['split']);rows=[]
        groups={}
        for name,cell in cfg['cells'].items():groups.setdefault(cell.get('paired_family') or name,[]).append((name,cell))
        with (out/'predictions.jsonl').open('w') as f:
            for group,cells in groups.items():
                for offset in range(0,job['n_per_cell'],cfg['batch_size']):
                    state=copy.deepcopy(evaluation.state_dict());reference=None
                    for name,cell in cells:
                        evaluation.load_state_dict(state);family=cell['family'];condition=cell['condition']
                        n=min(cfg['batch_size'],job['n_per_cell']-offset)
                        if time.time()>=deadline-10:raise TimeoutError('Evaluation reached cap')
                        x,y,metadata=evaluation.batch(n,family,condition)
                        stable=x if condition['protocol']=='anchor' else (torch.cat((x[:,:10],x[:,-1:]),1) if family=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1))
                        if reference is None:reference=(stable,y)
                        elif not torch.equal(reference[0],stable) or not torch.equal(reference[1],y):raise RuntimeError('Delay-paired evidence/cue/probe mismatch')
                        with torch.no_grad():probabilities=model(x.cuda(),family).softmax(1).cpu().tolist()
                        for p,label,meta in zip(probabilities,y.tolist(),metadata):
                            paired_id=meta['trial_id'];meta['paired_base_id']=paired_id
                            row=dict(task=family,condition=name,protocol=condition['protocol'],label=label,probabilities=p,base_id=None,trial_id=paired_id+'/'+name,paired_base_id=paired_id,metadata=meta,difficulty=meta.get('difficulty','unspecified'))
                            rows.append(row);f.write(json.dumps(row)+'\n')
                        f.flush()
                print(json.dumps(dict(evaluated=group,episodes_per_delay=job['n_per_cell'])),flush=True)
        result=evaluate_rows(rows,resamples=job.get('resamples',0))
        result['all_ten_cell_mean_auc']=result['selection_mean_auc']
        result['selection_mean_auc']=float(np.mean([v['overall']['macro_ovr_auc'] for v in result['cells'].values() if not v['condition'].endswith('_anchor')]))
        result['interpretation']='Equal eight primary retention cells select by mean AUC; anchors descriptive; delays share evidence and are paired presentations.'
        result.update(status='completed',step=step,split=job['split'],predictions=str(out/'predictions.jsonl'),checkpoint=job.get('checkpoint',job['parent_checkpoint']),worker_seconds=time.monotonic()-started)
        write(out/'summary.json',result)
    else:raise ValueError('Unknown worker kind')
    write(job['result'],result);timer.cancel()

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: train.py EXPLICIT_JOB.json')
    worker(read(sys.argv[1]))
