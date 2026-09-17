"""Bounded motion-only continuation; no cloud or other-task computations."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,random,threading,copy,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,sha
from WorkingMemory.TrainingExposure.sweep import write
VERSION='single_task_motion_continuation_v1'
TASK='motion_duration_cued'

def worker(job):
    import torch,numpy as np
    from WorkingMemory.SpatialTaskBattery.BiasedTraining.model import BiasedMemory,groups
    from WorkingMemory.SpatialTaskBattery.stimuli import SpatialBatteryStream
    from PreAttentiveVision.evaluate_multitask import summarize
    deadline=job['deadline'];timer=threading.Timer(max(.01,deadline-time.time()),lambda:os._exit(124));timer.daemon=True;timer.start()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    for n,h in job['source_hashes'].items():assert sha(ROOT/n)==h,n
    assert sha(job['parent'])==job['parent_sha256']
    parent=torch.load(job['parent'],map_location='cpu');assert parent['version']=='spatial_five_task_original_attention_biases_v1' and parent['step']==10000
    cfg=parent['config'];model=BiasedMemory(cfg);model.load_state_dict(parent['model'],strict=True);opt=torch.optim.Adam(groups(model,cfg),eps=cfg['adam_eps']);opt.load_state_dict(parent['optimizer'])
    stream=SpatialBatteryStream(cfg['train_seed'],'train');stream.load_state_dict(parent['stream']);step=parent['step'];counts=dict(episodes=0,frames=0);cell_counts={str(d):0 for d in (0,4,12,24)}
    lineage=dict(parent_step=step,parent_sha256=job['parent_sha256'],parent_counts=parent['counts'],parent_cell_counts=parent['cell_counts'],optimizer='Exact compatible Adam state/LRs retained; no LR compensation',loss='One full mean motion CE per update; cloud uses CE/5 plus four other gradients',stream='Exact checkpoint family-local streams and RNG; only motion is advanced; original motion delay scheduler continues at same global step')
    def restore_rng(rng):
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
    model.cuda()
    for state in opt.state.values():
        for k,v in state.items():
            if torch.is_tensor(v) and k!='step':state[k]=v.cuda()
    restore_rng(parent['rng']);del parent
    if job.get('checkpoint'):
        p=Path(job['checkpoint']);idx=[json.loads(l) for l in (p.parent/'checkpoint_index.jsonl').read_text().splitlines()];assert any(x['file']==p.name and x['sha256']==sha(p) for x in idx)
        cp=torch.load(p,map_location='cpu');assert cp['version']==VERSION and cp['source_hashes']==job['source_hashes'];model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);stream.load_state_dict(cp['stream']);step=cp['step'];counts=cp['counts'];cell_counts=cp['cell_counts'];restore_rng(cp['rng']);del cp
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);write(out/'lineage.json',lineage)
    def save():
        p=out/f'checkpoint_{step:06d}.pt'
        if not p.exists():
            cp=dict(version=VERSION,config=cfg,source_hashes=job['source_hashes'],lineage=lineage,step=step,model=model.state_dict(),optimizer=opt.state_dict(),stream=stream.state_dict(),counts=counts,cell_counts=cell_counts,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            temp=p.with_suffix('.tmp');torch.save(cp,temp);os.replace(temp,p)
            with (out/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(file=p.name,step=step,sha256=sha(p)))+'\n')
        return str(p)
    def train_batch(delay,record=False):
        tick=time.time();x,y,meta=stream.batch(8,TASK,dict(delay=delay));model.train();opt.zero_grad(set_to_none=True)
        if record:logits,diag=model(x.cuda(),TASK,True)
        else:logits=model(x.cuda(),TASK);diag=None
        loss=torch.nn.functional.cross_entropy(logits,y.cuda());loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite loss or gradient')
        diagnostics={}
        if record:
            diagnostics=dict(first_field_gradient=float(diag['first_field'].grad.norm()),source_bias_gradient=float(model.attention.source_bias.grad.norm()),locality_gradient=float(model.attention.raw_locality.grad.norm()),per_frame_attention=diag['records'])
        opt.step();torch.cuda.synchronize()
        return dict(loss=float(loss),accuracy=float((logits.argmax(1).cpu()==y).float().mean()),gradient_norm=float(norm),clipped=int(norm>cfg['clip']),frames=x.shape[1],seconds=time.time()-tick,diagnostics=diagnostics)
    started=time.time();kind=job['kind']
    if kind=='profile':
        rows=[];torch.cuda.reset_peak_memory_stats()
        for delay in (0,24):
            v=train_batch(delay,True);model.eval();x,y,_=stream.batch(8,TASK,dict(delay=delay));tick=time.time()
            with torch.no_grad():model(x.cuda(),TASK)
            torch.cuda.synchronize();v.update(delay=delay,eval_seconds=time.time()-tick);rows.append(v)
        result=dict(status='completed',rows=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),profile_exposure_discarded=True)
    elif kind=='train':
        save();names=cfg['train_names'][TASK];ti=list(cfg['task_classes']).index(TASK)
        with (out/'metrics.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,['step','added_episodes','delay','loss','accuracy','gradient_norm','clipped','frames','seconds','diagnostics'])
            if f.tell()==0:writer.writeheader()
            while step<job['target']:
                if time.time()>deadline-15:break
                block,pos=divmod(step-8400,len(names));rng=np.random.default_rng(cfg['scheduler_seed']+100003*ti+block);name=names[int(rng.permutation(len(names))[pos])];delay=cfg['cells'][name]['condition']['delay'];value=train_batch(delay,step%128==0);step+=1;counts['episodes']+=8;counts['frames']+=8*value['frames'];cell_counts[str(delay)]+=8
                value['diagnostics']=json.dumps(value['diagnostics']);writer.writerow(dict(step=step,added_episodes=counts['episodes'],delay=delay,**value));f.flush()
                if step%32==0:print(json.dumps(dict(step=step,added_episodes=counts['episodes'],loss=value['loss'])),flush=True)
                if step%256==0:save()
        result=dict(status='completed' if step==job['target'] else 'budget_stopped',step=step,checkpoint=save(),counts=counts,cell_counts=cell_counts)
    elif kind=='eval':
        model.eval();rows=[]
        with (out/'predictions.jsonl').open('w') as f:
            for delay in (0,4,12,24):
                evaluation=SpatialBatteryStream(job['eval_seed'],job['split'])
                for offset in range(0,job['n'],8):
                    x,y,meta=evaluation.batch(min(8,job['n']-offset),TASK,dict(delay=delay))
                    with torch.no_grad():prob=model(x.cuda(),TASK).softmax(1).cpu().tolist()
                    for j,(pr,label,m) in enumerate(zip(prob,y.tolist(),meta)):
                        row=dict(task=TASK,condition=f'D{delay}',protocol=TASK,label=label,probabilities=pr,paired_base_id=str(offset+j),trial_id=f'{offset+j}/D{delay}',metadata=m);rows.append(row);f.write(json.dumps(row)+'\n')
                    if time.time()>deadline-15:raise TimeoutError('Evaluation deadline')
                f.flush()
        cells={f'D{d}':summarize([r for r in rows if r['condition']==f'D{d}'],0) for d in (0,4,12,24)}
        result=dict(status='completed',step=step,cells=cells,rank=[min(m['balanced_accuracy'] for m in cells.values()),float(np.mean([m['macro_ovr_auc'] for m in cells.values()]))],predictions=str(out/'predictions.jsonl'));write(out/'summary.json',result)
    else:raise ValueError(kind)
    result['worker_seconds']=time.time()-started;write(job['result'],result);timer.cancel()
if __name__=='__main__':worker(read(sys.argv[1]))
