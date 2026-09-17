"""Architecture-only continuation worker under an absolute deadline."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,copy,random,threading,csv
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha
def worker(job):
    import numpy as np
    import torch
    from WorkingMemory.PreUpdateAttention.model import migrate,VERSION,groups
    from WorkingMemory.SpatialComparison.stimuli import SpatialStream,FAMILY_HEAD
    from WorkingMemory.RecurrentComparison.evaluate import evaluate_rows
    deadline=job['deadline'];timer=threading.Timer(max(.01,deadline-time.time()),lambda:os._exit(124));timer.daemon=True;timer.start()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    for name,digest in job['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Pinned source changed '+name)
    if sha(job['parent'])!=job['parent_sha256']:raise RuntimeError('Parent hash mismatch')
    cfg=job['config'];arm=job['arm'];out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['model_seed']);np.random.seed(cfg['model_seed']);torch.manual_seed(cfg['model_seed']);torch.cuda.manual_seed_all(cfg['model_seed'])
    parent=torch.load(job['parent'],map_location='cpu');model,opt,lineage=migrate(parent,arm,cfg)
    initializer=ROOT/'portable/attention_initialization.pt'
    if initializer.exists():
        initial=torch.load(initializer,map_location='cpu');model.load_state_dict(initial,strict=True)
        for name,value in parent['model'].items():assert torch.equal(initial[name],value),name
    stream=SpatialStream(cfg['train_seed'],'train');stream.load_state_dict(parent['stream']);step=parent['step'];counts=Counter(parent['counts']);cell_counts=Counter(parent['cell_counts']);model.cuda()
    random.setstate(parent['rng']['python']);np.random.set_state(parent['rng']['numpy']);torch.set_rng_state(parent['rng']['torch']);torch.cuda.set_rng_state(parent['rng']['cuda']);del parent
    for state in opt.state.values():
        for k,v in state.items():
            if torch.is_tensor(v) and k!='step':state[k]=v.cuda()
    if job.get('checkpoint'):
        cp_path=Path(job['checkpoint']);idx=[json.loads(x) for x in (cp_path.parent/'checkpoint_index.jsonl').read_text().splitlines()]
        assert any(r['file']==cp_path.name and r['sha256']==sha(cp_path) for r in idx)
        cp=torch.load(cp_path,map_location='cpu');assert cp['version']==VERSION and cp['arm']==arm and cp['config']==cfg and cp['source_hashes']==job['source_hashes']
        model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);stream.load_state_dict(cp['stream']);step=cp['step'];counts=Counter(cp['counts']);cell_counts=Counter(cp['cell_counts'])
        random.setstate(cp['rng']['python']);np.random.set_state(cp['rng']['numpy']);torch.set_rng_state(cp['rng']['torch']);torch.cuda.set_rng_state(cp['rng']['cuda']);del cp
    write(out/'initialization.json',dict(lineage=lineage,total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),state_entries=21632))
    def save():
        path=out/f'checkpoint_{step:06d}.pt'
        if not path.exists():
            data=dict(version=VERSION,arm=arm,config=cfg,source_hashes=job['source_hashes'],lineage=lineage,step=step,model=model.state_dict(),optimizer=opt.state_dict(),stream=stream.state_dict(),counts=dict(counts),cell_counts=dict(cell_counts),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
            tmp=path.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,path)
            with (out/'checkpoint_index.jsonl').open('a') as f:f.write(json.dumps(dict(file=path.name,step=step,sha256=sha(path)))+'\n')
        return str(path)
    def train_batch(name,record=False):
        tick=time.time();c=cfg['cells'][name];x,y,meta=stream.batch(cfg['batch_size'],c['family'],c['condition']);model.train();opt.zero_grad(set_to_none=True)
        if record:logits,diag=model(x.cuda(),FAMILY_HEAD[c['family']],True)
        else:logits=model(x.cuda(),FAMILY_HEAD[c['family']]);diag=None
        loss=torch.nn.functional.cross_entropy(logits,y.cuda().long());loss.backward();detail={}
        if record:
            detail['first_sensory_field_gradient_norm']=float(diag['first_field'].grad.norm())
            for position,state in [('early',diag['states'][0]),('late',diag['states'][-1])]:
                for kind,tensor in zip(('r','a'),state):detail[position+'_'+kind+'_gradient_norm']=None if tensor.grad is None else float(tensor.grad.norm());detail[position+'_'+kind+'_rms']=float(tensor.detach().square().mean().sqrt())
            detail['attention_gradient_norm']=float(sum(p.grad.square().sum() for p in model.attention.parameters() if p.grad is not None).sqrt())
            detail['encoder_gradient_norm']=float(sum(p.grad.square().sum() for p in model.encoder.parameters() if p.grad is not None).sqrt())
            detail['raw_recurrent_gradient_rms']=float(model.memory.raw_recurrent.grad.square().mean().sqrt());detail['state_stats']=diag['records'][-1]
            before=model.memory.recurrent_weight().detach().clone()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite loss/gradient')
        opt.step()
        if record:detail['effective_recurrent_relative_update']=float((model.memory.recurrent_weight().detach()-before).norm()/before.norm())
        torch.cuda.synchronize()
        return dict(loss=float(loss),accuracy=float((logits.argmax(1).cpu()==y).float().mean()),gradient_norm=float(norm),clipped=int(float(norm)>cfg['clip']),frames=int(x.shape[1]),seconds=time.time()-tick,diagnostics=detail)
    started=time.time();kind=job['kind']
    if kind=='profile':
        rows=[];torch.cuda.reset_peak_memory_stats()
        for name in job['profile_names']:
            value=train_batch(name,True);c=cfg['cells'][name];x,y,m=stream.batch(cfg['batch_size'],c['family'],c['condition']);tick=time.time();model.eval()
            with torch.no_grad():model(x.cuda(),FAMILY_HEAD[c['family']])
            torch.cuda.synchronize();value.update(name=name,eval_seconds=time.time()-tick);rows.append(value);step+=1;counts['episodes']+=cfg['batch_size']
        result=dict(status='completed',rows=rows,checkpoint=save(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=sum(p.numel() for p in model.parameters()),state_entries=21632)
    elif kind=='train':
        if not job.get('checkpoint'):save()
        with (out/'metrics.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,['step','episodes','frames_total','cell','loss','accuracy','gradient_norm','clipped','frames','seconds','diagnostics'])
            if f.tell()==0:writer.writeheader()
            while step<job['target']:
                if time.time()>deadline-15:break
                block,pos=divmod(step,len(cfg['cycle']));order=np.random.default_rng(cfg['scheduler_seed']+block).permutation(len(cfg['cycle']));name=cfg['cycle'][int(order[pos])]
                value=train_batch(name,step%128==0);step+=1;counts['episodes']+=cfg['batch_size'];counts['frames']+=cfg['batch_size']*value['frames'];cell_counts[name]+=cfg['batch_size'];value['diagnostics']=json.dumps(value['diagnostics'])
                writer.writerow(dict(step=step,episodes=counts['episodes'],frames_total=counts['frames'],cell=name,**value));f.flush()
                if step%64==0:print(json.dumps(dict(arm=arm,step=step,episodes=counts['episodes'],cell=name,loss=value['loss'])),flush=True)
                if step%256==0:save()
        result=dict(status='completed' if step==job['target'] else 'budget_stopped',step=step,checkpoint=save(),counts=dict(counts),cell_counts=dict(cell_counts))
    elif kind=='eval':
        model.eval();rows=[];evaluators={}
        with (out/'predictions.jsonl').open('w') as f:
            for family in ('orientation_single','orientation_binding','motion_direction'):
                modes=('id','locations') if family=='orientation_binding' and job.get('heldout_locations') else ('id',)
                for mode in modes:
                    split='locations' if mode=='locations' else job['split'];evaluation=SpatialStream(job['eval_seed'],split)
                    cells=[(name,c) for name,c in cfg['cells'].items() if c['family']==family]
                    for offset in range(0,job['n'],cfg['batch_size']):
                        state=copy.deepcopy(evaluation.state_dict());reference=None
                        for name,c in cells:
                            evaluation.load_state_dict(state);x,y,meta=evaluation.batch(min(cfg['batch_size'],job['n']-offset),family,c['condition']);stable=torch.cat((x[:,:10],x[:,-1:]),1) if family=='motion_direction' else torch.cat((x[:,:3],x[:,-2:]),1)
                            if reference is None:reference=(stable,y)
                            else:assert torch.equal(stable,reference[0]) and torch.equal(y,reference[1])
                            with torch.no_grad():prob=model(x.cuda(),FAMILY_HEAD[family]).softmax(1).cpu().tolist()
                            for j,(pr,label,m) in enumerate(zip(prob,y.tolist(),meta)):
                                row=dict(task=FAMILY_HEAD[family],condition=name+('_locations' if mode=='locations' else ''),protocol=family,label=label,probabilities=pr,base_id=None,paired_base_id=f'{family}/{mode}/{offset+j}',trial_id=f'{family}/{mode}/{offset+j}/{name}',difficulty=m.get('spacing','unspecified'),metadata=m)
                                rows.append(row);f.write(json.dumps(row)+'\n')
                            f.flush()
                            if time.time()>deadline-15:raise TimeoutError('Evaluation deadline')
                    print(json.dumps(dict(evaluated=family,mode=mode,n=job['n'])),flush=True)
        result=evaluate_rows(rows,resamples=0);selected=[v['overall']['macro_ovr_auc'] for v in result['cells'].values() if not v['condition'].endswith('_locations') and not v['condition'].startswith('motion')]
        result.update(status='completed',step=step,selection_mean_auc=float(np.mean(selected)),predictions=str(out/'predictions.jsonl'),checkpoint=job.get('checkpoint') or job['parent'])
        write(out/'summary.json',result)
    else:raise ValueError(kind)
    result['worker_seconds']=time.time()-started;write(job['result'],result);timer.cancel()
if __name__=='__main__':worker(read(sys.argv[1]))
