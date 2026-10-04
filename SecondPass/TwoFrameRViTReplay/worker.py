"""Versioned 1000-movie/ten-epoch replay; unchanged native renderer and RViT."""
import argparse, copy, json, math, os, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from SecondPass.TwoFrameRViT import worker as base

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.TwoFrameRViTReplay.worker'
VERSION=base.VERSION
PROTOCOL=VERSION+'_fresh_native_pool1000_epoch10_v1'
TASK,TASKS,CELLS=base.TASK,base.TASKS,base.CELLS
INIT_SEED,CUDA_SEED,SCHEDULER_SEED=base.INIT_SEED,base.CUDA_SEED,base.SCHEDULER_SEED
TARGET=4216
POOL_SIZE,EPOCHS=1000,10
POOL_UPDATES=330
EXPECTED_PARAMETERS=base.EXPECTED_PARAMETERS
atomic_json,append_jsonl,cpu_tree,tree_equal=base.atomic_json,base.append_jsonl,base.cpu_tree,base.tree_equal
digest,load_verified=base.digest,base.load_verified
evaluate,selection_key,validation_steps_for=base.evaluate,base.selection_key,base.validation_steps_for
validate_budget,verify_sources=base.validate_budget,base.verify_sources

def clone(fn,**scope):
    out=types.FunctionType(fn.__code__,dict(fn.__globals__,**scope),fn.__name__,fn.__defaults__,fn.__closure__)
    out.__kwdefaults__=fn.__kwdefaults__; return out

def provenance():
    return dict(base.provenance(),training_policy='1000 fresh native movies, ten shuffled epochs, then new pool',
        pool_size=POOL_SIZE,replay_epochs=EPOCHS,updates_per_epoch=33,updates_per_pool=330,
        initialization='whole model direct fresh constructor; no prior model/profile inputs')

def epoch_batches(sizes,rng):
    """Shuffle movies within conditions and condition order in each batch round."""
    batches={}
    for cell,size in sizes.items():
        order=list(range(size)); rng.shuffle(order)
        batches[cell]=[order[i:i+32] for i in range(0,size,32)]
    result=[]
    for round_index in range(max(map(len,batches.values()))):
        cells=list(sizes); rng.shuffle(cells)
        result.extend((cell,batches[cell][round_index]) for cell in cells if round_index<len(batches[cell]))
    return result

class ReplayStream(base.FreshStream):
    def __init__(self,split='train'):
        if split!='train': raise ValueError('Replay is training-only; evaluation uses independent native streams')
        super().__init__(split)
        self.pool_index=-1; self.epoch=0; self.cursor=0; self.pre_generation=None
        self.pool_ids={}; self.sizes={}; self.order=[]; self.data={}; self.used=set()
        self.rng=random.Random(SCHEDULER_SEED+71); self.presentations=0; self.presentation_counts={c:0 for c in CELLS}
        self.generation_seconds=0.; self.total_generation_seconds=0.; self.current=None
    def _render(self,native):
        data={}; ids={}
        for cell,count in self.sizes.items():
            movies=[]; identities=[]
            for start in range(0,count,32):
                images,labels,metadata=native.batch(min(32,count-start),TASK,cell)
                movies.extend((images[i],int(labels[i])) for i in range(len(labels)))
                identities.extend(m['suite_trial_id'] for m in metadata)
            data[cell]=movies; ids[cell]=identities
        self.data=data
        return ids
    def _new_pool(self):
        started=time.perf_counter(); self.data={}; self.pool_index+=1; self.epoch=0; self.cursor=0; self.used=set()
        self.sizes={c:333+int(i==self.pool_index%3) for i,c in enumerate(CELLS)}
        self.pre_generation=super().state_dict()
        native=base.FreshStream('train'); native.load_state_dict(self.pre_generation)
        self.pool_ids=self._render(native)
        super().load_state_dict(native.state_dict())
        self.order=epoch_batches(self.sizes,self.rng)
        self.generation_seconds=time.perf_counter()-started; self.total_generation_seconds+=self.generation_seconds
    def next_plan(self):
        self.generation_seconds=0.
        if self.pool_index<0: self._new_pool()
        if self.cursor==len(self.order):
            self.epoch+=1; self.cursor=0
            if self.epoch==EPOCHS: self._new_pool()
            else: self.order=epoch_batches(self.sizes,self.rng)
        self.current=self.order[self.cursor]; self.cursor+=1
        return TASK,self.current[0]
    def current_batch(self):
        cell,indices=self.current
        return torch.stack([self.data[cell][i][0] for i in indices]),torch.tensor([self.data[cell][i][1] for i in indices]),cell,indices
    def commit(self,cell,indices):
        self.presentations+=len(indices); self.presentation_counts[cell]+=len(indices)
        self.used.update((cell,i) for i in indices)
    def state_dict(self):
        state=super().state_dict()
        state['replay']=copy.deepcopy(dict(schema=1,pool_index=self.pool_index,epoch=self.epoch,cursor=self.cursor,
            pre_generation=self.pre_generation,pool_ids=self.pool_ids,sizes=self.sizes,order=self.order,rng=self.rng.getstate(),
            used=sorted(self.used),presentations=self.presentations,presentation_counts=self.presentation_counts,
            total_generation_seconds=self.total_generation_seconds,current=self.current))
        return state
    def load_state_dict(self,state):
        super().load_state_dict(state); r=copy.deepcopy(state['replay'])
        self.pool_index=r['pool_index']; self.epoch=r['epoch']; self.cursor=r['cursor']; self.pre_generation=r['pre_generation']
        self.pool_ids=r['pool_ids']; self.sizes=r['sizes']; self.order=r['order']; self.rng.setstate(r['rng'])
        self.used=set(map(tuple,r['used'])); self.presentations=r['presentations']; self.presentation_counts=r['presentation_counts']
        self.total_generation_seconds=r['total_generation_seconds']; self.current=r['current']; self.generation_seconds=0.
        if self.pool_index>=0:
            native=base.FreshStream('train'); native.load_state_dict(self.pre_generation)
            ids=self._render(native)
            assert ids==self.pool_ids and tree_equal(native.state_dict(),{k:v for k,v in state.items() if k!='replay'})

class ReplayScheduler:
    def __init__(self,stream): self.stream=stream; self.updates=0
    def next(self): self.updates+=1; return self.stream.next_plan()
    def state_dict(self): return dict(updates=self.updates,policy='replay shuffled homogeneous rounds')
    def load_state_dict(self,state): self.updates=state['updates']

class Session(base.Session):
    def __init__(self,directory,config):
        super().__init__(directory,config)
        self.stream=ReplayStream(); self.scheduler=ReplayScheduler(self.stream)
        self.state.update(unique_episodes_generated=0,unique_episodes_presented=0,replay_presentations=0)
    def status(self,phase,**extra):
        return super().status(phase,pool_index=self.stream.pool_index,epoch=self.stream.epoch,
            unique_episodes_generated=self.state['unique_episodes_generated'],
            unique_episodes_presented=self.state['unique_episodes_presented'],replay_presentations=self.state['episodes'],**extra)
    checkpoint=clone(base.Session.checkpoint,provenance=provenance,verify_progress=lambda d:verify_progress(d))
    def train_update(self,task,cell):
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        images,labels,actual_cell,indices=self.stream.current_batch(); assert task==TASK and cell==actual_cell
        actual=len(labels); micro=self.config['microbatch']; loss_sum=0.
        base.sync(self.device); started=time.perf_counter()
        for i in range(0,actual,micro):
            x,y=images[i:i+micro].to(self.device),labels[i:i+micro].to(self.device)
            logits=self.model(x,task); loss=F.cross_entropy(logits,y)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite loss')
            weight=len(y)/actual; (loss*weight).backward(); loss_sum+=float(loss.detach())*weight
        norms={}
        for name,p in self.model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()): raise FloatingPointError('missing/nonfinite gradient '+name)
            norms[name]=float(p.grad.square().sum())
        norm=math.sqrt(sum(norms.values()))
        if not math.isfinite(norm): raise FloatingPointError('nonfinite gradient norm')
        self.optimizer.step(); base.sync(self.device)
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()): raise FloatingPointError('nonfinite parameter')
        self.stream.commit(cell,indices); elapsed=time.perf_counter()-started
        s=self.state; frames=actual*images.shape[1]
        s['step']+=1; s['episodes']+=actual; s['frames']+=frames; s['optimizer_seconds']+=elapsed
        s.update(unique_episodes_generated=(self.stream.pool_index+1)*1000,
            unique_episodes_presented=self.stream.pool_index*1000+len(self.stream.used),replay_presentations=s['episodes'])
        e=s['exposure'][task]; e['updates']+=1; e['episodes']+=actual; e['frames']+=frames; e['cells'][cell]+=actual
        row=dict(utc=base.utc(),step=s['step'],task=task,cell=cell,loss=loss_sum,seconds=elapsed,episodes=actual,frames=frames,
            grad_norm=norm,cumulative_episodes=s['episodes'],cumulative_frames=s['frames'],optimizer_seconds=s['optimizer_seconds'],
            pool_index=self.stream.pool_index,epoch=self.stream.epoch,cursor=self.stream.cursor,
            unique_episodes_generated=s['unique_episodes_generated'],unique_episodes_presented=s['unique_episodes_presented'],
            pool_generation_seconds=self.stream.generation_seconds,partial_batch=actual<32,initialization='fresh',clipping=None)
        append_jsonl(self.directory/'progress.jsonl',row); print(json.dumps(row),flush=True)
        if s['step']==3 and not self.config.get('disposable_profile') and (self.directory/'profile/profile.json').exists():
            p=json.loads((self.directory/'profile/profile.json').read_text())
            costs={c:max(r['seconds'] for r in p['rows'] if r['cell']==c) for c in CELLS}
            rows=[json.loads(line) for line in (self.directory/'progress.jsonl').read_text().splitlines()]
            ratio=sum(r['seconds'] for r in rows)/sum(costs[r['cell']] for r in rows)
            atomic_json(self.directory/'first_cycle_timing.json',dict(updates=3,matched_profile_ratio=ratio,material_slowdown=ratio>1.35))
        return row

def verify_progress(directory):
    d=Path(directory); initial=load_verified(json.loads((d/'initial_checkpoint.json').read_text()))
    receipt=json.loads((d/'latest_checkpoint.json').read_text()); saved=load_verified(receipt)
    assert initial['schema']==saved['schema']==3 and initial['provenance']==saved['provenance']==provenance()
    assert initial['state']['step']==initial['scheduler']['updates']==0 and not initial['optimizer']['state'] and initial['stream']['streams']==[]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INIT_SEED); direct=base.TwoFrameRViT(checkpoint_encoder=True)
    assert tree_equal(initial['model'],direct.state_dict())
    names=[n for n,_ in direct.named_parameters()]; assert len(names)==92 and names==saved['optimizer_names']==initial['optimizer_names']
    step=saved['state']['step']; replay=saved['stream']['replay']
    assert step>0 and step==saved['scheduler']['updates']==receipt['step']
    assert saved['state']['episodes']==replay['presentations']==saved['state']['replay_presentations']
    assert sum(replay['presentation_counts'].values())==replay['presentations']
    assert replay['presentation_counts']==saved['state']['exposure'][TASK]['cells']
    assert saved['state']['unique_episodes_generated']==(replay['pool_index']+1)*1000
    assert sum(r['state']['counts'][TASK] for r in saved['stream']['streams'])==saved['state']['unique_episodes_generated']
    assert sum(map(len,replay['pool_ids'].values()))==1000
    assert set(saved['optimizer']['state'])==set(range(92))
    changed=[]
    for i,name in enumerate(names):
        opt=saved['optimizer']['state'][i]; assert float(opt['step'])==step
        assert torch.isfinite(saved['model'][name]).all()
        assert all(torch.isfinite(opt[k]).all() and opt[k].shape==saved['model'][name].shape for k in ('exp_avg','exp_avg_sq'))
        if not torch.equal(initial['model'][name],saved['model'][name]): changed.append(name)
    assert len(changed)==92
    groups={p:[n for n in changed if n.startswith(p)] for p in ('encoder.','row_embedding','column_embedding','token_norm.',
        'recurrent_block.query_norm.','recurrent_block.memory_norm.','recurrent_block.visual_attention.','recurrent_block.memory_attention.',
        'recurrent_block.ffn_norm.','recurrent_block.ffn.','readout_norm.','token_readout.','classifier.')}
    assert all(groups.values())
    out=dict(verified=True,step=step,episodes=saved['state']['episodes'],checkpoint=receipt,parameter_count=EXPECTED_PARAMETERS,
        changed_parameters=92,changed_by_module=groups,all_active_parameters_changed=True,persisted_constructor_equality=True,
        unique_episodes_generated=saved['state']['unique_episodes_generated'],unique_episodes_presented=saved['state']['unique_episodes_presented'],
        optimizer_seconds=saved['state']['optimizer_seconds'],fresh_initial_optimizer_empty=True)
    atomic_json(d/'persisted_progress_verification.json',out); return out

profile=clone(base.profile,Session=Session,verify_progress=verify_progress)
_run=clone(base.run,Session=Session,provenance=provenance,verify_progress=verify_progress,PROTOCOL=PROTOCOL)
def run(directory):
    code=_run(directory)
    report=json.loads((directory/'report.json').read_text()); terminal=load_verified(report['terminal_checkpoint'])
    report.update(training_policy=provenance(),unique_episodes_generated=terminal['state']['unique_episodes_generated'],
        unique_episodes_presented=terminal['state']['unique_episodes_presented'],replay_presentations=terminal['state']['episodes'])
    atomic_json(directory/'report.json',report)
    md=directory/'REPORT.md'
    md.write_text(md.read_text().replace('; episodes ', '; presentations ')+
        f"\nGenerated unique movies: {report['unique_episodes_generated']}; unique movies presented: {report['unique_episodes_presented']}; replay presentations: {report['replay_presentations']}.\n")
    return code
completion=clone(base.completion,TARGET=TARGET)
supervise=clone(base.supervise,MODULE=MODULE,completion=completion)

def config_for(budget):
    validate_budget(budget,budget)
    manifest=json.loads((ROOT.parent/'replay_deployment_manifest.json').read_text())
    assert manifest['protocol']==PROTOCOL
    hashes={str(ROOT.parent/r['path']):r['sha256'] for r in manifest['files']}
    cfg=dict(budget,protocol=PROTOCOL,device='cuda',checkpoint_encoder=True,effective_batch=32,microbatch=4,eval_microbatch=4,
        val_n=100,test_n=200,checkpoint_every=100,cpu_threads=2,source_hashes=hashes,runtime_root=str(ROOT),
        initialization=provenance(),all_trainable=True,lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,
        clipping=None,precision='fp32',bptt='full',tf32=False,training_policy='pool1000epoch10')
    verify_sources(cfg); return cfg

def measured_plan(directory,budget,now=None):
    now=time.time() if now is None else now; validate_budget(budget,budget,now)
    p=json.loads((directory/'profile/profile.json').read_text()); assert p['complete'] and len(p['rows'])==6
    costs={c:max(r['seconds'] for r in p['rows'] if r['cell']==c) for c in CELLS}
    assert all(sum(r['cell']==c and r['episodes']==32 for r in p['rows'])==2 for c in CELLS)
    val=1.35*sum(r['seconds']/r['n'] for r in p['evaluation_cells'])*100; test=2*val
    generation=sum(r.get('pool_generation_seconds',0.) for r in p['warmup_rows']); assert generation>0
    sizes={c:333+int(i==0) for i,c in enumerate(CELLS)}; rng=random.Random(SCHEDULER_SEED+71)
    schedule=[]; training=0.; best=None
    for i in range(TARGET):
        if i%330==0: sizes={c:333+int(j==(i//330)%3) for j,c in enumerate(CELLS)}
        if i%33==0: schedule.extend(epoch_batches(sizes,rng))
        cell,indices=schedule[i]; training+=costs[cell]
        overhead=len(validation_steps_for(i+1))*val+2*test+900
        gen=math.ceil((i+1)/330)*generation
        total=1.25*(training+gen)+overhead
        if total+60<budget['deadline']-now: best=(i+1,training,gen,total)
    if best is None or best[0]<33: raise RuntimeError('No complete replay epoch fits original remaining allowance')
    count,_,_,_=best
    if count>=330: count=count//330*330
    training=sum(costs[c] for c,ix in schedule[:count]); gen=math.ceil(count/330)*generation
    total=1.25*(training+gen)+len(validation_steps_for(count))*val+2*test+900
    return dict(max_steps=count,target_requested=TARGET,exposure_reduced=count<TARGET,total_episodes=sum(len(ix) for _,ix in schedule[:count]),
        planned_presentations=sum(len(ix) for _,ix in schedule[:count]),planned_unique_movies=math.ceil(count/330)*1000,
        validation_steps=validation_steps_for(count),planned_validation_looks=len(validation_steps_for(count)),checkpoint_every=100,
        estimated_optimizer_seconds=training,estimated_pool_generation_seconds=gen,estimated_total_remaining_seconds=total,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values())+generation,
        method='remaining original absolute cap; measured native replay micro4 full32 rates conservatively charged for partialtails; measured poolgeneration each330updates; exact frequentvalidations; pinwhole10epochpools where feasible')
prepare=clone(base.prepare,config_for=config_for)
pin=clone(base.pin,config_for=config_for,measured_plan=measured_plan)

def main():
    base.cpu_setup(); parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','profile','pin','run','verify','supervise-profile','supervise-run'])
    parser.add_argument('directory',type=Path); parser.add_argument('--budget',type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='prepare':
        if args.budget is None: parser.error('--budget required')
        return prepare(directory,json.loads(args.budget.read_text()))
    if args.mode=='pin': print(json.dumps(pin(directory))); return
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode.startswith('supervise-'): raise SystemExit(supervise(directory,args.mode.split('-',1)[1])['returncode'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one CUDA GPU required')
    import fcntl
    with (ROOT.parent/'two_frame_rvit_replay_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))
if __name__=='__main__': main()
