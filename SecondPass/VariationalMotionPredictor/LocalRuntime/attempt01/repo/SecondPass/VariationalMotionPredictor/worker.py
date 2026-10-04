"""Bounded, fresh three-frame variational motion-prediction pilot."""
from __future__ import annotations
import argparse, copy, datetime as dt, fcntl, hashlib, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .model import VariationalMotionPredictor, losses
from .dataset import MotionPool, generate, SPEEDS, NAMESPACES

MODULE='SecondPass.VariationalMotionPredictor.worker'
ROOT=Path(__file__).resolve().parents[2]
INIT_SEED=184211
TARGET=1024
CAP=1200
EXPECTED_PARAMETERS=15463363
EXPECTED_TENSORS=138

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return copy.deepcopy(x)
def sync(device):
    if device=='mps':torch.mps.synchronize()
def setup():
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(2)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def provenance():
    return dict(initialization='whole model direct fresh constructor; no checkpoint inputs',inherited_weights=False,
        inherited_optimizer=False,inherited_rng=False,inherited_stream=False,all_trainable=True,initialization_seed=INIT_SEED,
        namespaces=NAMESPACES,pool_size=1000,pool_epochs=2,pool_updates=64,presentations_per_pool=2000,
        input='ordered three RGB frames only',output='fourth RGB frame via one sampled 512-dimensional vector',
        evaluation='deterministic decode(mu); no classifier labels',precision='fp32',optimizer='Adam1e-4 no clipping',
        kl_beta_max=1e-4,kl_warmup_updates=250,parameter_count=EXPECTED_PARAMETERS)

class Session:
    def __init__(self,directory,config):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.config=config;self.device=config['device']
        random.seed(INIT_SEED);np.random.seed(INIT_SEED);torch.manual_seed(INIT_SEED)
        if self.device=='mps':torch.mps.manual_seed(INIT_SEED)
        self.model=VariationalMotionPredictor()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(INIT_SEED);direct=VariationalMotionPredictor()
        assert all(torch.equal(v,direct.state_dict()[n]) for n,v in self.model.state_dict().items())
        del direct
        names=[n for n,_ in self.model.named_parameters()]
        assert len(names)==EXPECTED_TENSORS and sum(p.numel() for p in self.model.parameters())==EXPECTED_PARAMETERS
        hashes={n:tensor_digest(p) for n,p in self.model.named_parameters()}
        self.model.to(self.device);assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.pool=MotionPool();self.state=dict(step=0,episodes=0,optimizer_seconds=0.,best_step=None,best_key=None,
            unique_episodes_generated=0,selection_history=[],config=config)
        atomic_json(self.directory/'initialization.json',dict(verified=True,parameter_count=EXPECTED_PARAMETERS,
            optimizer_names=names,initial_parameter_hashes=hashes,direct_constructor_equality=True,empty_adam=True,
            empty_streams=self.pool.state_dict()['pool_index']==-1,zero_counters=True,checkpoint_inputs=[],provenance=provenance()))
    def update(self):
        before=time.perf_counter();past,target,meta=self.pool.next_batch();generation=time.perf_counter()-before
        count=len(meta);self.model.train();self.optimizer.zero_grad(set_to_none=True);sync(self.device);started=time.perf_counter()
        beta=1e-4*min(self.state['step']/250.,1.);totals={}
        for offset in range(0,count,self.config['microbatch']):
            x=past[offset:offset+self.config['microbatch']].to(self.device);y=target[offset:offset+self.config['microbatch']].to(self.device)
            out=self.model(x,sample=True);terms=losses(out['prediction'],y,x,out['mu'],out['logvar'],beta)
            if not all(bool(torch.isfinite(v)) for v in terms.values()):raise FloatingPointError('Nonfinite loss')
            weight=len(x)/count;(terms['loss']*weight).backward()
            for key,value in terms.items():totals[key]=totals.get(key,0.)+float(value.detach())*weight
        norms={}
        for name,p in self.model.named_parameters():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Missing/nonfinite gradient '+name)
            norms[name]=float(p.grad.square().sum())
        self.optimizer.step();sync(self.device)
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):raise FloatingPointError('Nonfinite parameter')
        elapsed=time.perf_counter()-started;s=self.state;s['step']+=1;s['episodes']+=count;s['optimizer_seconds']+=elapsed
        s['unique_episodes_generated']=self.pool.stream.next_index
        row=dict(step=s['step'],episodes=count,cumulative_episodes=s['episodes'],unique_episodes_generated=s['unique_episodes_generated'],
            seconds=elapsed,pool_generation_seconds=generation,pool_index=self.pool.pool_index,epoch=self.pool.epoch,
            beta=beta,gradient_norm=math.sqrt(sum(norms.values())),losses=totals,clipping=None)
        append(self.directory/'progress.jsonl',row);self.status('training',last_update=row);print(json.dumps(row),flush=True);return row
    def status(self,phase,**extra):
        atomic_json(self.directory/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            step=self.state['step'],episodes=self.state['episodes'],pool_index=self.pool.pool_index,epoch=self.pool.epoch,
            unique_episodes_generated=self.state['unique_episodes_generated'],
            latest_validation=self.state['selection_history'][-1] if self.state['selection_history'] else None,
            optimizer_seconds=self.state['optimizer_seconds'],deadline=self.config['hard_deadline'],best_step=self.state['best_step'],**extra))
    def checkpoint(self):
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        if self.device=='mps':rng['mps']=torch.mps.get_rng_state()
        payload=cpu_tree(dict(schema=1,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            optimizer_names=[n for n,_ in self.model.named_parameters()],state=self.state,pool=self.pool.state_dict(),rng=rng,provenance=provenance()))
        path=self.directory/'latest.pt';tmp=self.directory/'latest.pt.tmp'
        with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);receipt=dict(path=str(path),step=self.state['step'],bytes=path.stat().st_size,sha256=digest(path),verified=True)
        atomic_json(self.directory/'latest_checkpoint.json',receipt)
        if self.state['step'] in (1,3):verify_progress(self.directory)
        return receipt
    def best(self):
        tmp=self.directory/'best.pt.tmp';os.link(self.directory/'latest.pt',tmp);os.replace(tmp,self.directory/'best.pt')
        receipt=json.loads((self.directory/'latest_checkpoint.json').read_text());receipt['path']=str(self.directory/'best.pt')
        atomic_json(self.directory/'best_checkpoint.json',receipt)

def verify_progress(directory):
    directory=Path(directory);receipt=json.loads((directory/'latest_checkpoint.json').read_text());path=Path(receipt['path'])
    assert path.stat().st_size==receipt['bytes'] and digest(path)==receipt['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);initial=json.loads((directory/'initialization.json').read_text());step=saved['state']['step']
    assert step==receipt['step']==saved['pool']['updates'] and step>0 and saved['state']['optimizer_seconds']>0
    assert saved['pool']['presentations']==saved['state']['episodes']
    assert saved['optimizer_names']==initial['optimizer_names'] and len(saved['optimizer_names'])==EXPECTED_TENSORS
    assert set(saved['optimizer']['state'])==set(range(EXPECTED_TENSORS));changed=[]
    for index,name in enumerate(saved['optimizer_names']):
        opt=saved['optimizer']['state'][index];value=saved['model'][name]
        assert float(opt['step'])==step and torch.isfinite(value).all()
        assert all(torch.isfinite(opt[k]).all() and opt[k].shape==value.shape for k in ('exp_avg','exp_avg_sq'))
        if tensor_digest(value)!=initial['initial_parameter_hashes'][name]:changed.append(name)
    assert len(changed)==EXPECTED_TENSORS
    assert saved['provenance']==initial['provenance']==provenance() and saved['rng']['cpu'].numel()>0
    proof=dict(verified=True,step=step,episodes=saved['state']['episodes'],checkpoint=receipt,
        changed_parameter_tensors=len(changed),all_named_adam_steps_verified=True,whole_model_fresh=True)
    atomic_json(directory/'persisted_progress_verification.json',proof);return proof

@torch.no_grad()
def evaluate(model,split,counts,device,directory,role,deadline,visuals=False):
    model.eval();cells=[];total_seconds=0.;examples=[]
    for speed_index,speed in enumerate(SPEEDS):
        count=counts if isinstance(counts,int) else counts[speed_index];totals={};n=0;started=time.perf_counter()
        for offset in range(0,count,4):
            if time.time()>=deadline:break
            rows=[generate(speed_index+3*j,split) for j in range(offset,min(offset+4,count))]
            x=torch.stack([r[0] for r in rows]).to(device);y=torch.stack([r[1] for r in rows]).to(device)
            out=model(x,sample=False)
            for name,prediction in [('model',out['prediction']),('copy_last',x[:,-1]),('gray',torch.full_like(y,.5))]:
                terms=losses(prediction,y,x,out['mu'],out['logvar'],0.)
                for metric in ('recon','support_mse','background_mse','full_mse','change_mse'):
                    key=name+'_'+metric;totals[key]=totals.get(key,0.)+float(terms[metric])*len(rows)
            n+=len(rows)
            if visuals and offset==0:examples.append((speed,rows[0][0],rows[0][1],out['prediction'][0].cpu()))
        sync(device);elapsed=time.perf_counter()-started;total_seconds+=elapsed
        cells.append(dict(speed=speed,n=n,expected_n=count,complete=n==count,seconds=elapsed,**{k:v/max(n,1) for k,v in totals.items()}))
    complete=all(r['complete'] for r in cells)
    result=dict(split=split,role=role,complete=complete,cells=cells,n=sum(r['n'] for r in cells),seconds=total_seconds,
        deterministic_latent='decode(mu), not the expectation of decoded sampled z',namespace=NAMESPACES[split],
        balanced_model_mse=sum(r.get('model_recon',0.) for r in cells)/3 if complete else None)
    atomic_json(Path(directory)/(role+'.json'),result)
    if visuals:
        for speed,past,target,prediction in examples:save_panel(Path(directory)/f'{role}_speed{speed:g}.png',past,target,prediction)
    return result

def save_panel(path,past,target,prediction):
    frames=[*past,target,prediction,past[-1]];labels=['Past1','Past2','Past3','True4','Predicted4','Copy last']
    panel=Image.new('RGB',(600,122),'white');draw=ImageDraw.Draw(panel)
    for i,(frame,label) in enumerate(zip(frames,labels)):
        pixels=(frame.permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
        panel.paste(Image.fromarray(pixels),(100*i,22));draw.text((100*i+2,3),label,fill='black')
    panel.save(path)

def validation_steps(count):return sorted({n for n in [100,250,*range(500,count+1,500),count] if n<=count})
def verify_sources(config):
    for path,sha in config['source_hashes'].items():
        if digest(path)!=sha:raise ValueError('Source changed: '+path)

def profile(directory,config):
    session=Session(directory,config);rows=[]
    for _ in range(3):rows.append(session.update())
    timing=evaluate(session.model,'val',[11,11,10],session.device,directory,'eval_timing',config['deadline'])
    if not timing['complete']:raise RuntimeError('Profile evaluation incomplete')
    result=dict(complete=True,rows=rows,eval_timing=timing,discard_all_state=True,checkpoints_written=False,
        profile_optimizer_states=len(session.optimizer.state),profile_updates=3)
    atomic_json(Path(directory)/'profile.json',result);del session
    if config['device']=='mps':torch.mps.empty_cache()
    return result

def pin(measured,budget):
    costs=[r['seconds'] for r in measured['rows']];cost=max(costs);generation=sum(r['pool_generation_seconds'] for r in measured['rows'])
    evaluation=measured['eval_timing']['seconds']/32.
    if not all(math.isfinite(x) and x>0 for x in (cost,evaluation)):raise ValueError('Invalid profile timings')
    plan=None;available=budget['deadline']-time.time()
    for count in range(64,TARGET+1,64):
        checks=validation_steps(count);val=1.35*evaluation*192;final=1.35*evaluation*384
        total=1.25*(count*cost+(count//64)*generation)+len(checks)*val+final+90
        if total<available:plan=dict(max_steps=count,target_requested=TARGET,exposure_reduced=count<TARGET,
            planned_unique_movies=count//64*1000,planned_presentations=count//64*2000,validation_steps=checks,
            estimated_remaining_seconds=total,estimated_validation_seconds=val,estimated_final_seconds=final,
            final_reserve_seconds=final+40,update_estimate_seconds=1.25*cost+generation)
    if plan is None:raise RuntimeError('One64-update pool cannot fit20-minutecap')
    return plan

def run(directory,config):
    verify_sources(config);session=Session(directory,config);stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    reason='planned_complete';failure=None;final=None
    def validate():
        session.status('validation')
        result=evaluate(session.model,'val',64,session.device,directory,f"validation_{session.state['step']:06d}",config['deadline']-config['estimated_final_seconds']-20)
        key=result['balanced_model_mse'];s=session.state
        s['selection_history'].append(dict(step=s['step'],complete=result['complete'],balanced_model_mse=key))
        better=result['complete'] and (s['best_key'] is None or key<s['best_key'])
        if better:s.update(best_key=key,best_step=s['step'])
        session.checkpoint()
        if better:session.best()
    try:
        while session.state['step']<config['max_steps']:
            if stopped[0]:reason='signal';break
            if time.time()+config['final_reserve_seconds']+config['update_estimate_seconds']>=config['deadline']:reason='wall_budget_reserve';break
            session.update();step=session.state['step']
            if step in (1,3) or step%100==0:session.checkpoint()
            if step in config['validation_steps']:validate()
        if not stopped[0]:
            if not session.state['selection_history'] or session.state['selection_history'][-1]['step']!=session.state['step']:validate()
            session.checkpoint()
            if session.state['best_step'] is not None and time.time()<config['deadline']:
                selected=torch.load(Path(directory)/'best.pt',map_location='cpu',weights_only=False)
                session.model.load_state_dict(selected['model']);del selected
                session.status('final_test_best')
                final=evaluate(session.model,'test',128,session.device,directory,'test_best',config['deadline'],visuals=True)
        elif session.state['step']>0:session.checkpoint()
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc));reason='worker_error';atomic_json(Path(directory)/'failure.json',failure)
        if session.state['step']>0:
            try:session.checkpoint()
            except Exception:pass
    report=dict(provenance=provenance(),stop_reason=reason,failure=failure,actual_updates=session.state['step'],
        pinned_updates=config['max_steps'],presentations=session.state['episodes'],unique_movies_generated=session.state['unique_episodes_generated'],
        selected_step=session.state['best_step'],selected_validation_balanced_mse=session.state['best_key'],
        selection_history=session.state['selection_history'],final=final,final_coverage_complete=bool(final and final['complete']),
        cap_started=config['cap_started'],hard_deadline=config['hard_deadline'],config=config,
        limitation='One short synthetic constant-velocity pilot; not Krauzlis task performance or capacity evidence.')
    atomic_json(Path(directory)/'report.json',report);session.status('finished',stop_reason=reason,final_coverage_complete=report['final_coverage_complete'])
    if session.state['step']>0:verify_progress(directory)
    return report

def local_supervise(directory):
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no CPU production fallback')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'budget.json').exists():raise RuntimeError('No restart/cap renewal')
    lock=Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();budget=dict(cap_started=start,hard_deadline=start+CAP,deadline=start+CAP-30,wall_cap_seconds=CAP,report_reserve_seconds=30)
    atomic_json(directory/'budget.json',budget);(directory/'budget.json').chmod(0o444)
    atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),worker_pid=os.getpid(),**budget))
    manifest=json.loads((ROOT/'runtime_manifest.json').read_text());config=dict(budget,device='mps',effective_batch=32,microbatch=4,
        cpu_threads=2,source_hashes=manifest['source_hashes'],lr=1e-4,clipping=None,precision='fp32',bptt='full')
    verify_sources(config);pd=directory/'profile';pd.mkdir()
    try:
        atomic_json(pd/'profile_config.json',dict(config,disposable_profile=True));measured=profile(pd,config)
        allocation=pin(measured,budget);config.update(allocation);atomic_json(directory/'allocation.json',allocation);atomic_json(directory/'config.json',config)
        result=run(directory,config);summary=dict(status='complete' if result['failure'] is None and result['final_coverage_complete'] else 'incomplete',actual_updates=result['actual_updates'],**budget)
    except Exception as exc:
        summary=dict(status='failed',error_type=type(exc).__name__,message=str(exc),**budget);atomic_json(directory/'failure.json',summary)
    atomic_json(directory/'local_supervisor_result.json',summary);lock.close();return summary

def main():
    setup();parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['local-supervise','verify']);parser.add_argument('directory',type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    if args.mode=='verify':print(json.dumps(verify_progress(directory)));return
    result=local_supervise(directory);print(json.dumps(result),flush=True);raise SystemExit(0 if result['status']=='complete' else 2)
if __name__=='__main__':main()
