"""Fresh CUDA/A40 weighted-frame RViT under an existing pod-creation budget."""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from . import worker as base

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.WeightedMeanRViT.cloud_worker'
VERSION=base.VERSION
PROTOCOL=VERSION+'_pool1000_epoch10_cuda_a40_v1'
TARGET=base.TARGET
TASK,TASKS,CELLS=base.TASK,base.TASKS,base.CELLS
INIT_SEED,SCHEDULER_SEED=base.INIT_SEED,base.SCHEDULER_SEED
CUDA_SEED=base.replay.CUDA_SEED
atomic_json,append_jsonl,cpu_tree,tree_equal,digest,load_verified=base.atomic_json,base.append_jsonl,base.cpu_tree,base.tree_equal,base.digest,base.load_verified
verify_sources=base.verify_sources


def provenance(config=None):
    return dict(base.provenance(config or {}),execution_placement='single A40 CUDA',precision='fp32',
        full_sequence_bptt=True,cuda_seed=CUDA_SEED,tf32=False,sdpa='PyTorch math; flash and memory-efficient disabled')


def validate_budget(budget,config,now=None):
    base.local.validate_budget(budget,config,now)
    if budget.get('max_usd')!=5: raise ValueError('Require unchanged new8h/$5 podcreation cap')


def cuda_setup():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one CUDA accelerator required')
    if 'A40' not in torch.cuda.get_device_name(0): raise RuntimeError('Authorized single A40 required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda,'enable_cudnn_sdp'): torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision('highest')


class Session(base.Session):
    def __init__(self,directory,config):
        super().__init__(directory,config)
        if str(self.device).startswith('cuda'): torch.cuda.manual_seed_all(CUDA_SEED)
        initialization=json.loads((self.directory/'initialization.json').read_text())
        initialization['provenance']=provenance(config)
        initialization['cuda_seed']=CUDA_SEED; initialization['precision']='fp32'
        atomic_json(self.directory/'initialization.json',initialization)

    def checkpoint(self):
        self.state['elapsed_cap_seconds']=time.time()-self.config['cap_started']
        rng=dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        if str(self.device).startswith('cuda'): rng['cuda']=torch.cuda.get_rng_state_all()
        payload=cpu_tree(dict(schema=3,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),state=self.state,
            optimizer_names=[n for n,_ in self.model.named_parameters()],scheduler=self.scheduler.state_dict(),stream=self.stream.state_dict(),
            rng=rng,provenance=provenance(self.config)))
        path=self.directory/'latest.pt'; temporary=self.directory/'latest.pt.tmp'
        with temporary.open('wb') as stream: torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        receipt=dict(path=str(path),step=self.state['step'],bytes=path.stat().st_size,sha256=digest(path),verified=True)
        assert tree_equal(payload,load_verified(receipt)); self.latest=receipt; atomic_json(self.directory/'latest_checkpoint.json',receipt)
        append_jsonl(self.directory/'checkpoints.jsonl',dict(utc=base.local.utc(),**receipt))
        if self.state['step'] in (1,2,3):
            proof=verify_progress(self.directory); atomic_json(self.directory/f"optimizer_proof_{self.state['step']:06d}.json",proof)
            if self.state['step']==3: atomic_json(self.directory/'startup_ready.json',dict(verified=True,checkpoint=receipt,persisted_optimizer_evidence=proof))
        return receipt


def verify_progress(directory):
    directory=Path(directory); initial=json.loads((directory/'initialization.json').read_text())
    receipt=json.loads((directory/'latest_checkpoint.json').read_text()); saved=load_verified(receipt); step=saved['state']['step']
    names=saved['optimizer_names']; assert len(names)==92 and names==initial['optimizer_names']
    assert saved['provenance']==initial['provenance']==provenance(saved['state']['config'])
    assert initial['direct_constructor_equality'] and initial['empty_adam'] and initial['empty_streams'] and initial['checkpoint_inputs']==[]
    assert step>0 and step==receipt['step']==saved['scheduler']['updates']
    assert set(saved['optimizer']['state'])==set(range(92)); changed=[]
    groups=saved['optimizer']['param_groups']; assert len(groups)==1
    assert groups[0]['lr']==1e-4 and groups[0]['betas']==(.9,.999) and groups[0]['eps']==1e-8 and groups[0]['weight_decay']==0
    for i,name in enumerate(names):
        value=saved['model'][name]; opt=saved['optimizer']['state'][i]
        assert torch.isfinite(value).all() and float(opt['step'])==step
        assert all(torch.isfinite(opt[k]).all() and opt[k].shape==value.shape for k in ('exp_avg','exp_avg_sq'))
        if base.harness.tensor_digest(value)!=initial['initial_parameter_hashes'][name]: changed.append(name)
    assert len(changed)==92
    r=saved['stream']['replay']; assert r['presentations']==saved['state']['episodes']==saved['state']['replay_presentations']
    assert r['presentation_counts']==saved['state']['exposure'][TASK]['cells'] and sum(r['presentation_counts'].values())==r['presentations']
    assert sum(m['state']['counts'][TASK] for m in saved['stream']['streams'])==saved['state']['unique_episodes_generated']
    assert len(saved['rng']['cuda'])==1 and saved['rng']['cuda'][0].numel()>0
    result=dict(verified=True,step=step,checkpoint=receipt,episodes=saved['state']['episodes'],parameter_count=initial['parameter_count'],
        changed_parameter_tensors=92,all_trainable_parameters_changed=True,all_named_adam_steps_verified=True,
        direct_constructor_equality=True,empty_initial_adam=True,fresh_streams=True)
    atomic_json(directory/'persisted_progress_verification.json',result); return result


def profile(directory):
    cfg=json.loads((directory/'profile_config.json').read_text()); verify_sources(cfg)
    session=Session(directory,cfg); warm=[]; rows=[]; torch.cuda.reset_peak_memory_stats()
    for i in range(6):
        if time.time()>=cfg['deadline']-60: raise RuntimeError('Native profile cap exhausted')
        row=session.train_update(*session.scheduler.next()); (warm if i<3 else rows).append(row)
    evaluation=evaluate(session.model,'val',20,4,session.device,directory/'eval_timing.json',cfg['deadline']-30)
    assert evaluation['complete']
    atomic_json(directory/'profile.json',dict(complete=True,rows=rows,warmup_rows=warm,evaluation_cells=evaluation['cells'],
        disposable=True,discard_all_state=True,optimizer_tensors=len(session.optimizer.state),optimizer_updates=session.state['step'],
        checkpoints_written=False,peak_cuda_bytes=torch.cuda.max_memory_allocated(),precision='fp32',sdpa='math',tf32=False))

def evaluate(model,split,count,microbatch,device,output,deadline):
    return base.evaluate(model,split,count,4,device,output,deadline)

def measured_plan(directory,budget):
    validate_budget(budget,budget)
    return base.measured_plan(directory,budget)

run=base.replay.clone(base.run,Session=Session,evaluate=evaluate,provenance=provenance,PROTOCOL=PROTOCOL,validate_budget=validate_budget)
completion=base.completion
supervise=base.replay.clone(base.supervise,MODULE=MODULE,completion=completion,validate_budget=validate_budget)


def config_for(budget):
    validate_budget(budget,budget)
    manifest=json.loads((ROOT.parent/'deployment_manifest.json').read_text())
    assert manifest['protocol']==PROTOCOL and manifest['checkpoints_included'] is False
    hashes={str(ROOT.parent/row['path']):row['sha256'] for row in manifest['files']}
    cfg=dict(budget,protocol=PROTOCOL,device='cuda',effective_batch=32,microbatch=4,eval_microbatch=4,cpu_threads=2,
        source_hashes=hashes,initialization=provenance(),precision='fp32',bptt='full',tf32=False,sdpa='math',all_trainable=True,
        lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clipping=None,checkpoint_every=100,val_n=100,test_n=200)
    verify_sources(cfg); return cfg


def prepare(directory,budget):
    validate_budget(budget,budget); directory.mkdir(parents=True,exist_ok=True)
    with (directory/'budget.json').open('x') as stream: json.dump(budget,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
    (directory/'budget.json').chmod(0o444)
    cfg=config_for(budget); pd=directory/'profile'; pd.mkdir()
    atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],time.time()+3600)))


def pin(directory):
    budget=json.loads((directory/'budget.json').read_text()); cfg=config_for(budget); plan=measured_plan(directory,budget); cfg.update(plan)
    for name,value in [('allocation.json',plan),('config.json',cfg)]:
        with (directory/name).open('x') as stream: json.dump(value,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
        (directory/name).chmod(0o444)
    return plan


def main():
    base.local.cpu_setup(); parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','profile','pin','run','verify','supervise-profile','supervise-run']); parser.add_argument('directory',type=Path); parser.add_argument('--budget',type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='prepare':
        if args.budget is None: parser.error('--budget required')
        return prepare(directory,json.loads(args.budget.read_text()))
    if args.mode=='pin': print(json.dumps(pin(directory))); return
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode.startswith('supervise-'): raise SystemExit(supervise(directory,args.mode.split('-',1)[1])['returncode'])
    cuda_setup()
    import fcntl
    with (ROOT.parent/'weighted_mean_rvit_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))
if __name__=='__main__': main()
