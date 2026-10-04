"""Fresh weighted-raw-frame CNN/RViT classification, queued local execution."""
import argparse, json, os, random, time, types
from pathlib import Path
import numpy as np
import torch
from SecondPass.VAERViT import worker as harness
from SecondPass.SingleStimulusRViT import worker as replay
from SecondPass.SingleStimulusRViT.stimuli import SingleStimulusStream
from .model import WeightedMeanRViT

ROOT=Path(__file__).resolve().parents[2]
MODULE='SecondPass.WeightedMeanRViT.worker'
VERSION='weighted_raw_mean_0p5_0p4_0p1_fresh_cnn_rvit'
PROTOCOL=VERSION+'_pool1000_epoch10_mps_v1'
TARGET=2310
TASK,TASKS,CELLS=harness.TASK,harness.TASKS,harness.CELLS
INIT_SEED,SCHEDULER_SEED=harness.INIT_SEED,harness.SCHEDULER_SEED
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=183201,183202,183203
atomic_json,append_jsonl,cpu_tree,tree_equal,digest,load_verified=harness.atomic_json,harness.append_jsonl,harness.cpu_tree,harness.tree_equal,harness.digest,harness.load_verified
verify_sources,validate_budget,local=harness.verify_sources,harness.validate_budget,harness.local
validation_steps_for,selection_key=harness.validation_steps_for,harness.selection_key

class FreshStream(SingleStimulusStream):
    def stream_seed(self,task,cell):
        index,_=self._cell(task,cell)
        return dict(train=TRAIN_NAMESPACE,val=VAL_NAMESPACE,test=FINAL_NAMESPACE)[self.split]*100000+TASKS[task]['stream_id']*1000+index

class ReplayStream(replay.ReplayStream):
    stream_seed=FreshStream.stream_seed
    _new_pool=replay.clone(replay.ReplayStream._new_pool,base=types.SimpleNamespace(FreshStream=FreshStream))
    load_state_dict=replay.clone(replay.ReplayStream.load_state_dict,base=types.SimpleNamespace(FreshStream=FreshStream))

evaluate=torch.no_grad()(replay.clone(replay.original.evaluate.__wrapped__,FreshStream=FreshStream,score_cell=replay.score_cell,
    sync=local.sync,VAL_NAMESPACE=VAL_NAMESPACE,FINAL_NAMESPACE=FINAL_NAMESPACE))

def provenance(config):
    return dict(architecture=VERSION,initialization='whole CNN, RViT, positions and decoder direct fresh constructor',
        inherited_weights=False,inherited_encoder=False,inherited_recurrent=False,inherited_classifier=False,
        inherited_optimizer=False,inherited_rng=False,inherited_stream=False,profile_state_inherited=False,
        all_trainable=True,initialization_seed=INIT_SEED,
        train_namespace=TRAIN_NAMESPACE,validation_namespace=VAL_NAMESPACE,final_namespace=FINAL_NAMESPACE,
        weighted_raw_frames=[.5,.4,.1],startup='repeat first frame for missing previous frames',
        training='1000 fresh movies, ten shuffled epochs, then replace; final binary crossentropy only',
        stimulus='existing no-cue single-stimulus29/37/45frames, native26/28changes; change57/nochange43',
        checkpoint_policy='only best.pt and latest.pt')

verify_progress=harness.verify_progress

class Session(harness.Session):
    def __init__(self,directory,config):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True); self.config=config; self.device=config['device']; self.latest=None
        torch.manual_seed(INIT_SEED); random.seed(INIT_SEED); np.random.seed(INIT_SEED)
        if self.device=='mps': torch.mps.manual_seed(INIT_SEED)
        self.model=WeightedMeanRViT(checkpoint_encoder=True,checkpoint_recurrent=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(INIT_SEED)
            direct=WeightedMeanRViT(checkpoint_encoder=True,checkpoint_recurrent=True)
        assert tree_equal(self.model.state_dict(),direct.state_dict())
        names=[n for n,_ in self.model.named_parameters()]
        initial_hashes={n:harness.tensor_digest(p) for n,p in self.model.named_parameters()}
        del direct
        self.model=self.model.to(self.device)
        assert all(p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters())
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.stream=ReplayStream(); self.scheduler=replay.ReplayScheduler(self.stream)
        self.state=dict(step=0,episodes=0,frames=0,optimizer_seconds=0.,best_step=None,best_key=None,best_checkpoint=None,
            selection_history=[],config=config,unique_episodes_generated=0,unique_episodes_presented=0,replay_presentations=0,
            exposure={TASK:dict(updates=0,episodes=0,frames=0,cells={c:0 for c in CELLS})})
        assert not self.optimizer.state and not self.stream.state_dict()['streams']
        atomic_json(self.directory/'initialization.json',dict(verified=True,provenance=provenance(config),
            parameter_count=sum(p.numel() for p in self.model.parameters()),optimizer_names=names,initial_parameter_hashes=initial_hashes,
            direct_constructor_equality=True,empty_adam=True,zero_classification_counters=True,empty_streams=True,checkpoint_inputs=[]))
    status=replay.clone(harness.Session.status,VERSION=VERSION)
    checkpoint=replay.clone(harness.Session.checkpoint,provenance=provenance,verify_progress=verify_progress)

profile=replay.clone(harness.profile,Session=Session,evaluate=evaluate)
measured_plan=harness.measured_plan
run=replay.clone(harness.run,Session=Session,evaluate=evaluate,provenance=provenance,VERSION=VERSION,PROTOCOL=PROTOCOL)
completion=harness.completion
supervise=replay.clone(harness.supervise,MODULE=MODULE,completion=completion)

def local_supervise(directory):
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'budget.json').exists(): raise RuntimeError('No restart/cap renewal')
    # No budget starts while queued; this owner is called only after the preceding worker exits.
    start=time.time(); budget=dict(cap_started=start,deadline=start+28200,hard_deadline=start+28800,
        retrieval_reserve_seconds=600,wall_cap_seconds=28800,execution_placement='local')
    with (directory/'budget.json').open('x') as stream: json.dump(budget,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
    (directory/'budget.json').chmod(0o444); atomic_json(directory/'activation.json',dict(supervisor_pid=os.getpid(),owner_ppid=os.getppid(),**budget))
    manifest_path=ROOT/'runtime_manifest.json'; manifest=json.loads(manifest_path.read_text())
    assert not manifest.get('checkpoint_inputs'), 'Fresh model accepts no checkpoint input'
    cfg=dict(budget,protocol=PROTOCOL,device='mps',source_hashes=dict(manifest['source_hashes']),
        effective_batch=32,microbatch=1,eval_microbatch=1,cpu_threads=2,lr=1e-4,betas=[.9,.999],eps=1e-8,
        weight_decay=0,precision='fp32',clipping=None,bptt='full',checkpoint_every=100,initialization=provenance({}))
    cfg['source_hashes'][str(manifest_path)]=digest(manifest_path); verify_sources(cfg)
    pd=directory/'profile'; pd.mkdir(); atomic_json(pd/'profile_config.json',dict(cfg,disposable_profile=True,deadline=min(budget['deadline'],start+3600)))
    outcomes=[]
    try:
        outcome=supervise(directory,'profile'); outcomes.append(outcome)
        if outcome['returncode']!=0: raise RuntimeError('Native profile failed')
        plan=measured_plan(directory,budget); cfg.update(plan)
        for name,value in [('allocation.json',plan),('config.json',cfg)]:
            with (directory/name).open('x') as stream: json.dump(value,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
            (directory/name).chmod(0o444)
        outcome=supervise(directory,'run'); outcomes.append(outcome); result=dict(status=outcome['status'],outcomes=outcomes,**budget)
    except Exception as exc:
        result=dict(status='error',error=repr(exc),outcomes=outcomes,**budget); atomic_json(directory/'failure.json',result); completion(directory,'incomplete',result['error'])
    atomic_json(directory/'local_supervisor_result.json',result); return result


def main():
    local.cpu_setup(); parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['local-supervise','profile','run','verify']); parser.add_argument('directory',type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode=='local-supervise':
        result=local_supervise(directory); print(json.dumps(result),flush=True); raise SystemExit(0 if result['status']=='complete' else 2)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    import fcntl
    with Path('/Users/jonathanmorgan/VAWMRuntime/local_gpu_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))
if __name__=='__main__': main()
