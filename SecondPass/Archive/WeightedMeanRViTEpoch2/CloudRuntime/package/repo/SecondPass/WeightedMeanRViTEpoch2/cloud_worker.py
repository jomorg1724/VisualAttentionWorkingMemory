"""The sole protocol change is ten to two epochs per1000-movie pool."""
import argparse, json, random, time
from pathlib import Path
from SecondPass.WeightedMeanRViT import worker as original
from SecondPass.WeightedMeanRViT import cloud_worker as cloud

MODULE='SecondPass.WeightedMeanRViTEpoch2.cloud_worker'
VERSION=original.VERSION
PROTOCOL=VERSION+'_pool1000_epoch2_cuda_a40_v1'
TARGET=2310
POOL_SIZE,EPOCHS,POOL_UPDATES=1000,2,66
TASK,TASKS,CELLS=original.TASK,original.TASKS,original.CELLS
INIT_SEED,SCHEDULER_SEED,CUDA_SEED=cloud.INIT_SEED,cloud.SCHEDULER_SEED,cloud.CUDA_SEED
TRAIN_NAMESPACE,VAL_NAMESPACE,FINAL_NAMESPACE=original.TRAIN_NAMESPACE,original.VAL_NAMESPACE,original.FINAL_NAMESPACE
ROOT=Path(__file__).resolve().parents[2]
atomic_json,append_jsonl,cpu_tree,tree_equal,digest,load_verified=cloud.atomic_json,cloud.append_jsonl,cloud.cpu_tree,cloud.tree_equal,cloud.digest,cloud.load_verified
validate_budget,verify_sources,cuda_setup=cloud.validate_budget,cloud.verify_sources,cloud.cuda_setup
FreshStream=original.FreshStream
WeightedMeanRViT=original.WeightedMeanRViT
selection_key,evaluate=original.selection_key,cloud.evaluate
validation_steps_for=original.validation_steps_for

class ReplayStream(original.ReplayStream):
    # next_plan's module-global EPOCHS is rebound; renderer, RNG, permutation and commits stay identical.
    next_plan=original.replay.clone(original.replay.ReplayStream.next_plan,EPOCHS=EPOCHS)
    def state_dict(self):
        state=super().state_dict(); state['replay'].update(replay_epochs=EPOCHS,updates_per_pool=POOL_UPDATES)
        return state
    def load_state_dict(self,state):
        assert state['replay']['replay_epochs']==EPOCHS and state['replay']['updates_per_pool']==POOL_UPDATES
        super().load_state_dict(state)

class ReplayScheduler(original.replay.ReplayScheduler):
    def state_dict(self): return dict(super().state_dict(),replay_epochs=EPOCHS,updates_per_pool=POOL_UPDATES)
    def load_state_dict(self,state):
        assert state['replay_epochs']==EPOCHS and state['updates_per_pool']==POOL_UPDATES
        super().load_state_dict(state)


def provenance(config=None):
    return dict(cloud.provenance(config),training='1000 fresh movies, two shuffled epochs, then replace; final binary crossentropy only',
        pool_size=POOL_SIZE,replay_epochs=EPOCHS,updates_per_epoch=33,updates_per_pool=POOL_UPDATES,
        sole_change='replay ten epochs to two; architecture, optimization, data rendering, seeds and namespaces unchanged')


def verify_progress(directory):
    result=_verify(directory)
    saved=load_verified(result['checkpoint']); state=saved['stream']['replay']
    assert state['replay_epochs']==EPOCHS and state['updates_per_pool']==POOL_UPDATES and state['epoch']<EPOCHS
    assert saved['scheduler']['replay_epochs']==EPOCHS and saved['scheduler']['updates_per_pool']==POOL_UPDATES
    result.update(replay_epochs=EPOCHS,updates_per_pool=POOL_UPDATES)
    atomic_json(Path(directory)/'persisted_progress_verification.json',result); return result

_verify=original.replay.clone(cloud.verify_progress,provenance=provenance)

class Session(cloud.Session):
    def __init__(self,directory,config):
        super().__init__(directory,config)
        self.stream=ReplayStream(); self.scheduler=ReplayScheduler(self.stream)
        initialization=json.loads((self.directory/'initialization.json').read_text())
        initialization['provenance']=provenance(config)
        atomic_json(self.directory/'initialization.json',initialization)
    checkpoint=original.replay.clone(cloud.Session.checkpoint,provenance=provenance,verify_progress=verify_progress)

profile=original.replay.clone(cloud.profile,Session=Session)
run=original.replay.clone(cloud.run,Session=Session,provenance=provenance,PROTOCOL=PROTOCOL)
completion=cloud.completion
supervise=original.replay.clone(cloud.supervise,MODULE=MODULE,completion=completion)


def measured_plan(directory,budget):
    validate_budget(budget,budget)
    profile=json.loads((Path(directory)/'profile/profile.json').read_text()); assert profile['complete'] and len(profile['rows'])==3
    costs={r['cell']:r['seconds'] for r in profile['rows']}; assert set(costs)==set(CELLS)
    val=1.35*sum(r['seconds']/r['n'] for r in profile['evaluation_cells'])*100; test=2*val
    generation=sum(r['pool_generation_seconds'] for r in profile['warmup_rows']); rng=random.Random(SCHEDULER_SEED+71)
    schedule=[]; training=0.; best=None; now=time.time()
    for i in range(TARGET):
        if i%POOL_UPDATES==0: sizes={c:333+int(j==(i//POOL_UPDATES)%3) for j,c in enumerate(CELLS)}
        if i%33==0: schedule.extend(original.replay.epoch_batches(sizes,rng))
        training+=costs[schedule[i][0]]; generation_total=(i//POOL_UPDATES+1)*generation
        total=1.25*(training+generation_total)+len(validation_steps_for(i+1))*val+2*test+900
        if total+60<budget['deadline']-now: best=i+1
    if best is None or best<POOL_UPDATES: raise RuntimeError('A complete two-epoch pool cannot fit remaining original cap')
    count=best//POOL_UPDATES*POOL_UPDATES
    training=sum(costs[cell] for cell,_ in schedule[:count]); generation_total=count//POOL_UPDATES*generation
    return dict(max_steps=count,target_requested=TARGET,exposure_reduced=count<TARGET,
        total_episodes=sum(len(indices) for _,indices in schedule[:count]),planned_unique_movies=count//POOL_UPDATES*POOL_SIZE,
        replay_epochs=EPOCHS,updates_per_pool=POOL_UPDATES,validation_steps=validation_steps_for(count),
        estimated_total_remaining_seconds=1.25*(training+generation_total)+len(validation_steps_for(count))*val+2*test+900,
        estimated_validation_seconds=val,estimated_one_test_seconds=test,final_reserve_seconds=val+2*test+180,
        update_estimate_seconds=1.25*max(costs.values())+generation)


def main():
    original.local.cpu_setup(); parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['profile','pin','run','verify','supervise-profile','supervise-run']); parser.add_argument('directory',type=Path)
    args=parser.parse_args(); directory=args.directory.resolve()
    if args.mode=='verify': print(json.dumps(verify_progress(directory))); return
    if args.mode=='pin':
        budget=json.loads((directory/'budget.json').read_text()); cfg=json.loads((directory/'config.json').read_text())
        validate_budget(budget,cfg); plan=measured_plan(directory,budget); cfg.update(plan)
        # Directly supplied source-bound config; standalone deployment_manifest is intentionally unnecessary.
        atomic_json(directory/'config.json',cfg)
        atomic_json(directory/'allocation.json',plan); print(json.dumps(plan)); return
    if args.mode.startswith('supervise-'): raise SystemExit(supervise(directory,args.mode.split('-',1)[1])['returncode'])
    cuda_setup()
    import fcntl
    with (ROOT.parent/'weighted_mean_rvit_worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='profile': profile(directory)
        else: raise SystemExit(run(directory))
if __name__=='__main__': main()
