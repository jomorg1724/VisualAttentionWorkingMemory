"""Exact-state continuation; original training implementation remains unchanged."""
import argparse,copy,json,os,random,shutil,time,types
from pathlib import Path
import numpy as np
import torch
from .. import train as base
from ..dataset import Stream,TASKS,all_cells
from SecondPass.JointTraining.core import BalancedScheduler,tree_equal

class FinalStream(Stream):
    def stream_seed(self,task,cell):
        if self.split!='test':return super().stream_seed(task,cell)
        index,_=self._cell(task,cell)
        return 199104*100000+TASKS[task]['stream_id']*1000+index

# Reuse the exact evaluator with only its test namespace changed.
original_evaluate=base.evaluate.__wrapped__
evaluate=torch.no_grad()(types.FunctionType(original_evaluate.__code__,dict(original_evaluate.__globals__,Stream=FinalStream),
    original_evaluate.__name__,original_evaluate.__defaults__,original_evaluate.__closure__))

def verify_sources(saved):
    for name,digest in saved['config']['source_hashes'].items():
        assert base.sha(base.ROOT/name)==digest, 'Original source changed: '+name
    assert base.sha(base.DATA_ROOT/'manifest.json')==saved['config']['dataset_manifest_sha256']

def restore(saved,device):
    model,opt=base.build(device);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    scheduler=BalancedScheduler(199212);scheduler.load_state_dict(saved['scheduler'])
    stream=Stream('train');stream.load_state_dict(saved['stream'])
    torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return model,opt,scheduler,stream,copy.deepcopy(saved['state'])

def verify_checkpoint(path,original=None):
    saved=torch.load(path,map_location='cpu')
    with torch.random.fork_rng(devices=[]):model=base.StabilizedAttentionSuite()
    model.load_state_dict(saved['model'])
    named=list(model.named_parameters());ids=[i for g in saved['optimizer']['param_groups'] for i in g['params']]
    states=saved['optimizer']['state'];state=saved['state'];assert len(named)==len(ids)==len(states)==80
    for (name,p),i in zip(named,ids):
        s=states[i];expected=state['task_updates'][name.split('.')[1]] if name.startswith('heads.') else state['step']
        assert int(s['step'])==expected
        assert s['exp_avg'].shape==s['exp_avg_sq'].shape==p.shape
        assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    result=dict(step=state['step'],finite_aligned_states=80,task_updates=state['task_updates'])
    if original:
        parent=torch.load(original,map_location='cpu')
        changed=[name for name,p in model.named_parameters() if not torch.equal(saved['model'][name],parent['model'][name])]
        assert changed
        result.update(changed_parameter_tensors=len(changed),added=state['step']-parent['state']['step'])
    return result

def checks(parent):
    torch.set_num_threads(2)
    saved=torch.load(parent/'latest.pt',map_location='cpu');verify_sources(saved)
    assert saved['state']['step']==13702
    model,opt,scheduler,stream,state=restore(saved,'cpu')
    assert tree_equal(model.state_dict(),saved['model']) and tree_equal(opt.state_dict(),saved['optimizer'])
    assert tree_equal(scheduler.state_dict(),saved['scheduler']) and tree_equal(stream.state_dict(),saved['stream'])
    assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
    assert tree_equal(np.random.get_state(),saved['numpy_rng']) and random.getstate()==saved['python_rng']
    other_scheduler=BalancedScheduler(0);other_scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.next()==other_scheduler.next()
    task,cell=scheduler.next();other=Stream('train');other.load_state_dict(saved['stream'])
    assert tree_equal(stream.batch(2,task,cell),other.batch(2,task,cell))
    print(json.dumps(dict(cpu_restore_exact=True,next_native_draw_exact=True,latest=verify_checkpoint(parent/'latest.pt'),
        best=verify_checkpoint(parent/'best.pt')),indent=2))

def production(run,parent):
    assert not (run/'production_config.json').exists()
    saved=torch.load(parent/'latest.pt',map_location='cpu');verify_sources(saved)
    budget=json.loads((run/'budget.json').read_text());deadline=budget['deadline'];start_step=saved['state']['step']
    assert start_step==13702;target=start_step+13702
    originals={str(p.relative_to(base.ROOT)):base.sha(p) for p in (parent/'best.pt',parent/'latest.pt')}
    base.atomic_json(run/'preserved_originals.json',originals)
    base.configure('cuda');model,opt,scheduler,stream,state=restore(saved,'cuda')
    assert tree_equal(base.cpu_tree(model.state_dict()),saved['model'])
    assert tree_equal(base.cpu_tree(opt.state_dict()),saved['optimizer'])
    assert tree_equal(stream.state_dict(),saved['stream']) and tree_equal(scheduler.state_dict(),saved['scheduler'])
    assert tree_equal(torch.cuda.get_rng_state_all(),saved['cuda_rng'])
    config=copy.deepcopy(saved['config'])
    config.update(fresh_whole_model=False,checkpoint_inputs=[str(parent/'latest.pt')],continuation=True,
        initial_step=start_step,additional_updates=13702,target_updates=target,deadline=deadline,started=budget['started'],
        final_test_namespace=199104,parent_checkpoint_hashes=originals,
        continuation_source_hashes={str(p.relative_to(base.ROOT)):base.sha(p) for p in Path(__file__).parent.glob('*.py')})
    base.atomic_json(run/'production_config.json',config)
    for name in dict(config['source_hashes'],**config['continuation_source_hashes']):
        destination=run/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.ROOT/name,destination)
    shutil.copy2(parent/'best.pt',run/'best.pt')
    base.atomic_json(run/'restored.json',dict(step=start_step,model_optimizer_stream_scheduler_exact=True,cuda_rng_exact=True,
        inherited_best_step=state['best_step'],inherited_best_key=state['best_key']))
    checks_at={start_step+130,target}|set(range(((start_step//520)+1)*520,target+1,520))
    def status(phase,**kw):base.atomic_json(run/'live_status.json',dict(phase=phase,pid=os.getpid(),utc=base.now(),
        initial_step=start_step,added_updates=state['step']-start_step,target_updates=target,**state,**kw))
    selected_loaded=False
    try:
        while state['step']<target:
            base.check_time(deadline)
            if (run/'stop_requested.json').exists():
                base.save(run,model,opt,scheduler,stream,state,config);status('user_stopped');return
            task,cell=scheduler.next();status('rendering',task=task,cell=cell)
            begin=time.time();x,y,_=stream.batch(32,task,cell);values=base.update(model,opt,x,y,task,'cuda')
            state['step']+=1;state['episodes']+=32;state['frames']+=values['frames']
            state['task_updates'][task]+=1;state['task_episodes'][task]+=32;state['cell_updates'][task+'/'+cell]+=1
            added=state['step']-start_step
            base.append_json(run/'progress.jsonl',dict(step=state['step'],added_updates=added,task=task,cell=cell,
                utc=base.now(),update_seconds=time.time()-begin,**values))
            status('training',task=task,cell=cell,**{k:v for k,v in values.items() if k not in ('episodes','frames')})
            if added in (1,13) or state['step']%130==0:
                base.save(run,model,opt,scheduler,stream,state,config)
                if added in (1,13):base.atomic_json(run/('verified_added_%03d.json'%added),verify_checkpoint(run/'latest.pt',parent/'latest.pt'))
            if state['step'] in checks_at:
                status('full_validation');result=base.evaluate(model,'val',64,100,'cuda',deadline,run/('validation_%06d.json'%state['step']))
                key=result['summary']['selection_key'];better=tuple(key)>tuple(state['best_key'])
                if better:state.update(best_key=key,best_step=state['step'])
                base.save(run,model,opt,scheduler,stream,state,config,best=better)
            elif state['step']%130==0:
                status('krauzlis_monitor');base.evaluate(model,'val',64,100,'cuda',deadline,run/('krauzlis_monitor_%06d.json'%state['step']),
                    cells=[(t,c) for t,c in all_cells() if t=='krauzlis_cued_motion'])
        final_latest=verify_checkpoint(run/'latest.pt',parent/'latest.pt')
        best=torch.load(run/'best.pt',map_location='cpu');model.load_state_dict(best['model']);selected_loaded=True
        selected=best['state']['step'];status('final_test',selected_step=selected)
        result=evaluate(model,'test',256,1000,'cuda',deadline,run/'final_test.json')
        assert all(base.sha(base.ROOT/n)==h for n,h in originals.items())
        base.atomic_json(run/'completion_verified.json',dict(status='complete',updates=state['step'],additional_updates=13702,
            episodes=state['episodes'],selected_step=selected,final_n=result['n'],test_namespace=199104,all35cells=True,
            latest_verification=final_latest,best_verification=verify_checkpoint(run/'best.pt'),originals_unchanged=True,completed=base.now()))
        status('complete',selected_step=selected)
    except BaseException as exc:
        if state['step']>start_step and not selected_loaded:base.save(run,model,opt,scheduler,stream,state,config)
        base.atomic_json(run/'failure.json',dict(error=repr(exc),step=state['step'],utc=base.now()));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--parent',type=Path,required=True);p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:checks(args.parent.resolve());return
    lock=base.GPULock(base.ROOT/'LocalRuntime/local_gpu_worker.lock')
    try:production(args.run.resolve(),args.parent.resolve())
    finally:lock.close()

if __name__=='__main__':main()
