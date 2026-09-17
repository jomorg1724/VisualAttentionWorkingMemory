"""Three temporal accumulators under a new explicit7200-second allowance."""
import os
import sys
import time
import json
import math
import shutil
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAV=HERE.parent
sys.path.insert(0,str(PAV.parent))
from PreAttentiveVision.train import read,write,sha

ARMS=['spatial_kda','convgru','opponent']
PARENT_ROOT=PAV/'runs'/'allocation_20260912_160414'
PARENT_CP=PARENT_ROOT/'contour_focus_seed20271'/'checkpoint_002268.pt'


def run(root):
    MODEL_NAMES=ARMS
    from PreAttentiveVision.neuroscience_stimuli import TASK_CLASSES
    remaining_seconds=7200.0
    parent_hash=sha(PARENT_CP)
    index=[json.loads(line) for line in (PARENT_CP.parent/'checkpoint_index.jsonl').read_text().splitlines()]
    if not any(x['file']==PARENT_CP.name and x['sha256']==parent_hash for x in index):raise RuntimeError('Parent index identity failed')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Cannot renew an existing run budget')
    if len(TASK_CLASSES)!=7:raise RuntimeError('Final defined seven-task registry required')
    # Archive executable dependency closure before any GPU work. Dataset identities are copied separately.
    names=['models.py','decoder.py','decoder_multitask.py','hybrid_models.py','train.py','evaluate.py','evaluate_multitask.py',
           'neuroscience_stimuli.py','natural_stimuli.py','allocation_sampler.py',
           'TemporalIntegration/accumulators.py','TemporalIntegration/train_temporal.py','TemporalIntegration/eval_temporal.py','TemporalIntegration/sweep_temporal.py']
    hashes={name:sha(PAV/name) for name in names}
    (root/'source').mkdir()
    for name in names:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PAV/name,target)
    shutil.copy2(PAV/'data'/'bsds500'/'manifest.json',root/'dataset_manifest.json')
    base=dict(batch_size=32,lr=.001,weight_decay=.0001,grad_clip=5.,fp32=True,amp=False,
        task_classes=TASK_CLASSES,decoder='current_plus_causal_state_v1',optimizer='Adam',common_seed=30312,core_seed=30313,train_base_seed=310001,
        training_recipe='Same focused12cycle for all three; frozenencoder, freshprojection/core/readout; task-local streams310001+100003*(index+1)',sampler_protocol='task_local_allocation_v1',
        checkpoint_selection='Highest minimum task BA, then mean task OVR-AUC; exact ties earliest checkpoint',
        decision_rule='Fixed argmax for every task; no test tuning')
    now=time.time();deadline=now+remaining_seconds
    ledger=dict(started_unix=now,deadline_unix=deadline,hard_seconds=remaining_seconds,
        allowance_origin='NEW user-authorized7200s temporal experiment; does not use old378.55s remainder',total_authorized_seconds=7200,
        source_hashes=hashes,active=None,events=[],status='profiling')
    aggregate=dict(status='profiling',run_root=str(root),config=dict(models=MODEL_NAMES,task_classes=TASK_CLASSES,
        batch_size=32,seeds=[30301],budget_seconds=remaining_seconds,parent_checkpoint=str(PARENT_CP),parent_sha256=parent_hash,parent_step=2268,engineering_criterion='At least.95balanced accuracy on EVERYtask; ambitious engineering target, not guaranteed horizon'),profiles={},runs=[])
    def publish():
        write(root/'aggregate.json',aggregate);write(HERE/'results_temporal.json',aggregate)
    write(root/'budget.json',ledger);publish()

    def launch(kind,cfg,out,**kwargs):
        number=len(ledger['events']);job_path=root/f'job_{number:03d}.json';result=root/f'job_{number:03d}_result.json'
        job=dict(kind=kind,config=cfg,out=str(out),deadline=deadline,result=str(result),source_hashes=hashes,parent_checkpoint=str(PARENT_CP),parent_sha256=parent_hash,**kwargs)
        write(job_path,job);log_path=root/f'worker_{number:03d}.log';tick=time.time()
        with log_path.open('w') as log:
            process=subprocess.Popen([sys.executable,'-B',str(HERE/'train_temporal.py'),str(job_path)],
                stdout=log,stderr=subprocess.STDOUT,cwd=PAV.parent)
            ledger['active']=dict(pid=process.pid,kind=kind,model=cfg['model'],seed=cfg['seed'],started_unix=tick,log=str(log_path))
            write(root/'budget.json',ledger)
            try:code=process.wait(timeout=max(0,deadline-time.time()))
            except subprocess.TimeoutExpired:process.kill();process.wait();code=124
            finally:
                if process.poll() is None:process.kill();process.wait()
        elapsed=time.time()-tick
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=elapsed))
        ledger['active']=None;write(root/'budget.json',ledger)
        if code:raise RuntimeError(f'{kind} {cfg["model"]} exit{code}; see {log_path}')
        return read(result),elapsed

    try:
        for name in MODEL_NAMES:
            cfg=dict(base,model=name,seed=30301,purpose='separate_parent_derived_profile')
            profile,wall=launch('profile',cfg,root/'profiles'/name,target=48)
            profile['full_profile_wall_seconds']=wall;aggregate['profiles'][name]=profile;publish()
        traincost=sum(p['train_step_mean']*1.25 for p in aggregate['profiles'].values())
        evalcost=sum(p['eval_batch_mean']*1.3 for p in aggregate['profiles'].values())
        # Four proposed validation looks, three-mode final tests, preserved pair reference, and analysis reserve.
        val_n=224;test_n=448;remaining=deadline-time.time()
        def estimate(updates):
            eval_batches=(updates//1008)*7*math.ceil(val_n/32)+3*7*math.ceil(test_n/32)
            reference_cost=7*math.ceil(test_n/32)*max(p['eval_batch_mean'] for p in aggregate['profiles'].values())*1.5
            return updates*traincost+eval_batches*evalcost+reference_cost+180
        updates=next((u for u in [4032,3024,2016,1008] if estimate(u)<remaining*.94),None)
        if updates is None:raise RuntimeError('Measured costs cannot fit all three1008updates plus evaluation under7200s')
        targets=list(range(1008,updates+1,1008))
        from PreAttentiveVision.allocation_sampler import exposure_counts,TaskLocalStreams
        aggregate['config'].update(updates=updates,targets=targets,training_pairs_per_arm=updates*32,
            training_pairs_per_task={task:n*32 for task,n in exposure_counts(updates,'contour_focus').items()},
            training_derived_seeds=TaskLocalStreams(310001,'train').derived_seeds,
            val_pairs_per_task=val_n,test_pairs_per_task=test_n,recipe=base,estimated_remaining_seconds=estimate(updates),
            uncertainty='1000 source-photo-cluster or generated-pair bootstrap replicates per normal task; paired intervals versus pair reference; diagnostic repetitions not independent samples')
        write(root/'fixed_config.json',aggregate['config']);print('FIXED_ALLOCATION '+json.dumps(aggregate['config']),flush=True)
        aggregate['status']='training';ledger['status']='training';write(root/'budget.json',ledger)
        for name in MODEL_NAMES:
            aggregate['runs'].append(dict(model=name,seed=30301,status='pending',step=0,training_seconds=0.,val_curve=[],test=None))
        publish()
        for target in targets:
            for record in aggregate['runs']:
                name=record['model'];out=root/f'{name}_seed30301';cfg=dict(base,model=name,seed=30301,purpose='new_temporal_module_production')
                trained,wall=launch('train',cfg,out,target=target,checkpoint=record.get('checkpoint'))
                record.update(status='running',step=trained['step'],checkpoint=trained['checkpoint'],train_pairs=trained['fresh_pairs'],per_task_updates=trained['per_task_updates'])
                record['training_seconds']+=wall;publish()
                if trained['step']!=target:raise TimeoutError('Training deadline prevented equal fixed endpoint')
                val,_=launch('eval',cfg,out/f'eval_val_{target:06d}',checkpoint=record['checkpoint'],split='val',eval_seed=910001,n_per_task=val_n,resamples=0)
                minimum=min(v['overall']['balanced_accuracy'] for v in val['tasks'].values())
                record['val_curve'].append(dict(step=target,min_task_ba=minimum,per_task_updates=trained['per_task_updates'],macro_ovr_auc=val['macro_ovr_auc'],mean_normalized_balanced_accuracy=val['mean_normalized_balanced_accuracy'],tasks=val['tasks']))
                if record.get('best_min_task_ba') is None or (minimum,val['macro_ovr_auc'])>(record['best_min_task_ba'],record['best_auc']):
                    record.update(best_min_task_ba=minimum,best_auc=val['macro_ovr_auc'],best_step=target,best_checkpoint=record['checkpoint'],selected_per_task_updates=trained['per_task_updates'])
                publish()
        aggregate['status']='final_test';publish()
        for record in aggregate['runs']:
            out=root/f"{record['model']}_seed30301";cfg=dict(base,model=record['model'],seed=30301,purpose='new_temporal_module_production')
            test,_=launch('eval',cfg,out/'eval_test',checkpoint=record['best_checkpoint'],split='test',eval_seed=1010001,n_per_task=test_n,resamples=1000)
            record.update(test=test,status='completed');publish()
        reference_cfg=dict(base,model='pair_reference',seed=30301,purpose='preserved_two_frame_reference')
        reference,_=launch('reference',reference_cfg,root/'pair_reference',split='test',eval_seed=1010001,n_per_task=test_n,resamples=1000)
        aggregate['pair_reference']=reference;publish()
        from PreAttentiveVision.TemporalIntegration.eval_temporal import paired_comparison
        reference_rows=[json.loads(line) for line in Path(reference['predictions']).read_text().splitlines()]
        comparisons=[]
        for record in aggregate['runs']:
            candidate_rows=[json.loads(line) for line in Path(record['test']['predictions']).read_text().splitlines()]
            comparisons.append(dict(model=record['model'],reference='pair_reference',tasks=paired_comparison(candidate_rows,reference_rows,deadline=deadline)))
        aggregate['paired_comparisons']=comparisons
        write(root/'paired_analysis.json',dict(comparisons=comparisons))
        aggregate['status']='completed'
    except BaseException as error:
        aggregate.update(status='budget_stopped' if time.time()>=deadline-5 else 'failed',error=repr(error))
        raise
    finally:
        aggregate['wall_seconds']=time.time()-now;publish();ledger['status']=aggregate['status'];write(root/'budget.json',ledger)
        write(root/'exit.json',dict(status=aggregate['status'],wall_seconds=time.time()-now,deadline_unix=deadline,error=aggregate.get('error')))


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: sweep_temporal.py NEW_RUN_ROOT')
    run(sys.argv[1])
