"""Three-arm parent-derived hybrid comparison within the remaining allowance."""
import os
import sys
import time
import json
import math
import shutil
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from PreAttentiveVision.train import read,write,sha

ARMS=['convnext_grn','convnext_gabor_residual','convnext_se_residual']
PARENT_ROOT=HERE/'runs'/'multitask_20260912_141316'
PARENT_CP=PARENT_ROOT/'convnext_grn_seed20271'/'checkpoint_000756.pt'


def run(root):
    MODEL_NAMES=ARMS
    from PreAttentiveVision.neuroscience_stimuli import TASK_CLASSES
    spent=[read(HERE/'runs'/'benchmark_20260912_135139'/'exit.json')['wall_seconds'],read(PARENT_ROOT/'exit.json')['wall_seconds'],read(HERE/'combination_analysis.json')['cpu_seconds']]
    remaining_seconds=3600-sum(spent)
    parent_hash=sha(PARENT_CP)
    index=[json.loads(line) for line in (PARENT_CP.parent/'checkpoint_index.jsonl').read_text().splitlines()]
    if not any(x['file']==PARENT_CP.name and x['sha256']==parent_hash for x in index):raise RuntimeError('Parent index identity failed')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Cannot renew an existing run budget')
    if len(TASK_CLASSES)!=7:raise RuntimeError('Final defined seven-task registry required')
    # Archive executable dependency closure before any GPU work. Dataset identities are copied separately.
    names=['models.py','decoder.py','decoder_multitask.py','train.py','train_multitask.py',
           'evaluate.py','evaluate_multitask.py','neuroscience_stimuli.py','natural_stimuli.py','train_hybrids.py','hybrid_transfer.py','hybrid_models.py','sweep_hybrids.py']
    hashes={name:sha(HERE/name) for name in names}
    (root/'source').mkdir()
    for name in names:shutil.copy2(HERE/name,root/'source'/name)
    shutil.copy2(HERE/'data'/'bsds500'/'manifest.json',root/'dataset_manifest.json')
    base=dict(batch_size=32,lr=.001,weight_decay=.0001,grad_clip=5.,fp32=True,amp=False,
        task_classes=TASK_CLASSES,decoder='ordered_multiscale_correlation_v1',decoder_seed_offset=500000,
        training_recipe='One full32-pair task batch per update, seven-task round robin, mean crossentropy',
        checkpoint_selection='Highest equal-task macro OVR-AUC on fixed validation; exact ties earliest checkpoint',
        decision_rule='Fixed argmax for every task; no test tuning')
    now=time.time();deadline=now+remaining_seconds
    ledger=dict(started_unix=now,deadline_unix=deadline,hard_seconds=remaining_seconds,
        previous_compute_charges_seconds=spent,total_authorized_seconds=3600,
        source_hashes=hashes,active=None,events=[],status='profiling')
    aggregate=dict(status='profiling',run_root=str(root),config=dict(models=MODEL_NAMES,task_classes=TASK_CLASSES,
        batch_size=32,seeds=[20271],budget_seconds=remaining_seconds,parent_checkpoint=str(PARENT_CP),parent_sha256=parent_hash,parent_step=756,engineering_criterion='Contour BA gain>=.03 versus continued control AND motion BA loss<=.02; point-estimate engineering screen, paired CIs reported'),profiles={},runs=[])
    def publish():
        write(root/'aggregate.json',aggregate);write(HERE/'results_hybrids.json',aggregate)
    write(root/'budget.json',ledger);publish()

    def launch(kind,cfg,out,**kwargs):
        number=len(ledger['events']);job_path=root/f'job_{number:03d}.json';result=root/f'job_{number:03d}_result.json'
        job=dict(kind=kind,config=cfg,out=str(out),deadline=deadline,result=str(result),source_hashes=hashes,parent_checkpoint=str(PARENT_CP),parent_sha256=parent_hash,**kwargs)
        write(job_path,job);log_path=root/f'worker_{number:03d}.log';tick=time.time()
        with log_path.open('w') as log:
            process=subprocess.Popen([sys.executable,'-B',str(HERE/'train_hybrids.py'),str(job_path)],
                stdout=log,stderr=subprocess.STDOUT,cwd=HERE.parent)
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
            cfg=dict(base,model=name,seed=20271,purpose='separate_parent_derived_profile')
            profile,wall=launch('profile',cfg,root/'profiles'/name,target=777)
            profile['full_profile_wall_seconds']=wall;aggregate['profiles'][name]=profile;publish()
        traincost=sum(max(p['train_step_median']*1.35,p['train_step_p90']*1.15) for p in aggregate['profiles'].values())
        evalcost=sum(p['eval_batch_mean']*1.3 for p in aggregate['profiles'].values())
        # Three validation looks, final tests for three arms plus frozen parent, and overhead reserve.
        val_n=224;test_n=448;eval_batches=3*7*math.ceil(val_n/32)+7*math.ceil(test_n/32)*4/3
        reserve=eval_batches*evalcost+250
        remaining=deadline-time.time()
        updates=next((u for u in [1512,1260,1008,756,504,252] if u*traincost+reserve<remaining*.92),None)
        if updates is None:raise RuntimeError('Remaining allowance cannot fit balanced three-arm training plus evaluation')
        targets=[756+updates//3,756+2*updates//3,756+updates]
        aggregate['config'].update(additional_updates=updates,final_step=756+updates,targets=targets,additional_training_pairs_per_arm=updates*32,
            additional_training_pairs_per_task_per_arm=updates*32//7,val_pairs_per_task=val_n,test_pairs_per_task=test_n,
            recipe=base,estimated_remaining_seconds=updates*traincost+reserve,
            uncertainty='1000 source-image-cluster or generated-pair bootstrap replicates per overall task; difficulty points and class-wise Wilson intervals; single seed')
        write(root/'fixed_config.json',aggregate['config']);print('FIXED_ALLOCATION '+json.dumps(aggregate['config']),flush=True)
        aggregate['status']='training';ledger['status']='training';write(root/'budget.json',ledger)
        for name in MODEL_NAMES:
            aggregate['runs'].append(dict(model=name,seed=20271,status='pending',step=756,training_seconds=0.,val_curve=[],test=None))
        publish()
        for target in targets:
            for record in aggregate['runs']:
                name=record['model'];out=root/f'{name}_seed20271';cfg=dict(base,model=name,seed=20271,purpose='parent_derived_hybrid_production')
                trained,wall=launch('train',cfg,out,target=target,checkpoint=record.get('checkpoint'))
                record.update(status='running',step=trained['step'],checkpoint=trained['checkpoint'],train_pairs=trained['fresh_pairs'],additional_pairs=trained['additional_pairs'],additional_updates=trained['additional_updates'],per_task_updates=trained['per_task_updates'])
                record['training_seconds']+=wall;publish()
                if trained['step']!=target:raise TimeoutError('Training deadline prevented equal fixed endpoint')
                val,_=launch('eval',cfg,out/f'eval_val_{target:06d}',checkpoint=record['checkpoint'],split='val',eval_seed=850001,n_per_task=val_n,resamples=0)
                record['val_curve'].append(dict(step=target,macro_ovr_auc=val['macro_ovr_auc'],mean_normalized_balanced_accuracy=val['mean_normalized_balanced_accuracy'],tasks=val['tasks']))
                if record.get('best_auc') is None or val['macro_ovr_auc']>record['best_auc']:
                    record.update(best_auc=val['macro_ovr_auc'],best_step=target,best_checkpoint=record['checkpoint'])
                publish()
        aggregate['status']='final_test';publish()
        for record in aggregate['runs']:
            out=root/f"{record['model']}_seed20271";cfg=dict(base,model=record['model'],seed=20271,purpose='parent_derived_hybrid_production')
            test,_=launch('eval',cfg,out/'eval_test',checkpoint=record['best_checkpoint'],split='test',eval_seed=950001,n_per_task=test_n,resamples=1000)
            record.update(test=test,status='completed');publish()
        if deadline-time.time()>90:
            parent_cfg=dict(base,model='convnext_grn',seed=20271,purpose='frozen_parent_reference')
            reference,_=launch('eval',parent_cfg,root/'parent_reference',allow_parent_eval=True,split='test',eval_seed=950001,n_per_task=test_n,resamples=1000)
            aggregate['parent_reference']=reference
        else:aggregate['parent_reference']=dict(status='omitted_to_preserve_hard_deadline')
        aggregate['status']='completed'
    except BaseException as error:
        aggregate.update(status='budget_stopped' if time.time()>=deadline-5 else 'failed',error=repr(error))
        raise
    finally:
        aggregate['wall_seconds']=time.time()-now;publish();ledger['status']=aggregate['status'];write(root/'budget.json',ledger)
        write(root/'exit.json',dict(status=aggregate['status'],wall_seconds=time.time()-now,deadline_unix=deadline,error=aggregate.get('error')))


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: sweep_hybrids.py NEW_RUN_ROOT')
    run(sys.argv[1])
