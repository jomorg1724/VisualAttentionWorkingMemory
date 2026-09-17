"""One warm-started opponent learner under a new explicit four-hour allowance."""
import os,sys,time,json,shutil,subprocess,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'WorkingMemory'
sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha

PARENT=ROOT/'PreAttentiveVision/TemporalIntegration/runs/temporal_20260912_165510/opponent_seed30301/checkpoint_004032.pt'


def run(root):
    from WorkingMemory.stimuli import TASK_CLASSES,BRIDGE_CONDITIONS,TRAIN_CONDITIONS,condition_frames,VERSION
    from WorkingMemory.train import schedule
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('An existing budget cannot be renewed')
    parent_hash=sha(PARENT)
    if not any(x['file']==PARENT.name and x['sha256']==parent_hash for x in map(json.loads,(PARENT.parent/'checkpoint_index.jsonl').read_text().splitlines())):raise RuntimeError('Parent index identity failed')
    names=['WorkingMemory/model.py','WorkingMemory/train.py','WorkingMemory/sweep.py','WorkingMemory/evaluate.py','WorkingMemory/stimuli.py',
           'PreAttentiveVision/models.py','PreAttentiveVision/hybrid_models.py','PreAttentiveVision/decoder.py','PreAttentiveVision/TemporalIntegration/accumulators.py',
           'PreAttentiveVision/neuroscience_stimuli.py','PreAttentiveVision/natural_stimuli.py','PreAttentiveVision/evaluate.py','PreAttentiveVision/evaluate_multitask.py','PreAttentiveVision/train.py']
    hashes={name:sha(ROOT/name) for name in names}
    for name in names:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(ROOT/'PreAttentiveVision/data/bsds500/manifest.json',root/'dataset_manifest.json')
    train_conditions={**{k:v for k,v in BRIDGE_CONDITIONS.items() if v['protocol']=='anchor'},**TRAIN_CONDITIONS}
    all_conditions={**BRIDGE_CONDITIONS,**TRAIN_CONDITIONS}
    base=dict(seed=48301,train_seed=483001,batch_size=4,lr=.0003,weight_decay=.0001,grad_clip=5.,
        task_classes=TASK_CLASSES,activation_checkpoint=True,fp32=True,amp=False,optimizer='fresh_Adam',
        bridge_steps=140,bridge_conditions=BRIDGE_CONDITIONS,train_conditions=train_conditions,
        generator_version=VERSION,training_protocol='wm_bridge_joint_v1',
        checkpoint_selection='Max mean cell AUC, averaged equally within family/protocol groups; ties earlier; worst BA descriptive',
        trainable='All learned encoder/projection/core-output/readout weights; fixed quadrature and.25/.75 retention')
    # One bounded scheduling check: it catches omitted protocol/family exposure.
    from collections import Counter
    allocation=Counter((stage,family,base['bridge_conditions'][cond]['protocol'] if stage=='bridge' else train_conditions[cond]['protocol'])
                       for step in range(1400) for family,cond,stage in [schedule(step,base)])
    assert all(allocation['bridge',f,p]==allocation['bridge',next(iter(TASK_CLASSES)),p] for f in TASK_CLASSES for p in ('anchor','integration','recall'))
    assert all([allocation['joint',f,p] for p in ('anchor','integration','recall')]==[18,81,81] for f in TASK_CLASSES)
    write(root/'schedule_check.json',dict(status='passed',counts={'/'.join(k):v for k,v in allocation.items()}))
    now=time.time();deadline=now+14400
    ledger=dict(started_unix=now,deadline_unix=deadline,hard_seconds=14400,status='profiling',active=None,events=[],
        allowance_origin='New user-authorized14400s WM allowance; separate from both earlier ledgers',source_hashes=hashes)
    aggregate=dict(status='profiling',run_root=str(root),profiles=[],config=dict(parent_checkpoint=str(PARENT),parent_sha256=parent_hash,
        allowance_seconds=14400,task_classes=TASK_CLASSES),training={},validation=[],test=None)
    def publish():write(root/'aggregate.json',aggregate);write(HERE/'results.json',aggregate)
    write(root/'budget.json',ledger);publish()

    def launch(kind,cfg,out,allow_failure=False,**kw):
        number=len(ledger['events']);result=root/f'job_{number:03d}_result.json';job_path=root/f'job_{number:03d}.json';log_path=root/f'worker_{number:03d}.log'
        write(job_path,dict(kind=kind,config=cfg,out=str(out),deadline=deadline,result=str(result),source_hashes=hashes,
            parent_checkpoint=str(PARENT),parent_sha256=parent_hash,**kw))
        tick=time.time()
        with log_path.open('w') as log:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'train.py'),str(job_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ledger['active']=dict(pid=p.pid,kind=kind,started_unix=tick,log=str(log_path));write(root/'budget.json',ledger)
            try:code=p.wait(timeout=max(.01,deadline-time.time()))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        elapsed=time.time()-tick;ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=elapsed));ledger['active']=None;write(root/'budget.json',ledger)
        if code:
            if allow_failure:return dict(status='failed',returncode=code,log=str(log_path)),elapsed
            raise RuntimeError(f'{kind} exit{code}; see {log_path}')
        return read(result),elapsed

    try:
        # Accounted short longest-trained-condition profiles, no production reuse.
        for batch in (2,4):
            cfg=dict(base,batch_size=batch,purpose='separate_profile')
            p,wall=launch('profile',cfg,root/f'profile_batch{batch}',allow_failure=True,
                profile_cells=[('motion_direction',TRAIN_CONDITIONS['integrate_L32']),('natural_spectrum',TRAIN_CONDITIONS['integrate_L32'])])
            p.update(batch_size=batch,wall_seconds=wall);aggregate['profiles'].append(p);publish()
            if p['status']=='failed' and batch==2:raise RuntimeError('Batch2 longest-sequence profile failed')
        p4=aggregate['profiles'][-1]
        batch=4 if p4['status']=='completed' and p4['peak_allocated_bytes']<4*2**30 else 2
        cfg=dict(base,batch_size=batch,purpose='separate_profile')
        representative=[(family,condition) for family in TASK_CLASSES for condition in
            (BRIDGE_CONDITIONS['anchor'],TRAIN_CONDITIONS['integrate_L8'],TRAIN_CONDITIONS['integrate_L32'],TRAIN_CONDITIONS['recall_D0'],TRAIN_CONDITIONS['recall_N4'])]
        profile,wall=launch('profile',cfg,root/'profile_recipe',profile_cells=representative)
        profile['wall_seconds']=wall;aggregate['profiles'].append(profile);publish()
        measurements=profile['measurements']
        def price(family,condition,kind):
            group=[m for m in measurements if m['family']==family and m['condition']['protocol']==condition['protocol']]
            group=sorted(group,key=lambda m:m['frames']);frames=condition_frames(condition,family)
            key='step_seconds' if kind=='train' else 'eval_seconds'
            if len(group)>1 and group[-1]['frames']!=group[0]['frames']:
                slope=max(0,(group[-1][key]-group[0][key])/(group[-1]['frames']-group[0]['frames']))
                seconds=max(.5*group[0][key],group[0][key]+slope*(frames-group[0]['frames']))
            else:seconds=group[0][key]*max(1,frames/group[0]['frames'])
            return seconds/batch
        def mean_price(conditions,kind):return sum(price(f,c,kind) for c in conditions.values() for f in TASK_CLASSES)/(len(conditions)*7)
        bridge_price=mean_price(BRIDGE_CONDITIONS,'train')
        joint_price=sum(weight*mean_price({k:v for k,v in train_conditions.items() if v['protocol']==p},'train') for p,weight in [('anchor',.1),('integration',.45),('recall',.45)])
        val_n=32;test_n=128
        evaluation_seconds=(3*val_n+test_n)*sum(price(f,c,'eval') for c in all_conditions.values() for f in TASK_CLASSES)
        evaluation_seconds+=val_n*sum(price(f,c,'eval') for c in BRIDGE_CONDITIONS.values() for f in TASK_CLASSES)
        remaining=deadline-time.time();reserve=1.35*evaluation_seconds+420
        max_units=int(40000/(batch*1400))
        units=next((u for u in range(max_units,0,-1) if 1.35*(batch*1400*u)*(.1*bridge_price+.9*joint_price)+reserve<remaining*.96),None)
        if units is None:raise RuntimeError('Profiled minimum allocation cannot fit training plus all required evaluations')
        total_steps=1400*units;bridge_steps=140*units;targets=[bridge_steps,560*units,980*units,total_steps]
        cfg=dict(base,batch_size=batch,bridge_steps=bridge_steps,purpose='pretrained_sequence_learning')
        aggregate['config'].update(recipe=cfg,total_steps=total_steps,bridge_steps=bridge_steps,targets=targets,
            total_episodes=total_steps*batch,bridge_episodes=bridge_steps*batch,joint_episodes=(total_steps-bridge_steps)*batch,
            validation_episodes_per_cell=val_n,test_episodes_per_cell=test_n,validation_conditions=all_conditions,
            validation_seed=983001,test_seed=1083001,estimated_training_seconds=1.35*(total_steps*batch)*(.1*bridge_price+.9*joint_price),
            estimated_evaluation_seconds=1.35*evaluation_seconds,report_checkpoint_reserve_seconds=420,
            optional_extrapolation='Not included; prioritize every declared trained-condition final cell')
        write(root/'fixed_config.json',aggregate['config']);print('FIXED_ALLOCATION '+json.dumps(aggregate['config']),flush=True)
        aggregate['status']='training';ledger['status']='training';write(root/'budget.json',ledger);publish()
        checkpoint=None;best=None;training_wall=0
        for target in targets:
            trained,wall=launch('train',cfg,root/'opponent',target=target,checkpoint=checkpoint)
            checkpoint=trained['checkpoint'];training_wall+=wall;aggregate['training']=dict(trained,training_wall_seconds=training_wall);publish()
            if trained['step']!=target:raise RuntimeError('Deadline prevented planned training endpoint')
            conditions=BRIDGE_CONDITIONS if target==bridge_steps else all_conditions
            val,_=launch('eval',cfg,root/f'validation_{target:06d}',checkpoint=checkpoint,conditions=conditions,split='val',eval_seed=983001,n_per_cell=val_n,resamples=0)
            aggregate['validation'].append(dict(stage='bridge' if target==bridge_steps else 'joint',**val))
            if target!=bridge_steps and (best is None or val['selection_mean_auc']>best['selection_mean_auc']):best=val;aggregate['selected_checkpoint']=checkpoint;aggregate['selected_step']=target
            publish()
        aggregate['status']='final_test';ledger['status']='final_test';write(root/'budget.json',ledger);publish()
        test,_=launch('eval',cfg,root/'test',checkpoint=aggregate['selected_checkpoint'],conditions=all_conditions,
            split='test',eval_seed=1083001,n_per_cell=test_n,resamples=500)
        aggregate['test']=test;aggregate['status']='completed'
    except BaseException as error:
        aggregate.update(status='budget_stopped' if time.time()>=deadline-10 else 'failed',error=repr(error));raise
    finally:
        aggregate['wall_seconds']=time.time()-now;publish();ledger['status']=aggregate['status'];write(root/'budget.json',ledger)
        write(root/'exit.json',dict(status=aggregate['status'],wall_seconds=time.time()-now,deadline_unix=deadline,error=aggregate.get('error')))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: sweep.py NEW_RUN_ROOT')
    run(sys.argv[1])
