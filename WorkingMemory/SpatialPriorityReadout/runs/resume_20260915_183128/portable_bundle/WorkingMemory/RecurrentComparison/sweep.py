"""Serial two-arm comparison under one new, immutable four-hour ledger."""
import os,sys,time,json,shutil,subprocess,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha

PARENT=ROOT/'WorkingMemory/runs/wm_20260912_181219/opponent/checkpoint_006860.pt'
ARMS=('lstm','ei_adaptive')


def protocol():
    from WorkingMemory.stimuli import BRIDGE_CONDITIONS,TRAIN_CONDITIONS
    cells={
        'motion_anchor':dict(family='motion_direction',condition=BRIDGE_CONDITIONS['anchor']),
        'orientation_anchor':dict(family='orientation',condition=BRIDGE_CONDITIONS['anchor']),
        'motion_L2':dict(family='motion_direction',condition=BRIDGE_CONDITIONS['bridge_integrate_visible']),
        'motion_L8':dict(family='motion_direction',condition=TRAIN_CONDITIONS['integrate_L8']),
        'orientation_recall_minimal':dict(family='orientation',condition=BRIDGE_CONDITIONS['bridge_recall']),
        'orientation_recall_D4':dict(family='orientation',condition=TRAIN_CONDITIONS['recall_N1']),
    }
    groups=(['anchor']+['motion','orientation']*4+['motion','anchor','orientation']+['motion','orientation']*4)*2
    options={'anchor':['motion_anchor','orientation_anchor'],'motion':['motion_L2','motion_L8'],
             'orientation':['orientation_recall_minimal','orientation_recall_D4']}
    counts=Counter();cycle=[]
    for group in groups:
        cycle.append(options[group][counts[group]%2]);counts[group]+=1
    assert Counter(cycle)==Counter(motion_anchor=2,orientation_anchor=2,motion_L2=9,motion_L8=9,orientation_recall_minimal=9,orientation_recall_D4=9)
    return cells,cycle


def run(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Existing allowance cannot be renewed')
    cells,cycle=protocol()
    parent_hash=sha(PARENT)
    if not any(x['file']==PARENT.name and x['sha256']==parent_hash for x in map(json.loads,(PARENT.parent/'checkpoint_index.jsonl').read_text().splitlines())):raise RuntimeError('Parent identity failed')
    names=['WorkingMemory/RecurrentComparison/'+n for n in ('model.py','train.py','sweep.py','evaluate.py')]
    names+=list(read(ROOT/'WorkingMemory/runs/wm_20260912_181219/budget.json')['source_hashes'])
    hashes={name:sha(ROOT/name) for name in names}
    for name in names:
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(HERE/'focused_check.json',root/'focused_check.json')
    shutil.copy2(ROOT/'PreAttentiveVision/data/bsds500/manifest.json',root/'inherited_dataset_manifest.json')
    base=dict(seed=59301,common_seed=59311,core_seed=59312,train_seed=593001,batch_size=8,
        new_lr=.0003,parent_lr=.00003,weight_decay=.0001,grad_clip=1.,adam_eps=1e-10,activation_checkpoint=True,
        task_classes={'motion_direction':4,'orientation':2},cells=cells,cycle=cycle,
        fp32=True,amp=False,optimizer='fresh_Adam_parameter_groups',purpose='focused_recurrent_learning',
        training_protocol='focused_six_cell_v1',generator_version='wm_visual_sequences_v1',
        checkpoint_selection='Maximum arithmetic mean six-cell OVR-AUC; exact ties earlier')
    write(root/'schedule_check.json',dict(status='passed',cycle=cycle,updates_per_cycle=dict(Counter(cycle))))
    # First actual GPU worker starts immediately after this ledger is created.
    now=time.time();deadline=now+14400
    ledger=dict(started_unix=now,deadline_unix=deadline,hard_seconds=14400,status='profiling',active=None,events=[],source_hashes=hashes,
        allowance_origin='New user-authorized 14400-second two-arm recurrent comparison; prior ledgers untouched')
    aggregate=dict(status='profiling',run_root=str(root),profiles=[],config=dict(parent_checkpoint=str(PARENT),parent_sha256=parent_hash,
        parent_step=6860,allowance_seconds=14400,cells=cells,arms=list(ARMS)),runs={})
    def publish():write(root/'aggregate.json',aggregate);write(HERE/'results.json',aggregate)
    write(root/'budget.json',ledger);publish()

    def launch(kind,cfg,out,allow_failure=False,**kw):
        number=len(ledger['events']);result=root/f'job_{number:03d}_result.json';job_path=root/f'job_{number:03d}.json';log_path=root/f'worker_{number:03d}.log'
        write(job_path,dict(kind=kind,config=cfg,out=str(out),deadline=deadline,result=str(result),source_hashes=hashes,
            parent_checkpoint=str(PARENT),parent_sha256=parent_hash,**kw))
        tick=time.time()
        with log_path.open('w') as log:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'train.py'),str(job_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ledger['active']=dict(pid=p.pid,kind=kind,arm=cfg['arm'],started_unix=tick,log=str(log_path));write(root/'budget.json',ledger)
            try:code=p.wait(timeout=max(.01,deadline-time.time()))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        elapsed=time.time()-tick;ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=elapsed));ledger['active']=None;write(root/'budget.json',ledger)
        if code:
            if allow_failure:return dict(status='failed',returncode=code,log=str(log_path)),elapsed
            raise RuntimeError(f'{cfg["arm"]}/{kind} exit {code}; see {log_path}')
        return read(result),elapsed

    try:
        profile_cells=[(cells[n]['family'],cells[n]['condition']) for n in ['motion_L8','orientation_recall_D4','motion_L2','orientation_recall_minimal','motion_anchor','orientation_anchor']]
        for batch in (4,8):
            for arm in ARMS:
                cfg=dict(base,arm=arm,batch_size=batch,purpose='separate_accounted_profile')
                p,wall=launch('profile',cfg,root/f'profile_{arm}_batch{batch}',allow_failure=True,profile_cells=profile_cells)
                p.update(arm=arm,batch_size=batch,wall_seconds=wall);aggregate['profiles'].append(p);publish()
                if p['status']!='completed' and batch==4:raise RuntimeError('Required batch4 profile failed')
        p8=[p for p in aggregate['profiles'] if p['batch_size']==8]
        batch=8 if all(p['status']=='completed' and p['peak_allocated_bytes']<4*2**30 for p in p8) else 4
        profiles={p['arm']:p for p in aggregate['profiles'] if p['batch_size']==batch}
        val_n,test_n=128,512
        def costs(arm):
            ms=profiles[arm]['measurements'];lookup={(m['family'],json.dumps(m['condition'],sort_keys=True)):m for m in ms}
            prices={n:lookup[(v['family'],json.dumps(v['condition'],sort_keys=True))] for n,v in cells.items()}
            train=sum(prices[n]['step_seconds'] for n in cycle)/len(cycle)
            eval_time=sum(m['eval_seconds'] for m in prices.values())/batch
            return train,eval_time
        reserve=480+1.3*sum(costs(arm)[1]*(4*val_n+2*test_n) for arm in ARMS)
        max_steps=(40000//batch//40)*40
        steps=next((n for n in range(max_steps,0,-40) if 1.3*n*sum(costs(arm)[0] for arm in ARMS)+reserve<(deadline-time.time())*.96),None)
        if not steps:raise RuntimeError('No positive equal exposure fits both models and reserved evaluation')
        targets=sorted(set([max(1,round(steps*f)) for f in (.25,.5,.75,1.)]))
        cfg=dict(base,batch_size=batch)
        aggregate['config'].update(recipe=cfg,total_steps=steps,episodes_per_arm=steps*batch,targets=targets,
            validation_n_per_cell=val_n,test_n_per_cell=test_n,validation_seed=1193001,test_seed=1293001,
            estimated_training_seconds=1.3*steps*sum(costs(arm)[0] for arm in ARMS),reserved_evaluation_report_seconds=reserve,
            initial_state_notes=dict(lstm='Forget bias nominal tau4..128, gate LN before bias, input bias-2',
                ei_adaptive='Balanced initial incoming totals E=.6/I=.6; tau_r2..8 bounded1..32; tau_a16..64 bounded4..128; adaptation .05 bounded0.. .5'),
            test_intervention='Reset added memory before every frame; opponent traces retain their history')
        write(root/'fixed_config.json',aggregate['config']);print('FIXED_ALLOCATION '+json.dumps(aggregate['config']),flush=True)
        # Researcher records the measured allocation and sends its required
        # launch notice. This local handoff never renews the absolute deadline.
        while not (root/'production_launch.json').exists():
            if time.time()>=deadline-10:raise TimeoutError('Deadline during measured-allocation handoff')
            time.sleep(.25)
        aggregate['status']='training';ledger['status']='training';write(root/'budget.json',ledger);publish()
        for arm in ARMS:
            armcfg=dict(cfg,arm=arm);record=dict(status='training',training={},validation=[],test=None,reset_test=None);aggregate['runs'][arm]=record;publish()
            cp=None;best=None;training_wall=0
            for target in targets:
                trained,wall=launch('train',armcfg,root/arm,target=target,checkpoint=cp)
                cp=trained['checkpoint'];training_wall+=wall;record['training']=dict(trained,training_wall_seconds=training_wall);publish()
                if trained['step']!=target:raise RuntimeError('Deadline prevented fixed endpoint')
                val,_=launch('eval',armcfg,root/f'{arm}_validation_{target:06d}',checkpoint=cp,split='val',eval_seed=1193001,n_per_cell=val_n,resamples=0)
                record['validation'].append(val)
                if best is None or val['selection_mean_auc']>best['selection_mean_auc']:
                    best=val;record.update(selected_checkpoint=cp,selected_step=target)
                publish()
            record['status']='training_completed';publish()
        # Reserve held-out inference for both completed arms before analysis.
        aggregate['status']='final_test';ledger['status']='final_test';write(root/'budget.json',ledger);publish()
        for arm in ARMS:
            record=aggregate['runs'][arm];armcfg=dict(cfg,arm=arm)
            for reset in (False,True):
                out=root/(arm+('_reset_test' if reset else '_test'))
                test,_=launch('eval',armcfg,out,checkpoint=record['selected_checkpoint'],split='test',eval_seed=1293001,n_per_cell=test_n,resamples=500,reset_memory=reset)
                record['reset_test' if reset else 'test']=test;publish()
            record['status']='completed';publish()
        aggregate['status']='completed'
    except BaseException as error:
        aggregate.update(status='budget_stopped' if time.time()>=deadline-10 else 'failed',error=repr(error));raise
    finally:
        aggregate['wall_seconds']=time.time()-now;publish();ledger['status']=aggregate['status'];write(root/'budget.json',ledger)
        write(root/'exit.json',dict(status=aggregate['status'],wall_seconds=time.time()-now,deadline_unix=deadline,error=aggregate.get('error')))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: sweep.py NEW_RUN_ROOT')
    run(sys.argv[1])
