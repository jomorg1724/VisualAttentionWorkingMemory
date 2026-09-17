"""One cloud GPU, two sequential arms, original absolute pod deadline."""
import os,sys,json,time,subprocess,shutil,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,sha
from WorkingMemory.TrainingExposure.sweep import write
from WorkingMemory.UnbiasedAttention.protocol import recipe,assess,VAL_SEED,TEST_SEED,SPATIAL_VAL_SEED,SPATIAL_TEST_SEED

def run(manifest_path):
    import torch,numpy,scipy,PIL
    manifest=read(manifest_path);root=ROOT/'biased_results';root.mkdir(exist_ok=True)
    deadline=manifest['deadline_unix'];parent=ROOT/'portable/parent_checkpoint_008400.pt'
    for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,name
    hashes=manifest['source_hashes'];versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pillow=PIL.__version__,platform=platform.platform())
    assert (versions['torch'],versions['numpy'],versions['scipy'],versions['pillow'])==('1.13.1+cu117','1.23.1','1.8.1','9.1.1'),versions
    budget_path=root/'budget.json';agg_path=root/'aggregate.json'
    if budget_path.exists():
        budget=read(budget_path);agg=read(agg_path);assert budget['deadline_unix']==deadline and budget['source_hashes']==hashes
        if agg['status']=='completed':return
    else:
        budget=dict(started_unix=time.time(),deadline_unix=deadline,source_hashes=hashes,events=[],active=None)
        agg=dict(status='profiling',platform=versions,runs={},parent=str(parent),parent_sha256=sha(parent),interpretation='Five-task training with original learned source/locality priors, from intact attention8400. Cancelled bias-free experiment is preserved separately; no old-task rerun.')
    def publish():write(budget_path,budget);write(agg_path,agg)
    def launch(arm,cfg,tag,kind,**kw):
        jobfile=root/(arm+'_'+tag+'_job.json');result=root/(arm+'_'+tag+'_result.json');log=root/(arm+'_'+tag+'.log');out=root/arm/tag
        if result.exists() and read(result).get('status')=='completed':return read(result)
        if kind=='train':
            out=root/arm/'training'
            if (out/'checkpoint_index.jsonl').exists():
                candidates=[json.loads(line) for line in (out/'checkpoint_index.jsonl').read_text().splitlines()]
                candidates=[r for r in candidates if r['step']<=kw['target'] and sha(out/r['file'])==r['sha256']]
                if candidates:kw['checkpoint']=str(out/max(candidates,key=lambda r:r['step'])['file'])
        job=dict(kind=kind,arm=arm,config=cfg,out=str(out),result=str(result),deadline=deadline-240,parent=str(parent),parent_sha256=sha(parent),source_hashes=hashes,**kw);write(jobfile,job)
        with log.open('a') as f:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'worker.py'),str(jobfile)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            budget['active']=dict(pid=p.pid,arm=arm,tag=tag,log=str(log));publish();tick=time.time()
            try:code=p.wait(timeout=max(.01,deadline-time.time()-240))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        budget['events'].append(dict(arm=arm,tag=tag,pid=p.pid,returncode=code,seconds=time.time()-tick));budget['active']=None;publish()
        if code:raise RuntimeError(f'{arm}/{tag} exited {code}; see {log}')
        value=read(result)
        if value['status']!='completed':raise RuntimeError('Incomplete bounded worker '+tag)
        return value
    publish()
    try:
        configs={'spatial_biased':recipe('spatial_unbiased')}
        if 'pinned_updates' not in agg:
            new=configs['spatial_biased']
            from WorkingMemory.SpatialTaskBattery.stimuli import frame_count
            new_names=[]
            for task,names in new['train_names'].items():
                ordered=sorted(names,key=lambda n:frame_count(task,new['cells'][n]['condition']));new_names.extend(dict.fromkeys([ordered[0],ordered[-1]]))
            profiles={'spatial_biased':launch('spatial_biased',new,'profile','profile',profile_names=new_names)};agg['profiles']=profiles
            oldtrain=0.;oldeval=0.
            npf={r['name']:r for r in profiles['spatial_biased']['rows']};ncost={}
            for task,names in new['train_names'].items():
                measured=[n for n in new_names if new['cells'][n]['task']==task];lo,hi=measured[0],measured[-1];fl=frame_count(task,new['cells'][lo]['condition']);fh=frame_count(task,new['cells'][hi]['condition'])
                for n in names:
                    f=0 if fh==fl else (frame_count(task,new['cells'][n]['condition'])-fl)/(fh-fl);ncost[n]={k:(1-f)*npf[lo][k]+f*npf[hi][k] for k in ('seconds','eval_seconds')}
            newstep=1.25*sum(sum(ncost[n]['seconds'] for n in names)/len(names) for names in new['train_names'].values())
            neweval=1.4*sum(v['eval_seconds']*(5*(100 if new['cells'][n]['task']=='krauzlis_cued_motion' else 64)+2*(400 if new['cells'][n]['task']=='krauzlis_cued_motion' else 256))/8 for n,v in ncost.items())
            available=deadline-time.time()-240-oldtrain-oldeval-neweval-600
            updates=min(4000,int(available/newstep)//200*200)
            if updates<200:raise RuntimeError('Even200 new-task updates cannot fit under remaining original deadline')
            agg['pinned_updates']=dict(spatial_biased=updates);agg['profile_costs']=dict(old_training=oldtrain,old_evaluation=oldeval,new_update=newstep,new_evaluation=neweval,spatial_target_reduced=updates<4000);publish()
        for arm,cfg in configs.items():
            nupdates=agg['pinned_updates'][arm];cfg['updates']=nupdates;spatial=cfg['battery']=='spatial';record=agg['runs'].setdefault(arm,dict(config=cfg,validation=[],updates=nupdates,added_episodes=nupdates*(40 if spatial else 8)));agg['status']='training';publish()
            if record.get('status')=='completed':continue
            targets=[8400+nupdates*i//5 for i in range(1,6)];vseed=SPATIAL_VAL_SEED if spatial else VAL_SEED;tseed=SPATIAL_TEST_SEED if spatial else TEST_SEED
            if spatial:baseline=None;best=None
            else:
                baseline=launch(arm,cfg,'biased_parent_validation','eval',biased_reference=True,split='val',eval_seed=vseed,n=128,heldout_locations=False);record['parent_validation']=baseline;best=assess(baseline,baseline);record.update(selected_checkpoint=str(parent),selected_step=8400,selected_is_parent=True)
            cp=None;record['validation']=[]
            for target in targets:
                train=launch(arm,cfg,f'train_{target}','train',checkpoint=cp,target=target);cp=train['checkpoint'];record['training']=train;publish()
                val=launch(arm,cfg,f'validation_{target}','eval',checkpoint=cp,split='val',eval_seed=vseed,n=64 if spatial else 128,heldout_locations=False)
                decision=dict(eligible=True,rank=val['rank']) if spatial else assess(val,baseline);val['assessment']=decision;record['validation'].append(val)
                if decision['eligible'] and (best is None or tuple(decision['rank'])>tuple(best['rank'])):best=decision;record.update(selected_checkpoint=cp,selected_step=target,selected_is_parent=False,selected_assessment=decision)
                publish()
            record['terminal_checkpoint']=cp
            if not spatial:record['parent_test']=launch(arm,cfg,'biased_parent_test','eval',biased_reference=True,split='test',eval_seed=tseed,n=512,heldout_locations=True)
            record['selected_test']=record['parent_test'] if record.get('selected_is_parent') else launch(arm,cfg,'selected_test','eval',checkpoint=record['selected_checkpoint'],split='test',eval_seed=tseed,n=256 if spatial else 512,heldout_locations=not spatial)
            record['terminal_test']=record['selected_test'] if record['selected_step']==8400+nupdates else launch(arm,cfg,'terminal_test','eval',checkpoint=cp,split='test',eval_seed=tseed,n=256 if spatial else 512,heldout_locations=not spatial)
            record['status']='completed';publish()
        agg['status']='completed'
    except BaseException as e:agg.update(status='failed',error=repr(e));raise
    finally:
        publish();write(root/'exit.json',dict(status=agg['status'],error=agg.get('error'),deadline_unix=deadline,wall_seconds=time.time()-budget['started_unix']))
        write(root/'artifact_index.json',{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_index.json'})

if __name__=='__main__':run(sys.argv[1])
