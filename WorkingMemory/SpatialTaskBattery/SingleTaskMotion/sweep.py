"""One local GPU worker under a new, nonrenewable 7200-second allowance."""
import os,sys,time,json,subprocess,datetime,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,sha
from WorkingMemory.TrainingExposure.sweep import write
VAL_SEED=71973001
TEST_SEED=72973001

def run():
    import torch
    cloud=read(ROOT/'WorkingMemory/SpatialTaskBattery/BiasedTraining/cloud_provisioning.json');cloudrun=Path(cloud['local_run']);parent=cloudrun/'incremental/biased_results/spatial_biased/training/checkpoint_010000.pt'
    transfers=read(cloudrun/'incremental_transfers.json');entry=next(r for r in transfers if r['file']=='spatial_biased/training/checkpoint_010000.pt');assert sha(parent)==entry['sha256'];cp=torch.load(parent,map_location='cpu');assert cp['step']==10000
    hashes=dict(cp['source_hashes'])
    for n,h in hashes.items():assert sha(ROOT/n)==h,n
    for name in ('worker.py','sweep.py','report.py'):hashes[str((HERE/name).relative_to(ROOT)).replace('\\','/')]=sha(HERE/name)
    root=HERE/'runs'/datetime.datetime.now().strftime('motion_%Y%m%d_%H%M%S');root.mkdir(parents=True)
    started=time.time();deadline=started+7200;budget=dict(status='profiling',started_unix=started,deadline_unix=deadline,events=[],active=None,source_hashes=hashes)
    aggregate=dict(status='profiling',run_root=str(root),parent=str(parent),parent_sha256=sha(parent),parent_step=10000,parent_motion_episodes=sum(v for k,v in cp['cell_counts'].items() if k.startswith('motion_duration_cued')),parent_config=cp['config'],val_seed=VAL_SEED,test_seed=TEST_SEED,validation_n=128,test_n=512,validation=[],platform=dict(python=platform.python_version(),torch=torch.__version__,platform=platform.platform()),selection='Maximize minimum delay BA, then mean delay AUC; earlier ties; parent is eligible fallback',interpretation='Task-focused acquisition continuation. Same motion episodes/update as cloud, but full task CE versus CE/5 plus other gradients and inherited Adam make this an optimization-policy change, not a causal interference test.')
    write(HERE/'local_run.json',dict(run=str(root),parent=str(parent),parent_sha256=sha(parent),deadline_unix=deadline));del cp
    def publish():write(root/'budget.json',budget);write(root/'aggregate.json',aggregate)
    def launch(tag,kind,**kw):
        job=root/(tag+'_job.json');result=root/(tag+'_result.json');log=root/(tag+'.log');out=root/('training' if kind=='train' else tag)
        write(job,dict(kind=kind,parent=str(parent),parent_sha256=aggregate['parent_sha256'],source_hashes=hashes,out=str(out),result=str(result),deadline=deadline-120,**kw))
        with log.open('w') as f:
            p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(HERE/'worker.py'),str(job)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));tick=time.time();budget['active']=dict(pid=p.pid,tag=tag,log=str(log));publish()
            try:code=p.wait(timeout=max(.01,deadline-time.time()-120))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        budget['events'].append(dict(pid=p.pid,tag=tag,returncode=code,seconds=time.time()-tick));budget['active']=None;publish()
        if code:raise RuntimeError(f'{tag} failed exit{code}: {log}')
        value=read(result)
        if value['status']!='completed':raise RuntimeError('Incomplete worker '+tag)
        return value
    publish()
    try:
        profile=launch('profile','profile');aggregate['profile']=profile;lo,hi=profile['rows'];train=[];ev=[]
        for d in (0,4,12,24):
            f=d/24;train.append((1-f)*lo['seconds']+f*hi['seconds']);ev.append((1-f)*lo['eval_seconds']+f*hi['eval_seconds'])
        reserve=1.5*sum(ev)*(6*128/8+3*512/8)+240;per_update=1.25*sum(train)/4;updates=min(4000,int((deadline-time.time()-reserve-120)/per_update)//200*200)
        if updates<200:raise RuntimeError('Insufficient allowance for minimal fixed exposure')
        aggregate.update(pinned_updates=updates,pinned_added_episodes=updates*8,targets=[10000+updates*i//5 for i in range(1,6)],training_estimate_seconds=updates*per_update,evaluation_reporting_reserve_seconds=reserve)
        write(root/'fixed_config.json',{k:v for k,v in aggregate.items() if k not in ('profile','validation')});publish()
        baseline=launch('parent_validation','eval',split='val',eval_seed=VAL_SEED,n=128);aggregate['parent_validation']=baseline;best=baseline['rank'];selected=None;aggregate.update(selected_step=10000,selected_is_parent=True)
        budget['status']=aggregate['status']='training';publish();checkpoint=None
        for target in aggregate['targets']:
            trained=launch(f'train_{target}','train',checkpoint=checkpoint,target=target);checkpoint=trained['checkpoint'];aggregate['training']=trained;publish()
            val=launch(f'validation_{target}','eval',checkpoint=checkpoint,split='val',eval_seed=VAL_SEED,n=128);aggregate['validation'].append(val)
            if tuple(val['rank'])>tuple(best):best=val['rank'];selected=checkpoint;aggregate.update(selected_step=target,selected_is_parent=False,selected_checkpoint=checkpoint)
            publish()
        aggregate['terminal_checkpoint']=checkpoint;budget['status']=aggregate['status']='evaluating';publish()
        aggregate['parent_test']=launch('parent_test','eval',split='test',eval_seed=TEST_SEED,n=512)
        aggregate['selected_test']=aggregate['parent_test'] if selected is None else launch('selected_test','eval',checkpoint=selected,split='test',eval_seed=TEST_SEED,n=512)
        aggregate['terminal_test']=aggregate['selected_test'] if aggregate['selected_step']==10000+updates else launch('terminal_test','eval',checkpoint=checkpoint,split='test',eval_seed=TEST_SEED,n=512)
        budget['status']=aggregate['status']='completed';publish()
        from WorkingMemory.SpatialTaskBattery.SingleTaskMotion.report import report
        report(root)
    except BaseException as e:budget['status']=aggregate['status']='failed';aggregate['error']=repr(e);raise
    finally:
        publish();write(root/'exit.json',dict(status=aggregate['status'],error=aggregate.get('error'),wall_seconds=time.time()-started,deadline_unix=deadline));write(root/'artifact_index.json',{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_index.json'})
if __name__=='__main__':run()
