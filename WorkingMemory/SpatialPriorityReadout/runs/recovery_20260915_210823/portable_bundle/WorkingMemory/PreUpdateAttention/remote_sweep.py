"""One attention arm, fixed 4000-update exposure, finite paid wall deadline."""
import sys,os,time,json,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha

def protocol():
    cells={}
    for family,label in [('orientation_single','single'),('orientation_binding','binding'),('motion_direction','motion')]:
        for d in ((0,24) if label=='motion' else (0,4,12,24)):cells[f'{label}_D{d}']=dict(family=family,condition=dict(delay=d,spacing='mixed'))
    return cells,[name for name in cells for _ in range(4 if name.startswith('motion') else 9)]

def recipe():
    cells,cycle=protocol()
    return dict(cells=cells,cycle=cycle,batch_size=8,train_seed=32973001,scheduler_seed=33973001,model_seed=41973001,new_lr=.0003,parent_lr=.00003,adam_eps=1e-10,weight_decay=.0001,clip=1.,activation_checkpoint=True)

def run():
    manifest=read(ROOT/'portable/manifest.json');root=ROOT/'remote_results';root.mkdir(exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Existing run cannot renew budget')
    deadline=manifest['deadline_unix'];start=time.time();cfg=recipe();arm='preupdate_attention'
    if start>=deadline-300:raise TimeoutError('Insufficient remaining cloud time')
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source/package changed '+name)
    import torch,numpy,scipy,PIL
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pillow=PIL.__version__,platform=platform.platform())
    if (versions['torch'],versions['numpy'],versions['scipy'],versions['pillow'])!=('1.13.1+cu117','1.23.1','1.8.1','9.1.1'):raise RuntimeError('Unpinned runtime '+str(versions))
    versions['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip();write(root/'platform.json',versions)
    ledger=dict(started_unix=start,pod_created_unix=manifest['created_unix'],deadline_unix=deadline,status='profiling',active=None,events=[],source_hashes=manifest['source_hashes'])
    record=dict(validation=[])
    agg=dict(status='profiling',run_root=str(root),platform=versions,config=dict(recipe=cfg,parent='portable/parent_checkpoint_004400.pt',parent_sha256=manifest['parent_sha256'],parent_step=4400,updates_per_arm=4000,episodes_per_arm=32000,targets=[5400,6400,7400,8400],val_seed=45973001,test_seed=46973001,validation_n_per_cell=128,test_n_per_cell=512,selection='equal mean eight primary validation AUC; ties earlier',profile_exposure_not_continued=True),profiles={},runs={arm:record})
    def publish():write(root/'budget.json',ledger);write(root/'aggregate.json',agg)
    def launch(kind,out,**kw):
        i=len(ledger['events']);job=root/f'job_{i:03d}.json';result=root/f'job_{i:03d}_result.json';log=root/f'worker_{i:03d}.log'
        write(job,dict(kind=kind,arm=arm,config=cfg,out=str(out),result=str(result),deadline=deadline-180,parent=str(ROOT/'portable/parent_checkpoint_004400.pt'),parent_sha256=manifest['parent_sha256'],source_hashes=manifest['source_hashes'],**kw));tick=time.time()
        with log.open('w') as f:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'train.py'),str(job)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);ledger['active']=dict(pid=p.pid,arm=arm,kind=kind,log=str(log));publish()
            try:code=p.wait(timeout=max(.01,deadline-time.time()-180))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=time.time()-tick));ledger['active']=None;publish()
        if code:raise RuntimeError(f'{kind} failed exit {code}: {log}')
        return read(result)
    publish()
    try:
        prof=launch('profile',root/'profile',profile_names=['binding_D24','single_D24','motion_D24','binding_D0','single_D0','motion_D0']);agg['profiles'][arm]=prof
        rows={r['name']:r for r in prof['rows']};costs={}
        for name in cfg['cells']:
            family,d=name.split('_D');fraction=int(d)/24
            costs[name]={k:(1-fraction)*rows[family+'_D0'][k]+fraction*rows[family+'_D24'][k] for k in ('seconds','eval_seconds')}
        perupdate=sum(costs[n]['seconds'] for n in cfg['cycle'])/80
        evalcost=(sum(c['eval_seconds'] for c in costs.values())*(4*128+512)+sum(costs[n]['eval_seconds'] for n in costs if n.startswith('binding'))*512)/8
        estimate=1.3*4000*perupdate+1.6*evalcost+300
        agg['config'].update(training_estimate_seconds=1.3*4000*perupdate,evaluation_reporting_reserve_seconds=1.6*evalcost+300,remaining_seconds_after_profile=deadline-time.time(),total_estimate_seconds=estimate)
        write(root/'fixed_config.json',agg['config']);publish()
        if estimate>deadline-time.time()-180:raise RuntimeError('Fixed4000 exposure does not fit: notify coordinator, no automatic reduced run or cap extension')
        review=read(HERE/'review_ready.json')
        if review.get('status')!='passed':raise RuntimeError('Single independent review not passed')
        for name,digest in review['source_hashes'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Reviewed source mismatch '+name)
        agg['status']=ledger['status']='training';publish();cp=None;best=None
        for target in (5400,6400,7400,8400):
            result=launch('train',root/arm,checkpoint=cp,target=target);record['training']=result;cp=result['checkpoint'];publish()
            if result['step']!=target:raise RuntimeError('Fixed exposure interrupted')
            val=launch('eval',root/f'validation_{target:06d}',checkpoint=cp,split='val',eval_seed=45973001,n=128,heldout_locations=False);record['validation'].append(val)
            if best is None or val['selection_mean_auc']>best['selection_mean_auc']:best=val;record.update(selected_checkpoint=cp,selected_step=target)
            publish()
        record['test']=launch('eval',root/'test',checkpoint=record['selected_checkpoint'],split='test',eval_seed=46973001,n=512,heldout_locations=True)
        record['status']=agg['status']=ledger['status']='completed'
    except BaseException as e:
        agg.update(status='failed',error=repr(e));ledger['status']=agg['status'];raise
    finally:
        agg['wall_seconds']=time.time()-start;publish()
        write(root/'exit.json',dict(status=agg['status'],error=agg.get('error'),wall_seconds=agg['wall_seconds'],deadline_unix=deadline))
        write(root/'artifact_index.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_index.json'})
if __name__=='__main__':run()
