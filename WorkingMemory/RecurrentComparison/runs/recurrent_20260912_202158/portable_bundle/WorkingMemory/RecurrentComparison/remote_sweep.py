"""One authorized RunPod EI worker at a time; fixed exposure and cloud cap."""
import os,sys,time,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,write,sha


def run():
    manifest=read(ROOT/'portable/manifest.json');config=manifest['fixed_config'];cfg=dict(config['recipe'],arm='ei_adaptive')
    root=ROOT/'remote_results';root.mkdir(exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Remote run already exists; allowance cannot be renewed')
    deadline=manifest['cloud_deadline_unix']
    if time.time()>=deadline:raise TimeoutError('Paid cloud deadline passed')
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Portable file mismatch '+name)
    import torch,numpy,scipy,PIL
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pillow=PIL.__version__,platform=platform.platform())
    if (versions['torch'],versions['numpy'],versions['scipy'],versions['pillow'])!=('1.13.1+cu117','1.23.1','1.8.1','9.1.1'):
        raise RuntimeError('Runtime versions differ from declared matching environment: '+str(versions))
    versions['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    write(root/'platform.json',versions)
    now=time.time();ledger=dict(status='training',started_unix=now,deadline_unix=deadline,pod_id=manifest['pod_id'],active=None,events=[],
        origin='User-directed migration of EI arm; fresh exact local initializer, same fixed40000episode protocol, earlier paid cloud cap')
    record=dict(status='training',training={},validation=[],test=None,reset_test=None)
    aggregate=dict(status='training',run_root=str(root),config=config,runs={'ei_adaptive':record},platform=versions,migration=manifest['migration'])
    def publish():write(root/'results.json',aggregate);write(root/'budget.json',ledger)
    publish()
    def launch(kind,out,**kw):
        n=len(ledger['events']);job_path=root/f'job_{n:03d}.json';result=root/f'job_{n:03d}_result.json';log_path=root/f'worker_{n:03d}.log'
        write(job_path,dict(kind=kind,config=cfg,out=str(out),deadline=deadline,result=str(result),source_hashes=manifest['source_hashes'],
            parent_checkpoint=str(ROOT/'portable/parent_checkpoint_006860.pt'),parent_sha256=config['parent_sha256'],
            initializer_sha256=manifest['files']['portable/fresh_ei_initialization.pt'],**kw))
        tick=time.time()
        with log_path.open('w') as log:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'remote_worker.py'),str(job_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ledger['active']=dict(pid=p.pid,kind=kind,arm='ei_adaptive',started_unix=tick,log=str(log_path));publish()
            try:code=p.wait(timeout=max(.01,deadline-time.time()))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        wall=time.time()-tick;ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=wall));ledger['active']=None;publish()
        if code:raise RuntimeError('Remote '+kind+' exit '+str(code))
        return read(result),wall
    try:
        cp=None;best=None;wall=0
        for target in config['targets']:
            trained,elapsed=launch('train',root/'ei_adaptive',target=target,checkpoint=cp)
            cp=trained['checkpoint'];wall+=elapsed;record['training']=dict(trained,training_wall_seconds=wall);publish()
            if trained['step']!=target:raise RuntimeError('Cloud deadline prevented fixed endpoint')
            val,_=launch('eval',root/f'validation_{target:06d}',checkpoint=cp,split='val',eval_seed=config['validation_seed'],n_per_cell=config['validation_n_per_cell'],resamples=0)
            record['validation'].append(val)
            if best is None or val['selection_mean_auc']>best['selection_mean_auc']:
                best=val;record.update(selected_checkpoint=cp,selected_step=target)
            publish()
        for reset in (False,True):
            name='reset_test' if reset else 'test'
            test,_=launch('eval',root/name,checkpoint=record['selected_checkpoint'],split='test',eval_seed=config['test_seed'],n_per_cell=config['test_n_per_cell'],resamples=500,reset_memory=reset)
            record[name]=test;publish()
        record['status']='completed';aggregate['status']='completed';ledger['status']='completed'
    except BaseException as e:
        aggregate.update(status='budget_stopped' if time.time()>=deadline-10 else 'failed',error=repr(e));ledger['status']=aggregate['status'];raise
    finally:
        aggregate['wall_seconds']=time.time()-now;publish()
        write(root/'exit.json',dict(status=aggregate['status'],wall_seconds=time.time()-now,deadline_unix=deadline,error=aggregate.get('error'),pod_id=manifest['pod_id']))
        # A final immutable file index supports verified local retrieval before
        # the explicitly authorized pod termination by the coordinating agent.
        index={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_index.json'}
        write(root/'artifact_index.json',index)

if __name__=='__main__':run()
