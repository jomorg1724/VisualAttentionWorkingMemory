"""One bounded exposure arm, portable local/cloud worker and shared selection."""
import sys,os,time,json,subprocess,shutil,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from PreAttentiveVision.train import read,sha
from WorkingMemory.TrainingExposure.protocol import recipe,assess,VAL_SEED,TEST_SEED,TARGETS
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.sweep.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(40):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)
def parent_path():return Path(read(ROOT/'WorkingMemory/PreUpdateAttention/retrieval_receipt.json')['results'])/'preupdate_attention/checkpoint_008400.pt'
def source_hashes(parent):
    import torch
    cp=torch.load(parent,map_location='cpu')
    names=list(cp['source_hashes'])+['WorkingMemory/TrainingExposure/'+n for n in ('model.py','protocol.py','worker.py','sweep.py')]
    return {n:sha(ROOT/n) for n in names}
def run(arm,out,parent=None,external_manifest=None):
    import torch,numpy,scipy,PIL
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'budget.json').exists():raise RuntimeError('Existing allowance cannot renew')
    parent=Path(parent).resolve() if parent else parent_path();cfg=recipe(arm)
    hashes=external_manifest['source_hashes'] if external_manifest else source_hashes(parent)
    review=read(HERE/'review_ready.json');assert review['status']=='passed'
    for name,digest in review['source_hashes'].items():assert sha(ROOT/name)==digest,name
    for name,digest in hashes.items():
        assert sha(ROOT/name)==digest,name
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pillow=PIL.__version__,platform=platform.platform())
    assert (versions['torch'],versions['numpy'],versions['scipy'],versions['pillow'])==('1.13.1+cu117','1.23.1','1.8.1','9.1.1'),versions
    started=time.time();deadline=external_manifest['deadline_unix'] if external_manifest else started+14400
    if deadline-started<600:raise TimeoutError('Insufficient original allowance')
    ledger=dict(status='profiling',started_unix=started,deadline_unix=deadline,external_placement=external_manifest,active=None,events=[],source_hashes=hashes)
    record=dict(validation=[]);agg=dict(status='profiling',run_root=str(root),platform=versions,config=dict(arm=arm,recipe=cfg,parent=str(parent),parent_sha256=sha(parent),parent_step=8400,updates=4000,episodes=32000,targets=TARGETS,val_seed=VAL_SEED,test_seed=TEST_SEED,validation_n_per_cell=128,test_n_per_cell=512,selection='Allten trainedcell BA loss <=2pp vs paired parent; then best minmotionBA, meanmotionAUC, earlier; include parentfallback',stream_matching='Exact parent family-local stream state; shared per-family evidence prefixes, not necessarily same per-delay assignment'),runs={arm:record})
    def publish():write(root/'budget.json',ledger);write(root/'aggregate.json',agg)
    def launch(kind,out,**kw):
        i=len(ledger['events']);job=root/f'job_{i:03d}.json';result=root/f'job_{i:03d}_result.json';log=root/f'worker_{i:03d}.log'
        write(job,dict(kind=kind,arm=arm,config=cfg,out=str(out),result=str(result),deadline=deadline-180,parent=str(parent),parent_sha256=agg['config']['parent_sha256'],source_hashes=hashes,**kw));tick=time.time()
        with log.open('w') as f:
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'worker.py'),str(job)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);ledger['active']=dict(pid=p.pid,arm=arm,kind=kind,log=str(log));publish()
            try:code=p.wait(timeout=max(.01,deadline-time.time()-180))
            except subprocess.TimeoutExpired:p.kill();p.wait();code=124
            finally:
                if p.poll() is None:p.kill();p.wait()
        ledger['events'].append(dict(ledger['active'],returncode=code,wall_seconds=time.time()-tick));ledger['active']=None;publish()
        if code:raise RuntimeError(f'{arm}/{kind} failed exit{code}: {log}')
        return read(result)
    publish()
    try:
        prof=launch('profile',root/'profile',profile_names=['binding_D24','single_D24','motion_D24','binding_D0','single_D0','motion_D0']);agg['profile']=prof
        rows={r['name']:r for r in prof['rows']};costs={}
        for name in cfg['cells']:
            fam,d=name.split('_D');f=int(d)/24;costs[name]={k:(1-f)*rows[fam+'_D0'][k]+f*rows[fam+'_D24'][k] for k in ('seconds','eval_seconds')}
        traincost=1.3*4000*sum(costs[n]['seconds'] for n in cfg['cycle'])/80
        valcost=sum(c['eval_seconds'] for c in costs.values())*6*128/8
        testcost=(sum(c['eval_seconds'] for c in costs.values())+sum(costs[n]['eval_seconds'] for n in costs if n.startswith('binding')))*3*512/8
        reserve=1.6*(valcost+testcost)+360
        agg['config'].update(training_estimate_seconds=traincost,evaluation_reporting_reserve_seconds=reserve,profile_exposure_discarded=True)
        write(root/'fixed_config.json',agg['config']);publish()
        if traincost+reserve>deadline-time.time()-180:raise RuntimeError('Fixed4000 exposure cannot fit; no automatic unequal reduction or renewal')
        baseline=launch('eval',root/'parent_validation',checkpoint=None,split='val',eval_seed=VAL_SEED,n=128,heldout_locations=False);agg['parent_validation']=baseline
        best=assess(baseline,baseline);record.update(selected_checkpoint=str(parent),selected_step=8400,selected_is_parent=True,selected_assessment=best);agg['status']=ledger['status']='training';publish();cp=None
        for target in TARGETS:
            trained=launch('train',root/arm,checkpoint=cp,target=target);cp=trained['checkpoint'];record['training']=trained;publish()
            if trained['step']!=target:raise RuntimeError('Production target interrupted')
            val=launch('eval',root/f'validation_{target:06d}',checkpoint=cp,split='val',eval_seed=VAL_SEED,n=128,heldout_locations=False);decision=assess(val,baseline);val['assessment']=decision;record['validation'].append(val)
            if decision['eligible'] and tuple(decision['rank'])>tuple(best['rank']):best=decision;record.update(selected_checkpoint=cp,selected_step=target,selected_is_parent=False,selected_assessment=best)
            publish()
        record['terminal_checkpoint']=cp
        agg['parent_test']=launch('eval',root/'parent_test',checkpoint=None,split='test',eval_seed=TEST_SEED,n=512,heldout_locations=True)
        record['selected_test']=agg['parent_test'] if record['selected_is_parent'] else launch('eval',root/'selected_test',checkpoint=record['selected_checkpoint'],split='test',eval_seed=TEST_SEED,n=512,heldout_locations=True)
        record['terminal_test']=record['selected_test'] if record['selected_step']==12400 else launch('eval',root/'terminal_test',checkpoint=cp,split='test',eval_seed=TEST_SEED,n=512,heldout_locations=True)
        record['status']=agg['status']=ledger['status']='completed'
    except BaseException as e:agg.update(status='failed',error=repr(e));ledger['status']=agg['status'];raise
    finally:
        agg['wall_seconds']=time.time()-started;publish();write(root/'exit.json',dict(status=agg['status'],error=agg.get('error'),wall_seconds=agg['wall_seconds'],deadline_unix=deadline))
        write(root/'artifact_index.json',{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_index.json'})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--out',required=True);p.add_argument('--parent');p.add_argument('--manifest');a=p.parse_args();run(a.arm,a.out,a.parent,read(a.manifest) if a.manifest else None)
